#!/usr/bin/env python3
from __future__ import annotations

"""Convert a raw recorded RGB-D folder into lightweight browser-viewer JSON.

The viewer format deliberately duplicates some geometry and metadata into small
static JSON files so the HTML viewers can be opened from any simple static file
server without Python dependencies in the browser.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for recording viewer export."""

    parser = argparse.ArgumentParser(
        description="Export a recorded RGB-D folder into static viewer JSON files."
    )
    parser.add_argument("dataset", type=Path, help="Path to the recording folder")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("viewer_data"),
        help="Directory where viewer assets are written",
    )
    parser.add_argument(
        "--point-stride",
        type=int,
        default=6,
        help="Pixel stride for point cloud sampling",
    )
    parser.add_argument(
        "--depth-stride",
        type=int,
        default=4,
        help="Pixel stride for depth preview downsampling",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=1.0,
        help="Ignore depths beyond this distance in meters",
    )
    return parser.parse_args()


def extract_timestamp(path: Path) -> str:
    """Extract the timestamp suffix from a per-frame file path."""

    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected filename format: {path.name}")
    return "_".join(parts[-2:])


def load_json(path: Path) -> dict[str, Any]:
    """Load a UTF-8 encoded JSON dictionary."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_transform(dataset_dir: Path, timestamp: str) -> np.ndarray:
    """Load the best available camera-to-world transform for one frame.

    The exporter checks multiple conventions because the recordings in this
    project evolved over time and not every dataset encodes pose exactly the
    same way.
    """

    for candidate in (
        dataset_dir / f"tf_color_{timestamp}.npz",
        dataset_dir / f"tf_{timestamp}.npz",
    ):
        if candidate.exists():
            with np.load(candidate) as data:
                return np.asarray(data["transform"], dtype=np.float64)

    pose_path = dataset_dir / f"camera_pose_{timestamp}.json"
    if pose_path.exists():
        pose = load_json(pose_path)
        return np.asarray(pose["transform_matrix"], dtype=np.float64)

    raise FileNotFoundError(f"Missing transform for timestamp {timestamp}")


def make_depth_preview(depth: np.ndarray, stride: int, max_depth: float) -> tuple[int, int, list[int]]:
    """Build a compact grayscale depth preview for the viewer side-panel."""

    reduced = depth[::stride, ::stride]
    preview = np.zeros_like(reduced, dtype=np.uint8)
    valid = np.isfinite(reduced) & (reduced > 0.0) & (reduced < max_depth)
    if np.any(valid):
        clipped = reduced[valid]
        lo = float(np.min(clipped))
        hi = float(np.max(clipped))
        if hi - lo < 1e-6:
            scaled = np.full(clipped.shape, 255.0, dtype=np.float32)
        else:
            scaled = 255.0 * (1.0 - ((clipped - lo) / (hi - lo)))
        preview[valid] = scaled.astype(np.uint8)
    height, width = preview.shape
    return width, height, preview.reshape(-1).tolist()


def depth_to_world_points(
    depth: np.ndarray,
    intrinsics: dict[str, Any],
    transform: np.ndarray,
    stride: int,
    max_depth: float,
) -> np.ndarray:
    """Back-project a sampled depth map into world-space points."""

    sampled = depth[::stride, ::stride]
    rows, cols = np.indices(sampled.shape, dtype=np.float32)
    u = cols * stride
    v = rows * stride

    z = sampled
    valid = np.isfinite(z) & (z > 0.0) & (z < max_depth)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])

    z = z[valid]
    u = u[valid]
    v = v[valid]

    # Convert sampled pixels from image space into camera-space metric points.
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    camera_points = np.stack((x, y, z), axis=1)

    # The stored transform is interpreted as camera-to-world, so points can be
    # transformed directly with `R * p + t`.
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    world_points = (camera_points @ rotation.T) + translation
    return world_points.astype(np.float32)


def relative_path(path: Path, root: Path) -> str:
    """Return a POSIX relative path for browser manifests."""

    return path.relative_to(root).as_posix()


def export_dataset(
    dataset_dir: Path,
    output_root: Path,
    point_stride: int,
    depth_stride: int,
    max_depth: float,
) -> Path:
    """Export every frame in a recording folder to the browser manifest format."""

    repo_root = Path.cwd()
    output_dir = output_root / dataset_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_entries: list[dict[str, Any]] = []
    bounds_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    bounds_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    depth_files = sorted(dataset_dir.glob("aligned_depth_to_color_*.npz"))
    if not depth_files:
        raise FileNotFoundError(f"No aligned depth files found in {dataset_dir}")

    for depth_path in depth_files:
        timestamp = extract_timestamp(depth_path)
        color_info_path = dataset_dir / f"color_camera_info_{timestamp}.json"
        image_path = dataset_dir / f"image_{timestamp}.jpg"
        if not color_info_path.exists():
            raise FileNotFoundError(f"Missing color camera info for {timestamp}")
        if not image_path.exists():
            raise FileNotFoundError(f"Missing RGB image for {timestamp}")

        intrinsics = load_json(color_info_path)
        transform = load_transform(dataset_dir, timestamp)

        with np.load(depth_path) as depth_data:
            depth = np.asarray(depth_data["depth"], dtype=np.float32)

        world_points = depth_to_world_points(
            depth=depth,
            intrinsics=intrinsics,
            transform=transform,
            stride=point_stride,
            max_depth=max_depth,
        )

        if world_points.size:
            bounds_min = np.minimum(bounds_min, np.min(world_points, axis=0))
            bounds_max = np.maximum(bounds_max, np.max(world_points, axis=0))

        preview_width, preview_height, preview_pixels = make_depth_preview(
            depth=depth,
            stride=depth_stride,
            max_depth=max_depth,
        )

        valid_depth = depth[np.isfinite(depth) & (depth > 0.0) & (depth < max_depth)]
        frame_json_path = output_dir / f"frame_{timestamp}.json"
        frame_payload = {
            "timestamp": timestamp,
            "image_path": relative_path(image_path, repo_root),
            "camera_position": transform[:3, 3].round(6).tolist(),
            "transform": transform.round(6).tolist(),
            "point_count": int(world_points.shape[0]),
            # Points are stored as a flat xyzxyz... array to keep the manifest
            # simple and cheap to parse in browser JavaScript.
            "points": np.round(world_points, 5).reshape(-1).tolist(),
            "depth_preview": {
                "width": preview_width,
                "height": preview_height,
                "pixels": preview_pixels,
            },
            "depth_stats": {
                "min": None if valid_depth.size == 0 else round(float(np.min(valid_depth)), 4),
                "max": None if valid_depth.size == 0 else round(float(np.max(valid_depth)), 4),
            },
        }
        frame_json_path.write_text(json.dumps(frame_payload), encoding="utf-8")

        frame_entries.append(
            {
                "timestamp": timestamp,
                "image_path": relative_path(image_path, repo_root),
                "frame_path": relative_path(frame_json_path, repo_root),
                "camera_position": transform[:3, 3].round(6).tolist(),
                "point_count": int(world_points.shape[0]),
            }
        )

    bounds = {
        "min": bounds_min.round(5).tolist() if np.isfinite(bounds_min).all() else [-1, -1, -1],
        "max": bounds_max.round(5).tolist() if np.isfinite(bounds_max).all() else [1, 1, 1],
    }
    manifest = {
        "dataset": dataset_dir.name,
        "source_dir": relative_path(dataset_dir, repo_root),
        "point_stride": point_stride,
        "depth_stride": depth_stride,
        "max_depth": max_depth,
        "bounds": bounds,
        "frames": frame_entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def main() -> None:
    """Entry point for recording viewer export."""

    args = parse_args()
    manifest_path = export_dataset(
        dataset_dir=args.dataset.resolve(),
        output_root=args.output_root.resolve(),
        point_stride=args.point_stride,
        depth_stride=args.depth_stride,
        max_depth=args.max_depth,
    )
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()

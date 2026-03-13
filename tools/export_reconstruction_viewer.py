#!/usr/bin/env python3
from __future__ import annotations

"""Export TSDF reconstruction outputs into the browser viewer manifest format."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for TSDF reconstruction viewer export."""

    parser = argparse.ArgumentParser(
        description="Export fused TSDF geometry and recovered camera trajectory for the browser viewer."
    )
    parser.add_argument(
        "reconstruction_dir",
        type=Path,
        help="Directory containing aruco_pose_summary.json and tsdf_pointcloud.ply",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("viewer_data/reconstruction"),
        help="Directory where viewer JSON is written",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=45000,
        help="Maximum number of geometry points exported to the viewer",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Load a UTF-8 JSON file from disk."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compute_bounds(points: np.ndarray) -> dict[str, list[float]]:
    """Compute an axis-aligned bounding box for viewer auto-framing."""

    if points.size == 0:
        return {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]}
    return {
        "min": np.min(points, axis=0).round(5).tolist(),
        "max": np.max(points, axis=0).round(5).tolist(),
    }


def sample_point_cloud(points: np.ndarray, colors: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Downsample a cloud by uniform stride so the browser payload stays manageable."""

    if len(points) <= max_points:
        return points, colors
    stride = int(np.ceil(len(points) / max_points))
    return points[::stride], colors[::stride]


def export_viewer_data(reconstruction_dir: Path, output_root: Path, max_points: int) -> Path:
    """Convert TSDF outputs into a single JSON manifest consumed by the viewer."""

    summary_path = reconstruction_dir / "aruco_pose_summary.json"
    pcd_path = reconstruction_dir / "tsdf_pointcloud.ply"
    if not summary_path.exists() or not pcd_path.exists():
        raise FileNotFoundError("Expected aruco_pose_summary.json and tsdf_pointcloud.ply in reconstruction_dir")

    summary = load_json(summary_path)
    point_cloud = o3d.io.read_point_cloud(str(pcd_path))
    points = np.asarray(point_cloud.points, dtype=np.float32)
    colors = np.asarray(point_cloud.colors, dtype=np.float32)
    if colors.size == 0:
        colors = np.full_like(points, 0.75, dtype=np.float32)

    sampled_points, sampled_colors = sample_point_cloud(points, colors, max_points)

    # The pose summary already stores camera-to-world transforms, so the camera
    # centers are simply the translation components of those matrices.
    trajectory = []
    for pose in summary["poses"]:
        transform = np.asarray(pose["transform_world_from_camera"], dtype=np.float64)
        trajectory.append(
            {
                "timestamp": pose["timestamp"],
                "position": transform[:3, 3].round(6).tolist(),
                "mean_error_m": pose["mean_error_m"],
                "inlier_count": pose["inlier_count"],
                "correspondence_count": pose["correspondence_count"],
            }
        )

    payload = {
        "dataset": Path(summary["dataset"]).name,
        "dictionary": summary["dictionary"],
        "successful_pose_count": summary["successful_pose_count"],
        "geometry_point_count": int(len(points)),
        "viewer_point_count": int(len(sampled_points)),
        "bounds": compute_bounds(points),
        "points": np.round(sampled_points, 5).reshape(-1).tolist(),
        "colors": np.clip(np.round(sampled_colors * 255.0), 0, 255).astype(np.uint8).reshape(-1).tolist(),
        "trajectory": trajectory,
    }

    output_dir = output_root / reconstruction_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "reconstruction_manifest.json"
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    return output_path


def main() -> None:
    """Entry point for TSDF reconstruction viewer export."""

    args = parse_args()
    output_path = export_viewer_data(
        reconstruction_dir=args.reconstruction_dir.resolve(),
        output_root=args.output_root.resolve(),
        max_points=args.max_points,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

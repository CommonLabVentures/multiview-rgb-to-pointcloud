#!/usr/bin/env python3
from __future__ import annotations

"""Export sparse or dense pycolmap geometry into the browser viewer format."""

import argparse
import json
from pathlib import Path

import open3d as o3d
import pycolmap
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for photogrammetry viewer export."""

    parser = argparse.ArgumentParser(
        description="Export a pycolmap reconstruction to the browser viewer format."
    )
    parser.add_argument("photogrammetry_dir", type=Path, help="Directory with summary.json and model/")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("viewer_data/photogrammetry"),
        help="Directory to write viewer manifests",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=30000,
        help="Maximum points for the browser manifest",
    )
    parser.add_argument(
        "--geometry",
        choices=("auto", "sparse", "dense"),
        default="auto",
        help="Which reconstructed geometry to export",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    """Load a UTF-8 JSON file from disk."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compute_bounds(points: np.ndarray) -> dict[str, list[float]]:
    """Compute a viewer-friendly axis-aligned bounding box."""

    if points.size == 0:
        return {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]}
    return {
        "min": np.min(points, axis=0).round(5).tolist(),
        "max": np.max(points, axis=0).round(5).tolist(),
    }


def choose_geometry_path(photogrammetry_dir: Path, geometry: str) -> tuple[Path, str]:
    """Resolve which photogrammetry geometry artifact should be exported."""

    dense_path = photogrammetry_dir / "dense_fused.ply"
    sparse_path = photogrammetry_dir / "sparse_points.ply"
    if geometry == "dense":
        return dense_path, "dense"
    if geometry == "sparse":
        return sparse_path, "sparse"
    if dense_path.exists():
        return dense_path, "dense"
    return sparse_path, "sparse"


def main() -> None:
    """Export a photogrammetry reconstruction into a static browser manifest."""

    args = parse_args()
    photogrammetry_dir = args.photogrammetry_dir.resolve()
    summary = load_json(photogrammetry_dir / "summary.json")
    reconstruction = pycolmap.Reconstruction(str(photogrammetry_dir / "model"))
    geometry_path, geometry_kind = choose_geometry_path(photogrammetry_dir, args.geometry)
    point_cloud = o3d.io.read_point_cloud(str(geometry_path))

    points = np.asarray(point_cloud.points, dtype=np.float32)
    colors = np.asarray(point_cloud.colors, dtype=np.float32)
    if colors.size == 0:
        colors = np.full_like(points, 0.85, dtype=np.float32)

    if len(points) > args.max_points:
        stride = int(np.ceil(len(points) / args.max_points))
        points = points[::stride]
        colors = colors[::stride]

    # pycolmap stores camera centers in world coordinates, so the browser
    # trajectory can be derived directly from each registered image.
    trajectory = []
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.image(image_id)
        center = image.projection_center()
        trajectory.append(
            {
                "timestamp": image.name,
                "position": [float(v) for v in center],
                "mean_error_m": float(summary["mean_reprojection_error"]),
                "inlier_count": int(image.num_points3D),
                "correspondence_count": int(image.num_points2D()),
            }
        )

    trajectory.sort(key=lambda item: item["timestamp"])
    payload = {
        "dataset": Path(summary["dataset"]).name,
        "dictionary": f"RGB photogrammetry ({geometry_kind}, pycolmap)",
        "successful_pose_count": len(trajectory),
        "geometry_kind": geometry_kind,
        "geometry_source": geometry_path.name,
        "geometry_point_count": int(len(np.asarray(point_cloud.points))),
        "viewer_point_count": int(len(points)),
        "bounds": compute_bounds(points),
        "points": np.round(points, 5).reshape(-1).tolist(),
        "colors": np.clip(np.round(colors * 255.0), 0, 255).astype(np.uint8).reshape(-1).tolist(),
        "trajectory": trajectory,
    }

    output_dir = (args.output_root / photogrammetry_dir.name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = "reconstruction_manifest.json" if geometry_kind == "sparse" else "reconstruction_manifest_dense.json"
    output_path = output_dir / output_name
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

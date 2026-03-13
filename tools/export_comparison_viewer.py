#!/usr/bin/env python3
from __future__ import annotations

"""Export a comparison manifest that overlays TSDF and photogrammetry geometry.

The exporter first aligns photogrammetry camera centers to the TSDF trajectory
using a similarity transform, then optionally refines the geometry alignment
with point-to-plane ICP. The resulting static JSON is consumed by the browser
comparison viewer.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import pycolmap


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the comparison viewer export."""

    parser = argparse.ArgumentParser(
        description="Export a combined TSDF vs photogrammetry comparison viewer with automatic trajectory alignment."
    )
    parser.add_argument("tsdf_dir", type=Path, help="Directory containing aruco_pose_summary.json and tsdf_pointcloud.ply")
    parser.add_argument("photogrammetry_dir", type=Path, help="Directory containing summary.json, model/, and optionally dense_fused.ply")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("viewer_data/comparison"),
        help="Directory for the comparison manifest",
    )
    parser.add_argument("--max-tsdf-points", type=int, default=30000)
    parser.add_argument("--max-photo-points", type=int, default=40000)
    parser.add_argument("--photo-geometry", choices=("auto", "dense", "sparse"), default="auto")
    parser.add_argument("--icp-threshold", type=float, default=0.02)
    parser.add_argument("--icp-max-iterations", type=int, default=80)
    parser.add_argument("--icp-normal-radius", type=float, default=0.03)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Load a UTF-8 JSON file from disk."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_timestamp(value: str) -> str:
    """Normalize image names and raw timestamps into a shared comparison key."""

    stem = Path(value).stem
    if stem.startswith("image_"):
        return stem.split("image_", 1)[1]
    return stem


def compute_bounds(point_sets: list[np.ndarray]) -> dict[str, list[float]]:
    """Compute one bounding box that covers every non-empty point set."""

    valid = [pts for pts in point_sets if pts.size]
    if not valid:
        return {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]}
    merged = np.concatenate(valid, axis=0)
    return {
        "min": np.min(merged, axis=0).round(5).tolist(),
        "max": np.max(merged, axis=0).round(5).tolist(),
    }


def sample_cloud(points: np.ndarray, colors: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly stride-sample a point cloud for browser rendering."""

    if len(points) <= max_points:
        return points, colors
    stride = int(np.ceil(len(points) / max_points))
    return points[::stride], colors[::stride]


def choose_photo_geometry(photogrammetry_dir: Path, geometry: str) -> tuple[Path, str]:
    """Resolve which photogrammetry geometry artifact should be aligned and shown."""

    dense = photogrammetry_dir / "dense_fused.ply"
    sparse = photogrammetry_dir / "sparse_points.ply"
    if geometry == "dense":
        return dense, "dense"
    if geometry == "sparse":
        return sparse, "sparse"
    if dense.exists():
        return dense, "dense"
    return sparse, "sparse"


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate a similarity transform with Umeyama alignment.

    `src` and `dst` must already be ordered so row `i` refers to the same frame
    in both trajectories.
    """

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    covariance = (dst_centered.T @ src_centered) / len(src)
    u_matrix, singular_values, v_transpose = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(u_matrix) * np.linalg.det(v_transpose) < 0:
        sign[-1, -1] = -1
    rotation = u_matrix @ sign @ v_transpose
    src_variance = np.mean(np.sum(src_centered ** 2, axis=1))
    scale = float(np.trace(np.diag(singular_values) @ sign) / src_variance)
    translation = dst_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def apply_similarity(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Apply a similarity transform to a point set."""

    if points.size == 0:
        return points
    return (scale * (points @ rotation.T)) + translation


def similarity_to_transform(scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Pack a similarity transform into a homogeneous 4x4 matrix."""

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    return transform


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply the affine part of a homogeneous 4x4 transform to a point set."""

    if points.size == 0:
        return points
    return (points @ transform[:3, :3].T) + transform[:3, 3]


def run_icp_refinement(
    source_points: np.ndarray,
    target_points: np.ndarray,
    threshold: float,
    max_iterations: int,
    normal_radius: float,
) -> o3d.pipelines.registration.RegistrationResult:
    """Refine a cloud alignment with Open3D point-to-plane ICP."""

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_points)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_points)

    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
    )
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
    )

    # ICP is run after the trajectory-based similarity estimate so the optimizer
    # only needs to solve a small residual rigid correction.
    return o3d.pipelines.registration.registration_icp(
        source=source,
        target=target,
        max_correspondence_distance=threshold,
        init=np.eye(4, dtype=np.float64),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations),
    )


def main() -> None:
    """Build the combined comparison manifest used by `viewer/compare.html`."""

    args = parse_args()
    tsdf_dir = args.tsdf_dir.resolve()
    photogrammetry_dir = args.photogrammetry_dir.resolve()

    tsdf_summary = load_json(tsdf_dir / "aruco_pose_summary.json")
    tsdf_cloud = o3d.io.read_point_cloud(str(tsdf_dir / "tsdf_pointcloud.ply"))
    photo_summary = load_json(photogrammetry_dir / "summary.json")
    reconstruction = pycolmap.Reconstruction(str(photogrammetry_dir / "model"))
    photo_geometry_path, photo_geometry_kind = choose_photo_geometry(photogrammetry_dir, args.photo_geometry)
    photo_cloud = o3d.io.read_point_cloud(str(photo_geometry_path))

    tsdf_traj = {
        pose["timestamp"]: np.asarray(np.asarray(pose["transform_world_from_camera"])[:3, 3], dtype=np.float64)
        for pose in tsdf_summary["poses"]
    }
    photo_traj = {
        normalize_timestamp(reconstruction.image(image_id).name): np.asarray(
            reconstruction.image(image_id).projection_center(), dtype=np.float64
        )
        for image_id in reconstruction.reg_image_ids()
    }

    common = sorted(set(tsdf_traj) & set(photo_traj))
    if len(common) < 3:
        raise RuntimeError("Need at least 3 shared timestamps to align photogrammetry and TSDF trajectories")

    # First align the two camera trajectories with a scale-aware similarity fit.
    src = np.asarray([photo_traj[key] for key in common], dtype=np.float64)
    dst = np.asarray([tsdf_traj[key] for key in common], dtype=np.float64)
    scale, rotation, translation = umeyama_similarity(src, dst)
    aligned_traj = apply_similarity(src, scale, rotation, translation)
    rms_error = float(np.sqrt(np.mean(np.sum((aligned_traj - dst) ** 2, axis=1))))

    tsdf_points = np.asarray(tsdf_cloud.points, dtype=np.float32)
    tsdf_colors = np.asarray(tsdf_cloud.colors, dtype=np.float32)
    if tsdf_colors.size == 0:
        tsdf_colors = np.tile(np.array([[0.85, 0.47, 0.22]], dtype=np.float32), (len(tsdf_points), 1))

    photo_points = np.asarray(photo_cloud.points, dtype=np.float32)
    photo_colors = np.asarray(photo_cloud.colors, dtype=np.float32)
    if photo_colors.size == 0:
        photo_colors = np.tile(np.array([[0.13, 0.48, 0.73]], dtype=np.float32), (len(photo_points), 1))
    init_transform = similarity_to_transform(scale, rotation, translation)
    aligned_photo_points = apply_transform(photo_points.astype(np.float64), init_transform)

    # Then use geometry to tighten the alignment in the already matched frame.
    icp_result = run_icp_refinement(
        source_points=aligned_photo_points,
        target_points=tsdf_points.astype(np.float64),
        threshold=args.icp_threshold,
        max_iterations=args.icp_max_iterations,
        normal_radius=args.icp_normal_radius,
    )
    final_transform = icp_result.transformation @ init_transform
    refined_photo_points = apply_transform(photo_points.astype(np.float64), final_transform).astype(np.float32)

    tsdf_points, tsdf_colors = sample_cloud(tsdf_points, tsdf_colors, args.max_tsdf_points)
    refined_photo_points, photo_colors = sample_cloud(refined_photo_points, photo_colors, args.max_photo_points)

    trajectory = []
    for key in common:
        photo_pose_init = apply_transform(photo_traj[key][None, :], init_transform)[0]
        photo_pose_refined = apply_transform(photo_traj[key][None, :], final_transform)[0]
        trajectory.append(
            {
                "timestamp": key,
                "tsdf_position": tsdf_traj[key].round(6).tolist(),
                "photo_position_aligned": photo_pose_init.round(6).tolist(),
                "photo_position_icp": photo_pose_refined.round(6).tolist(),
            }
        )

    payload = {
        "dataset": Path(tsdf_summary["dataset"]).name,
        "comparison": {
            "shared_timestamps": len(common),
            "photo_geometry_kind": photo_geometry_kind,
            "photo_geometry_source": photo_geometry_path.name,
            "trajectory_rms_error_m": round(rms_error, 6),
            "photo_to_tsdf_similarity": {
                "scale": scale,
                "rotation": np.round(rotation, 8).tolist(),
                "translation": np.round(translation, 8).tolist(),
            },
            "icp": {
                "threshold_m": args.icp_threshold,
                "max_iterations": args.icp_max_iterations,
                "fitness": round(float(icp_result.fitness), 6),
                "inlier_rmse_m": round(float(icp_result.inlier_rmse), 6),
                "delta_transform": np.round(icp_result.transformation, 8).tolist(),
                "photo_to_tsdf_refined_transform": np.round(final_transform, 8).tolist(),
            },
        },
        "bounds": compute_bounds([tsdf_points, refined_photo_points]),
        "tsdf": {
            "point_count": int(len(tsdf_points)),
            "points": np.round(tsdf_points, 5).reshape(-1).tolist(),
            "colors": np.clip(np.round(tsdf_colors * 255.0), 0, 255).astype(np.uint8).reshape(-1).tolist(),
        },
        "photogrammetry": {
            "point_count": int(len(refined_photo_points)),
            "points": np.round(refined_photo_points, 5).reshape(-1).tolist(),
            "colors": np.clip(np.round(photo_colors * 255.0), 0, 255).astype(np.uint8).reshape(-1).tolist(),
        },
        "trajectory": trajectory,
    }

    output_dir = (args.output_root / Path(tsdf_summary["dataset"]).name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "compare_manifest.json"
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

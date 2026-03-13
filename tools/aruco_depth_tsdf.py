#!/usr/bin/env python3
from __future__ import annotations

"""Recover camera poses from ArUco corners plus aligned depth and fuse a TSDF model.

This script implements the geometry-first reconstruction path in this repository:

1. Read RGB images, aligned depth maps, and per-frame color intrinsics from a
   recorded RealSense-like dataset folder.
2. Detect ArUco corners in each color image.
3. Back-project the corner pixels into 3D using the aligned depth map.
4. Use repeated marker corners as frame-to-frame 3D correspondences.
5. Estimate a rigid transform from each camera frame into the first frame using
   a minimal-sample RANSAC fit followed by closed-form refinement.
6. Integrate all successful RGB-D frames into an Open3D TSDF volume.

The result is a fused mesh, fused point cloud, and a JSON summary that captures
pose diagnostics for downstream inspection and browser export.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d


COMMON_ARUCO_DICTIONARIES = [
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_4X4_250",
    "DICT_4X4_1000",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_5X5_250",
    "DICT_5X5_1000",
    "DICT_6X6_50",
    "DICT_6X6_100",
    "DICT_6X6_250",
    "DICT_6X6_1000",
    "DICT_7X7_50",
    "DICT_7X7_100",
    "DICT_7X7_250",
    "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
    "DICT_APRILTAG_16h5",
    "DICT_APRILTAG_25h9",
    "DICT_APRILTAG_36h10",
    "DICT_APRILTAG_36h11",
]


@dataclass
class FrameData:
    """Bundle together the files required to process one RGB-D frame."""

    timestamp: str
    image_path: Path
    depth_path: Path
    camera_info_path: Path


@dataclass
class PoseEstimate:
    """Pose estimate and quality metrics for one frame in the reconstructed world."""

    timestamp: str
    transform_world_from_camera: np.ndarray
    correspondence_count: int
    inlier_count: int
    mean_error_m: float
    median_error_m: float


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the ArUco-plus-depth TSDF pipeline."""

    parser = argparse.ArgumentParser(
        description="Estimate camera poses from ArUco corners plus aligned depth and fuse a TSDF."
    )
    parser.add_argument("dataset", type=Path, help="Folder containing image/depth/camera info files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to write poses and reconstruction artifacts",
    )
    parser.add_argument(
        "--dictionary",
        type=str,
        default="auto",
        help="OpenCV aruco dictionary name, or 'auto'",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=0.9,
        help="Ignore depth beyond this distance in meters",
    )
    parser.add_argument(
        "--depth-patch-radius",
        type=int,
        default=2,
        help="Radius around each ArUco corner for robust depth lookup",
    )
    parser.add_argument(
        "--point-match-threshold",
        type=float,
        default=0.01,
        help="RANSAC inlier threshold in meters for 3D corner registration",
    )
    parser.add_argument(
        "--min-correspondences",
        type=int,
        default=12,
        help="Minimum 3D corner correspondences required to estimate a pose",
    )
    parser.add_argument(
        "--ransac-iters",
        type=int,
        default=300,
        help="RANSAC iterations for rigid alignment",
    )
    parser.add_argument(
        "--voxel-length",
        type=float,
        default=0.002,
        help="TSDF voxel size in meters",
    )
    parser.add_argument(
        "--sdf-trunc",
        type=float,
        default=0.01,
        help="TSDF truncation distance in meters",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=0.05,
        help="Ignore depth below this distance in meters",
    )
    return parser.parse_args()


def extract_timestamp(path: Path) -> str:
    """Extract the `<date>_<time>` suffix shared by all per-frame files."""

    parts = path.stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {path.name}")
    return "_".join(parts[-2:])


def list_frames(dataset: Path) -> list[FrameData]:
    """Enumerate usable RGB-D frames by matching image, depth, and intrinsics files."""

    frames: list[FrameData] = []
    for image_path in sorted(dataset.glob("image_*.jpg")):
        timestamp = extract_timestamp(image_path)
        depth_path = dataset / f"aligned_depth_to_color_{timestamp}.npz"
        camera_info_path = dataset / f"color_camera_info_{timestamp}.json"
        if not depth_path.exists() or not camera_info_path.exists():
            continue
        frames.append(
            FrameData(
                timestamp=timestamp,
                image_path=image_path,
                depth_path=depth_path,
                camera_info_path=camera_info_path,
            )
        )
    if not frames:
        raise FileNotFoundError(f"No frames found in {dataset}")
    return frames


def load_json(path: Path) -> dict[str, Any]:
    """Load a UTF-8 encoded JSON file into a dictionary."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_dictionary_name(frames: list[FrameData], sample_count: int = 3) -> str:
    """Choose the ArUco dictionary that yields the strongest early-frame detections.

    The heuristic is intentionally simple: evaluate a shortlist of common
    dictionaries on the first few frames, then choose the one with the largest
    total detection count, breaking ties by the weakest per-frame result.
    """

    candidate_frames = frames[:sample_count]
    best_name = None
    best_score = -1
    best_min_count = -1

    for name in COMMON_ARUCO_DICTIONARIES:
        # Recreate the detector for each candidate dictionary so detection counts
        # are comparable across the same image subset.
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        counts = []
        for frame in candidate_frames:
            image = cv2.imread(str(frame.image_path), cv2.IMREAD_GRAYSCALE)
            corners, ids, _ = detector.detectMarkers(image)
            counts.append(0 if ids is None else len(corners))
        score = sum(counts)
        min_count = min(counts)
        if score > best_score or (score == best_score and min_count > best_min_count):
            best_name = name
            best_score = score
            best_min_count = min_count

    if best_name is None or best_score <= 0:
        raise RuntimeError("Failed to infer an ArUco dictionary from the dataset")
    return best_name


def make_detector(dictionary_name: str) -> cv2.aruco.ArucoDetector:
    """Create a detector with sub-pixel corner refinement enabled."""

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def backproject_pixel(
    depth: np.ndarray,
    intrinsics: dict[str, float],
    u: float,
    v: float,
    patch_radius: int,
    min_depth: float,
    max_depth: float,
) -> np.ndarray | None:
    """Turn one image pixel into a 3D camera-space point using local depth.

    A small pixel neighborhood is sampled and summarized with the median depth so
    the reconstruction is less sensitive to holes or one-pixel depth spikes.
    """

    x0 = max(0, int(round(u)) - patch_radius)
    x1 = min(depth.shape[1], int(round(u)) + patch_radius + 1)
    y0 = max(0, int(round(v)) - patch_radius)
    y1 = min(depth.shape[0], int(round(v)) + patch_radius + 1)
    patch = depth[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch >= min_depth) & (patch <= max_depth)]
    if valid.size == 0:
        return None

    z = float(np.median(valid))
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.asarray([x, y, z], dtype=np.float64)


def detect_marker_corners_3d(
    frame: FrameData,
    detector: cv2.aruco.ArucoDetector,
    patch_radius: int,
    min_depth: float,
    max_depth: float,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[str, Any]]:
    """Detect ArUco markers and recover 3D corner locations for one frame.

    The returned point dictionary uses `(marker_id, corner_index)` keys so the
    same physical marker corner can be matched across frames without solving a
    separate data-association problem.
    """

    color_image = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
    grayscale = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(grayscale)
    intrinsics = load_json(frame.camera_info_path)
    with np.load(frame.depth_path) as npz:
        depth = np.asarray(npz["depth"], dtype=np.float32)

    points: dict[tuple[int, int], np.ndarray] = {}
    detected_marker_count = 0 if ids is None else int(len(ids))
    if ids is not None:
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            marker_id_int = int(marker_id)
            for corner_index, corner in enumerate(marker_corners.reshape(-1, 2)):
                point = backproject_pixel(
                    depth=depth,
                    intrinsics=intrinsics,
                    u=float(corner[0]),
                    v=float(corner[1]),
                    patch_radius=patch_radius,
                    min_depth=min_depth,
                    max_depth=max_depth,
                )
                if point is not None:
                    points[(marker_id_int, corner_index)] = point

    metadata = {
        "timestamp": frame.timestamp,
        "image_path": str(frame.image_path),
        "marker_count": detected_marker_count,
        "corner_3d_count": len(points),
        "rejected_count": len(rejected),
    }
    return points, metadata


def rigid_transform(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """Solve the least-squares rigid transform from `src_points` to `dst_points`.

    This is a standard SVD-based absolute orientation solve with the usual
    determinant check to prevent unintended reflections.
    """

    src_centroid = src_points.mean(axis=0)
    dst_centroid = dst_points.mean(axis=0)
    covariance = (src_points - src_centroid).T @ (dst_points - dst_centroid)
    u_matrix, _, v_transpose = np.linalg.svd(covariance)
    rotation = v_transpose.T @ u_matrix.T
    if np.linalg.det(rotation) < 0:
        v_transpose[-1, :] *= -1
        rotation = v_transpose.T @ u_matrix.T

    translation = dst_centroid - rotation @ src_centroid
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def ransac_rigid_transform(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    threshold_m: float,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robustly estimate a rigid transform using 3-point RANSAC samples."""

    if len(src_points) < 3:
        raise ValueError("Need at least 3 correspondences for rigid transform estimation")

    rng = np.random.default_rng(0)
    indices = np.arange(len(src_points))
    best_inliers = None
    best_transform = None

    for _ in range(iterations):
        # Three points are the minimal sample for a non-degenerate rigid 3D fit.
        sample = rng.choice(indices, size=3, replace=False)
        candidate = rigid_transform(src_points[sample], dst_points[sample])
        predicted = (src_points @ candidate[:3, :3].T) + candidate[:3, 3]
        errors = np.linalg.norm(predicted - dst_points, axis=1)
        inliers = errors <= threshold_m
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_transform = candidate

    if best_inliers is None or int(best_inliers.sum()) < 3 or best_transform is None:
        raise RuntimeError("RANSAC failed to find a valid rigid transform")

    # Refit on the consensus set so the final estimate uses all inliers, not just
    # the minimal sample that happened to score best.
    refined = rigid_transform(src_points[best_inliers], dst_points[best_inliers])
    predicted = (src_points @ refined[:3, :3].T) + refined[:3, 3]
    errors = np.linalg.norm(predicted - dst_points, axis=1)
    return refined, best_inliers, errors


def estimate_poses(
    frames: list[FrameData],
    detector: cv2.aruco.ArucoDetector,
    patch_radius: int,
    min_depth: float,
    max_depth: float,
    min_correspondences: int,
    threshold_m: float,
    ransac_iters: int,
) -> tuple[list[PoseEstimate], list[dict[str, Any]]]:
    """Estimate per-frame poses relative to the first valid RGB-D frame.

    The first frame defines the project-local world frame. Each subsequent frame
    is aligned directly to that reference using repeated 3D marker corners.
    Diagnostics for both successful and rejected frames are returned so the
    caller can inspect why individual frames failed.
    """

    frame_points: dict[str, dict[tuple[int, int], np.ndarray]] = {}
    diagnostics: list[dict[str, Any]] = []
    for frame in frames:
        points, metadata = detect_marker_corners_3d(
            frame=frame,
            detector=detector,
            patch_radius=patch_radius,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        frame_points[frame.timestamp] = points
        diagnostics.append(metadata)

    reference_timestamp = frames[0].timestamp
    reference_points = frame_points[reference_timestamp]
    if len(reference_points) < min_correspondences:
        raise RuntimeError(
            f"Reference frame {reference_timestamp} only has {len(reference_points)} usable 3D marker corners"
        )

    poses: list[PoseEstimate] = [
        PoseEstimate(
            timestamp=reference_timestamp,
            transform_world_from_camera=np.eye(4, dtype=np.float64),
            correspondence_count=len(reference_points),
            inlier_count=len(reference_points),
            mean_error_m=0.0,
            median_error_m=0.0,
        )
    ]

    diagnostic_index = {item["timestamp"]: item for item in diagnostics}
    diagnostic_index[reference_timestamp]["status"] = "ok"
    diagnostic_index[reference_timestamp]["correspondence_count"] = len(reference_points)
    diagnostic_index[reference_timestamp]["inlier_count"] = len(reference_points)
    diagnostic_index[reference_timestamp]["mean_error_m"] = 0.0
    diagnostic_index[reference_timestamp]["median_error_m"] = 0.0

    for frame in frames[1:]:
        current_points = frame_points[frame.timestamp]
        common_keys = sorted(set(reference_points) & set(current_points))
        if len(common_keys) < min_correspondences:
            # Keep a record of the failure mode so downstream debugging can
            # distinguish "marker not seen" from "registration was unstable".
            diagnostic_index[frame.timestamp]["status"] = "insufficient_correspondences"
            diagnostic_index[frame.timestamp]["correspondence_count"] = len(common_keys)
            continue

        src = np.asarray([current_points[key] for key in common_keys], dtype=np.float64)
        dst = np.asarray([reference_points[key] for key in common_keys], dtype=np.float64)
        try:
            transform, inliers, errors = ransac_rigid_transform(
                src_points=src,
                dst_points=dst,
                threshold_m=threshold_m,
                iterations=ransac_iters,
            )
        except RuntimeError:
            diagnostic_index[frame.timestamp]["status"] = "ransac_failed"
            diagnostic_index[frame.timestamp]["correspondence_count"] = len(common_keys)
            continue

        pose = PoseEstimate(
            timestamp=frame.timestamp,
            transform_world_from_camera=transform,
            correspondence_count=len(common_keys),
            inlier_count=int(inliers.sum()),
            mean_error_m=float(np.mean(errors)),
            median_error_m=float(np.median(errors)),
        )
        poses.append(pose)
        diagnostic_index[frame.timestamp]["status"] = "ok"
        diagnostic_index[frame.timestamp]["correspondence_count"] = pose.correspondence_count
        diagnostic_index[frame.timestamp]["inlier_count"] = pose.inlier_count
        diagnostic_index[frame.timestamp]["mean_error_m"] = pose.mean_error_m
        diagnostic_index[frame.timestamp]["median_error_m"] = pose.median_error_m

    return poses, diagnostics


def load_rgbd(
    frame: FrameData,
    min_depth: float,
    max_depth: float,
) -> tuple[o3d.geometry.RGBDImage, o3d.camera.PinholeCameraIntrinsic]:
    """Load one RGB-D frame in the format Open3D expects for TSDF integration."""

    intrinsics = load_json(frame.camera_info_path)
    color_bgr = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    with np.load(frame.depth_path) as npz:
        depth = np.asarray(npz["depth"], dtype=np.float32)

    depth = depth.copy()
    # Open3D treats zero depth as invalid, so clamp all out-of-range values there
    # before integration.
    invalid = ~np.isfinite(depth) | (depth < min_depth) | (depth > max_depth)
    depth[invalid] = 0.0

    color_o3d = o3d.geometry.Image(color_rgb)
    depth_o3d = o3d.geometry.Image(depth)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=color_o3d,
        depth=depth_o3d,
        depth_scale=1.0,
        depth_trunc=max_depth,
        convert_rgb_to_intensity=False,
    )
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=int(intrinsics["width"]),
        height=int(intrinsics["height"]),
        fx=float(intrinsics["fx"]),
        fy=float(intrinsics["fy"]),
        cx=float(intrinsics["cx"]),
        cy=float(intrinsics["cy"]),
    )
    return rgbd, intrinsic


def integrate_tsdf(
    frames: list[FrameData],
    poses: list[PoseEstimate],
    min_depth: float,
    max_depth: float,
    voxel_length: float,
    sdf_trunc: float,
) -> tuple[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud]:
    """Fuse all successfully posed frames into a shared TSDF volume."""

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    frame_by_timestamp = {frame.timestamp: frame for frame in frames}
    for pose in poses:
        frame = frame_by_timestamp[pose.timestamp]
        rgbd, intrinsic = load_rgbd(frame=frame, min_depth=min_depth, max_depth=max_depth)
        # `transform_world_from_camera` maps camera-space points into the world
        # frame, but Open3D integration expects the extrinsic as world-to-camera.
        world_to_camera = np.linalg.inv(pose.transform_world_from_camera)
        volume.integrate(rgbd, intrinsic, world_to_camera)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    point_cloud = volume.extract_point_cloud()
    return mesh, point_cloud


def save_pose_summary(
    output_dir: Path,
    dataset: Path,
    dictionary_name: str,
    poses: list[PoseEstimate],
    diagnostics: list[dict[str, Any]],
) -> Path:
    """Persist pose estimates and frame diagnostics for downstream tooling."""

    payload = {
        "dataset": str(dataset),
        "dictionary": dictionary_name,
        "reference_timestamp": poses[0].timestamp if poses else None,
        "successful_pose_count": len(poses),
        "poses": [
            {
                "timestamp": pose.timestamp,
                "correspondence_count": pose.correspondence_count,
                "inlier_count": pose.inlier_count,
                "mean_error_m": round(pose.mean_error_m, 6),
                "median_error_m": round(pose.median_error_m, 6),
                "transform_world_from_camera": np.round(
                    pose.transform_world_from_camera, 8
                ).tolist(),
            }
            for pose in poses
        ],
        "frame_diagnostics": diagnostics,
    }
    summary_path = output_dir / "aruco_pose_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path


def main() -> None:
    """Run the full ArUco depth registration and TSDF fusion pipeline."""

    args = parse_args()
    dataset = args.dataset.resolve()
    output_dir = (args.output_dir / dataset.name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = list_frames(dataset)
    dictionary_name = (
        infer_dictionary_name(frames) if args.dictionary == "auto" else args.dictionary
    )
    detector = make_detector(dictionary_name)

    poses, diagnostics = estimate_poses(
        frames=frames,
        detector=detector,
        patch_radius=args.depth_patch_radius,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        min_correspondences=args.min_correspondences,
        threshold_m=args.point_match_threshold,
        ransac_iters=args.ransac_iters,
    )
    if len(poses) < 2:
        raise RuntimeError("Pose estimation succeeded for too few frames to build a reconstruction")

    mesh, point_cloud = integrate_tsdf(
        frames=frames,
        poses=poses,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        voxel_length=args.voxel_length,
        sdf_trunc=args.sdf_trunc,
    )

    mesh_path = output_dir / "tsdf_mesh.ply"
    pcd_path = output_dir / "tsdf_pointcloud.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    o3d.io.write_point_cloud(str(pcd_path), point_cloud)
    summary_path = save_pose_summary(
        output_dir=output_dir,
        dataset=dataset,
        dictionary_name=dictionary_name,
        poses=poses,
        diagnostics=diagnostics,
    )

    print(f"Dictionary: {dictionary_name}")
    print(f"Frames found: {len(frames)}")
    print(f"Frames fused: {len(poses)}")
    print(f"Pose summary: {summary_path}")
    print(f"Mesh: {mesh_path}")
    print(f"Point cloud: {pcd_path}")


if __name__ == "__main__":
    main()

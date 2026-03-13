#!/usr/bin/env python3
from __future__ import annotations

"""Generate tracked README preview assets from the sample dataset and local outputs.

The script produces:

1. A 2x2 contact sheet of representative RGB frames from the tracked sample
   dataset.
2. A side-by-side reconstruction preview where the TSDF and dense
   photogrammetry clouds are aligned into the same frame, normalized by the same
   scale, and rendered from the same viewpoint for easy visual comparison.

The photogrammetry-to-TSDF alignment reuses the same trajectory-similarity plus
ICP strategy that powers the browser comparison exporter.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pycolmap
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

from export_comparison_viewer import (
    apply_transform,
    choose_photo_geometry,
    normalize_timestamp,
    run_icp_refinement,
    similarity_to_transform,
    umeyama_similarity,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for README preview generation."""

    parser = argparse.ArgumentParser(
        description="Generate README preview images for the tracked sample dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/sample_eye1_2026-03-10A"),
        help="Tracked sample dataset folder containing image_*.jpg",
    )
    parser.add_argument(
        "--tsdf-dir",
        type=Path,
        default=Path("outputs/record_eye1__2026-03-10A"),
        help="Local TSDF output directory containing aruco_pose_summary.json and tsdf_pointcloud.ply",
    )
    parser.add_argument(
        "--photogrammetry-dir",
        type=Path,
        default=Path("outputs_photogrammetry/record_eye1__2026-03-10A"),
        help="Local photogrammetry output directory containing summary.json, model/, and dense_fused.ply",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("docs/assets"),
        help="Destination for generated README assets",
    )
    parser.add_argument(
        "--photo-geometry",
        choices=("auto", "dense", "sparse"),
        default="auto",
        help="Which photogrammetry geometry artifact to render",
    )
    parser.add_argument(
        "--max-tsdf-points",
        type=int,
        default=80000,
        help="Maximum TSDF points rendered in the comparison image",
    )
    parser.add_argument(
        "--max-photo-points",
        type=int,
        default=80000,
        help="Maximum photogrammetry points rendered in the comparison image",
    )
    parser.add_argument(
        "--icp-threshold",
        type=float,
        default=0.02,
        help="ICP correspondence threshold in meters",
    )
    parser.add_argument(
        "--icp-max-iterations",
        type=int,
        default=80,
        help="Maximum point-to-plane ICP iterations",
    )
    parser.add_argument(
        "--icp-normal-radius",
        type=float,
        default=0.03,
        help="Neighborhood radius used when estimating normals for ICP",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    """Load a UTF-8 JSON document from disk."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sample_points(points: np.ndarray, colors: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly stride-sample a colored point cloud for plotting."""

    if len(points) <= max_points:
        return points, colors
    stride = int(np.ceil(len(points) / max_points))
    return points[::stride], colors[::stride]


def make_rgb_contact_sheet(dataset_dir: Path, output_path: Path, dataset_label: str) -> None:
    """Create a 2x2 contact sheet from representative timestamps in the sample dataset."""

    frame_names = [
        "image_20260310_171557.jpg",
        "image_20260310_171629.jpg",
        "image_20260310_171707.jpg",
        "image_20260310_171803.jpg",
    ]
    thumb_w = 520
    thumb_h = 294
    pad = 20
    header_h = 70
    label_h = 42
    canvas_w = pad * 3 + thumb_w * 2
    canvas_h = header_h + pad * 3 + (thumb_h + label_h) * 2
    canvas = Image.new("RGB", (canvas_w, canvas_h), (247, 245, 240))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((pad, 20), "Tracked sample RGB frames", fill=(28, 28, 28), font=font)
    draw.text((pad, 40), dataset_label, fill=(90, 90, 90), font=font)

    for idx, name in enumerate(frame_names):
        image = Image.open(dataset_dir / name).convert("RGB")
        thumb = image.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        col = idx % 2
        row = idx // 2
        x = pad + col * (thumb_w + pad)
        y = header_h + pad + row * (thumb_h + label_h + pad)
        canvas.paste(thumb, (x, y))
        timestamp = name.removeprefix("image_").removesuffix(".jpg")
        draw.rectangle((x, y + thumb_h, x + thumb_w, y + thumb_h + label_h), fill=(255, 255, 255))
        draw.text((x + 10, y + thumb_h + 12), timestamp, fill=(55, 55, 55), font=font)

    canvas.save(output_path, quality=92)


def align_photogrammetry_to_tsdf(
    tsdf_dir: Path,
    photogrammetry_dir: Path,
    photo_geometry: str,
    icp_threshold: float,
    icp_max_iterations: int,
    icp_normal_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load local outputs and align the photogrammetry geometry into the TSDF frame."""

    tsdf_summary = load_json(tsdf_dir / "aruco_pose_summary.json")
    tsdf_cloud = o3d.io.read_point_cloud(str(tsdf_dir / "tsdf_pointcloud.ply"))
    reconstruction = pycolmap.Reconstruction(str(photogrammetry_dir / "model"))
    photo_geometry_path, _ = choose_photo_geometry(photogrammetry_dir, photo_geometry)
    photo_cloud = o3d.io.read_point_cloud(str(photo_geometry_path))

    tsdf_traj = {
        pose["timestamp"]: np.asarray(pose["transform_world_from_camera"], dtype=np.float64)[:3, 3]
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
        raise RuntimeError("Need at least 3 shared timestamps to align photogrammetry and TSDF outputs")

    src = np.asarray([photo_traj[key] for key in common], dtype=np.float64)
    dst = np.asarray([tsdf_traj[key] for key in common], dtype=np.float64)
    scale, rotation, translation = umeyama_similarity(src, dst)
    init_transform = similarity_to_transform(scale, rotation, translation)

    tsdf_points = np.asarray(tsdf_cloud.points, dtype=np.float64)
    tsdf_colors = np.asarray(tsdf_cloud.colors, dtype=np.float32)
    if tsdf_colors.size == 0:
        tsdf_colors = np.tile(np.array([[0.85, 0.47, 0.22]], dtype=np.float32), (len(tsdf_points), 1))

    photo_points = np.asarray(photo_cloud.points, dtype=np.float64)
    photo_colors = np.asarray(photo_cloud.colors, dtype=np.float32)
    if photo_colors.size == 0:
        photo_colors = np.tile(np.array([[0.13, 0.48, 0.73]], dtype=np.float32), (len(photo_points), 1))

    aligned_photo_points = apply_transform(photo_points, init_transform)
    icp_result = run_icp_refinement(
        source_points=aligned_photo_points,
        target_points=tsdf_points,
        threshold=icp_threshold,
        max_iterations=icp_max_iterations,
        normal_radius=icp_normal_radius,
    )
    final_transform = icp_result.transformation @ init_transform
    refined_photo_points = apply_transform(photo_points, final_transform)

    return (
        tsdf_points.astype(np.float32),
        tsdf_colors,
        refined_photo_points.astype(np.float32),
        photo_colors,
    )


def compute_display_transform(
    tsdf_points: np.ndarray,
    photo_points: np.ndarray,
    camera_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive a stable shared render frame from the aligned reconstruction geometry.

    The TSDF cloud defines the reference PCA basis. The smallest-variance axis is
    treated as the board normal, and its sign is chosen so the camera trajectory
    sits above the board rather than below it.
    """

    combined = np.concatenate([tsdf_points, photo_points], axis=0).astype(np.float64)
    center = combined.mean(axis=0)
    centered_tsdf = tsdf_points.astype(np.float64) - center
    covariance = np.cov(centered_tsdf.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    basis = eigenvectors[:, order]

    # Flip the plane normal so the cameras sit on the positive side of the board.
    centered_cameras = camera_positions.astype(np.float64) - center
    if np.mean(centered_cameras @ basis[:, 2]) < 0:
        basis[:, 2] *= -1

    # Keep a right-handed basis after any sign flips.
    basis[:, 1] = np.cross(basis[:, 2], basis[:, 0])
    basis[:, 1] /= np.linalg.norm(basis[:, 1])
    basis[:, 0] = np.cross(basis[:, 1], basis[:, 2])
    basis[:, 0] /= np.linalg.norm(basis[:, 0])
    rotation = basis
    return center, rotation


def transform_for_display(points: np.ndarray, center: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Apply the shared display transform to a point set."""

    return (points.astype(np.float64) - center) @ rotation


def crop_to_limits(points: np.ndarray, colors: np.ndarray, limits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the points that fall inside the shared display crop."""

    mask = np.all((points >= limits[:, 0]) & (points <= limits[:, 1]), axis=1)
    return points[mask], colors[mask]


def crop_tsdf_to_photo_support(
    tsdf_points: np.ndarray,
    tsdf_colors: np.ndarray,
    photo_points: np.ndarray,
    support_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop TSDF points that are far from the photogrammetry-supported region."""

    tree = cKDTree(photo_points)
    distances, _ = tree.query(tsdf_points, k=1, workers=-1)
    mask = distances <= support_radius
    return tsdf_points[mask], tsdf_colors[mask]


def render_comparison_preview(
    tsdf_points: np.ndarray,
    tsdf_colors: np.ndarray,
    photo_points: np.ndarray,
    photo_colors: np.ndarray,
    output_path: Path,
    max_tsdf_points: int,
    max_photo_points: int,
    camera_positions: np.ndarray,
) -> None:
    """Render TSDF and aligned photogrammetry previews with matched pose and scale."""

    tsdf_points, tsdf_colors = sample_points(tsdf_points, tsdf_colors, max_tsdf_points)
    photo_points, photo_colors = sample_points(photo_points, photo_colors, max_photo_points)

    center, rotation = compute_display_transform(
        tsdf_points=tsdf_points,
        photo_points=photo_points,
        camera_positions=camera_positions,
    )
    tsdf_display = transform_for_display(tsdf_points, center, rotation)
    photo_display = transform_for_display(photo_points, center, rotation)

    # Use the aligned photogrammetry cloud to set the zoom level. It is already
    # focused on the board/object region, whereas the TSDF often includes a much
    # larger swath of surrounding table or floor that would make the interesting
    # geometry appear tiny in the README.
    mins = np.quantile(photo_display, 0.05, axis=0)
    maxs = np.quantile(photo_display, 0.95, axis=0)
    span = np.max(maxs - mins)
    midpoint = 0.5 * (mins + maxs)
    limits = np.stack([midpoint - span * 0.62, midpoint + span * 0.62], axis=1)

    tsdf_display, tsdf_colors = crop_tsdf_to_photo_support(
        tsdf_points=tsdf_display,
        tsdf_colors=tsdf_colors,
        photo_points=photo_display,
        support_radius=span * 0.04,
    )
    tsdf_display, tsdf_colors = crop_to_limits(tsdf_display, tsdf_colors, limits)
    photo_display, photo_colors = crop_to_limits(photo_display, photo_colors, limits)

    fig = plt.figure(figsize=(13.5, 6.6), dpi=180)
    fig.patch.set_facecolor((0.98, 0.98, 0.98))
    titles = [
        ("TSDF fusion", tsdf_display, tsdf_colors),
        ("Dense photogrammetry aligned to TSDF frame", photo_display, photo_colors),
    ]

    for index, (title, points, colors) in enumerate(titles, start=1):
        axis = fig.add_subplot(1, 2, index, projection="3d")
        axis.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=colors,
            s=0.24,
            linewidths=0,
            depthshade=False,
            alpha=0.86,
        )
        axis.set_title(title, fontsize=12, pad=12)
        axis.set_xlim(limits[0])
        axis.set_ylim(limits[1])
        axis.set_zlim(limits[2])
        axis.set_box_aspect((1.0, 1.0, 0.55))
        axis.view_init(elev=46, azim=-48)
        axis.set_axis_off()
        axis.set_facecolor((0.985, 0.985, 0.985))

    fig.suptitle(
        "Same orientation and scale for direct visual comparison",
        fontsize=13,
        y=0.97,
    )
    plt.tight_layout(pad=0.3, rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def load_camera_positions(tsdf_dir: Path) -> np.ndarray:
    """Extract TSDF camera centers for display-orientation disambiguation."""

    summary = load_json(tsdf_dir / "aruco_pose_summary.json")
    return np.asarray(
        [
            np.asarray(pose["transform_world_from_camera"], dtype=np.float64)[:3, 3]
            for pose in summary["poses"]
        ],
        dtype=np.float64,
    )


def main() -> None:
    """Generate all tracked README preview assets."""

    args = parse_args()
    dataset_dir = args.dataset.resolve()
    tsdf_dir = args.tsdf_dir.resolve()
    photogrammetry_dir = args.photogrammetry_dir.resolve()
    assets_dir = args.assets_dir.resolve()
    assets_dir.mkdir(parents=True, exist_ok=True)

    make_rgb_contact_sheet(
        dataset_dir=dataset_dir,
        output_path=assets_dir / "sample_rgb_grid.jpg",
        dataset_label=args.dataset.as_posix(),
    )

    tsdf_points, tsdf_colors, photo_points, photo_colors = align_photogrammetry_to_tsdf(
        tsdf_dir=tsdf_dir,
        photogrammetry_dir=photogrammetry_dir,
        photo_geometry=args.photo_geometry,
        icp_threshold=args.icp_threshold,
        icp_max_iterations=args.icp_max_iterations,
        icp_normal_radius=args.icp_normal_radius,
    )
    render_comparison_preview(
        tsdf_points=tsdf_points,
        tsdf_colors=tsdf_colors,
        photo_points=photo_points,
        photo_colors=photo_colors,
        output_path=assets_dir / "reconstruction_comparison_preview.png",
        max_tsdf_points=args.max_tsdf_points,
        max_photo_points=args.max_photo_points,
        camera_positions=load_camera_positions(tsdf_dir),
    )

    print(f"Wrote {assets_dir / 'sample_rgb_grid.jpg'}")
    print(f"Wrote {assets_dir / 'reconstruction_comparison_preview.png'}")


if __name__ == "__main__":
    main()

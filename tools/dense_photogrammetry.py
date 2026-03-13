#!/usr/bin/env python3
from __future__ import annotations

"""Densify an existing pycolmap sparse reconstruction.

This stage expects the output directory from `tools/rgb_photogrammetry.py`. It
undistorts images into a COLMAP dense workspace, runs PatchMatch stereo, fuses
depth maps into a dense point cloud, and optionally extracts a Poisson mesh.
"""

import argparse
from pathlib import Path

import pycolmap


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dense photogrammetry generation."""

    parser = argparse.ArgumentParser(
        description="Run dense photogrammetry on top of an existing pycolmap sparse reconstruction."
    )
    parser.add_argument(
        "photogrammetry_dir",
        type=Path,
        help="Directory produced by tools/rgb_photogrammetry.py",
    )
    parser.add_argument(
        "--pmvs-option-name",
        type=str,
        default="option-all",
        help="PatchMatch stereo workspace option name",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=1600,
        help="Maximum undistorted image size for dense stereo",
    )
    parser.add_argument(
        "--depth-min",
        type=float,
        default=-1.0,
        help="Optional minimum depth for PatchMatch",
    )
    parser.add_argument(
        "--depth-max",
        type=float,
        default=-1.0,
        help="Optional maximum depth for PatchMatch",
    )
    parser.add_argument(
        "--no-mesh",
        action="store_true",
        help="Skip Poisson meshing after stereo fusion",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the dense photogrammetry refinement pipeline."""

    args = parse_args()
    photogrammetry_dir = args.photogrammetry_dir.resolve()
    sparse_model_dir = photogrammetry_dir / "model"
    summary_json = photogrammetry_dir / "summary.json"
    dense_dir = photogrammetry_dir / "dense"
    fused_ply = photogrammetry_dir / "dense_fused.ply"
    mesh_ply = photogrammetry_dir / "dense_poisson_mesh.ply"

    if not sparse_model_dir.exists() or not summary_json.exists():
        raise FileNotFoundError(
            "Expected a sparse photogrammetry output directory with model/ and summary.json"
        )

    # The sparse summary stores the original dataset path so this dense stage can
    # be rerun later without requiring the caller to pass the image folder again.
    dataset_dir = Path(__import__("json").load(summary_json.open("r", encoding="utf-8"))["dataset"]).resolve()
    dense_dir.mkdir(parents=True, exist_ok=True)

    undistort_options = pycolmap.UndistortCameraOptions()
    undistort_options.max_image_size = args.max_image_size

    pycolmap.undistort_images(
        output_path=str(dense_dir),
        input_path=str(sparse_model_dir),
        image_path=str(dataset_dir),
        output_type="COLMAP",
        copy_policy=pycolmap.CopyType.copy,
        num_patch_match_src_images=20,
        undistort_options=undistort_options,
    )

    patch_match_options = pycolmap.PatchMatchOptions()
    patch_match_options.max_image_size = args.max_image_size
    patch_match_options.depth_min = args.depth_min
    patch_match_options.depth_max = args.depth_max

    pycolmap.patch_match_stereo(
        workspace_path=str(dense_dir),
        workspace_format="COLMAP",
        pmvs_option_name=args.pmvs_option_name,
        options=patch_match_options,
    )

    # The project currently exports geometric fusion because it tends to be the
    # more stable comparison target against TSDF geometry than photometric fusion.
    pycolmap.stereo_fusion(
        output_path=str(fused_ply),
        workspace_path=str(dense_dir),
        workspace_format="COLMAP",
        pmvs_option_name=args.pmvs_option_name,
        input_type="geometric",
    )

    if not args.no_mesh:
        pycolmap.poisson_meshing(
            input_path=str(fused_ply),
            output_path=str(mesh_ply),
        )

    print(f"Dense workspace: {dense_dir}")
    print(f"Fused dense cloud: {fused_ply}")
    if not args.no_mesh:
        print(f"Poisson mesh: {mesh_ply}")


if __name__ == "__main__":
    main()

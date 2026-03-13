#!/usr/bin/env python3
from __future__ import annotations

"""Run a pure-RGB COLMAP/pycolmap sparse reconstruction on recorded images.

The script assumes a dataset directory that contains:

- `image_<timestamp>.jpg`
- `color_camera_info_<timestamp>.json`

It uses the first frame's intrinsics as the shared camera model, performs SIFT
feature extraction and exhaustive matching, runs incremental mapping, then
exports both COLMAP-native model files and a lightweight JSON summary for later
inspection or viewer export.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pycolmap


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the sparse photogrammetry pipeline."""

    parser = argparse.ArgumentParser(
        description="Run pure-RGB photogrammetry with pycolmap on the recorded JPG images."
    )
    parser.add_argument("dataset", type=Path, help="Folder containing image_*.jpg files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_photogrammetry"),
        help="Root folder for COLMAP database and reconstruction outputs",
    )
    parser.add_argument(
        "--camera-model",
        type=str,
        default="PINHOLE",
        help="COLMAP camera model to use",
    )
    parser.add_argument(
        "--single-camera",
        action="store_true",
        default=True,
        help="Treat all frames as a single shared camera",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=1600,
        help="Maximum image size used for SIFT extraction",
    )
    parser.add_argument(
        "--max-num-features",
        type=int,
        default=8192,
        help="Feature cap per image for SIFT extraction",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable GPU if available for feature extraction and matching",
    )
    return parser.parse_args()


def extract_timestamp(path: Path) -> str:
    """Extract the shared timestamp suffix from an image path."""

    parts = path.stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {path.name}")
    return "_".join(parts[-2:])


def load_json(path: Path) -> dict[str, Any]:
    """Load a UTF-8 JSON document from disk."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_images(dataset: Path) -> list[Path]:
    """Return all recorded RGB frames in sorted timestamp order."""

    images = sorted(dataset.glob("image_*.jpg"))
    if not images:
        raise FileNotFoundError(f"No JPG images found in {dataset}")
    return images


def load_intrinsics(dataset: Path, image_path: Path) -> tuple[int, int, str]:
    """Read intrinsics for one image and serialize them for pycolmap."""

    timestamp = extract_timestamp(image_path)
    info_path = dataset / f"color_camera_info_{timestamp}.json"
    info = load_json(info_path)
    width = int(info["width"])
    height = int(info["height"])
    camera_params = ",".join(
        str(value)
        for value in (
            float(info["fx"]),
            float(info["fy"]),
            float(info["cx"]),
            float(info["cy"]),
        )
    )
    return width, height, camera_params


def prepare_output_dir(output_dir: Path) -> tuple[Path, Path]:
    """Reset the mutable COLMAP working directories for a fresh run."""

    database_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    if database_path.exists():
        database_path.unlink()
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    return database_path, sparse_dir


def choose_largest_reconstruction(reconstructions: dict[int, pycolmap.Reconstruction]) -> pycolmap.Reconstruction:
    """Pick the biggest connected model when COLMAP returns multiple reconstructions."""

    if not reconstructions:
        raise RuntimeError("Incremental mapping returned no reconstructions")
    return max(reconstructions.values(), key=lambda rec: rec.num_reg_images())


def save_summary(
    output_dir: Path,
    dataset: Path,
    reconstruction: pycolmap.Reconstruction,
) -> Path:
    """Write a human-readable reconstruction summary for inspection and tooling."""

    images_summary = []
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.image(image_id)
        center = image.projection_center()
        images_summary.append(
            {
                "image_id": int(image_id),
                "name": image.name,
                "camera_id": int(image.camera_id),
                "num_points2D": int(image.num_points2D()),
                "num_points3D": int(image.num_points3D),
                "projection_center": [float(v) for v in center],
                "viewing_direction": [float(v) for v in image.viewing_direction()],
            }
        )

    payload = {
        "dataset": str(dataset),
        "num_registered_images": int(reconstruction.num_reg_images()),
        "num_points3D": int(reconstruction.num_points3D()),
        "mean_reprojection_error": float(reconstruction.compute_mean_reprojection_error()),
        "mean_track_length": float(reconstruction.compute_mean_track_length()),
        "mean_observations_per_image": float(reconstruction.compute_mean_observations_per_reg_image()),
        "images": sorted(images_summary, key=lambda item: item["name"]),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path


def main() -> None:
    """Execute sparse RGB-only photogrammetry from feature extraction to export."""

    args = parse_args()
    dataset = args.dataset.resolve()
    images = collect_images(dataset)
    output_dir = (args.output_dir / dataset.name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path, sparse_dir = prepare_output_dir(output_dir)

    width, height, camera_params = load_intrinsics(dataset, images[0])
    image_names = [image.name for image in images]

    # The project assumes a single physical camera intrinsics model across all
    # captured frames. If a future dataset mixes cameras, this is one place that
    # would need to change.
    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = args.camera_model
    reader_options.camera_params = camera_params

    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.max_image_size = args.max_image_size
    extraction_options.use_gpu = args.use_gpu

    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.use_gpu = args.use_gpu

    pycolmap.extract_features(
        database_path=str(database_path),
        image_path=str(dataset),
        image_names=image_names,
        camera_mode=pycolmap.CameraMode.SINGLE if args.single_camera else pycolmap.CameraMode.AUTO,
        camera_model=args.camera_model,
        reader_options=reader_options,
        extraction_options=extraction_options,
        device=pycolmap.Device.auto if args.use_gpu else pycolmap.Device.cpu,
    )

    pycolmap.match_exhaustive(
        database_path=str(database_path),
        device=pycolmap.Device.auto if args.use_gpu else pycolmap.Device.cpu,
        matching_options=matching_options,
    )

    mapper_options = pycolmap.IncrementalPipelineOptions()
    mapper_options.multiple_models = False
    mapper_options.extract_colors = True
    mapper_options.min_model_size = 6
    mapper_options.ba_refine_principal_point = False

    # Incremental mapping returns a dictionary of candidate models keyed by model
    # id; for this workflow we retain the largest reconstruction only.
    reconstructions = pycolmap.incremental_mapping(
        database_path=str(database_path),
        image_path=str(dataset),
        output_path=str(sparse_dir),
        options=mapper_options,
    )
    reconstruction = choose_largest_reconstruction(reconstructions)

    model_dir = output_dir / "model"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write(str(model_dir))

    ply_path = output_dir / "sparse_points.ply"
    reconstruction.export_PLY(str(ply_path))
    summary_path = save_summary(output_dir=output_dir, dataset=dataset, reconstruction=reconstruction)

    print(f"Registered images: {reconstruction.num_reg_images()}/{len(images)}")
    print(f"Sparse points: {reconstruction.num_points3D()}")
    print(f"Mean reprojection error: {reconstruction.compute_mean_reprojection_error():.6f}")
    print(f"Database: {database_path}")
    print(f"Model: {model_dir}")
    print(f"Sparse PLY: {ply_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

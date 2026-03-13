# Sample datasets

This directory contains versioned input recordings that are intended to be small
enough to clone directly and use as reproducible fixtures for the reconstruction
pipeline.

## Tracked sample

Current tracked dataset:

- [`sample_eye1_2026-03-10A`](/home/sambit/Code/3Dstitch/datasets/sample_eye1_2026-03-10A)

Observed properties of the sample:

- approximately `53 MB` on disk
- `16` timestamps
- color image resolution `848 x 480`
- aligned depth maps stored as `.npz`
- one RGB image, aligned depth map, color intrinsics, depth intrinsics, pose
  metadata, and raw point cloud dump per timestamp

## File naming pattern

For a timestamp like `20260310_171557`, the sample contains:

- `image_20260310_171557.jpg`
- `aligned_depth_to_color_20260310_171557.npz`
- `color_camera_info_20260310_171557.json`
- `depth_camera_info_20260310_171557.json`
- `tf_20260310_171557.npz`
- `tf_color_20260310_171557.npz`
- `camera_pose_20260310_171557.json`
- `cloud_20260310_171557.pcd`

## What is and is not versioned

Tracked:

- raw sample input data under this directory

Not tracked:

- `outputs/`
- `outputs_photogrammetry/`
- `viewer_data/`
- `.venv/`

Those directories are intentionally regenerated from the tracked sample using the
commands in [`README.md`](/home/sambit/Code/3Dstitch/README.md).

## Note on Git LFS

`git-lfs` was not available in the environment used to prepare this repository,
so this sample is committed with normal git objects. That is acceptable for this
sample because the dataset is small and the largest individual files are only a
few megabytes.

If future recordings become substantially larger, move new datasets to Git LFS
before committing them.

Here’s a compact context dump you can paste into Codex / VS Code as a project handoff.

````md
# Project Context Dump — 3d camera pointcloud Doggy

## Goal
Build a practical perception stack for robot manipulation / assembly using RGB-D, starting from noisy RealSense D435 data and moving toward robust multi-view reconstruction and tracking for downstream motion planning.

## Current hardware / setups discussed
- RealSense D435 as eye-in-hand RGB-D camera
- UFactory 850 arms
- Variants considered:
  - 2 arms, each with a D435
  - 1 arm + 1 D435
  - 1 arm + ArUco board under workspace
  - possible 3-camera wrist rig looking inward

## Core conclusions so far

### 1) Single-frame RGB-D is not enough
A single D435 frame is too noisy / holey for accurate pose estimation and motion planning.
Typical solution:
- short active scan from multiple wrist poses
- fuse depth into a local object-centric model
- estimate pose from fused geometry
- then track over time

Recommended pattern:
- multi-view active perception
- temporal fusion
- model-based tracking
- uncertainty-aware planning

### 2) Best library direction for desktop workcell
For a desktop workcell / eye-in-hand setup, prefer:
- **Open3D TSDF fusion** as the default practical choice

Why:
- simpler than open-world mapping stacks
- works well with known camera poses
- good fit for small local reconstruction around object/workspace ROI

Alternatives considered:
- **nvblox**: stronger if already in ROS2 + NVIDIA GPU stack
- **Voxblox**: ROS-centric volumetric mapping option
- but for this project, Open3D is the recommended starting point

### 3) Best operating mode
Do **short burst fusion**, not permanent global mapping:
- move wrist through small arc
- capture ~10–30 RGB-D frames
- crop to workspace or object ROI
- fuse into TSDF
- extract fused cloud / mesh
- run pose estimation / registration

## Open3D recipes already defined

### A) Dual-arm recipe
For 2 x UFactory 850 arms each with D435:
- define `world`
- know:
  - `T_world_left_base`
  - `T_world_right_base`
  - `T_left_tool_left_cam`
  - `T_right_tool_right_cam`
- at runtime:
  - get FK poses for each arm
  - compute `T_world_left_cam`, `T_world_right_cam`
  - integrate both into one TSDF in world frame

Suggested development path:
1. single-arm TSDF first
2. then dual-arm shared-world fusion
3. then registration on fused cloud

### B) Simplified recipe with board under workspace
Assume:
- one eye-in-hand D435
- ArUco GridBoard below workspace
- board defines `world`

Per frame:
1. align depth to color
2. detect board in RGB
3. solve board pose with OpenCV ArUco
4. get `T_cam_world`
5. invert to `T_world_cam`
6. integrate RGB-D into Open3D TSDF
7. crop workspace above board plane
8. use fused cloud for downstream pose estimation

Important simplification:
- no robot FK needed for fusion if board is visible every frame

## Single-file runnable prototype already produced
A full single-file Python prototype was sketched with:
- `pyrealsense2`
- `opencv-contrib-python`
- `open3d`
- `numpy`

It does:
- RealSense streaming
- depth-to-color alignment
- ArUco GridBoard detection
- board-pose-based world frame
- Open3D `ScalableTSDFVolume` integration
- workspace crop
- save `.ply` and `.obj`
- interactive capture:
  - Space = integrate frame
  - V = finish/visualize
  - R = reset
  - Q/Esc = quit

Dependencies:
```bash
pip install numpy open3d opencv-contrib-python pyrealsense2
````

Main implementation ideas from prototype:

* use `rs.align(rs.stream.color)`
* use `cv2.aruco.GridBoard` + `estimatePoseBoard`
* use `o3d.pipelines.integration.ScalableTSDFVolume`

## Dynamic scene / tracking conclusions

### Reconstruction vs tracking split

Recommended architecture:

* **Mode 1: reconstruction / initialization**

  * short TSDF scan burst to build clean object model
* **Mode 2: online tracking**

  * use each new single RGB-D frame only as a measurement update

### Best practical online tracking loop

Use:

* predicted pose from previous frame
* crop tight ROI from new RGB-D frame
* align reconstructed model to current depth with ICP
* fuse with motion model / filter
* re-detect if confidence drops

Recommended online stack:

* state estimator:

  * constant-velocity EKF / UKF
  * state includes pose + linear/angular velocity
* measurement:

  * coarse RGB detection / segmentation if needed
  * fine depth ICP against reconstructed model
* gating:

  * ICP fitness
  * inlier RMSE
  * visible fraction
  * consistency with predicted motion
* recovery:

  * re-detect / reinitialize after several bad frames

Key conclusion:

* yes, efficient single-frame RGB-D tracking is practical **after** a good multi-view initialization
* do not re-run TSDF fusion every frame unless needed

## Wrist camera rig discussion

### Can 3 cameras be mounted on the wrist looking inward?

Yes, conceptually viable.

Benefits:

* fewer blind spots near tool/gripper
* better occlusion handling
* stronger multi-view pose estimation

Problems:

* bulk / collision / reduced access
* calibration complexity
* timestamp sync
* compute and bandwidth
* depth interference if all are active RGB-D devices

### Recommended practical version

Instead of 3 RGB-D cameras, prefer:

* **1 RGB-D camera**
* **2 RGB cameras** at oblique inward angles

This gives:

* one reliable depth source
* multiple viewpoints for visibility, silhouettes, tracking
* less active-depth interference
* easier mechanical integration

Suggested inward-looking rig geometry:

* one forward-ish view
* one left oblique
* one right oblique
* optical axes converge ~5–15 cm in front of tool center
* angular separation ~25–45 degrees

## Important implementation principles

### Calibration

Critical items:

* camera intrinsics
* depth-to-color alignment convention
* hand-eye (if using FK path)
* board/world registration
* timestamp alignment between image and robot pose

### ROI and workspace handling

Prefer:

* fuse only workspace or object ROI
* not the whole scene
* crop using:

  * known workspace bounds
  * detector ROI
  * segmentation mask

### Good initial Open3D parameters for D435

Reasonable starting values:

* `voxel_length = 0.002` m
* `sdf_trunc = 0.01` m
* `depth_trunc = 0.6–0.9` m
* `10–30` frames per scan burst
* object distance roughly `25–50 cm`

### Common gotchas

* board visibility loss
* depth/RGB misalignment
* confusion between `T_cam_world` and `T_world_cam`
* motion blur during capture
* shiny / black / specular objects causing bad depth
* hand/gripper occlusion near contact

## Architecture direction implied by the design doc in this project

The uploaded design doc was repeatedly referenced as supporting:

* continuous RGB-D capture
* multi-view fusion into unified cloud
* model-based 6D tracking/refinement
* per-object filters / pose tracking
* occlusion handling
* adding oblique cameras
* deterministic raw + derived logging

## Recommended next coding steps in VS Code / Codex

1. Implement the single-camera ArUco-board Open3D TSDF prototype first.
2. Validate on empty workspace:

   * board plane flat
   * no doubled surfaces
   * correct scale
3. Add one known object and verify fused cloud quality.
4. Add model registration / ICP on fused cloud.
5. Build online tracker:

   * last pose prediction
   * ROI crop
   * ICP refine
   * confidence gating
6. Only then add:

   * robot FK + hand-eye
   * board-occlusion fallback
   * multi-camera wrist rig if still needed

## Suggested repo structure

```text
project/
  README.md
  requirements.txt
  scripts/
    realsense_aruco_tsdf.py
    visualize_ply.py
  src/
    camera/
      realsense.py
      intrinsics.py
    fiducials/
      aruco_board.py
    fusion/
      tsdf_fuser.py
      workspace_crop.py
    tracking/
      icp_tracker.py
      motion_filter.py
    geometry/
      transforms.py
      se3.py
  data/
    calibration/
    logs/
    outputs/
```

## Suggested immediate milestone

“Reconstruct once, track thereafter”

Meaning:

* one short scan burst with board-visible frames
* build fused object cloud
* then track pose frame-to-frame with ICP + simple motion prior

## Open questions to continue on in Codex

* best object model representation:

  * fused point cloud
  * mesh
  * CAD model
* ICP implementation details:

  * point-to-plane vs generalized ICP
* how to gate bad updates robustly
* board-occlusion fallback strategy:

  * pure vision
  * robot FK + hand-eye
  * hybrid
* whether wrist rig should be:

  * 1 RGB-D
  * 1 RGB-D + 2 RGB
  * more

````

A shorter copy-paste prompt for Codex:

```txt
I’m building a robot perception pipeline for a desktop workcell. Current best direction is: one eye-in-hand RealSense D435, ArUco GridBoard under workspace as world frame, depth aligned to color, OpenCV ArUco board pose each frame, Open3D TSDF fusion over a short scan burst (10–30 frames), crop workspace ROI, then use fused cloud/model for pose estimation. For dynamic scenes, reconstruct once and then track online using predicted pose + ROI crop + depth ICP + simple EKF/constant-velocity filter + re-detect on confidence drop. Also considering later upgrade to 1 RGB-D + 2 wrist RGB cameras for occlusion robustness. Help me implement this as clean modular Python in VS Code.
````

If you want, I can also turn this into a `README.md` plus a starter file tree you can drop directly into a repo.

<!--
Maintenance instructions for future updates:
- Purpose: maintain a factual, per-commit project history timeline for this repository.
- Use git committer time (`%cI`), not author time.
- Inspect actual diffs, touched files, and resulting code/docs/config structure; do not rely on commit subjects alone.
- Append only new commits the next time this file is updated.
- Preserve chronological order.
- Keep exactly one entry per commit.
- Do not rewrite older entries unless correcting a factual error.
- Header format: `## <ISO-8601 committer timestamp> | <7-char sha> | <author>`
- Subject format: `**Subject:** \`<full commit subject>\``
-->
# Project Timeline

This document reconstructs repository history from inspected git diffs rather than commit messages alone. Timestamps use git committer time, and entries are listed in chronological order.

## 2026-03-13T14:43:30+08:00 | 0d78947 | BotCyborg
**Subject:** `Initial commit`

This commit created the repository with a single tracked file: an MIT `LICENSE` for Common Labs. No code, dataset, viewer, or project documentation landed yet in this commit.

## 2026-03-13T14:53:07+08:00 | 15dfb18 | Pal Sambit
**Subject:** `Document pipelines and add tracked sample dataset`

This commit introduced the working project tree: `.gitignore`, `requirements.txt`, a large README handoff document, `datasets/README.md`, `context-from-chatgpt.md`, seven Python pipeline/export scripts under `tools/`, three static viewer pages under `viewer/`, and a tracked sample recording at `datasets/sample_eye1_2026-03-10A` with 16 timestamps of RGB images, aligned depth maps, intrinsics, transforms, poses, and point-cloud dumps. Operationally, the repository became a reproducible offline reconstruction workspace with an ArUco-plus-depth TSDF pipeline, sparse and dense pycolmap photogrammetry path, JSON-export-based browser viewers, and explicit gitignore rules for generated outputs and `.venv`.

## 2026-03-13T14:54:24+08:00 | d5d0b7a | Pal Sambit
**Subject:** `Merge remote-tracking branch 'origin/main'`

This merge joined the working project history with the separate remote main-line commit that contained only the MIT `LICENSE`. Relative to the first parent, the tree change was just addition of `LICENSE`; no reconstruction scripts, viewer files, dataset contents, or documentation behavior changed in this merge.

## 2026-03-13T15:02:48+08:00 | a623344 | Pal Sambit
**Subject:** `Add README visual overview assets`

This commit added two tracked image assets under `docs/assets/` and expanded `README.md` with a new visual-overview section plus an updated repository tree that mentioned those assets. The documentation now showed a 2x2 sample RGB contact sheet and a dense photogrammetry point-cloud preview, but no pipeline, export, or viewer code changed.

## 2026-03-13T15:06:12+08:00 | 750cc43 | Pal Sambit
**Subject:** `Fix Markdown links for GitHub`

This commit was a documentation portability cleanup in `README.md` and `datasets/README.md`: absolute local filesystem links such as `/home/sambit/Code/3Dstitch/...` were replaced with repository-relative links, and machine-specific working-directory/output references were rewritten as generic clone paths or plain directory literals where linking to generated artifacts would not make sense. No source code, config, or dataset files changed.

## 2026-03-13T15:19:21+08:00 | 0c74971 | Pal Sambit
**Subject:** `Add aligned reconstruction preview assets`

This commit replaced the earlier dense-point-cloud README preview with a side-by-side reconstruction comparison image in `docs/assets/reconstruction_comparison_preview.png`, updated `README.md` to describe that aligned TSDF-versus-photogrammetry view, and added `tools/generate_readme_previews.py`. That new script generates the tracked README assets by building a sample RGB contact sheet and rendering TSDF and photogrammetry point clouds from local outputs in the same frame, reusing the comparison exporter's similarity alignment and ICP refinement helpers; existing reconstruction and viewer code paths were not modified.

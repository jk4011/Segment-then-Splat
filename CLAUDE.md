# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Segment then Splat** (NeurIPS 2025) — unified 3D open-vocabulary segmentation via Gaussian Splatting. Each 3D Gaussian is assigned an integer object ID at three granularity levels (default/large, middle, small) instead of storing language features. At evaluation time, CLIP embeddings from multi-view masked crops match text queries to 3D objects via cosine similarity.

Based on: Deformable-3DGS (dynamic scenes), AutoSeg-SAM2 (multi-level tracking), 3DGS rasterizer.

## Environment Setup

```bash
conda create -n segment_then_splat python=3.10
conda activate segment_then_splat
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt --no-build-isolation
# CUDA submodules (must compile):
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

Also install segment-anything-1 and segment-anything-2 per `third_party/AutoSeg-SAM2/` instructions.

## Pipeline Commands

```bash
# 1. Object Tracking (modify data paths in autoseg.sh first)
cd third_party/AutoSeg-SAM2/ && bash autoseg.sh

# 2. Mask Preprocessing
python ./helpers/preprocess_mask.py --mask_root third_party/AutoSeg-SAM2/output/<scene> --out_root <data_dir>/<scene>/ --image_path <data_dir>/<scene>/images

# 3. Object-specific Initialization (requires COLMAP sparse reconstruction)
python ./helpers/object_specific_initialization.py --scene_root <data_dir>/<scene>/

# 4. Training (40k iterations)
python train.py -s <data_dir>/<scene>/ -m output/<scene> --eval --iterations 40000 --num_sample_objects 3 --densify_until_iter 20000 --partial_mask_iou 0.3
# Dynamic scenes: add --is_6dof --deform

# 5. Per-object Rendering
python render_objs.py -m ./output/<scene>/ --mode render --skip_train
# Dynamic scenes: add --is_6dof --deform

# 6. CLIP Association & mIoU Evaluation
python ./helpers/evaluation.py --scene <data_dir>/<scene>/ --render_dir ./output/<scene>/ --label_dir <label_dir>/gt
```

## Architecture

### Core Pipeline (4 stages)

1. **AutoSeg-SAM2 tracking** (`third_party/AutoSeg-SAM2/`) → per-frame `.npy` binary masks at 2-3 scales
2. **Mask preprocessing** (`helpers/preprocess_mask.py`) → deduplicated per-object mask directories (`multiview_masks_{default,middle,small}_merged/`)
3. **Object-specific init** (`helpers/object_specific_initialization.py`) → augments COLMAP `.ply` with `obj_id_default`, `obj_id_middle`, `obj_id_small` per 3D point (255 = background)
4. **3DGS training** (`train.py`) → per-object rendering loss with curriculum training

### Key Modules

- **`scene/gaussian_model.py`** — `GaussianModel` extended with three integer object ID tensors (`default_object_id`, `middle_object_id`, `small_object_id`). Densification/pruning propagates IDs; per-object minimum point counts prevent object collapse.
- **`gaussian_renderer/__init__.py`** — `render()` uses `depth-diff-gaussian-rasterization` CUDA kernel. When `obj_id` is given, masks Gaussians to render single objects in isolation.
- **`utils/time_utils.py`** — `DeformNetwork`: 8-layer MLP (256 hidden) with positional encoding of (xyz, time). Outputs position/rotation/scale deltas. `is_6dof=True` uses SE(3) screw-axis transforms.
- **`helpers/evaluation.py`** — CLIP ViT-L/14@336px encodes multi-view masked crops per object → averaged embeddings. Text query matching via cosine similarity across all granularity levels.

### Training Details

- **3-stage curriculum**: stage 1 (small only) → stage 2 (small+middle, until iter 5000) → stage 3 (all levels, until 40000)
- **Per-object rendering loss**: each iteration samples `num_sample_objects` objects per level, renders in isolation, computes L1+SSIM against GT masked by 2D object mask
- **Partial mask IoU filter** (`--partial_mask_iou 0.3`, after iter 30000): skips object loss when rendered vs. supervision mask IoU is below threshold

### Dataset Layout

```
<scene_dir>/
├── images/                              # RGB frames
├── sparse/0/                            # COLMAP: cameras.bin, images.bin, points3D.bin/.ply
├── multiview_masks_default_merged/      # 000/, 001/, ... per-object mask dirs
├── multiview_masks_middle_merged/
├── multiview_masks_small_merged/
├── train.txt / test.txt                 # Image filename splits
```

## Key Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `-s` | — | Source data path |
| `-m` | — | Model output path |
| `--iterations` | 30000 | Training iterations |
| `--num_sample_objects` | 3 | Objects sampled per level per iteration |
| `--densify_until_iter` | 15000 | Stop densification at this iter |
| `--partial_mask_iou` | 0.0 | IoU threshold for mask filtering (0 = disabled) |
| `--deform` | False | Enable deformation network |
| `--is_6dof` | False | Use SE(3) deformation (requires --deform) |
| `--is_blender` | False | Blender synthetic dataset mode |

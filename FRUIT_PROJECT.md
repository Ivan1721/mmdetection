# Fruit occlusion project (Mask2Former side)

This fork of `open-mmlab/mmdetection` hosts the Mask2Former (and exploratory DETR / Mask-RCNN) side
of a fruit-occlusion instance-segmentation study. The YOLO11/YOLO26 side, the dataset pipeline, and
the paper manuscript live in the sibling repo `../Ultralytics`; both repos read the shared dataset at
`C:\Users\garci\OneDrive - UNIVERSIDAD ANDRES BELLO\Desktop\Bases de Datos\dataset\` (not tracked by either repo's git). See `../Ultralytics/README.md` and
`../Ultralytics/CLAUDE.md` for the project as a whole.

## What's project-specific here (vs. upstream mmdetection)

Everything else in this repo is the vendored `open-mmlab/mmdetection` library — don't expect project
context from it. The project's own additions:

- `configs/mask2former/mask2former_fruits_r50.py` — the active 6-class Mask2Former config.
  `mask2former_fruits_visibility.py` builds on it per visibility-level (`VIS_LEVEL`).
- `configs/mask_rcnn/maskRCNNoclussion.py` — Mask-RCNN baseline for the same 6 classes.
- `configs/*/*_apples*.py`, `configs/detr/detr_apples_instance.py` — earlier 2-class
  (apple_green/apple_red only) exploratory configs, superseded by the ones above. `configs/detr/DETR_Oclusion.py`
  is an empty placeholder for unfinished DETR work.
- `mask2former_visibility_eval.py` — runs the trained Mask2Former checkpoint against each
  visibility-level validation split across saved epoch checkpoints, producing
  `mask2former_visibility_epoch_eval/mask2former_visibility_by_epoch.csv` (consumed by the paper's
  combined YOLO+Mask2Former occlusion-robustness figure).
- `generate_qualitative_mask2former.py` — runs the best Mask2Former checkpoint on the exact 6 source
  frames used by the paper's other qualitative-comparison images, saving directly into
  `../Ultralytics/paper/Articulo/*_mask2former.jpeg`. This was the one piece of the pipeline with no
  reproducible source before this script existed — run it again any time those 6 images need refreshing.
- `work_dirs/` (gitignored, local-only) — training checkpoints. The one this project currently cares
  about is `work_dirs/mask2former_fruits_r50_50e_v2_epoch_ckpts/best_coco_segm_mAP_epoch_50.pth`.

## A config gotcha worth knowing

mmengine loads a config file via `eval()` of its compiled code, not as a normal imported module — so
`__file__` is **not defined** inside a config. `data_root` in the three active configs above is
therefore a plain absolute path (`C:\Users\garci\OneDrive - UNIVERSIDAD ANDRES BELLO\Desktop\Bases de Datos\dataset\coco`), not computed relative to the
config file. If `dataset/` or this repo ever moves again, update those three `data_root` lines by hand
(and `mask2former_visibility_eval.py`'s `DATA_ROOT`, which *can* use `__file__` since it runs as a
normal script, not as a loaded config).

## Environment

Two `openmmlab` conda envs exist on this machine (`openmmlab`, torch 2.4.1; `openmmlab241fix`, torch
2.1.0) — pick whichever one actually imports `mmcv` without a `DLL load failed` error (binary
compatibility between the installed mmcv build and the CUDA/PyTorch pair varies by env). A conda
recipe for one of them is exported at `openmmlab_environment.yml` / `openmmlab_environment_minimo.yml`.

## Known gaps

- Only `mask2former_fruits_r50.py`'s family has real trained checkpoints and reproducible eval. The
  Mask-RCNN and DETR configs exist but aren't part of the paper's reported results.
- No independent test set anywhere in this project (train/val only) — see the Discussion section of
  the paper for this caveat.

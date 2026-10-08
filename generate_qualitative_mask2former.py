"""
Generates the *_mask2former.jpeg qualitative-comparison images used in
paper/Articulo/ (Figure 4 of sn-articleOK.tex), matching the exact source
frames already used for the *_train/*_yolo11/*_yolo26 versions of the same
figure (see Ultralytics/notebooks/Instance_segV1.ipynb).

Run from the mmdetection repo root with the openmmlab241fix conda env (the
plain openmmlab env fails with a DLL load error for mmcv extensions):
    C:\\Users\\garci\\anaconda3\\envs\\openmmlab241fix\\python.exe generate_qualitative_mask2former.py
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS

MMDET_ROOT = Path(__file__).resolve().parent
CONFIG_FILE = MMDET_ROOT / "configs/mask2former/mask2former_fruits_r50.py"
CHECKPOINT_FILE = (
    MMDET_ROOT
    / "work_dirs/mask2former_fruits_r50_50e_v2_epoch_ckpts/best_coco_segm_mAP_epoch_50.pth"
)

COCO_IMAGES_TRAIN = (MMDET_ROOT / ".." / "dataset" / "coco" / "images" / "train").resolve()
OUT_DIR = (MMDET_ROOT / ".." / "Ultralytics" / "paper" / "Articulo").resolve()

# fruit -> exact source frame used by the *_train/*_yolo11/*_yolo26 figures
FRAMES = {
    "apple_green": "20260112_183202_d405_color.png",
    "apple_red": "20260112_185149_d405_color.png",
    "peach": "20260122_131447_d405_color.png",
    "pear": "20260122_140212_d405_color.png",
    "avocado": "20260122_140551_d405_color.png",
    "orange": "20260122_133339_d405_color.png",
}


def main():
    if not CHECKPOINT_FILE.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_FILE}\n"
            "This script expects the mask2former_fruits_r50_50e_v2_epoch_ckpts "
            "work_dir (gitignored, local-only) to be present."
        )

    model = init_detector(str(CONFIG_FILE), str(CHECKPOINT_FILE), device="cuda:0")

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    # Match Ultralytics' default mask overlay transparency (alpha=0.5) so the
    # qualitative comparison figure isn't visually biased by rendering style;
    # mmdetection's DetLocalVisualizer otherwise defaults to alpha=0.8 (much
    # more opaque).
    visualizer.alpha = 0.5

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for fruit, frame_name in FRAMES.items():
        img_path = COCO_IMAGES_TRAIN / frame_name
        if not img_path.is_file():
            print(f"[WARN] missing source frame for {fruit}: {img_path}")
            continue

        result = inference_detector(model, str(img_path))

        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        visualizer.add_datasample(
            name=fruit,
            image=img_rgb,
            data_sample=result,
            draw_gt=False,
            show=False,
            pred_score_thr=0.5,
        )

        out_rgb = visualizer.get_image()

        out_path = OUT_DIR / f"{fruit}_mask2former.jpeg"
        plt.imsave(str(out_path), out_rgb, format="jpeg")
        print(f"Saved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

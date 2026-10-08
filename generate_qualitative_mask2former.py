"""
Generates the *_mask2former.jpeg qualitative-comparison images used in
paper/Articulo/ (Figure 4 of sn-articleOK.tex), matching the exact source
frames AND the exact rendering style already used for the
*_train/*_yolo11/*_yolo26 versions of the same figure (see the custom
cv2-based overlay cell in Ultralytics/notebooks/Instance_segV1.ipynb) --
same mask alpha, same per-class color palette, same darker-shade contour,
same "name conf" text with a black outline and no label background box,
and no drawn bounding box.

Run from the mmdetection repo root with the openmmlab241fix conda env (the
plain openmmlab env fails with a DLL load error for mmcv extensions):
    C:\\Users\\garci\\anaconda3\\envs\\openmmlab241fix\\python.exe generate_qualitative_mask2former.py
"""

from pathlib import Path

import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector

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

# Must match the Ultralytics-side rendering cell exactly (same CONF, ALPHA,
# colors, contour/text style) so the three models' qualitative figures are
# visually comparable.
CONF = 0.25
ALPHA = 0.45
DRAW_BOXES = False
DRAW_LABELS = True

CLASS_COLORS = {
    "apple_green": (60, 180, 75),
    "apple_red": (40, 40, 220),
    "peach": (120, 180, 255),
    "avocado": (35, 110, 35),
    "pear": (80, 220, 180),
    "orange": (0, 140, 255),
}

CONTOUR_THICKNESS = 2
TEXT_SCALE = 0.6
TEXT_THICKNESS = 2
TEXT_OUTLINE_THICKNESS = 4


def class_color(name):
    return CLASS_COLORS.get(name, (255, 255, 255))


def darker(color, factor=0.55):
    return tuple(int(ch * factor) for ch in color)


def draw_text_with_outline(img, text, org, color):
    cv2.putText(
        img, text, org,
        cv2.FONT_HERSHEY_SIMPLEX,
        TEXT_SCALE,
        (0, 0, 0),
        TEXT_OUTLINE_THICKNESS,
        cv2.LINE_AA,
    )
    cv2.putText(
        img, text, org,
        cv2.FONT_HERSHEY_SIMPLEX,
        TEXT_SCALE,
        color,
        TEXT_THICKNESS,
        cv2.LINE_AA,
    )


def main():
    if not CHECKPOINT_FILE.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_FILE}\n"
            "This script expects the mask2former_fruits_r50_50e_v2_epoch_ckpts "
            "work_dir (gitignored, local-only) to be present."
        )

    model = init_detector(str(CONFIG_FILE), str(CHECKPOINT_FILE), device="cuda:0")
    class_names = model.dataset_meta["classes"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for fruit, frame_name in FRAMES.items():
        img_path = COCO_IMAGES_TRAIN / frame_name
        if not img_path.is_file():
            print(f"[WARN] missing source frame for {fruit}: {img_path}")
            continue

        img_bgr = cv2.imread(str(img_path))
        H, W = img_bgr.shape[:2]

        result = inference_detector(model, str(img_path))
        instances = result.pred_instances
        keep = instances.scores.cpu().numpy() >= CONF
        masks = instances.masks[keep].cpu().numpy()  # (N,h,w) bool
        labels = instances.labels[keep].cpu().numpy()
        scores = instances.scores[keep].cpu().numpy()
        bboxes = instances.bboxes[keep].cpu().numpy()

        out = img_bgr.copy()

        if len(masks) > 0:
            overlay = np.zeros((H, W, 3), dtype=np.uint8)
            contour_items = []

            for i in range(masks.shape[0]):
                name = class_names[int(labels[i])]
                color = class_color(name)

                m = masks[i]
                if m.shape[0] != H or m.shape[1] != W:
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)

                mask_u8 = (m > 0.5).astype(np.uint8)
                overlay[mask_u8.astype(bool)] = color

                contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour_items.append((contours, darker(color)))

            out = cv2.addWeighted(out, 1.0, overlay, ALPHA, 0)

            for contours, contour_color in contour_items:
                cv2.drawContours(out, contours, -1, contour_color, CONTOUR_THICKNESS, cv2.LINE_AA)

            for i in range(len(labels)):
                name = class_names[int(labels[i])]
                color = class_color(name)
                x1, y1, x2, y2 = bboxes[i].astype(int)

                if DRAW_BOXES:
                    cv2.rectangle(out, (x1, y1), (x2, y2), darker(color), 2)

                if DRAW_LABELS:
                    txt = f"{name} {scores[i]:.2f}"
                    tx, ty = x1, max(18, y1 - 8)
                    draw_text_with_outline(out, txt, (tx, ty), color)

        out_path = OUT_DIR / f"{fruit}_mask2former.jpeg"
        cv2.imwrite(str(out_path), out)
        print(f"Saved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

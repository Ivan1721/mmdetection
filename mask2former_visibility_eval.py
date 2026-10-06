import subprocess
import pandas as pd
import re
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================

MMDET_ROOT = Path(r"C:\workspace\mmdetection")

CONFIG_FILE = MMDET_ROOT / "configs/mask2former/mask2former_fruits_r50.py"

CHECKPOINT_DIR = MMDET_ROOT / "work_dirs/mask2former_fruits_r50_50e_v2_epoch_ckpts"

DATA_ROOT = Path(
    r"C:\Users\garci\OneDrive - UNIVERSIDAD ANDRES BELLO\Desktop\1.Universidad\PhdDISA\vision\Transformer\Manzana\coco_dataset"
)

VISIBILITY_LEVELS = ["25", "50", "75", "100"]

OUT_DIR = MMDET_ROOT / "mask2former_visibility_epoch_eval"
OUT_DIR.mkdir(exist_ok=True)

CSV_OUT = OUT_DIR / "mask2former_visibility_by_epoch.csv"

# =========================================================
# CREAR CONFIG TEMPORAL PARA VALIDACIÓN
# =========================================================

def create_temp_config(level):
    cfg_path = OUT_DIR / f"cfg_val_{level}.py"

    content = f"""
_base_ = ['{CONFIG_FILE.as_posix()}']

data_root = r'{DATA_ROOT.as_posix()}/'

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/instances_val_{level}.json',
        data_prefix=dict(img=''),
        test_mode=True,
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val_{level}.json',
    metric=['bbox', 'segm']
)

test_evaluator = val_evaluator
"""

    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(content)

    return cfg_path

# =========================================================
# EXTRAER EPOCH
# =========================================================

def get_epoch(ckpt):
    match = re.search(r"epoch_(\d+)", ckpt.name)
    return int(match.group(1)) if match else None

# =========================================================
# PARSEAR OUTPUT MMDETECTION
# =========================================================

def parse_metrics(output):
    metrics = {}

    # bbox
    bbox_match = re.search(
        r"bbox_mAP_copypaste:\s([\d\.]+)\s([\d\.]+)",
        output
    )

    if bbox_match:
        metrics["bbox_mAP50_95"] = float(bbox_match.group(1))
        metrics["bbox_mAP50"] = float(bbox_match.group(2))

    # segm
    segm_match = re.search(
        r"segm_mAP_copypaste:\s([\d\.]+)\s([\d\.]+)",
        output
    )

    if segm_match:
        metrics["segm_mAP50_95"] = float(segm_match.group(1))
        metrics["segm_mAP50"] = float(segm_match.group(2))

    return metrics

# =========================================================
# LOOP PRINCIPAL
# =========================================================

results = []

checkpoints = sorted(
    CHECKPOINT_DIR.glob("epoch_*.pth"),
    key=lambda p: get_epoch(p)
)

print(f"Checkpoints encontrados: {len(checkpoints)}")

for ckpt in checkpoints:

    epoch = get_epoch(ckpt)
    if epoch is None:
        continue

    print(f"\n=== Epoch {epoch} ===")

    for level in VISIBILITY_LEVELS:

        print(f"Evaluando visibilidad {level}%")

        cfg = create_temp_config(level)

        cmd = [
            "python",
            "tools/test.py",
            str(cfg),
            str(ckpt),
            "--launcher", "none"
        ]

        result = subprocess.run(
            cmd,
            cwd=MMDET_ROOT,
            capture_output=True,
            text=True
        )

        output = result.stdout + result.stderr

        metrics = parse_metrics(output)

        if not metrics:
            print("❌ No se pudieron extraer métricas")
            continue

        results.append({
            "model": "Mask2Former",
            "epoch": epoch,
            "visibility": level,
            "bbox_mAP50_95": metrics.get("bbox_mAP50_95"),
            "bbox_mAP50": metrics.get("bbox_mAP50"),
            "mask_mAP50_95": metrics.get("segm_mAP50_95"),
            "mask_mAP50": metrics.get("segm_mAP50"),
        })

# =========================================================
# GUARDAR CSV
# =========================================================

df = pd.DataFrame(results)

df = df.sort_values(["visibility", "epoch"])

df.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")

print("\n✅ CSV generado:")
print(CSV_OUT)

print("\nPreview:")
print(df.head())
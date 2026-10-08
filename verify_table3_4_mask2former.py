"""
Re-validates the trained Mask2Former checkpoint against the full validation
set (not a visibility subset) for direct comparison against the published
paper numbers (Table 3: mAP; Table 4: Params/FLOPs/latency).

Run from the mmdetection repo root with the openmmlab241fix conda env:
    C:\\Users\\garci\\anaconda3\\envs\\openmmlab241fix\\python.exe verify_table3_4_mask2former.py
"""
import json
import re
import subprocess
import time
from pathlib import Path

import torch

MMDET_ROOT = Path(__file__).resolve().parent
CONFIG_FILE = MMDET_ROOT / "configs/mask2former/mask2former_fruits_r50.py"
CHECKPOINT = (
    MMDET_ROOT
    / "work_dirs/mask2former_fruits_r50_50e_v2_epoch_ckpts/best_coco_segm_mAP_epoch_50.pth"
)
OUT_JSON = MMDET_ROOT / "table3_4_verification_mask2former.json"

results = {}

# --- Table 3: mAP via tools/test.py on the full validation set ---
print("==== Running tools/test.py (full validation set) ====")
cmd = ["python", "tools/test.py", str(CONFIG_FILE), str(CHECKPOINT)]
proc = subprocess.run(cmd, cwd=MMDET_ROOT, capture_output=True, text=True)
output = proc.stdout + proc.stderr
print(output[-4000:])

bbox_match = re.search(r"bbox_mAP_copypaste:\s([\d.]+)\s([\d.]+)", output)
segm_match = re.search(r"segm_mAP_copypaste:\s([\d.]+)\s([\d.]+)", output)
if bbox_match:
    results["bbox_mAP50_95"] = float(bbox_match.group(1))
    results["bbox_mAP50"] = float(bbox_match.group(2))
if segm_match:
    results["segm_mAP50_95"] = float(segm_match.group(1))
    results["segm_mAP50"] = float(segm_match.group(2))

# --- Table 4: Params / FLOPs via tools/analysis_tools/get_flops.py ---
print("\n==== Running get_flops.py ====")
cmd = ["python", "tools/analysis_tools/get_flops.py", str(CONFIG_FILE)]
proc = subprocess.run(cmd, cwd=MMDET_ROOT, capture_output=True, text=True)
flops_output = proc.stdout + proc.stderr
print(flops_output[-2000:])
params_match = re.search(r"Params:\s*([\d.]+)\s*([GM])", flops_output)
flops_match = re.search(r"Flops:\s*([\d.]+)\s*([GM])", flops_output)
if params_match:
    val, unit = params_match.groups()
    results["params_M"] = float(val) if unit == "M" else float(val) * 1000
if flops_match:
    val, unit = flops_match.groups()
    results["gflops"] = float(val) if unit == "G" else float(val) / 1000

# --- Table 4: GPU latency at batch=1, matching the paper's protocol ---
print("\n==== Measuring GPU latency (batch=1) ====")
from mmdet.apis import init_detector, inference_detector

model = init_detector(str(CONFIG_FILE), str(CHECKPOINT), device="cuda:0")
import numpy as np

dummy = (np.random.rand(640, 640, 3) * 255).astype("uint8")

for _ in range(5):
    inference_detector(model, dummy)
torch.cuda.synchronize()
t0 = time.perf_counter()
N = 30
for _ in range(N):
    inference_detector(model, dummy)
torch.cuda.synchronize()
t1 = time.perf_counter()
results["latency_ms_batch1"] = (t1 - t0) / N * 1000

print(json.dumps(results, indent=2))
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved:", OUT_JSON)

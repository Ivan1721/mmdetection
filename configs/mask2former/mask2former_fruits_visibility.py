# =========================================================
# BASE CONFIG
# =========================================================

_base_ = './mask2former_fruits_r50.py'

# =========================================================
# PARAMETRO DE VISIBILIDAD (CAMBIAR ESTO)
# =========================================================

VIS_LEVEL = "100"   # <-- cambiar: "25", "50", "75", "100"

# =========================================================
# DATASET ROOT
# =========================================================
classes = (
    'apple_green',
    'apple_red',
    'peach',
    'avocado',
    'pear',
    'orange',
)

import os as _os
# relative to this file: configs/mask2former/<this file> -> repos/dataset/coco
data_root = _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'dataset', 'coco'
))

ann_file = f'annotations/instances_val_{VIS_LEVEL}.json'

# =========================================================
# DATALOADER
# =========================================================

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file=ann_file,
        data_prefix=dict(img=''),
        test_mode=True,
    )
)

test_dataloader = val_dataloader

# =========================================================
# EVALUATOR
# =========================================================

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/' + ann_file,
    metric=['bbox', 'segm'],
    format_only=False
)

test_evaluator = val_evaluator

# =========================================================
# WORK DIR
# =========================================================

work_dir = f'work_dirs/mask2former_visibility/vis_{VIS_LEVEL}'
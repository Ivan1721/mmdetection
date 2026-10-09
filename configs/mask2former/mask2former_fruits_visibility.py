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

# NOTE: mmengine loads config files via eval(), not as a normal Python
# module, so __file__ is not defined here - a __file__-relative path
# cannot be computed from inside a config. This must stay an absolute
# path; keep it in sync with the dataset/ directory's actual location
# (now centralized under Bases de Datos, see repos\Ultralytics\CLAUDE.md).
data_root = r'C:\Users\garci\OneDrive - UNIVERSIDAD ANDRES BELLO\Desktop\Bases de Datos\dataset\coco'

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
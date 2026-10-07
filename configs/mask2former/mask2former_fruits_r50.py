_base_ = './mask2former_r50_8xb2-lsj-50e_coco.py'

# =========================================================
# DATASET
# =========================================================

dataset_type = 'CocoDataset'

classes = (
    'apple_green',
    'apple_red',
    'peach',
    'avocado',
    'pear',
    'orange',
)

num_things_classes = len(classes)
num_stuff_classes = 0

# NOTE: mmengine loads config files via eval(), not as a normal Python
# module, so __file__ is not defined here - a __file__-relative path
# cannot be computed from inside a config. This must stay an absolute
# path; keep it in sync with the dataset/ directory's actual location
# (a sibling of this repo under C:\Users\garci\repos\).
data_root = r'C:\Users\garci\repos\dataset\coco' + '\\'

# =========================================================
# MODEL
# =========================================================

model = dict(
    panoptic_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_things_classes + [0.1],
        ),
    ),
    panoptic_fusion_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
    ),
    test_cfg=dict(
        panoptic_on=False,
        semantic_on=False,
        instance_on=True,
        max_per_image=100,
    ),
)

# =========================================================
# DATALOADERS
# =========================================================

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img=''),
        test_mode=True,
    ),
)

test_dataloader = val_dataloader

# =========================================================
# EVALUATORS
# =========================================================

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric=['bbox', 'segm'],
    format_only=False,
)

test_evaluator = val_evaluator

# =========================================================
# TRAINING LOOP
# =========================================================

max_epochs = 50

train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=5,
)

val_cfg = dict(
    _delete_=True,
    type='ValLoop',
)

test_cfg = dict(
    _delete_=True,
    type='TestLoop',
)

# =========================================================
# LR SCHEDULER
# =========================================================

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[35, 45],
        gamma=0.1,
    )
]

# =========================================================
# OPTIMIZER
# =========================================================

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
    ),
    clip_grad=dict(max_norm=0.01, norm_type=2),
)

# =========================================================
# HOOKS
# =========================================================

default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        interval=5,
        by_epoch=True,
        max_keep_ckpts=-1,
        save_best='coco/segm_mAP',
        rule='greater',
        save_optimizer=False,
        save_param_scheduler=False
    ),
    logger=dict(type='LoggerHook', interval=20)
)

# =========================================================
# VISUALIZER
# =========================================================

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer',
)

# =========================================================
# RUNTIME
# =========================================================

work_dir = './work_dirs/mask2former_fruits_r50_50e_v2_epoch_ckpts'

randomness = dict(seed=42, deterministic=False)
_base_ = './mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco.py'

# =========================================================
# DATASET
# =========================================================
dataset_type = 'CocoDataset'
classes = (
    'apple_green',
    'apple_red',
    'peach',
    'orange',
    'pear',
    'avocado',
)
num_classes = len(classes)

# Raíz de tu dataset COCO ya convertido
data_root = r'C:/Users/garci/OneDrive - UNIVERSIDAD ANDRES BELLO/Desktop/1.Universidad/PhdDISA/Tesis/vision/Transformer/Manzana/coco_dataset/'

# =========================================================
# MODEL
# =========================================================
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=num_classes),
        mask_head=dict(num_classes=num_classes)
    )
)

# =========================================================
# DATALOADERS
# =========================================================
train_dataloader = dict(
    batch_size=2,          # ajusta a tu VRAM
    num_workers=2,         # en Windows 2 suele ser más seguro que 4+
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img=''),
        test_mode=True,
    )
)

test_dataloader = val_dataloader

# =========================================================
# EVALUATION
# =========================================================
val_evaluator = dict(
    ann_file=data_root + 'annotations/instances_val.json',
    metric=['bbox', 'segm']
)
test_evaluator = val_evaluator

# =========================================================
# TRAINING SCHEDULE
# =========================================================
# Para dataset pequeño/mediano, conviene validar más seguido.
# Mantengo un schedule largo, pero más práctico para tu caso.
# max_epochs = 120

# train_cfg = dict(
#     type='EpochBasedTrainLoop',
#     max_epochs=max_epochs,
#     val_interval=5
# )
# val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')

# # Un step tardío suele funcionar bien al hacer fine-tuning
# param_scheduler = [
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=max_epochs,
#         by_epoch=True,
#         milestones=[90, 110],
#         gamma=0.1
#     )
# ]

max_epochs = 30

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=2
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[30, 36],
        gamma=0.1
    )
]

# =========================================================
# OPTIMIZER
# =========================================================
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.005,          # pensado para batch_size total ≈ 2
        momentum=0.9,
        weight_decay=0.0001
    )
)

# =========================================================
# HOOKS
# =========================================================
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        max_keep_ckpts=3,
        save_best='coco/segm_mAP',
        rule='greater'
    ),
    logger=dict(type='LoggerHook', interval=20)
)

# =========================================================
# PRETRAINED / WORKDIR
# =========================================================
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

work_dir = './work_dirs/mask_rcnn_fruits_r50_30ep'

# =========================================================
# OPCIONAL: visualizador
# =========================================================
visualizer = dict(
    type='DetLocalVisualizer',        
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer'
)
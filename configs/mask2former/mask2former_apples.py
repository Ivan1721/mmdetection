_base_ = 'mmdet::mask2former/mask2former_r50_8xb2-lsj-50e_coco.py'

data_root = r'C:\Users\garci\OneDrive - UNIVERSIDAD ANDRES BELLO\Desktop\1.Universidad\PhdDISA\Tesis\vision\dataset_coco\\'

metainfo = dict(
    classes=('apple_green', 'apple_red'),
    palette=[(220, 20, 60), (0, 128, 0)]
)

# -------------------
# Model (alinear TODO)
# -------------------
model = dict(
    # IMPORTANTE: fusion head también
    panoptic_fusion_head=dict(
        num_things_classes=2,
        num_stuff_classes=0
    ),
    panoptic_head=dict(
        num_things_classes=2,
        num_stuff_classes=0,
        num_classes=2,
        # 2 clases + no-object
        loss_cls=dict(class_weight=[1.0, 1.0, 1.0])
    ),
    test_cfg=dict(
        instance_on=True,
        panoptic_on=False,
        semantic_on=False,
        # ayuda a que no se “cuele” no-object como predicción
        filter_low_score=True,
        max_per_image=100
    )
)

train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/train/')
    )
)

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='images/val/')
    )
)
test_dataloader = val_dataloader

# Evaluator: evita conflicto list/dict del base
val_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric=['bbox', 'segm']
)
test_evaluator = val_evaluator

# Loop iter-based corto
train_cfg = dict(type='IterBasedTrainLoop', max_iters=2000, val_interval=200)

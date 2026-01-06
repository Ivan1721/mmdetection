_base_ = '../detr/detr_r50_8xb2-150e_coco.py'

data_root = r'C:\Users\garci\OneDrive - UNIVERSIDAD ANDRES BELLO\Desktop\1.Universidad\PhdDISA\Tesis\vision\dataset_coco\\'

metainfo = {
    'classes': ('apple_green', 'apple_red'),
    'palette': [
        (220, 20, 60),
        (0, 128, 0),
    ]
}

# Ajusta num_classes del head (la ruta exacta puede variar por versión)
model = dict(bbox_head=dict(num_classes=2))

train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/train/')))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='images/val/')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/instances_val.json')
test_evaluator = val_evaluator

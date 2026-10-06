
_base_ = ['C:/workspace/mmdetection/configs/mask2former/mask2former_fruits_r50.py']

data_root = r'C:/Users/garci/OneDrive - UNIVERSIDAD ANDRES BELLO/Desktop/1.Universidad/PhdDISA/vision/Transformer/Manzana/coco_dataset/'

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/instances_val_50.json',
        data_prefix=dict(img=''),
        test_mode=True,
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val_50.json',
    metric=['bbox', 'segm']
)

test_evaluator = val_evaluator

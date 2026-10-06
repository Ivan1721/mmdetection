
_base_ = r'C:/workspace/mmdetection/configs/mask2former/mask2former_fruits_visibility.py'

VIS_LEVEL = "75"

data_root = r'C:/Users/garci/OneDrive - UNIVERSIDAD ANDRES BELLO/Desktop/1.Universidad/PhdDISA/vision/Transformer/Manzana/coco_dataset'
ann_file = f'annotations/instances_val_{VIS_LEVEL}.json'

classes = (
    'apple_green',
    'apple_red',
    'peach',
    'avocado',
    'pear',
    'orange',
)

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

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/' + ann_file,
    metric=['bbox', 'segm'],
    format_only=False
)

test_evaluator = val_evaluator

work_dir = r'C:/workspace/mmdetection/mask2former_visibility_epoch_eval/work_vis_75'

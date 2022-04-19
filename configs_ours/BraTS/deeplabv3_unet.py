_base_ = [
    '../_base_/models/deeplabv3_unet_s5-d16.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(
    decode_head=dict(loss_decode=[
        dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
        dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)
    ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# evaluation = dict(metric='mDice')
dataset_type = 'CustomDataset'
transfer_data_root = 'data/Transfer/BraTS/'
source_data_root = 'data/Transfer/BraTS/'
transfer_model = 'demo_model'
domains='flair_to_t1'
BraTS_classes = ('background', 'enhancing tumor', 'tumor core', 'whole tumor')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_scale = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

train_A = dict(
    type = dataset_type,
    img_dir = f'{source_data_root}/train/flair',
    img_suffix='.png',
    ann_dir = f'{source_data_root}/annotation/flair',
    pipeline = train_pipeline, 
    classes = BraTS_classes, 
    reduce_zero_label=True,
)

train_B = dict(
    type = dataset_type,
    img_dir = f'{source_data_root}/train/t1',
    img_suffix='.png',
    ann_dir = f'{source_data_root}/annotation/t1',
    pipeline = train_pipeline, 
    classes = BraTS_classes, 
    reduce_zero_label=True,
)

train_C = dict(
    type = dataset_type,
    img_dir = f'{source_data_root}/train/t1ce',
    img_suffix='.png',
    ann_dir = f'{source_data_root}/annotation/t1ce',
    pipeline = train_pipeline, 
    classes = BraTS_classes, 
    reduce_zero_label=True,
)

train_D = dict(
    type = dataset_type,
    img_dir = f'{source_data_root}/train/t2',
    img_suffix='.png',
    ann_dir = f'{source_data_root}/annotation/t2',
    pipeline = train_pipeline, 
    classes = BraTS_classes, 
    reduce_zero_label=True,
)


test_A = dict(
    type = dataset_type,
    img_dir = f'{source_data_root}/test/t2',
    img_suffix='.png',
    ann_dir = f'{source_data_root}/annotation/t2',
    pipeline = test_pipeline, 
    classes = BraTS_classes, 
    reduce_zero_label=True,
)

test_B = dict(
    type = dataset_type,
    img_dir = f'{source_data_root}/test/flair',
    img_suffix='.png',
    ann_dir = f'{source_data_root}/annotation/flair',
    pipeline = test_pipeline, 
    classes = BraTS_classes, 
    reduce_zero_label=True,
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train = [train_A, train_B, train_C, train_D],
    test = [test_A, test_B])

runner = dict(type='IterBasedRunner', max_iters=100)
checkpoint_config = dict(by_epoch=False, interval=100)
evaluation = dict(interval=100, metric=['mIoU','mDice'], pre_eval=True)

log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=100)
evaluation = dict(interval=5, metric='PCK', key_indicator='PCK')
COLOR = 'black'

optimizer = dict(
    type='Adam',
    lr=5e-4,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200])
total_epochs = 210
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

channel_cfg = dict(
    num_output_channels=9,
    dataset_joints=9,
    dataset_channel=[
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8
        ],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8
    ])

# model settings
model = dict(
    type='TopDown',
    pretrained='torchvision://resnet152',
    backbone=dict(type='ResNet', depth=152),
    keypoint_head=dict(
        type='TopDownSimpleHead',
        in_channels=2048,
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))

data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'])

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]

test_pipeline = val_pipeline
dataset_type = 'AnimalHorse10Dataset'
data_root = 'data/cattle/training_data/cattle_leg_front_256_'+COLOR

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type=dataset_type,
        ann_file=f'{data_root}/annotations/train.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=f'{data_root}/annotations/test.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=f'{data_root}/annotations/test.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=test_pipeline),
)
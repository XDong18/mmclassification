# dataset settings
dataset_type = 'BddCls'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomCrop', size=640),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', size=(256, -1)),
    # dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        image_dir='/shared/xudongliu/bdd100k/' + '100k/val/',
        label_dir='/shared/xudongliu/bdd100k/' + 'labels/cls/cls_val.json',
        data_prefix = None,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        image_dir='/shared/xudongliu/bdd100k/' + '100k/val/',
        label_dir='/shared/xudongliu/bdd100k/' + 'labels/cls/cls_val.json',
        data_prefix = None,
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        image_dir='/shared/xudongliu/bdd100k/' + '100k/val/',
        label_dir='/shared/xudongliu/bdd100k/' + 'labels/cls/cls_val.json',
        data_prefix = None,
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='accuracy')

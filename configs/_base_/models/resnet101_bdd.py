# model settings
model = dict(
    type='ImageClassifier',
    pretrained='open-mmlab://detectron/resnet101_caffe',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        stem_channels=128,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='NewMultiLinearClsHead',
        multi_num_classes=[6,6,3],
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

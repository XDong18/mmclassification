# model settings
model = dict(
    type='ImageClassifier',
    pretrained='/shared/xudongliu/code/weights/dla34-ba72cf86.pth',
    backbone=dict(
        type='DLA',
        levels=[1, 1, 1, 2, 2, 1],
        channels=[16, 32, 64, 128, 256, 512],
        block_num=2,
        return_levels=True),
    head=dict(
        dict(
            type='DLAClsHead',
            num_convs=2,
            in_channels=512,
            conv_kernel_size=3,
            conv_out_channels=256,
            num_classes=[6, 6, 3],
            conv_cfg=None,
            norm_cfg=None,
            loss_cls=dict(
                type='CrossEntropyLoss',
                ignore_index=-1),
            topk=(1,),
        ),
    ))

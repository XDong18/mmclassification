import mmcv
import numpy as np
# import pycocotools.mask as mask_util
import torch
import torch.nn as nn

from ..builder import build_loss
from ..builder import HEADS
from ..model_utils import ConvModule
from .cls_head import ClsHead
# from mmdet.core import mask_target, force_fp32, auto_fp16


@HEADS.register_module
class DLAClsHead(ClsHead):

    def __init__(self,
                 num_convs=2,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 num_classes=[6, 6, 3],
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_cls=dict(
                     type='CrossEntropyLoss'),
                 topk=(1, )):
        super(DLAClsHead, self).__init__(loss=loss_cls, topk=topk)
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.num_classes = num_classes
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        # self.loss_cls = build_loss(loss_cls)
        # self.topk = topk

        self.convs = nn.ModuleList()
        for _ in range(3):
            convs = nn.ModuleList()
            for i in range(self.num_convs):
                in_channels = (
                    self.in_channels if i == 0 else self.conv_out_channels)
                padding = (self.conv_kernel_size - 1) // 2
                convs.append(
                    ConvModule(
                        in_channels,
                        self.conv_out_channels,
                        self.conv_kernel_size,
                        stride=2,
                        padding=padding,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg))
            self.convs.append(convs)


        self.fcs = nn.ModuleList([nn.Linear(conv_out_channels, n) for n in num_classes])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)

    def simple_test(self, x):
        x = x.unsqueeze(0).expand(3, -1, -1, -1, -1)
        for i in range(len(self.convs[0])):
            x = [convs[i](_x) for _x, convs in zip(x, self.convs)]
        # global avg pool
        x = [torch.mean(_x.view(_x.size(0), _x.size(1), -1), dim=2) for _x in x]
        cls_scores = [fc(_x) for _x, fc in zip(x, self.fcs)]
        preds = [F.softmax(cls_score, dim=1) if cls_score is not None else None for cls_score in cls_scores]
        if torch.onnx.is_in_onnx_export():
            return preds
        preds = [list(pred.detach().cpu().numpy()) for pred in preds]
        return preds

    def forward_train(self, x, gt_label):
        x = x.unsqueeze(0).expand(3, -1, -1, -1, -1)
        for i in range(len(self.convs[0])):
            x = [convs[i](_x) for _x, convs in zip(x, self.convs)]
        # global avg pool
        x = [torch.mean(_x.view(_x.size(0), _x.size(1), -1), dim=2) for _x in x]
        cls_scores = [fc(_x) for _x, fc in zip(x, self.fcs)]
        losses = self.loss(cls_scores, gt_label)
        return losses

    def loss(self, cls_scores, gt_label):
        losses = dict()
        loss_cls = 0
        for i in range(len(gt_label[0])):
            # if i != 2:
            #     continue
            labels_i = torch.stack([l[i] for l in gt_label])
            loss_cls += self.compute_loss(cls_scores[i], labels_i)
        # loss['loss_cls'] = loss_cls
        losses['loss'] = loss_cls
        accs = []
        for i in range(len(gt_label[0])):
            labels_i = torch.stack([l[i] for l in gt_label])
            acc = self.compute_accuracy(cls_scores[i], labels_i)
            accs.append(acc)

        # acc = self.compute_accuracy(cls_score, gt_label)
        losses['loss'] = loss
        assert len(accs[0]) == len(self.topk)
        losses['accuracy_weather'] = {f'weather_top-{k}': a for k, a in zip(self.topk, accs[0])}
        losses['accuracy_scene'] = {f'scene_top-{k}': a for k, a in zip(self.topk, accs[1])}
        losses['accuracy_time'] = {f'time_top-{k}': a for k, a in zip(self.topk, accs[2])}
        return losses
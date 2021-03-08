import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from ..builder import HEADS
from .cls_head import ClsHead


@HEADS.register_module()
class NewMultiLinearClsHead(ClsHead):
    """Linear classifier head.

    Args:
        num_classes list(int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
    """  # noqa: W605

    def __init__(self,
                 multi_num_classes,
                 in_channels,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, )):
        super(NewMultiLinearClsHead, self).__init__(loss=loss, topk=topk)
        self.in_channels = in_channels
        self.multi_num_classes = multi_num_classes

        if min(self.multi_num_classes) <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self._init_layers()

    def _init_layers(self):
        # self.fc = nn.Linear(self.in_channels, self.num_classes)
        self.fcs = nn.ModuleList([nn.Linear(self.in_channels, n) for n in self.multi_num_classes])

    def init_weights(self):
        for fc in self.fcs:
            normal_init(fc, mean=0, std=0.01, bias=0)

    def simple_test(self, img):
        """Test without augmentation."""
        cls_scores = [fc(img) for fc in self.fcs]
        # if isinstance(cls_score, list):
        #     cls_score = sum(cls_score) / float(len(cls_score))
        preds = [F.softmax(cls_score, dim=1) if cls_score is not None else None for cls_score in cls_scores]
        if torch.onnx.is_in_onnx_export():
            return preds
        preds = [list(pred.detach().cpu().numpy()) for pred in preds]
        return preds

    def loss(self, cls_scores, gt_label):
        num_samples = len(cls_scores)
        losses = dict()
        # compute loss
        loss = 0
        for i in range(len(gt_label[0])):
            # if i != 2:
            #     continue
            labels_i = torch.stack([l[i] for l in gt_label])
            # print('\npin\n', i, labels_i.max(), '\npin\n')
            loss += self.compute_loss(cls_scores[i], labels_i)

        # loss = self.compute_loss(cls_score, gt_label, avg_factor=num_samples)
        # compute accuracy
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

    def forward_train(self, x, gt_label):
        cls_scores = [fc(x) for fc in self.fcs]
        losses = self.loss(cls_scores, gt_label)
        return losses

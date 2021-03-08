import os

import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS
import mmcv
import json


@DATASETS.register_module
class BddCls(BaseDataset):

    def __init__(self,
                 image_dir,
                 label_dir,
                 flip_ratio=0.5,
                 with_lane=True,
                 with_drivable=True,
                 crop_size=None,
                 task='cls',
                 *args,
                 **kwargs):
        super(BddCls, self).__init__(*args, **kwargs)
        # data
        # self.test_mode = test_mode
        self.image_dir = image_dir
        self.img_prefix = self.image_dir
        with open(label_dir) as f:
            self.labels = json.load(f)
        # temporary
        for k in self.labels.keys():
            self.labels[k] = self.labels[k]
        # self.phase = phase.split('_')[-1]
        # self.flip_ratio = flip_ratio
        self.flag = np.ones(len(self), dtype=np.uint8)
        # self.task = task

    # def __getitem__(self, idx):
    #     # load image
    #     img = mmcv.imread(os.path.join(self.img_prefix, self.labels['names'][idx]))
    #     ori_shape = img.shape
    #     flip = np.random.rand() < self.flip_ratio
    #     img, img_shape, pad_shape, scale_factor = self.img_transform(img, 1, flip)
    #     # if not self.phase == 'test':
    #     gt_cls = np.array([
    #         self.labels['weather'][idx],
    #         self.labels['scene'][idx],
    #         self.labels['timeofday'][idx]])
    #     img_meta = dict(
    #         ori_shape=ori_shape,
    #         img_shape=img_shape,
    #         pad_shape=pad_shape,
    #         scale_factor=scale_factor,
    #         flip=flip,
    #         task='cls',
    #         file_name=self.labels['names'][idx],
    #         gt_cls=DC(to_tensor(gt_cls)))
    #     data = dict(
    #         img=DC(to_tensor(img), stack=True) if self.phase != 'test' else [img],
    #         img_meta=DC(img_meta, cpu_only=True) if self.phase != 'test' else [DC(img_meta, cpu_only=True)])
    #     if self.phase != 'test':
    #         data['gt_cls'] = DC(to_tensor(gt_cls))
    #     return data

    def load_annotations(self):
        data_infos = []
        for file_name, label_weather, label_scene, label_time in \
            zip(self.labels['name'], self.labels['weather'], \
                self.labels['scene'], self.labels['timeofday']):
            info = {'img_prefix': self.img_prefix}
            info['img_info'] = {'filename': file_name}
            gt_cls = np.array([
            label_weather,
            label_scene,
            label_time], dtype=np.int64)
            info['gt_label'] = gt_cls
            data_infos.append(info)

        return data_infos

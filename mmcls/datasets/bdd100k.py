import os

import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS
from mmcls.models.losses import accuracy, accuracy_class
from mmcls.core.evaluation import precision_recall_f1, support
import mmcv
import json


NAME_DICT = {0: 'weather', 1:'scene', 2:'time'}
CLASS_DICT = {0: 6, 1: 6, 2: 3}

@DATASETS.register_module
class BddCls(BaseDataset):

    def __init__(self,
                 image_dir,
                 label_dir,
                 *args,
                 **kwargs):
        # data
        # self.test_mode = test_mode
        self.image_dir = image_dir
        self.img_prefix = self.image_dir
        with open(label_dir) as f:
            label_data = json.load(f)
        self.labels = {}
        for k in label_data.keys():
            self.labels[k] = label_data[k]
        super(BddCls, self).__init__(*args, **kwargs)
        # self.phase = phase.split('_')[-1]
        # self.flip_ratio = flip_ratio
        # self.flag = np.ones(len(self), dtype=np.uint8)
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
            zip(self.labels['names'], self.labels['weather'], \
                self.labels['scene'], self.labels['timeofday']):
            info = {'img_prefix': self.img_prefix}
            info['img_info'] = {'filename': file_name}
            gt_cls = np.array([
            label_weather - 1,
            label_scene - 1,
            label_time - 1], dtype=np.int64)
            info['gt_label'] = gt_cls
            data_infos.append(info)

        return data_infos
    
    def evaluate(self,
                 results,
                 metric='accuracy',
                 metric_options=None,
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'topk', 'thrs' and 'average_mode'.
                Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
        Returns:
            dict: evaluation results
        """
        if metric_options is None:
            metric_options = {'topk': (1,)}
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'support'
        ]
        eval_results = {}
        results_0 = np.vstack(results[0])
        # print('\npin1\n', results_0.shape, '\npin1\n')
        results_1 = np.vstack(results[1])
        results_2 = np.vstack(results[2])
        list_results = [results_0, results_1, results_2]
        gt_labels_0, gt_labels_1, gt_labels_2 = self.get_gt_labels()
        list_gt_labels = [gt_labels_0, gt_labels_1, gt_labels_2]

        num_imgs = len(results[0])
        # print('\npin2\n', len(gt_labels_0), num_imgs, '\npin2\n')
        # assert len(gt_labels_0) == num_imgs, 'dataset testing results should '\
        #     'be of the same length as gt_labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metirc {invalid_metrics} is not supported.')

        topk = metric_options.get('topk', (1,))
        # print('\npin\n', topk, '\npin\n')
        thrs = metric_options.get('thrs')
        average_mode = metric_options.get('average_mode', 'macro')

        if 'accuracy' in metrics:
            list_acc = [accuracy_class(results_i, gt_labels_i, topk=topk, thrs=thrs, num_class=CLASS_DICT[idx]) \
                for idx, (results_i, gt_labels_i) in enumerate(zip(list_results, list_gt_labels))]

            if isinstance(topk, tuple):
                list_eval_results_ = []
                for i, acc in enumerate(list_acc):
                    eval_results_ = {
                        f'accuracy_top-{k}-{NAME_DICT[i]}-avg': a
                        for k, a in zip(topk, [acc.pop()])
                    }
                    for j, class_acc in enumerate(acc):
                        eval_results_.update({
                            f'accuracy_top-1-{NAME_DICT[i]}-{j}':class_acc[0]
                        })
                    list_eval_results_.append(eval_results_)
                    
            else:
                list_eval_results_ = [{f'accuracy_{NAME_DICT[i]}': acc} for i, acc in enumerate(list_acc)]
            if isinstance(thrs, tuple):
                for key, values in eval_results_.items():
                    eval_results.update({
                        f'{key}_thr_{thr:.2f}': value.item()
                        for thr, value in zip(thrs, values)
                    })
            else:
                for eval_results_i_ in list_eval_results_:
                    eval_results.update(
                        {k: v.item()
                        for k, v in eval_results_i_.items()})

        if 'support' in metrics:
            support_value = support(
                results, gt_labels, average_mode=average_mode)
            eval_results['support'] = support_value

        precision_recall_f1_keys = ['precision', 'recall', 'f1_score']
        if len(set(metrics) & set(precision_recall_f1_keys)) != 0:
            list_precision_recall_f1_keys = []
            list_precision_recall_f1_values = []
            for i, results, gt_labels in enumerate(zip(list_results, list_gt_labels)):
                list_precision_recall_f1_keys.extend([key_i + f'_{NAME_DICT[i]}' for key_i in precision_recall_f1_keys])
                precision_recall_f1_values = precision_recall_f1(
                    results, gt_labels, average_mode=average_mode, thrs=thrs)
                list_precision_recall_f1_values.extend(precision_recall_f1_values)

            for key, values in zip(precision_recall_f1_keys,
                                   precision_recall_f1_values):
                if key in metrics:
                    if isinstance(thrs, tuple):
                        eval_results.update({
                            f'{key}_thr_{thr:.2f}': value
                            for thr, value in zip(thrs, values)
                        })
                    else:
                        eval_results[key] = values

        return eval_results

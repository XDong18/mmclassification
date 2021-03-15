from .cls_head import ClsHead
from .linear_head import LinearClsHead
from .multi_label_head import MultiLabelClsHead
from .multi_label_linear_head import MultiLabelLinearClsHead
from .new_multi_linear_head import NewMultiLinearClsHead
from .dla_cls_head import DLAClsHead

__all__ = [
    'ClsHead', 'LinearClsHead', 'MultiLabelClsHead', 'MultiLabelLinearClsHead', 'NewMultiLinearClsHead',
    'DLAClsHead'
]

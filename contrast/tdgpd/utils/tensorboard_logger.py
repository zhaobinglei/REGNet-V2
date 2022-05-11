import time
import os.path as osp
import os
import torch

from .metric_logger import AverageMeter
from tensorboardX import SummaryWriter
#from .io_utils import mkdir

_KEYWORDS = ("loss", "pct", "acc", "err")


class TensorboardLogger(object):
    def __init__(self, log_dir, keywords=_KEYWORDS):
        self.log_dir = osp.join(log_dir, "events.{}".format(time.strftime("%m_%d_%H_%M_%S")))
        os.mkdir(self.log_dir)
        self.keywords = keywords
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def add_scalars(self, meters, step, prefix=""):
        for k, meter in meters.items():
            for keyword in _KEYWORDS:
                if keyword in k:
                    if isinstance(meter, AverageMeter):
                        v = meter.global_avg
                    elif isinstance(meter, (int, float)):
                        v = meter
                    elif isinstance(meter, torch.Tensor):
                        v = meter.mean().cpu().item()
                    else:
                        raise TypeError('Unknown meter type.')
                    self.writer.add_scalar(osp.join(prefix, k), v, global_step=step)

    def add_image(self, img, step, prefix=""):
        assert len(img.size()) == 3
        self.writer.add_image(osp.join(prefix, "_img"), img, global_step=step)

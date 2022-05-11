# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import defaultdict
from collections import deque

import numpy as np
import torch


class AverageMeter(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.values = deque(maxlen=window_size)
        self.counts = deque(maxlen=window_size)
        self.sum = 0.0
        self.count = 0

    def update(self, value, count=1):
        self.values.append(value)
        self.counts.append(count)
        self.sum += value
        self.count += count

    @property
    def avg(self):
        if np.sum(self.counts) == 0:
            return 0
        return np.sum(self.values) / np.sum(self.counts)

    @property
    def global_avg(self):
        if self.count == 0:
            return 0
        return self.sum / self.count


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                count = v.numel()
                value = v.item() if count == 1 else v.sum().item()
            elif isinstance(v, np.ndarray):
                count = v.size
                value = v.item() if count == 1 else v.sum().item()
            else:
                assert isinstance(v, (float, int))
                value = v
                count = 1
            self.meters[k].update(value, count)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        return object.__getattr__(self, attr)

    def __str__(self):
        metric_str = []
        for name, meter in self.meters.items():
            metric_str.append(
                "{}: Avg: {:.4f} Global Avg: {:.4f}".format(name, meter.avg, meter.global_avg)
            )
        return self.delimiter.join(metric_str)

    @property
    def summary_str(self):
        metric_str = []
        for name, meter in self.meters.items():
            metric_str.append(
                "{}: {:.4f}".format(name, meter.global_avg)
            )
        return self.delimiter.join(metric_str)

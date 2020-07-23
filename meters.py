import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sum_p2 = 0
        self.count = 0
        self._var = 0
        self._var_count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.sum_p2 += n * (val**2)
        self.count += n
        self.avg = self.sum / self.count


    @property
    def var(self):
        if self.count > 1 and self.count>self._var_count:
            self._var_count=self.count
            self._var = (self.sum_p2 - (self.sum**2)/self.count)/(self.count-1)
        return self._var

    @property
    def std(self):
        return self.var**0.5


class OnlineMeter(object):
    """Computes and stores the average and variance/std values of tensor"""

    def __init__(self,batched=False,track_cov=False):
        self.mean = torch.FloatTensor(1).fill_(-1)
        self.M2 = torch.FloatTensor(1).zero_()
        self.count = 0.
        self.needs_init = True
        self.batched = batched
        self.track_covariance = track_cov

    def reset(self, x):
        size=x[0].size()
        if self.track_covariance:
            cov_size = torch.prod(torch.tensor(size))
            self.xtx = x.new(cov_size, cov_size).zero_()

        self.mean = x.new(size).zero_()
        self.M2 = x.new(size).zero_()
        self.count = 0.
        self.needs_init = False

    def update(self, x):
        self.val = x
        if not self.batched:
            x_ = x.unsqueeze(0)
        else:
            x_ = x

        if self.needs_init:
            self.reset(x_)

        num_observations = x_.shape[0]
        self.count += num_observations
        delta = x_.mean(0) - self.mean
        scale = num_observations/self.count
        self.mean.add_(delta.mul_(scale))

        ## calc centered sum squares
        centered_x = x_ - self.mean
        if self.track_covariance:
            x_2d = centered_x.view(num_observations,-1)
            # outer product sum over all samples
            self.xtx += x_2d.transpose(1,0).matmul(x_2d)
        # keep this for now even if covariance if calculated
        delta_p2 = (centered_x * centered_x).sum(0)
        # reduce sum batch dimension
        self.M2.add_(delta_p2)

    @property
    def cov(self):
        if not self.track_covariance or self.count < 2:
            return 0
        return self.xtx / (self.count - 1)

    @property
    def var(self):
        if self.count < 2:
            return self.M2.clone().zero_()
        return self.M2 / (self.count - 1)

    @property
    def std(self):
        return self.var().sqrt()


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t().type_as(target)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AccuracyMeter(object):
    """Computes and stores the average and current topk accuracy"""

    def __init__(self, topk=(1,)):
        self.topk = topk
        self.reset()

    def reset(self):
        self._meters = {}
        for k in self.topk:
            self._meters[k] = AverageMeter()

    def update(self, output, target):
        n = target.nelement()
        acc_vals = accuracy(output, target, self.topk)
        for i, k in enumerate(self.topk):
            self._meters[k].update(acc_vals[i])

    @property
    def val(self):
        return {n: meter.val for (n, meter) in self._meters.items()}

    @property
    def avg(self):
        return {n: meter.avg for (n, meter) in self._meters.items()}

    @property
    def avg_error(self):
        return {n: 100. - meter.avg for (n, meter) in self._meters.items()}

class ConfusionMeter(AccuracyMeter):
    """Computes and stores the average and current topk accuracy"""

    def __init__(self, topk=(1,),nclasses=1000):
        super().__init__(topk)
        self.nclasses = nclasses
        self.reset()

    def reset(self):
        self._confusion_matrix =None
        super().reset()

    def update(self, output, target):
        if self._confusion_matrix is None:
            self._confusion_matrix = torch.zeros(self.nclasses, self.nclasses)
        pred = output.argmax(1)

        for p,t in zip(pred.view(-1),target.view(-1)):
            self._confusion_matrix[t.item(), p.item()] += 1

        super().update(output,target)

    @property
    def confusion(self):
        return self._confusion_matrix.clone()

    @property
    def confusion_normlized(self):
        return self._confusion_matrix / self._confusion_matrix.sum(1)

    @property
    def per_class_accuracy(self):
        return self._confusion_matrix.diag() / self._confusion_matrix.sum(1)

### a dictionary of meters, used in conjunction with CorrCriterionWrapper stats recorder to aggregate statistics
# (due to OnlineMeter restriction only expects a single observation at a time by default)
from _collections import OrderedDict
class MeterDict(OrderedDict):
    def __init__(self, online_meter_class=OnlineMeter,meter_factory=None):
        super().__init__()
        self.online_meter_class = online_meter_class
        self.meter_factory = meter_factory

    def __setitem__(self, key, value):
        if not isinstance(value, self.online_meter_class):
            meter_val = self.meter_factory(key,value) if self.meter_factory else self.online_meter_class()
            meter_val.update(value.detach())
            value = meter_val
        super().__setitem__(key, value)

    def update(self, *args, **kwargs):
        def _update(k_, v_):
            if k_ not in self.keys():
                self.__setitem__(k_, v_)
            else:
                v_old = self.__getitem__(k_)
                v_old.update(v_.detach())

        for other in args:
            if hasattr(other, 'keys'):
                for k in other.keys():
                    _update(k, other[k])
            else:
                for k, v in other:
                    _update(k, v)

        if kwargs:
            self.update(kwargs)

    def __repr__(self):
        return f'{dict([(k, v.mean) for (k, v) in self.items()])}'
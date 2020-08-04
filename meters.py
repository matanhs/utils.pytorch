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
    def __init__(self, batched=False, track_cov=False, track_percentiles=False, target_percentiles=None,
                 per_channel=False, number_edge_samples=0):
        self.mean = torch.FloatTensor(1).fill_(-1)
        self.M2 = torch.FloatTensor(1).zero_()
        self.count = 0.
        self.needs_init = True
        self.batched = batched
        self.track_covariance = track_cov
        self.track_percentiles = track_percentiles
        self.per_channel = per_channel
        self.target_percentiles = target_percentiles
        self.number_edge_samples = number_edge_samples

    def reset(self, x):
        self.sample_shape = x[0].size()
        if self.per_channel:
            self.num_variables = self.sample_shape[0]
        else:
            self.num_variables = x[0].numel()

        if self.track_percentiles:
            self.percentiles = AverageMeter()
            if self.number_edge_samples > 0:
                if self.per_channel:
                    perc_shape = (self.number_edge_samples, self.num_variables)
                else:
                    perc_shape = (self.number_edge_samples,)+ tuple(self.sample_shape)
                perc_shape = torch.Size(perc_shape)
                self.max_values_observed = x.new(perc_shape).fill_(-float("inf"))
                self.min_values_observed = x.new(perc_shape).fill_(float("inf"))

        if self.per_channel:
            self.mean = x.new(self.num_variables).zero_()
            self.M2 = x.new(self.num_variables).zero_()
        else:
            self.mean = x.new(self.sample_shape).zero_()
            self.M2 = x.new(self.sample_shape).zero_()

        if self.track_covariance:
            self.cov = x.new(self.num_variables, self.num_variables).zero_()

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

        num_observations = x_.numel()//self.num_variables
        self.count += num_observations

        if self.per_channel:
            # add spatial elements to the first dimension
            x_ = x_.transpose(1,0).contiguous().view(self.num_variables,num_observations).transpose(1,0)

        delta = x_.mean(0) - self.mean
        scale = num_observations/self.count
        self.mean.add_(delta.mul_(scale))

        ## calc centered sum squares
        centered_x = x_ - self.mean
        # calc variance for now even if covariance if calculated
        # reduce sum batch dimension
        delta_p2 = (centered_x * centered_x).sum(0)
        # update second moment accumulator
        self.M2.add_(delta_p2)
        if self.track_covariance:
            if not self.per_channel:
                #flatten the variable samples for covariance computation,
                centered_x = centered_x.view(num_observations,self.num_variables)
            new_covariance = centered_x.transpose(1,0).matmul(centered_x).div_(num_observations)
            delta = new_covariance.sub_(self.cov)
            # update mean covariance
            scale = num_observations / self.count
            self.cov.add_(delta.mul_(scale))

        if self.track_percentiles:
            x_sorted = x_.sort(0)[0]
            ## note that percentiles are observed only within a given batch
            # if the number of observation is too small to represent the requested percentiles the result will be
            # value duplication!
            percentile_ids = torch.repeat_interleave(
                torch.round(self.target_percentiles * (num_observations - 1)).long()[:, None],
                self.num_variables, 1).to(x_.device)
            if not self.per_channel:
                percentile_ids = percentile_ids.reshape(((percentile_ids.shape[0],)+self.sample_shape))
            # update averege meter that approximates the percentiles (tractable statistic)
            self.percentiles.update(x_sorted.gather(0, percentile_ids))
            self.min_values_observed = torch.cat([self.min_values_observed,
                                                  x_sorted[:self.number_edge_samples]]).topk(self.number_edge_samples,
                                                                                             dim=0,sorted=True,
                                                                                             largest=False)[0]
            self.max_values_observed = torch.cat([self.max_values_observed,
                                                  x_sorted[-self.number_edge_samples:]]).topk(self.number_edge_samples,
                                                                                              dim=0, sorted=True,
                                                                                              largest=True)[0]

    @property
    def var(self):
        if self.count < 2:
            return self.M2.clone().zero_()
        return self.M2 / (self.count - 1)

    @property
    def std(self):
        return self.var().sqrt()

    def get_distribution_histogram(self,edge_subsample_rate=10):
        edge_percentiles_ids_low = torch.arange(0, self.number_edge_samples, edge_subsample_rate)
        edge_percentiles_ids_high = self.number_edge_samples - reversed(edge_percentiles_ids_low) - 1
        quantiles = torch.cat([self.min_values_observed[edge_percentiles_ids_low], self.percentiles.avg,
                               self.max_values_observed[edge_percentiles_ids_high]])
        edge_percentiles = (edge_percentiles_ids_low+1) /self.count
        percentiles = torch.cat([edge_percentiles, self.target_percentiles , 1 - edge_percentiles])
        return percentiles,quantiles

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


if __name__ == '__main__':
    import numpy as np
    ## test only per channel mode
    m=OnlineMeter(batched=True,track_cov=True,track_percentiles=True,percentiles=torch.tensor([0.,1.]),per_channel=True,number_edge_samples=1)
    x= torch.randn(100000,8,3,3)
    cov_np=np.cov(x.transpose(1, 0).contiguous().view(100000 * 9, 8).transpose(1, 0).numpy())
    var_np=np.var(x.numpy(),(0,2,3))
    for i in range(x.shape[0]//1000):
        m.update(x[i*1000:(i+1)*1000])
    diff = 0
    for np_ref,meter_out in [[cov_np,m.cov],[np.diag(cov_np),m.var],[var_np,m.var],[var_np,torch.diag(m.cov)]]:
        for diff_reduce_fn in [np.median,np.max,np.mean]:
            diff = diff_reduce_fn(np.abs(np_ref-meter_out.numpy()))

    pass
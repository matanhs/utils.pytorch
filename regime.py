import torch
from copy import deepcopy
from six import string_types
import math


def eval_func(f, x):
    if isinstance(f, string_types):
        f = eval(f)
    return f(x)


def test_step_lambda(lambda_string,n_steps=10,plot=True):
    lrs=[eval_func(lambda_string, i)['lr'] for i in range(n_steps)]
    if plot:
        from matplotlib import pyplot as plt
        plt.plot(lrs)
    else:
        return lrs


def linear_lr(lr0, lrT, T0, T):
    rate = (lrT - lr0) / (T-T0)
    return f"lambda t: {{'lr': {lr0} - (t-{T0}) * {rate}}}"


def ramp_up_lr(lr0, lrT, T):
    rate = (lrT - lr0) / T
    return "lambda t: {'lr': %s + t * %s}" % (lr0, rate)


def exp_decay_lr(lr0, lrT,T0, T,factor=None,n_drops=None):
    tot_calls=T-T0
    n_drops=n_drops or (tot_calls - 1)
    drop_every_n_calles=max(1,tot_calls//(n_drops+1))
    assert lr0>0 and lrT>0 and tot_calls>0
    factor = factor or torch.exp(1/n_drops*torch.log(torch.tensor(lrT/lr0))).item()
    print('lr decay scale:',factor)
    return f"lambda t: {{'lr': max({lr0} * {factor} ** ((t-{T0})//{drop_every_n_calles}),{lrT})}}"


def lr_drops(lr0, lrT,T0, T,n_drops):
    steps_T=T-T0
    steps_per_drop=steps_T//(n_drops+1)
    factor= torch.exp(1/n_drops*torch.log(torch.tensor(lrT/lr0))).item()
    print('lr drop scale:',factor)
    return "lambda t: {'lr': max(%s * %s ** ((t-%s)//%s),%s)}" % (lr0, factor,T0,steps_per_drop,lrT)


def cosine_anneal_lr(lr0,lr_T,T0,T,n_drops=None):
    delta_T=T-T0
    if n_drops:
        steps_per_drop=delta_T//(n_drops+1)
        return f"lambda t: {{'lr': {lr_T} + ({lr0} - {lr_T})*(1+math.cos(math.pi*((t-{T0})//{steps_per_drop})/{n_drops}))/2}}"
    else:
        return f"lambda t: {{'lr': {lr_T} + ({lr0} - {lr_T})*(1+math.cos(math.pi*((t-{T0})/{delta_T})))/2}}"

class Regime(object):
    """
    Examples for regime:

    1)  "[{'epoch': 0, 'optimizer': 'Adam', 'lr': 1e-3},
          {'epoch': 2, 'optimizer': 'Adam', 'lr': 5e-4},
          {'epoch': 4, 'optimizer': 'Adam', 'lr': 1e-4},
          {'epoch': 8, 'optimizer': 'Adam', 'lr': 5e-5}
         ]"
    2)
        "[{'step_lambda':
            "lambda t: {
            'optimizer': 'Adam',
            'lr': 0.1 * min(t ** -0.5, t * 4000 ** -1.5),
            'betas': (0.9, 0.98), 'eps':1e-9}
         }]"
    """

    def __init__(self, regime, defaults={}):
        self.regime = regime
        self.current_regime_phase = None
        self.setting = defaults

    def update(self, epoch=None, train_steps=None):
        """adjusts according to current epoch or steps and regime.
        """
        if self.regime is None:
            return False
        epoch = -1 if epoch is None else epoch
        train_steps = -1 if train_steps is None else train_steps
        setting = deepcopy(self.setting)
        if self.current_regime_phase is None:
            # Find the first entry where the epoch is smallest than current
            for regime_phase, regime_setting in enumerate(self.regime):
                start_epoch = regime_setting.get('epoch', 0)
                start_step = regime_setting.get('step', 0)
                if epoch >= start_epoch or train_steps >= start_step:
                    self.current_regime_phase = regime_phase
                    break
                # each entry is updated from previous
                setting.update(regime_setting)
        if len(self.regime) > self.current_regime_phase + 1:
            next_phase = self.current_regime_phase + 1
            # Any more regime steps?
            start_epoch = self.regime[next_phase].get('epoch', float('inf'))
            start_step = self.regime[next_phase].get('step', float('inf'))
            if epoch >= start_epoch or train_steps >= start_step:
                self.current_regime_phase = next_phase
        setting.update(self.regime[self.current_regime_phase])

        if 'lr_decay_rate' in setting and 'lr' in setting:
            decay_steps = setting.pop('lr_decay_steps', 100)
            if train_steps % decay_steps == 0:
                decay_rate = setting.pop('lr_decay_rate')
                setting['lr'] *= decay_rate ** (train_steps / decay_steps)
        elif 'step_lambda' in setting:
            setting.update(eval_func(setting.pop('step_lambda'), train_steps))
        elif 'epoch_lambda' in setting:
            setting.update(eval_func(setting.pop('epoch_lambda'), epoch))

        if 'execute' in setting:
            setting.pop('execute')()

        if 'execute_once' in setting:
            setting.pop('execute_once')()
            # remove from regime, so won't happen again
            self.regime[self.current_regime_phase].pop('execute_once', None)

        if setting == self.setting:
            return False
        else:
            self.setting = setting
            return True

    def __repr__(self):
        return str(self.regime)
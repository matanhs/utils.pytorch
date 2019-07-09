import torch
import torch.nn as nn
import logging

_ABSORBING_CLASSES = [torch.nn.modules.conv.Conv2d, torch.nn.modules.linear.Linear]
def remove_bn_params(bn_module):
    bn_module.register_buffer('running_mean', None)
    bn_module.register_buffer('running_var', None)
    bn_module.register_parameter('weight', None)
    bn_module.register_parameter('bias', None)
    bn_module.register_buffer('num_batches_tracked', None)


def init_bn_params(bn_module,keep_modifiers=False):
    bn_module.num_batches_tracked.fill_(0)
    bn_module.running_mean.fill_(0)
    bn_module.running_var.fill_(1)
    if bn_module.affine and not keep_modifiers:
         bn_module.weight.fill_(1)
         bn_module.bias.fill_(0)


def absorb_bn(module, bn_module, remove_bn=True, verbose=False,keep_modifiers=False):
    with torch.no_grad():
        w = module.weight
        if module.bias is None:
            zeros = torch.zeros(module.out_channels,
                                dtype=w.dtype, device=w.device)
            bias = nn.Parameter(zeros)
            module.register_parameter('bias', bias)
        b = module.bias

        if hasattr(bn_module, 'running_mean'):
            b.add_(-bn_module.running_mean)
        if hasattr(bn_module, 'running_var'):
            #print(bn_module.running_var.abs().min())
            invstd = bn_module.running_var.clone().add_(bn_module.eps).pow_(-0.5)
            w.mul_(invstd.view(w.size(0), 1, 1, 1))
            b.mul_(invstd)

        if not keep_modifiers and bn_module.affine:
            if hasattr(bn_module, 'weight'):
                w.mul_(bn_module.weight.view(w.size(0), 1, 1, 1))
                b.mul_(bn_module.weight)
            if hasattr(bn_module, 'bias'):
                b.add_(bn_module.bias)

        if remove_bn:
            remove_bn_params(bn_module)
        else:
            init_bn_params(bn_module,keep_modifiers=keep_modifiers)

        if verbose:
            logging.info('BN module %s was asborbed into layer %s' %
                         (bn_module, module))

def absorb_bn_step(module, bn_module, remove_bn=True, verbose=False,keep_modifiers=False,lr=1.0):
    with torch.no_grad():
        w_ = module.weight
        w = module.weight.clone()
        if module.bias is None:
            zeros = torch.zeros(module.out_channels,
                                dtype=w.dtype, device=w.device)
            bias = nn.Parameter(zeros)
            module.register_parameter('bias', bias)
        b_ = module.bias
        b = module.bias.clone()

        if hasattr(bn_module, 'running_mean'):
            b.add_(-bn_module.running_mean)
        if hasattr(bn_module, 'running_var'):
            invstd = bn_module.running_var.clone().add_(bn_module.eps).pow_(-0.5)
            w.mul_(invstd.view(w.size(0), 1, 1, 1))
            b.mul_(invstd)

        if bn_module.affine:
            if hasattr(bn_module, 'weight'):
                w.mul_(bn_module.weight.view(w.size(0), 1, 1, 1))
                b.mul_(bn_module.weight)
            if hasattr(bn_module, 'bias'):
                b.add_(bn_module.bias)

        w_.mul_(1-lr).add_(w.mul_(lr))
        b_.mul_(1 - lr).add_(b.mul_(lr))

        if remove_bn:
            remove_bn_params(bn_module)
        else:
            init_bn_params(bn_module,keep_modifiers=keep_modifiers)

        if verbose:
            logging.info('BN module %s was asborbed into layer %s' %
                         (bn_module, module))

def get_absorb_bn_w_b(module, bn_module,keep_modifiers=False):
    w = module.weight.clone()
    b = module.bias.clone()

    b = b.add(-bn_module.running_mean)

    invstd = bn_module.running_var.clone().add_(bn_module.eps).pow_(-0.5)
    w = w.mul(invstd.view(w.size(0), 1, 1, 1))
    b = b.mul(invstd)

    if bn_module.affine:
        w = w.mul(bn_module.weight.view(w.size(0), 1, 1, 1))
        b = b.mul(bn_module.weight)
        b = b.add(bn_module.bias)

    init_bn_params(bn_module,keep_modifiers=keep_modifiers)
    return w,b

def absorb_bn_step_old(module, bn_module, remove_bn=True, verbose=False,keep_modifiers=False,lr=1.0):
    with torch.no_grad():
        w = module.weight
        w_ = w.copy()
        if module.bias is None:
            zeros = torch.zeros(module.out_channels,
                                dtype=w.dtype, device=w.device)
            bias = nn.Parameter(zeros)
            module.register_parameter('bias', bias)
        b = module.bias

        if hasattr(bn_module, 'running_mean'):
            b.add_(-bn_module.running_mean)
        if hasattr(bn_module, 'running_var'):
            invstd = bn_module.running_var.clone().add_(bn_module.eps).pow_(-0.5)
            w_ = w.mul(invstd.view(w.size(0), 1, 1, 1)).mul_(lr)
            w.mul_(1-lr).add_(w_)
            b_=b.mul(invstd).mul_(lr)
            b.mul_(1-lr).add_(b_)

        if not keep_modifiers and bn_module.affine:
            if hasattr(bn_module, 'weight'):
                w.mul_(bn_module.weight.view(w.size(0), 1, 1, 1))
                b.mul_(bn_module.weight)
            if hasattr(bn_module, 'bias'):
                b.add_(bn_module.bias)

        if remove_bn:
            remove_bn_params(bn_module)
        else:
            init_bn_params(bn_module,keep_modifiers=keep_modifiers)

        if verbose:
            logging.info('BN module %s was asborbed into layer %s' %
                         (bn_module, module))

def compare(w,w_):
    print(torch.abs(w-w_).max().item(),torch.abs(w-w_).mean().item(),torch.abs(w-w_).std().item())

def is_bn(m):
    return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)


def is_absorbing(m):
    return any(isinstance(m, x) for x in _ABSORBING_CLASSES)


def search_absorbe_bn(model, prev=None, remove_bn=True, verbose=False,keep_modifiers=False):
    with torch.no_grad():
        for m in model.children():
            if is_bn(m) and is_absorbing(prev):
                absorb_bn(prev, m, remove_bn=remove_bn, verbose=verbose,keep_modifiers=keep_modifiers)
            search_absorbe_bn(m, remove_bn=remove_bn, verbose=verbose,keep_modifiers=keep_modifiers)
            prev = m

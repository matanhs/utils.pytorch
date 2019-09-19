import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from argparse import ArgumentParser
torch_dtypes = {
    'float': torch.float,
    'float32': torch.float32,
    'float64': torch.float64,
    'double': torch.double,
    'float16': torch.float16,
    'half': torch.half,
    'uint8': torch.uint8,
    'int8': torch.int8,
    'int16': torch.int16,
    'short': torch.short,
    'int32': torch.int32,
    'int': torch.int,
    'int64': torch.int64,
    'long': torch.long
}


'''
strict dot dictionary, all attribute defined in _ATTRS must be provided.
usefull for storing meta-data, dependent configurations etc. 
'''
class _META(object):
    _ATTRS=[]
    def __init__(self,**kwargs):
        super(_META,self).__init__()
        self.attrs={}
        assert all([attr in kwargs for attr in _META._ATTRS])

        for attr,val in kwargs.items():
            setattr(self,attr,val)
            self.attrs[attr]=val

    def get_attrs(self):
        return self.attrs


class AutoArgParser(ArgumentParser):
    def __init__(self):
        super().__init__()

    def auto_add(self,settings,enforce_type=True):
        for k,v in settings.items():
            if enforce_type:
                self.add_argument(f'-{k}',default=v,type=type(v))
            else:
                self.add_argument(f'-{k}',default=v)



def onehot(indexes, N=None, ignore_index=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    ignore_index will be zero in onehot representation
    """
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().byte().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    if ignore_index is not None and ignore_index >= 0:
        output.masked_fill_(indexes.eq(ignore_index).unsqueeze(-1), 0)
    return output


def set_global_seeds(i):
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(i)
    np.random.seed(i)
    random.seed(i)


class CheckpointModule(nn.Module):
    def __init__(self, module, num_segments=1):
        super(CheckpointModule, self).__init__()
        assert num_segments == 1 or isinstance(module, nn.Sequential)
        self.module = module
        self.num_segments = num_segments

    def forward(self, x):
        if self.num_segments > 1:
            return checkpoint_sequential(self.module, self.num_segments, x)
        else:
            return checkpoint(self.module, x)

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)


import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F

class CosineSimilarityChannelWiseLoss(nn.Module):
    def __init__(self,reduction='mean'):
        assert reduction in ['mean','sum','max']
        super(CosineSimilarityChannelWiseLoss,self).__init__()
        self.reduction = reduction

    def forward(self, input,target):
        #per_channel_loss = torch.zeros((input.size(1)),device=input.device)
        # channel first

        per_channel_loss = F.cosine_similarity(input.transpose(1,0).view(input.size(1),input.size(0),-1),target.transpose(1,0).view(target.size(1),target.size(0),-1),-1).mean(1)
        # channel reduction:
        if self.reduction == 'mean':
            return per_channel_loss.mean()
        elif self.reduction == 'sum':
            return per_channel_loss.sum()
        elif self.reduction == 'max':
            return per_channel_loss.max()


class GaussianSmoothing(nn.Module):
    # code from https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/9
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim
        if isinstance(sigma, torch.Tensor):
            sigma = [sigma.item()] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.pad = kernel_size[0] // 2
        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups,padding = self.pad)


# smoothing = GaussianSmoothing(3, 5, 1)
# input = torch.rand(1, 3, 100, 100)
# input = F.pad(input, (2, 2, 2, 2), mode='reflect')
# output = smoothing(input)
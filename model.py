import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import (
    change_filters,
    change_repeats,
    get_configs
)


class MBConvBlock(nn.Module):
    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self._se_ratio = 0.25
        
        # Expansion phase
        inp = self._block_args.in_channels
        out = self._block_args.in_channels * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            self._expand_conv = nn.Conv2d(in_channels=inp, out_channels=out, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=out, eps=self._bn_eps, momentum=self._bn_mom)
            
        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = nn.Conv2d(in_channels=out, out_channels=out, groups=out, 
                                         kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out, eps=self._bn_eps, momentum=self._bn_mom)
        
        # Squeenze and Excitation layer
        num_squeezed_channels = max(1, int(inp * self._se_ratio))
        self._se_reduce = nn.Conv2d(in_channels=out, out_channels=num_squeezed_channels, kernel_size=1)
        self._se_expand = nn.Conv2d(in_channels=num_squeezed_channels, out_channels=out, kernel_size=1)
        
        # Pointwise convolutional phase
        final_out = self._block_args.out_channels
        self._project_conv = nn.Conv2d(in_channels=out, out_channels=final_out, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_out, eps=self._bn_eps, momentum=self._bn_mom)
        
    def forward(self, inputs, drop_connection_rate=None):
        
        x = inputs
        # Expansion phase
        if self._block_args._expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self.bn0(x)
            
        # Depthwise convolution phase
        x = self._depthwise_conv(x)
        x = self._bn1(x)
        
        # Squeenze and Excitation layer
        x_squeezed = F.adaptive_avg_pool2d(x, 1)
        x_squeezed = self._se_reduce(x_squeezed)
        x_squeezed = self._se_expand(x_squeezed)
        x = torch.sigmoid(x_squeezed) * x
        
        # Pointwise convolutional phase
        x = self._project_conv(x)
        x = self._bn2(x)
        
        # Skip connection and dropout
        inp, out = self._block_args.in_channels, self._block_args.out_channels
        if self._block_args.stride == 1 and inp == out:
            if drop_connection_rate:
                x = F.dropout2d(x, p=drop_connection_rate, training=self.training)
            x = x + inputs
            
        return x
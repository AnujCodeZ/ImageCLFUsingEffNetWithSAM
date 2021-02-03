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
            self._expand_conv = nn.Conv2d(in_channels=inp, out_channels=out, kernel_size=1, padding=0,bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=out, eps=self._bn_eps, momentum=self._bn_mom)
            
        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = nn.Conv2d(in_channels=out, out_channels=out, groups=out, 
                                         kernel_size=k, padding=(k-1)//2, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out, eps=self._bn_eps, momentum=self._bn_mom)
        
        # Squeenze and Excitation layer
        num_squeezed_channels = max(1, int(inp * self._se_ratio))
        self._se_reduce = nn.Conv2d(in_channels=out, out_channels=num_squeezed_channels, kernel_size=1, padding=0)
        self._se_expand = nn.Conv2d(in_channels=num_squeezed_channels, out_channels=out, kernel_size=1, padding=0)
        
        # Pointwise convolutional phase
        final_out = self._block_args.out_channels
        self._project_conv = nn.Conv2d(in_channels=out, out_channels=final_out, kernel_size=1, padding=0, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_out, eps=self._bn_eps, momentum=self._bn_mom)
        
    def forward(self, inputs, drop_connection_rate=None):
        
        x = inputs
        # Expansion phase
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            
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
                x = F.dropout(x, p=drop_connection_rate, training=self.training)
            x = x + inputs
            
        return x

class EfficientNet(nn.Module):
    def __init__(self, blocks_args, global_params):
        super().__init__()
        self._global_params = global_params
        self._blocks_args = blocks_args
        
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon
        
        # Stem
        in_channels = 3
        out_channels = change_filters(32, self._global_params)
        self._conv_stem = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, eps=bn_eps, momentum=bn_mom)
        
        # Building blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:
            
            block_args = block_args._replace(
                in_channels=change_filters(block_args.in_channels, self._global_params),
                out_channels=change_filters(block_args.out_channels, self._global_params),
                num_repeat=change_repeats(block_args.num_repeat, self._global_params)
            )
            
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(in_channels=block_args.out_channels, stride=1)
            
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))
        
        # Head
        in_channels = block_args.out_channels
        out_channels = change_filters(1280, self._global_params)
        self._conv_head = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, eps=bn_eps, momentum=bn_mom)
        
        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_class)
    
    def _extract_features(self, inputs):
        
        # Stem
        x = self._bn0(self._conv_stem(inputs))
        
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connection_rate = self._global_params.drop_connection_rate
            if drop_connection_rate:
                drop_connection_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connection_rate=drop_connection_rate)
        
        # Head
        x = self._bn1(self._conv_head(x))
        
        return x
    
    def forward(self, inputs):
        
        # Cnvolutional layers
        x = self._extract_features(inputs)
        
        # Pooling and final linear layer
        x = self._avg_pooling(x)
        if self._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self._dropout(x)
            x = self._fc(x)
        
        return x
    
    @classmethod
    def from_name(cls, model_name, in_channels=3):
        
        blocks_args, global_params = get_configs(model_name)
        model = cls(blocks_args, global_params)
        
        return model
                
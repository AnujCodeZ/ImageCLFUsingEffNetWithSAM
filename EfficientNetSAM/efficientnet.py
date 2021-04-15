import torch
import torch.nn as nn
from math import ceil


class CNNBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, groups=1,
    ):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()
    
    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))


class SqueezeAndExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeAndExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return x * self.se(x)


class InvertedResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels, 
        kernel_size,
        stride, 
        padding,
        expand_ratio,
        reduction=4, # Squeeze and Excitation
        survival_prob=0.8, # Stochastic depth
    ):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = survival_prob
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int( in_channels / reduction)
        
        if self.expand:
            self.expand_conv = CNNBlock(
                in_channels, hidden_dim, kernel_size=3, stride=1, padding=1,
            )
        
        self.conv = nn.Sequential(
            CNNBlock(
                hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim,
            ),
            SqueezeAndExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    
    def stochastic_depth(self, x):
        if not self.training:
            return x
        
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        
        return torch.div(x, self.survival_prob) * binary_tensor
    
    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs
        
        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)


class EfficientNet(nn.Module):
    def __init__(self, version, num_classes, phi_values, base_model):
        super(EfficientNet, self).__init__()
        self.phi_values = phi_values
        self.base_model = base_model
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = ceil(1280 * width_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(width_factor, depth_factor, last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes),
        )
    
    def calculate_factors(self, version, alpha=1.2, beta=1.2):
        phi, res, drop_rate = self.phi_values[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        return width_factor, depth_factor, drop_rate
    
    def create_features(self, width_factor, depth_factor, last_channels):
        channels = int(32 * width_factor)
        features = [CNNBlock(3, channels, 3, 2, 1)]
        in_channels = channels
        
        for expand_ratio, channels, repeats, stride, kernel_size in self.base_model:
            out_channels = 4 * ceil(int(channels * width_factor) / 4)
            layers_repeats = ceil(repeats * depth_factor)
            
            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride = stride if layer == 0 else 1,
                        padding=kernel_size // 2,
                        expand_ratio=expand_ratio
                    )
                )
                in_channels = out_channels
            
        features.append(
            CNNBlock(in_channels, last_channels, 1, 1, 0),
        )
        
        return nn.Sequential(*features)
    
    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x.view(x.shape[0], -1))
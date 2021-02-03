import math
import torch.nn as nn
from collections import namedtuple


GlobalParams = namedtuple('GlobalParams', [
    'width_coefficient', 'depth_coefficient', 'image_size', 'dropout_rate',
    'num_class', 'batch_norm_momentum', 'batch_norm_epsilon',
    'drop_connection_rate', 'depth_divisor', 'min_depth', 'include_top'
])

BlocksArgs = namedtuple('BlocksArgs', [
    'num_repeat', 'kernel_size', 'stride',
    'expand_ratio', 'in_channels', 'out_channels'
])

def change_filters(filters, global_params):
    
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    
    filters *= multiplier
    min_depth = min_depth or divisor
    
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    
    return int(new_filters)

def change_repeats(repeats, global_params):
    
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    
    return int(math.ceil(multiplier * repeats))

class BlockMaker:
    
    @staticmethod
    def _build_block_args(configs):
        return BlocksArgs(
            num_repeat=configs[0],
            kernel_size=configs[1],
            stride=configs[2],
            expand_ratio=configs[3],
            in_channels=configs[4],
            out_channels=configs[5]
        )
    
    @staticmethod
    def make(blocks_args_configs):
        blocks_args = []
        for configs in blocks_args_configs:
            blocks_args.append(BlockMaker._build_block_args(configs))
        return blocks_args    

def get_configs(model_name, num_class=10, include_top=True):
    
    params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }


    w, d, res, p = params_dict[model_name]
    
    blocks_args_configs = [
        # n, k, s, e, i, o
        [1, 3, 1, 1, 32, 16],
        [2, 3, 2, 6, 16, 24],
        [2, 5, 2, 6, 24, 40],
        [3, 3, 2, 6, 40, 80],
        [3, 5, 1, 6, 80, 112],
        [4, 5, 2, 6, 112, 192],
        [1, 3, 1, 6, 192, 320]
    ]
    
    blocks_args = BlockMaker.make(blocks_args_configs)
    
    global_params = GlobalParams(
        width_coefficient=w,
        depth_coefficient=d,
        image_size=res,
        dropout_rate=p,
        drop_connection_rate=p,
        num_class=num_class,
        batch_norm_momentum=0.01,
        batch_norm_epsilon=1e-3,
        depth_divisor=8,
        min_depth=None,
        include_top=include_top
    )
    
    return blocks_args, global_params


import torch
from EfficientNetModel.model import EfficientNet
from EfficientNetModel.utils import get_configs

blocks_args, global_params = get_configs('efficientnet-b1')
image_size = global_params.image_size
inputs = torch.rand(1, 3, image_size, image_size)
model = EfficientNet(blocks_args, global_params)
model.eval()
print(model(inputs))
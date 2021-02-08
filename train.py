import argparse
import torch

from EfficientNetSAM.model import EfficientNet
from EfficientNetSAM.utils import get_configs
from EfficientNetSAM.sam import SAM
from Utils.data import Cifar10
from Utils.logger import Logger
from Utils.step_lr import StepLR


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--threads", default=4, type=int)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--learning_rate", default=1e-3, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--rho", default=0.05, type=float)
parser.add_argument("--model_name", default="efficientnet-b0", type=str)
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

blocks_args, global_params = get_configs(args.model_name)
image_size = global_params.image_size

dataset = Cifar10(args.batch_size, image_size, args.threads)
model = EfficientNet(blocks_args, global_params).to(device)
log = Logger()

base_optimizer = torch.optim.SGD
optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, lr=args.learning_rate, 
                momentum=args.momentum, weight_decay=args.weight_decay)
schedular = StepLR(optimizer, args.learning_rate, args.epochs)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(args.epochs):
    model.train()
    log.train(len(dataset.train))
    
    for batch in dataset.train:
        inputs, targets = (b.to(device) for b in batch)
        
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            correct = torch.argmax(predictions.data, 1) == targets
            log(model, loss.cpu(), correct.cpu(), schedular.lr())
            schedular(epoch)
    
    model.eval()
    log.eval(len(dataset.test))
    
    with torch.no_grad():
        for batch in dataset.test:
            inputs, targets = (b.to(device) for b in batch)
            
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            correct = torch.argmax(predictions.data, 1) == targets
            log(model, loss.cpu(), correct.cpu())

log.flush()
import argparse
import torch

from EfficientNetSAM.efficientnet import EfficientNet
import EfficientNetSAM.utils as utils
from EfficientNetSAM.sam import SAM
from Utils.data import Cifar10
from Utils.logger import Logger
from Utils.step_lr import StepLR


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--threads", default=4, type=int)
parser.add_argument("--epochs", default=20, type=int)
parser.add_argument("--learning_rate", default=1e-1, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--rho", default=0.05, type=float)
parser.add_argument("--version", default="b0", type=str)
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

phi, image_size, drop_rate = config.phi_values[version]
num_classes = 10

dataset = Cifar10(args.batch_size, image_size, args.threads)
model = EfficientNet(version, num_classes).to(device)
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
        optimizer.first_step()
        
        criterion(model(inputs), targets).backward()
        optimizer.second_step()
        
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
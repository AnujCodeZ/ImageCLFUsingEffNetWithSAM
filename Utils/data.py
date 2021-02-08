import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


class Cifar10:
    def __init__(self, batch_size, image_size, threads):
        
        mean, std = self._get_mean_std()
        
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = datasets.CIFAR10(root='./data', train=True, transform=train_transform, download=True)
        test_set = datasets.CIFAR10(root='./data', train=False, transform=test_transform, download=True)
        
        self.train = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)
        
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        
    def _get_mean_std(self):
        
        train_set = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        test_set = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=True)
        
        full_data = torch.cat([s[0] for s in DataLoader(train_set)] + [s[0] for s in DataLoader(test_set)])
        
        return full_data.mean(dim=[0, 2, 3]), full_data.std(dim=[0, 2, 3])
        
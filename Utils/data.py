import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Cutout augmentation
class Cutout:
    def __init__(self, size=16, p=0.5):
        self.size = size
        self.half_size = size // 2
        self.p = p
        
    def __call__(self, image):
        if torch.rand([1]).item() > self.p:
            return image
        
        left = torch.randint(-self.half_size, image.size(1) - self.half_size, [1]).item()
        top = torch.randint(-self.half_size, image.size(2) - self.half_size, [1]).item()
        right = min(image.size(1), left + self.size)
        bottom = min(image.size(2), top + self.size)
        
        image[:, max(0, left): right, max(0, top): bottom] = 0
        
        return image

class Cifar10:
    def __init__(self, batch_size, threads):
        
        mean, std = self._get_mean_std()
        
        train_transform = transforms.Compose([
            transforms.RandomCrop(size=(32, 32), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout()
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = datasets.CIFAR10(root='./data', train=True, transform=train_transform, download=True)
        test_set = datasets.CIFAR10(root='./data', train=False, transform=test_transform, download=True)
        
        self.train = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.train = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)
        
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        
    def _get_mean_std():
        
        train_set = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        test_set = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=True)
        
        full_data = torch.cat([s[0] for s in DataLoader(train_set)] + [s[0] for s in DataLoader(test_set)])
        
        return full_data.mean(dim=[0, 2, 3]), full_data.std(dim=[0, 2, 3])
        
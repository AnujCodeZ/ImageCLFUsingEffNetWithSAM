class StepLR:
    def __init__(self, optimizer, learning_rate, total_epochs):
        self.optimizer = optimizer
        self.base = learning_rate
        self.total_epochs = total_epochs
    
    def __call__(self, epoch):
        
        if epoch < self.total_epochs * 3/10:
            lr = self.base
        elif epoch < self.total_epochs * 6/10:
            lr = self.base * 0.3 ** 2
        elif epoch < self.total_epochs * 8/10:
            lr = self.base * 0.3 ** 3
        
        for params in self.optimizer.param_groups:
            params['lr'] = lr
    
    def lr(self):
        return self.optimizer.param_groups[0]['lr']
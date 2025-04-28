import torch
from torch.utils.data import DataLoader, TensorDataset

class DummyDataLoader:
    def __init__(self, config=None):
        self.config = config

    def load_data(self):
        # Example: Random data for demonstration
        x_train = torch.rand(100, 4, 500)
        y_train = torch.randint(0, 15, (100,))
        x_test = torch.rand(20, 4, 500)
        y_test = torch.randint(0, 15, (20,))
        
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        return train_loader, test_loader
import unittest
import torch
from dataloaders.data_loader import DummyDataLoader

class TestDummyDataLoader(unittest.TestCase):
    def setUp(self):
        self.data_loader = DummyDataLoader()
    
    def test_load_dummy_data(self):
        train_loader, test_loader = self.data_loader.load_data()
        self.assertIsInstance(train_loader, torch.utils.data.DataLoader, "Train loader should be a DataLoader")
        self.assertIsInstance(test_loader, torch.utils.data.DataLoader, "Test loader should be a DataLoader")
        self.assertGreater(len(train_loader), 0, "Train loader should not be empty")
        self.assertGreater(len(test_loader), 0, "Test loader should not be empty")



if __name__ == "__main__":
    unittest.main()

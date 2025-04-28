import torch
from models.deepmsn import DeepMSN

def test_deepmsn():
    model = DeepMSN()
    x = torch.rand(2, 4, 500)  # Batch size of 2, 4 channels, 500 length
    output = model(x)
    assert output.shape == (2, 15), "Output shape mismatch"

if __name__ == "__main__":
    test_deepmsn()
    print("All tests passed!")

import torch
from models.deepflybrain import DeepFlyBrain

def test_DeepFlyBrain():
    model = DeepFlyBrain()
    x = torch.rand(2, 4, 500)  # Batch size of 2, 4 channels, 500 length
    output = model(x)
    assert output.shape == (2, 15), "Output shape mismatch"

if __name__ == "__main__":
    test_DeepFlyBrain()
    print("All tests passed!")

import torch
from models.deepmsn import DeepMSN
from dataloaders.data_loader import load_data

def evaluate():
    # Load data
    _, test_loader = load_data()

    # Load model
    model = DeepMSN()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    # Evaluation loop
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    evaluate()

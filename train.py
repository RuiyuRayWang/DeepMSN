import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets.topic_datasets import TopicDataset
from models.deepflybrain import DeepFlyBrain

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        
        # Move data to the device
        X, y = X.to(device), y.to(device)
        y = y.float()  # Cast y to float
        
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y = y.float()  # Cast y to float
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    
    # Hyperparameters
    learning_rate = 1e-3
    weight_decay = 1e-2
    batch_size = 32
    epochs = 100
    
    # Set fixed random number seed
    torch.manual_seed(42)
    
    # Prepare dataset
    dataset = TopicDataset(
        genome='data/resources/mm10.fa',
        region_topic_bed='data/CTdnsmpl_catlas_35_Topics_top_3k/regions_and_topics_sorted.bed',
        transform=None,  # Use default one-hot encoding
        target_transform=None  # Use default target transformation
    )
    
    # Split dataset into training and testing sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Define data loaders for training and testing
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    
    for X, y in test_dataloader:
        print(f"Shape of X [N, L, C]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    
    # Initialize the model
    model = DeepFlyBrain().to(device)
    
    # Define loss function
    loss_function = nn.BCELoss()
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_function, optimizer)
        test_loop(test_dataloader, model, loss_function)
    print("Done!")
    
    # Saving the model  
    save_path = f'./models/state_dict/model.pth'
    torch.save(model.state_dict(), save_path)
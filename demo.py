import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets.topic_datasets import TopicDataset
from models.deepmsn import DeepMSN

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
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
    learning_rate = 2e-4
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
    
    # x1 = dataset[0]
    # x2 = dataset[1]
    
    # final = torch.stack([x1['sequence'], x2['sequence']], dim=0).to(device)
    
    # # Initialize the model
    model = DeepMSN().to(device)
    # model.train()
    # output = model(final)
    
    # assert False
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Define loss function
    # loss_function = nn.BCELoss()
    # loss_function = nn.CrossEntropyLoss()
    loss_function = nn.BCEWithLogitsLoss()
    # loss = loss_function(output, torch.stack([x1['label'], x2['label']], dim=0).to(device))
    
    # assert False
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
        )
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(dataloader, model, loss_function, optimizer, batch_size)
        # test_loop(test_dataloader, model, loss_function)
    print("Done!")
    
    # # Saving the model  
    # save_path = f'./models/state_dict/model.pth'
    # torch.save(model.state_dict(), save_path)
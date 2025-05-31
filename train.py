import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets.topic_datasets import TopicDataset
from models.deepmsn import DeepMSN

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    # for batch, (X, y) in enumerate(dataloader):
    for batch_idx, batch in enumerate(dataloader):
        
        X = batch['sequence']
        y = batch['label']
        
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
        
        if batch_idx % 100 == 0:
            loss, current = loss.item(), batch_idx * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# def test_loop(dataloader, model, loss_fn):
#     model.eval()
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     test_loss = 0
#     all_preds = []
#     all_targets = []
    
#     with torch.no_grad():
#         for batch in dataloader:
            
#             X = batch['sequence']
#             y = batch['label']
            
#             X, y = X.to(device), y.to(device)
#             y = y.float()  # Cast y to float
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
            
#             # Collect predictions and targets for F1 calculation
#             all_preds.append(pred)
#             all_targets.append(y)
    
#     # Concatenate all predictions and targets
#     all_preds = torch.cat(all_preds, dim=0)
#     all_targets = torch.cat(all_targets, dim=0)
    
#     # Compute F1 score directly from collected predictions
#     pred_y_bin = (torch.sigmoid(all_preds) > 0.5).float()
#     y_bin = all_targets.float()
    
#     tp = torch.sum(pred_y_bin * y_bin, dim=0)
#     fp = torch.sum(pred_y_bin * (1 - y_bin), dim=0)
#     fn = torch.sum((1 - pred_y_bin) * y_bin, dim=0)
    
#     precision = tp / (tp + fp + 1e-7)
#     recall = tp / (tp + fn + 1e-7)
#     f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
#     f1_score = torch.mean(f1)
    
#     test_loss /= num_batches
#     print(f"Test Error: \n F1 Score: {f1_score:.4f}, Avg loss: {test_loss:>8f} \n")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            
            X = batch['sequence']
            y = batch['label']
            
            X, y = X.to(device), y.to(device)
            y = y.float()  # Cast y to float
            pred = model(X)
            prob = torch.sigmoid(pred)  # Convert logits to probabilities
            test_loss += loss_fn(pred, y).item()
            
            # Collect predictions and targets for F1 calculation
            all_probs.append(prob)
            all_targets.append(y)
            
            correct += (prob.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    
    # Concatenate all predictions and targets
    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Print diagnostic information
    print(f"Prediction probabilities - Min: {all_probs.min():.4f}, Max: {all_probs.max():.4f}, Mean: {all_probs.mean():.4f}")
    print(f"Target distribution - Mean: {all_targets.mean():.4f}, Sum: {all_targets.sum()}")
    
    # # Try different thresholds
    # for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
    #     pred_y_bin = (all_probs > threshold).float()
    #     y_bin = all_targets.float()
        
    #     tp = torch.sum(pred_y_bin * y_bin, dim=0)
    #     fp = torch.sum(pred_y_bin * (1 - y_bin), dim=0)
    #     fn = torch.sum((1 - pred_y_bin) * y_bin, dim=0)
        
    #     precision = tp / (tp + fp + 1e-7)
    #     recall = tp / (tp + fn + 1e-7)
    #     f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    #     f1_score = torch.mean(f1)
        
    #     print(f"Threshold {threshold}: F1 = {f1_score:.4f}")
    
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
    
    # Split dataset into training and testing sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Define data loaders for training and testing
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    
    for batch in test_dataloader:
        X = batch['sequence']
        y = batch['label']
        print(f"Shape of X [N, L, C]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    
    # Initialize the model
    model = DeepMSN().to(device)
    
    # Calculate expected positive rate: 1.26 active labels out of 18 = ~7%
    expected_pos_rate = 1.26 / 18  # ~0.07
    
    # Create balanced loss function
    # Since 7% should be positive, we want to slightly encourage positive predictions
    pos_weight = torch.ones(18) * (1 - expected_pos_rate) / expected_pos_rate  # ~13.3
    pos_weight = torch.clamp(pos_weight, min=1.0, max=8.0)  # Clip to reasonable range
    
    # Define loss function
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_function, optimizer)
        test_loop(test_dataloader, model, loss_function)
    print("Done!")
    
    # Saving the model  
    save_path = f'./checkpoints/model.pth'
    torch.save(model.state_dict(), save_path)
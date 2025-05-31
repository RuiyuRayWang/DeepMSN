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
    
    # Count predictions per class at 0.5 threshold  
    pred_counts = (all_probs > 0.5).sum(dim=0)
    target_counts = all_targets.sum(dim=0)
    print(f"Predictions at 0.5 threshold: {pred_counts.sum()} total")
    print(f"Actual targets: {target_counts.sum()} total")
    
    # Try different thresholds - UNCOMMENTED
    best_f1 = 0
    best_threshold = 0.5
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
        pred_y_bin = (all_probs > threshold).float()
        y_bin = all_targets.float()
        
        tp = torch.sum(pred_y_bin * y_bin, dim=0)
        fp = torch.sum(pred_y_bin * (1 - y_bin), dim=0)
        fn = torch.sum((1 - pred_y_bin) * y_bin, dim=0)
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        f1_score = torch.mean(f1)
        
        print(f"Threshold {threshold}: F1 = {f1_score:.4f}, Total preds: {pred_y_bin.sum()}")
        
        if f1_score > best_f1:
            best_f1 = f1_score
            best_threshold = threshold
    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.2f}%, Best F1: {best_f1:.4f} @ {best_threshold}, Avg loss: {test_loss:>8f} \n")
    return test_loss, best_f1

if __name__ == "__main__":
    
    # Hyperparameters
    learning_rate = 2e-4
    weight_decay = 1e-2
    batch_size = 64
    epochs = 500
    
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
    
    # Create balanced loss function - REDUCED pos_weight
    pos_weight = torch.ones(18) * 3.0  # Much smaller weight
    
    print(f"Using pos_weight: {pos_weight[0]:.1f}")
    
    # Define loss function
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init, weight_decay=weight_decay)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Add scheduler to track progress
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # # Track progress
    # losses = []
    # f1_scores = []
    
    lr_init = 3e-4
    decay_rate = 0.333
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_function, optimizer)
        test_loss, best_f1 = test_loop(test_dataloader, model, loss_function)
        
        new_lrate = lr_init * (decay_rate ** (t / epochs))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        
        # Save model checkpoint every 100 epochs
        if (t + 1) % 100 == 0:
            save_path = f'./checkpoints/model_epoch_{t+1}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"Model checkpoint saved at {save_path}")
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        
        # Train loop
        size = len(train_dataloader.dataset)
        model.train()
        
        for batch_idx, batch in enumerate(train_dataloader):
            
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
            
            if batch_idx % 300 == 0:
                loss, current = loss.item(), batch_idx * batch_size + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")    

        # Test loop
        model.eval()
        size = len(test_dataloader.dataset)
        num_batches = len(test_dataloader)
        test_loss, correct = 0, 0
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                
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
        
        # Count predictions per class at 0.5 threshold  
        pred_counts = (all_probs > 0.5).sum(dim=0)
        target_counts = all_targets.sum(dim=0)
        print(f"Predictions at 0.5 threshold: {pred_counts.sum()} total")
        print(f"Actual targets: {target_counts.sum()} total")
        
        # Try different thresholds
        best_f1 = 0
        best_threshold = 0.5
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
            pred_y_bin = (all_probs > threshold).float()
            y_bin = all_targets.float()
            
            tp = torch.sum(pred_y_bin * y_bin, dim=0)
            fp = torch.sum(pred_y_bin * (1 - y_bin), dim=0)
            fn = torch.sum((1 - pred_y_bin) * y_bin, dim=0)
            
            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
            f1_score = torch.mean(f1)
            
            print(f"Threshold {threshold}: F1 = {f1_score:.4f}, Total preds: {pred_y_bin.sum()}")
            
            if f1_score > best_f1:
                best_f1 = f1_score
                best_threshold = threshold
        
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.2f}%, Best F1: {best_f1:.4f} @ {best_threshold}, Avg loss: {test_loss:>8f} \n")
        
        new_lrate = lr_init * (decay_rate ** (t / epochs))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        
        # Save model checkpoint every 100 epochs
        if (t + 1) % 100 == 0:
            save_path = f'./checkpoints/model_epoch_{t+1}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"Model checkpoint saved at {save_path}")
        
        # # Track metrics
        # losses.append(test_loss)
        # f1_scores.append(best_f1)
        
        # # Print progress summary
        # if t >= 4:  # After 5 epochs, check progress
        #     recent_loss_trend = sum(losses[-3:]) / 3 - sum(losses[-6:-3]) / 3 if len(losses) >= 6 else 0
        #     recent_f1_trend = sum(f1_scores[-3:]) / 3 - sum(f1_scores[-6:-3]) / 3 if len(f1_scores) >= 6 else 0
        #     print(f"Recent loss trend: {recent_loss_trend:+.4f}, F1 trend: {recent_f1_trend:+.4f}")
        
        # Update scheduler
        # scheduler.step(test_loss)
        
        # # Early stopping if F1 isn't improving
        # if t > 15 and max(f1_scores[-10:]) < 0.01:
        #     print("F1 score not improving. Consider adjusting model architecture or hyperparameters.")
        #     break
    
    print("Done!")
    
    # Print final summary
    print(f"Final loss: {losses[-1]:.4f}, Best F1 achieved: {max(f1_scores):.4f}")

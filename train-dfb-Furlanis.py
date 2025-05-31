import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets.topic_datasets import TopicDataset
from models.deepflybrain import DeepFlyBrain

import os
import yaml
import json
from datetime import datetime

device = torch.device(f'cuda:0')
print(f"Using {device} device")

if __name__ == "__main__":
    
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Creating checkpoints directory: ./checkpoints/dfb_furlanis_{date_time}")
    if not os.path.exists(f'./checkpoints/dfb_furlanis_{date_time}'):
        os.makedirs(f'./checkpoints/dfb_furlanis_{date_time}')
    
    # Hyperparameters
    lr_init = 1e-3
    weight_decay = 1e-3
    batch_size = 64
    epochs = 200
    
    # # Early stopping parameters
    # patience = 15
    # min_delta = 0.001  # Minimum improvement to reset patience
    
    # Set fixed random number seed
    torch.manual_seed(42)
    
    # Load config
    with open('configs/config_furlanis.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataset
    dataset = TopicDataset(config=config['dataset'])
    
    # Split dataset into training and testing sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - val_size - train_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    
    # Define data loaders for training and testing
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    # This is useful for later analysis or backup
    print("Collecting train/val/test dataset indices...")
    train_indices = []
    for batch in train_dataloader:
        train_indices.extend(batch['index'].tolist())
    with open(f'./checkpoints/dfb_furlanis_{date_time}/train_indices.txt', 'w') as f:
        f.write('\n'.join(map(str, train_indices)))
        
    val_indices = []
    for batch in val_dataloader:
        val_indices.extend(batch['index'].tolist())
    with open(f'./checkpoints/dfb_furlanis_{date_time}/val_indices.txt', 'w') as f:
        f.write('\n'.join(map(str, val_indices)))
        
    test_indices = []
    for batch in test_dataloader:
        test_indices.extend(batch['index'].tolist())
    with open(f'./checkpoints/dfb_furlanis_{date_time}/test_indices.txt', 'w') as f:
        f.write('\n'.join(map(str, test_indices)))

    # Initialize the model
    model = DeepFlyBrain(config=config).to(device)
    
    # Label smoothing loss to prevent overconfident predictions
    class LabelSmoothingBCELoss(nn.Module):
        def __init__(self, smoothing=0.1):
            super().__init__()
            self.smoothing = smoothing
            self.bce = nn.BCEWithLogitsLoss()
        
        def forward(self, pred, target):
            # Apply label smoothing: y_smooth = y * (1-α) + α/2
            target_smooth = target * (1 - self.smoothing) + 0.5 * self.smoothing
            return self.bce(pred, target_smooth)
    
    loss_fn = LabelSmoothingBCELoss(smoothing=0.1)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init, weight_decay=weight_decay)
    
    # Learning rate scheduler - STEP-BASED, not every epoch
    # Reduce LR by 0.5 every 30 epochs if no improvement
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=10,  # Wait 10 epochs before reducing LR
        threshold=0.001,
        min_lr=1e-7
    )
    
    # Alternative: Step-based scheduler (uncomment to use instead)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)
    
    # # Early stopping variables
    # best_val_loss = float('inf')
    # patience_counter = 0
    # best_model_state = None
    
    # Tracking for analysis
    train_losses = []
    val_losses = []
    learning_rates = []
    exact_match_accs = []
    hamming_accs = []
    train_val_ratios = []
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        
        # Train loop
        size = len(train_dataloader.dataset)
        model.train()
        epoch_train_loss = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            
            X, y = batch['sequence'], batch['label']
            X, y = X.to(device), y.to(device)
            y = y.float()
            
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)
            
            # Backpropagation with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_train_loss += loss.item()
            
            if batch_idx % 200 == 0:
                loss_val, current = loss.item(), batch_idx * batch_size + len(X)
                print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")
        
        avg_train_loss = epoch_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        # Validation loop
        model.eval()
        val_size = len(val_dataloader.dataset)
        num_batches = len(val_dataloader)
        val_loss, correct = 0, 0
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                
                X, y = batch['sequence'], batch['label']
                X, y = X.to(device), y.to(device)
                y = y.float()
                pred = model(X)
                
                prob = torch.sigmoid(pred)
                val_loss += loss_fn(pred, y).item()
                
                all_probs.append(prob)
                all_targets.append(y)
                
                # Exact match accuracy
                pred_binary = (prob > 0.5).float()
                exact_match = ((pred_binary == y).sum(dim=1) == y.shape[1]).float().sum().item()
                correct += exact_match
        
        # Calculate metrics
        all_probs = torch.cat(all_probs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        val_loss /= num_batches
        val_losses.append(val_loss)
        exact_match_acc = correct / val_size
        exact_match_accs.append(exact_match_acc)
        
        # Calculate additional metrics
        pred_binary = (all_probs > 0.5).float()
        hamming_acc = (pred_binary == all_targets).float().mean().item()
        hamming_accs.append(hamming_acc)
        
        # Print metrics with overfitting indicators
        train_val_ratio = val_loss / avg_train_loss
        train_val_ratios.append(train_val_ratio)
        overfitting_indicator = "⚠️ OVERFITTING" if train_val_ratio > 1.5 else "✓ OK"
        
        # Store learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        print(f"Train Loss: {avg_train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f} (ratio: {train_val_ratio:.2f}) {overfitting_indicator}")
        print(f"Exact Match Acc: {exact_match_acc:.4f}, Hamming Acc: {hamming_acc:.4f}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")
        print()
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save training statistics after each epoch
        training_stats = {
            'epoch': t + 1,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'exact_match_acc': exact_match_acc,
            'hamming_acc': hamming_acc,
            'learning_rate': current_lr,
            'train_val_ratio': train_val_ratio
        }
        
        # Save as JSON (human readable)
        stats_file = f'./checkpoints/dfb_furlanis_{date_time}/training_stats_epoch_{t+1:03d}.json'
        with open(stats_file, 'w') as f:
            json.dump(training_stats, f, indent=2)
        
        # # Early stopping logic
        # if val_loss < best_val_loss - min_delta:
        #     best_val_loss = val_loss
        #     patience_counter = 0
        #     best_model_state = model.state_dict().copy()
        #     print("✓ New best model saved")
        # else:
        #     patience_counter += 1
        #     print(f"No improvement for {patience_counter}/{patience} epochs")
        
        # if patience_counter >= patience:
        #     print(f"Early stopping triggered after {patience} epochs without improvement")
        #     model.load_state_dict(best_model_state)
        #     break
        
        # Save checkpoints
        if (t + 1) % 50 == 0:
            save_path = f'./checkpoints/dfb_furlanis_{date_time}/model_epoch_{t+1}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"Model checkpoint saved at {save_path}")
    
    print("\nTraining completed!")
    
    # Final test evaluation
    print("\n" + "="*50)
    print("FINAL TEST EVALUATION")
    print("="*50)
    
    model.eval()
    test_loss = 0
    test_correct = 0
    all_test_probs = []
    all_test_targets = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            X, y = batch['sequence'], batch['label']
            X, y = X.to(device), y.to(device)
            y = y.float()
            
            pred = model(X)
            prob = torch.sigmoid(pred)
            test_loss += loss_fn(pred, y).item()
            
            all_test_probs.append(prob)
            all_test_targets.append(y)
            
            # Exact match accuracy
            pred_binary = (prob > 0.5).float()
            exact_match = ((pred_binary == y).sum(dim=1) == y.shape[1]).float().sum().item()
            test_correct += exact_match
    
    all_test_probs = torch.cat(all_test_probs, dim=0)
    all_test_targets = torch.cat(all_test_targets, dim=0)
    
    test_loss /= len(test_dataloader)
    test_exact_acc = test_correct / len(test_dataset)
    
    pred_binary = (all_test_probs > 0.5).float()
    test_hamming_acc = (pred_binary == all_test_targets).float().mean().item()
    
    print(f"Final Test Results:")
    print(f"  Test Loss: {test_loss:.6f}")
    print(f"  Exact Match Accuracy: {test_exact_acc:.4f}")
    print(f"  Hamming Accuracy: {test_hamming_acc:.4f}")
    
    # Save final statistics
    final_stats = {
        'train_stats': {
            'total_epochs': len(train_losses),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'exact_match_accs': exact_match_accs,
            'hamming_accs': hamming_accs,
            'learning_rates': learning_rates,
            'train_val_ratios': train_val_ratios,
            'hyperparameters': {
                'lr_init': lr_init,
                'weight_decay': weight_decay,
                'batch_size': batch_size,
                'epochs': epochs
            }
        },
        'test_stats': {
            'test_loss': test_loss,
            'test_exact_match_acc': test_exact_acc,
            'test_hamming_acc': test_hamming_acc
        }
    }
    
    # Save final statistics to a JSON file
    with open(f'./checkpoints/dfb_furlanis_{date_time}/final_stats.json', 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    # Save final model
    final_save_path = f'./checkpoints/dfb_furlanis_{date_time}/final_model.pth'
    torch.save(model.state_dict(), final_save_path)
    print(f"Final model saved at {final_save_path}")

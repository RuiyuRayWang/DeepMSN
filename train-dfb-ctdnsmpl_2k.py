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
    
    # # Early stopping parameters
    # patience = 15
    # min_delta = 0.001  # Minimum improvement to reset patience
    
    # Set fixed random number seed
    torch.manual_seed(42)
    
    # Load config
    with open('configs/config_ctdnsmpl_top2k.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Hyperparameters
    lr_init = float(config.get('train').get('lr_init', 1e-3))
    weight_decay = float(config.get('train').get('weight_decay', 1e-2))
    batch_size = int(config.get('train').get('batch_size', 32))
    epochs = int(config.get('train').get('epochs', 200))
    
    if config['train']['resume_from_checkpoint']:
        # Resume training from checkpoint
        epoch_start = config['train']['checkpoint']['resume_from_epoch'] + 1
        print(f"Resuming training from epoch {epoch_start} for {epochs - epoch_start + 1} additional epochs")
        
        # Resume model from checkpoint
        checkpoint_path = config['train']['checkpoint']['path']
        resume_from_epoch = config['train']['checkpoint']['resume_from_epoch']
        checkpoint_file = f'{checkpoint_path}/checkpoint_epoch_{resume_from_epoch}.pth'
        
        if os.path.isfile(checkpoint_file):
            print(f"Resuming from checkpoint: {checkpoint_file} at epoch {resume_from_epoch}")
        else:
            raise FileNotFoundError(f"Checkpoint file {checkpoint_file} does not exist.")
        
        # Load checkpoint
        checkpoint = torch.load(f'{checkpoint_path}/checkpoint_epoch_{resume_from_epoch}.pth')
        
        # Restore model
        model = DeepFlyBrain(config=config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore optimizer (includes learning rate)
        optimizer = torch.optim.AdamW(model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, threshold=0.001, min_lr=1e-7
        )
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Get hyperparameters
        hyperparams = checkpoint['hyperparameters']
        lr_init = hyperparams['lr_init']
        weight_decay = hyperparams['weight_decay']
        batch_size = hyperparams['batch_size']
        
        # Resume dataset and dataloaders from saved indices
        dataset = TopicDataset(config=config['dataset'])
        
        with open(f'{checkpoint_path}/train_indices.txt', 'r') as f:
            train_indices = [int(line.strip()) for line in f.readlines()]
        with open(f'{checkpoint_path}/val_indices.txt', 'r') as f:
            val_indices = [int(line.strip()) for line in f.readlines()]
        with open(f'{checkpoint_path}/test_indices.txt', 'r') as f:
            test_indices = [int(line.strip()) for line in f.readlines()]
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        print(f"Loaded train/val/test datasets from indices files.")
        
        # Define data loaders for training and testing
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    else:
        # Initialize checkpoints directory
        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        checkpoint_path = f'checkpoints/dfb_ctdnsmpl_2k_{date_time}'
        print(f"Creating checkpoints directory: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        
        epoch_start = 1  # Start from epoch 1 if not resuming
        
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
        with open(f'{checkpoint_path}/train_indices.txt', 'w') as f:
            f.write('\n'.join(map(str, train_indices)))
            
        val_indices = []
        for batch in val_dataloader:
            val_indices.extend(batch['index'].tolist())
        with open(f'{checkpoint_path}/val_indices.txt', 'w') as f:
            f.write('\n'.join(map(str, val_indices)))
            
        test_indices = []
        for batch in test_dataloader:
            test_indices.extend(batch['index'].tolist())
        with open(f'{checkpoint_path}/test_indices.txt', 'w') as f:
            f.write('\n'.join(map(str, test_indices)))

        # Initialize the model
        model = DeepFlyBrain(config=config).to(device)
        
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
    
    for t in range(epoch_start - 1, epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        
        # =================================
        # Train loop
        # =================================
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
        
        # =================================
        # Validation loop
        # =================================
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
        print("\n")
        
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
        
        # Append to a single CSV file instead of individual JSON files
        import csv
        csv_file = f'{checkpoint_path}/training_log.csv'
        file_exists = os.path.exists(csv_file)
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=training_stats.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(training_stats)
        
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
        
        # # Save checkpoints
        # if (t + 1) % 50 == 0:
        #     save_path = f'{checkpoint_path}/model_epoch_{t+1}.pth'
        #     torch.save(model.state_dict(), save_path)
        #     print(f"Model checkpoint saved at {save_path}")
            
        # --------------------------------------------------------------
        # Save model + hyperparameters + optimizer state
        if (t + 1) % 50 == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'hyperparameters': {
                    'lr_init': lr_init,
                    'weight_decay': weight_decay,
                    'batch_size': batch_size,
                    'epochs': epochs
                },
                'epoch': t + 1,
                'train_loss': avg_train_loss,
                'val_loss': val_loss
            }
            torch.save(checkpoint, f'{checkpoint_path}/checkpoint_epoch_{t+1}.pth')
    
    print("\nTraining completed!")
    
    # ============================================================================
    # Final test evaluation
    # ============================================================================
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
            'total_epochs': epochs,
            # 'train_losses': train_losses,
            # 'val_losses': val_losses,
            # 'exact_match_accs': exact_match_accs,
            # 'hamming_accs': hamming_accs,
            # 'learning_rates': learning_rates,
            # 'train_val_ratios': train_val_ratios,
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
    with open(f'{checkpoint_path}/final_stats.json', 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    # Save final model
    final_save_path = f'{checkpoint_path}/final_model.pth'
    torch.save(model.state_dict(), final_save_path)
    print(f"Final model saved at {final_save_path}")

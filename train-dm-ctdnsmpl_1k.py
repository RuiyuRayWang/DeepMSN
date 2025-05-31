import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets.topic_datasets import TopicDataset
from models.deepmsn import DeepMSN

import os
import yaml
import json
from datetime import datetime

# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
device = torch.device(f'cuda:1')
print(f"Using {device} device")

if __name__ == "__main__":
    
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Creating checkpoints directory: ./checkpoints/dm_ctdnsmpl_1k_{date_time}")
    if not os.path.exists(f'./checkpoints/dm_ctdnsmpl_1k_{date_time}'):
        os.makedirs(f'./checkpoints/dm_ctdnsmpl_1k_{date_time}')
    
    # Hyperparameters
    lr_init = 3e-4
    decay_rate = 0.333
    # learning_rate = 2e-4
    weight_decay = 1e-2
    batch_size = 64
    epochs = 500
    
    # Set fixed random number seed
    torch.manual_seed(42)
    
    # Load config
    with open('configs/config_ctdnsmpl_top1k.yaml', 'r') as f:
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
    with open(f'./checkpoints/dm_ctdnsmpl_1k_{date_time}/train_indices.txt', 'w') as f:
        f.write('\n'.join(map(str, train_indices)))
        
    val_indices = []
    for batch in val_dataloader:
        val_indices.extend(batch['index'].tolist())
    with open(f'./checkpoints/dm_ctdnsmpl_1k_{date_time}/val_indices.txt', 'w') as f:
        f.write('\n'.join(map(str, val_indices)))
        
    test_indices = []
    for batch in test_dataloader:
        test_indices.extend(batch['index'].tolist())
    with open(f'./checkpoints/dm_ctdnsmpl_1k_{date_time}/test_indices.txt', 'w') as f:
        f.write('\n'.join(map(str, test_indices)))
    
    # Initialize the model
    model = DeepMSN(config=config).to(device)
    
    # # Calculate expected positive rate: 1.26 active labels out of `18` = ~7%
    # expected_pos_rate = 1.26 / 15  # ~0.07
    
    # # Create balanced loss function - REDUCED pos_weight
    # pos_weight = torch.ones(15) * 3.0  # Much smaller weight
    
    # print(f"Using pos_weight: {pos_weight[0]:.1f}")
    
    # Define loss function
    # loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init, weight_decay=weight_decay)
    
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
            y = y.float()  # Cast y to float
            
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()  # Reset gradients
            
            epoch_train_loss += loss.item()
            
            if batch_idx % 200 == 0:
                loss, current = loss.item(), batch_idx * batch_size + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")    

        new_lrate = lr_init * (decay_rate ** (t / epochs))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        
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
        print("\n")
        
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
        csv_file = f'./checkpoints/dm_ctdnsmpl_1k_{date_time}/training_log.csv'
        file_exists = os.path.exists(csv_file)
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=training_stats.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(training_stats)
        
        # Save model checkpoint every 100 epochs
        if (t + 1) % 100 == 0:
            save_path = f'./checkpoints/dm_ctdnsmpl_1k_{date_time}/model_epoch_{t+1}.pth'
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
    
    print("\nTraining completed!")
    
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
    with open(f'./checkpoints/dm_ctdnsmpl_1k_{date_time}/final_stats.json', 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    # Save final model
    final_save_path = f'./checkpoints/dm_ctdnsmpl_1k_{date_time}/final_model.pth'
    torch.save(model.state_dict(), final_save_path)
    print(f"Final model saved at {final_save_path}")


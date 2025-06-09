import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets.topic_datasets import TopicDataset
from models.deepmsn import DeepMSN

import os
import yaml
import json
from datetime import datetime

device = torch.device(f'cuda:0')
print(f"Using {device} device")

if __name__ == "__main__":
    
    # Set fixed random number seed
    torch.manual_seed(42)
    
    # Load config
    with open('configs/config_dm_ctdnsmpl_top3k_5fold-rc.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Hyperparameters
    lr_init = float(config.get('train').get('lr_init', 3e-4))
    decay_rate = float(config.get('train').get('decay_rate', 0.333))
    weight_decay = float(config.get('train').get('weight_decay', 1e-2))
    batch_size = int(config.get('train').get('batch_size', 64))
    epochs = int(config.get('train').get('epochs', 500))
    save_every_n_epochs = int(config.get('train').get('save_every_n_epochs', 100))
    
    # Get dataset paths from config
    dataset_config = config['dataset']
    out_dir = dataset_config['out_dir']
    
    # Check if preprocessed split files exist
    train_file = os.path.join(out_dir, 'regions_and_topics_train.bed')
    val_file = os.path.join(out_dir, 'regions_and_topics_val.bed')
    test_file = os.path.join(out_dir, 'regions_and_topics_test.bed')
    split_info_file = os.path.join(out_dir, 'split_info.json')
    
    if not all(os.path.exists(f) for f in [train_file, val_file, test_file]):
        raise FileNotFoundError(
            f"Preprocessed data files not found in {out_dir}. "
            "Please run preprocessing first:\n"
            "python scripts/preprocess.py -c configs/config_dm_ctdnsmpl_top3k.yaml"
        )
    
    # Load preprocessing information
    if os.path.exists(split_info_file):
        with open(split_info_file, 'r') as f:
            split_info = json.load(f)
        
        print("Preprocessing information:")
        print(f"  Random seed: {split_info['preprocessing_info']['random_seed']}")
        print(f"  Augmentation: {split_info['preprocessing_info']['augmentation']}")
        if split_info['preprocessing_info']['augmentation']:
            method = split_info['preprocessing_info'].get('augmentation_method', 'unknown')
            print(f"  Augmentation method: {method}")
            if split_info['augmentation_stats']:
                factor = split_info['augmentation_stats']['augmentation_factor']
                print(f"  Augmentation factor: {factor:.2f}x")
        print(f"  Dataset sizes: {split_info['dataset_sizes']}")
        print()
    
    if config['train'].get('resume_from_checkpoint', False):
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
        checkpoint = torch.load(checkpoint_file)
        
        # Restore model
        model = DeepMSN(config=config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore optimizer (includes learning rate)
        optimizer = torch.optim.AdamW(model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Get hyperparameters
        hyperparams = checkpoint['hyperparameters']
        lr_init = hyperparams['lr_init']
        decay_rate = hyperparams['decay_rate']
        weight_decay = hyperparams['weight_decay']
        batch_size = hyperparams['batch_size']
        
        print(f"Loaded hyperparameters from checkpoint")
    else:
        # Initialize checkpoints directory
        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        proj_name = config.get('name', '')
        augment_suffix = "_augmented" if split_info.get('preprocessing_info', {}).get('augmentation', False) else ""
        checkpoint_path = f'checkpoints/dm_{proj_name}{augment_suffix}_{date_time}'
        print(f"Creating checkpoints directory: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        
        epoch_start = 1  # Start from epoch 1 if not resuming
        
        # Initialize the model
        model = DeepMSN(config=config).to(device)
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init, weight_decay=weight_decay)
    
    # Load pre-split datasets from preprocessed files
    print("Loading pre-split datasets...")
    
    # Create separate configs for each dataset pointing to the correct bed file
    train_config = config.copy()
    train_config['dataset'] = config['dataset'].copy()
    train_config['dataset']['bed_file'] = train_file
    
    val_config = config.copy()
    val_config['dataset'] = config['dataset'].copy()
    val_config['dataset']['bed_file'] = val_file
    
    test_config = config.copy()
    test_config['dataset'] = config['dataset'].copy()
    test_config['dataset']['bed_file'] = test_file
    
    # Create datasets
    train_dataset = TopicDataset(config=train_config)
    val_dataset = TopicDataset(config=val_config)
    test_dataset = TopicDataset(config=test_config)
    
    print(f"Dataset sizes:")
    print(f"  Training: {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    
    # Define data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    # Copy split_info.json to checkpoint directory for reference
    if not config['train'].get('resume_from_checkpoint', False):
        import shutil
        if os.path.exists(split_info_file):
            shutil.copy2(split_info_file, f'{checkpoint_path}/split_info.json')
            print(f"Copied preprocessing info to checkpoint directory")
    
    # Create balanced loss function
    pos_weight = torch.ones(len(config['dataset']['data_path'])) * 5.0
    print(f"Using pos_weight: {pos_weight[0]:.1f}")
    
    # Define loss function
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    # loss_fn = nn.BCEWithLogitsLoss()
    
    # Tracking for analysis
    train_losses = []
    val_losses = []
    learning_rates = []
    exact_match_accs = []
    hamming_accs = []
    train_val_ratios = []
    
    print(f"\nStarting training from epoch {epoch_start} to {epochs}...")
    print("="*50)
    
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
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_train_loss += loss.item()
            
            if batch_idx % 200 == 0:
                loss_val, current = loss.item(), batch_idx * batch_size + len(X)
                print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")
        
        # Update learning rate
        new_lrate = lr_init * (decay_rate ** (t / epochs))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        
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
        train_val_ratio = val_loss / avg_train_loss if avg_train_loss > 0 else float('inf')
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
        
        # Append to CSV file
        import csv
        csv_file = f'{checkpoint_path}/training_log.csv'
        file_exists = os.path.exists(csv_file)
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=training_stats.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(training_stats)
        
        # Save checkpoints
        if (t + 1) % save_every_n_epochs == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'hyperparameters': {
                    'lr_init': lr_init,
                    'weight_decay': weight_decay,
                    'decay_rate': decay_rate,
                    'batch_size': batch_size,
                    'epochs': epochs
                },
                'epoch': t + 1,
                'train_loss': avg_train_loss,
                'val_loss': val_loss
            }
            torch.save(checkpoint, f'{checkpoint_path}/checkpoint_epoch_{t+1}.pth')
            print(f"Checkpoint saved at {checkpoint_path}/checkpoint_epoch_{t+1}.pth")
    
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
        'preprocessing_info': split_info if 'split_info' in locals() else None,
        'train_stats': {
            'total_epochs': epochs,
            'hyperparameters': {
                'lr_init': lr_init,
                'weight_decay': weight_decay,
                'decay_rate': decay_rate,
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
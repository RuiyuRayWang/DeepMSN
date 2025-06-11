from accelerate import Accelerator
accelerator = Accelerator()

import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets.topic_datasets import TopicDataset
from models.deepmsn import DeepMSN

import os
import yaml
import json
from datetime import datetime

# device = torch.device(f'cuda:0')
# print(f"Using {device} device")
device = accelerator.device

if __name__ == "__main__":
    
    # Set fixed random number seed
    torch.manual_seed(42)
    
    # Load config
    with open('configs/config_dm_pb_ctdnsmpl_top3k.yaml', 'r') as f:
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
        
        if accelerator.is_main_process:
            print("Preprocessing information:")
            print(f"  Random seed: {split_info['preprocessing_info']['random_seed']}")
            # print(f"  Augmentation: {split_info['preprocessing_info']['augmentation']}")
            # if split_info['preprocessing_info']['augmentation']:
            #     method = split_info['preprocessing_info'].get('augmentation_method', 'unknown')
            #     print(f"  Augmentation method: {method}")
            #     if split_info['augmentation_stats']:
            #         factor = split_info['augmentation_stats']['augmentation_factor']
            #         print(f"  Augmentation factor: {factor:.2f}x")
            print(f"  Dataset sizes: {split_info['dataset_sizes']}")
            print()
    
    # Load pre-split datasets from preprocessed files
    if accelerator.is_main_process:
        print("Loading pre-split datasets...")
    
    # Create separate configs for each dataset pointing to the correct bed file
    train_config = config.copy()
    train_config['dataset'] = config['dataset'].copy()
    train_config['dataset']['bed_file'] = train_file
    
    val_config = config.copy()
    val_config['dataset'] = config['dataset'].copy()
    val_config['dataset']['bed_file'] = val_file
    val_config['dataset']['augment'] = False
    
    test_config = config.copy()
    test_config['dataset'] = config['dataset'].copy()
    test_config['dataset']['bed_file'] = test_file
    test_config['dataset']['augment'] = False
    
    # Create datasets
    train_dataset = TopicDataset(config=train_config)
    val_dataset = TopicDataset(config=val_config)
    test_dataset = TopicDataset(config=test_config)
    
    if accelerator.is_main_process:
        print(f"Dataset sizes:")
        print(f"  Training: {len(train_dataset)}")
        print(f"  Validation: {len(val_dataset)}")
        print(f"  Test: {len(test_dataset)}")
    
    # Define data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    # Initialize the model
    model = DeepMSN(config=config)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init, weight_decay=weight_decay)
    
    # Create balanced loss function
    pos_weight = torch.ones(len(config['dataset']['data_path'])) * 18.0
    if accelerator.is_main_process:
        print(f"Using pos_weight: {pos_weight[0]:.1f}")
    
    # Define loss function
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # loss_fn = nn.BCEWithLogitsLoss()
    
    # Prepare model, optimizer, and dataloaders with Accelerator
    model, optimizer, train_dataloader, val_dataloader, test_dataloader, loss_fn = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, test_dataloader, loss_fn
    )
    
    if config['train'].get('resume_from_checkpoint', False):
        epoch_start = config['train']['checkpoint']['resume_from_epoch'] + 1
        if accelerator.is_main_process:
            print(f"Resuming training from epoch {epoch_start}")
        
        checkpoint_dir = config['train']['checkpoint']['path']
        resume_epoch = config['train']['checkpoint']['resume_from_epoch']
        checkpoint_path_to_load = f'{checkpoint_dir}/checkpoint_epoch_{resume_epoch}'
        
        if os.path.exists(checkpoint_path_to_load):
            # Load all states (model + optimizer + lr_scheduler if any)
            accelerator.load_state(checkpoint_path_to_load)
            
            # Load additional metadata
            metadata_file = f'{checkpoint_path_to_load}/training_metadata.json'
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Restore hyperparameters if needed
                hyperparams = metadata['hyperparameters']
                lr_init = hyperparams['lr_init']
                decay_rate = hyperparams['decay_rate']
                weight_decay = hyperparams['weight_decay']
                batch_size = hyperparams['batch_size']
                
                if accelerator.is_main_process:
                    print(f"Loaded checkpoint from epoch {metadata['epoch']}")
                    print(f"Previous train loss: {metadata['train_loss']:.6f}")
                    print(f"Previous val loss: {metadata['val_loss']:.6f}")
            
            checkpoint_path = checkpoint_dir  # Use existing checkpoint directory
        else:
            raise FileNotFoundError(f"Checkpoint directory {checkpoint_path_to_load} does not exist.")
    else:
        # Initialize checkpoints directory
        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        proj_name = config.get('name', '')
        # augment_suffix = "_augmented" if split_info.get('preprocessing_info', {}).get('augmentation', False) else ""
        checkpoint_path = f'checkpoints/dm_{proj_name}_{date_time}'
        
        if accelerator.is_main_process:
            print(f"Creating checkpoints directory: {checkpoint_path}")
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
        
        # Ensure all processes wait for directory creation
        accelerator.wait_for_everyone()
        
        epoch_start = 1  # Start from epoch 1 if not resuming
    
    # Copy split_info.json to checkpoint directory for reference
    if not config['train'].get('resume_from_checkpoint', False) and accelerator.is_main_process:
        import shutil
        if os.path.exists(split_info_file):
            shutil.copy2(split_info_file, f'{checkpoint_path}/split_info.json')
            print(f"Copied preprocessing info to checkpoint directory")
    
    # ============================================================================
    
    # Tracking for analysis
    train_losses = []
    val_losses = []
    learning_rates = []
    exact_match_accs = []
    hamming_accs = []
    train_val_ratios = []
    
    if accelerator.is_main_process:
        print(f"\nStarting training from epoch {epoch_start} to {epochs}...")
        print("="*50)
    
    for t in range(epoch_start - 1, epochs):
        if accelerator.is_main_process:
            print(f"Epoch {t+1}\n-------------------------------")
        
        # =================================
        # Train loop
        # =================================
        size = len(train_dataloader.dataset)
        model.train()
        epoch_train_loss = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            
            X, y = batch['sequence'], batch['label']
            # X, y = X.to(device), y.to(device)  # Comment out to use Accelerator
            y = y.float()
            
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)
            
            # Backpropagation
            # loss.backward()  # Comment out to use Accelerator
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_train_loss += loss.item()
            
            if batch_idx % 200 == 0 and accelerator.is_main_process:
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
                # X, y = X.to(device), y.to(device)  # Comment out to use Accelerator
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
        
        # Calculate positive/negative agreement
        tp = (pred_binary * all_targets).sum().item()
        fp = (pred_binary * (1 - all_targets)).sum().item()
        fn = ((1 - pred_binary) * all_targets).sum().item()
        tn = ((1 - pred_binary) * (1 - all_targets)).sum().item()
        pos_agreement = tp / (tp + fp) if (tp + fp) > 0 else 0
        neg_agreement = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Print metrics with overfitting indicators
        train_val_ratio = val_loss / avg_train_loss if avg_train_loss > 0 else float('inf')
        train_val_ratios.append(train_val_ratio)
        overfitting_indicator = "⚠️ OVERFITTING" if train_val_ratio > 1.5 else "✓ OK"
        
        # Store learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        if accelerator.is_main_process:
            print(f"Train Loss: {avg_train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f} (ratio: {train_val_ratio:.2f}) {overfitting_indicator}")
            print(f"Exact Match Acc: {exact_match_acc:.4f}, Hamming Acc: {hamming_acc:.4f}")
            print(f"Positive Agreement: {pos_agreement:.4f}, Negative Agreement: {neg_agreement:.4f}")
            print(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")
            print("\n")
        
        if accelerator.is_main_process:
            # Save training statistics after each epoch
            training_stats = {
                'epoch': t + 1,
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'exact_match_acc': exact_match_acc,
                'hamming_acc': hamming_acc,
                'positive_agreement': pos_agreement,
                'negative_agreement': neg_agreement,
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
            # New code - saves model + optimizer + other states
            checkpoint_dir = f'{checkpoint_path}/checkpoint_epoch_{t+1}'
            accelerator.save_state(output_dir=checkpoint_dir)
            
            # Save additional metadata
            if accelerator.is_main_process:
                additional_data = {
                    'epoch': t + 1,
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                    'hyperparameters': {
                        'lr_init': lr_init,
                        'weight_decay': weight_decay,
                        'decay_rate': decay_rate,
                        'batch_size': batch_size,
                        'epochs': epochs
                    }
                }
                with open(f'{checkpoint_dir}/training_metadata.json', 'w') as f:
                    json.dump(additional_data, f, indent=2)
            
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                print(f"Checkpoint saved at {checkpoint_dir}")
    
    print("\nTraining completed!")
    
    # ============================================================================
    # Final test evaluation
    # ============================================================================
    if accelerator.is_main_process:
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
            # X, y = X.to(device), y.to(device)  # Comment out to use Accelerator
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
    
    tp = (pred_binary * all_test_targets).sum().item()
    fp = (pred_binary * (1 - all_test_targets)).sum().item()
    fn = ((1 - pred_binary) * all_test_targets).sum().item()
    tn = ((1 - pred_binary) * (1 - all_test_targets)).sum().item()
    pos_agreement = tp / (tp + fp) if (tp + fp) > 0 else 0
    neg_agreement = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    if accelerator.is_main_process:
        print(f"Final Test Results:")
        print(f"  Test Loss: {test_loss:.6f}")
        print(f"  Exact Match Accuracy: {test_exact_acc:.4f}")
        print(f"  Hamming Accuracy: {test_hamming_acc:.4f}")
        print(f"  Positive Agreement: {pos_agreement:.4f}")
        print(f"  Negative Agreement: {neg_agreement:.4f}")
    
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
            'test_hamming_acc': test_hamming_acc,
            'test_positive_agreement': pos_agreement,
            'test_negative_agreement': neg_agreement,
        }
    }
    
    # Save final statistics to a JSON file
    if accelerator.is_main_process:
        with open(f'{checkpoint_path}/final_stats.json', 'w') as f:
            json.dump(final_stats, f, indent=2)
    
    # Save final model and optimizer state
    final_checkpoint_dir = f'{checkpoint_path}/final_checkpoint'
    accelerator.save_state(output_dir=final_checkpoint_dir)

    # Also save just the model for easier loading during inference
    final_model_path = f'{checkpoint_path}/final_model'
    accelerator.wait_for_everyone()
    accelerator.save_model(model, final_model_path)

    if accelerator.is_main_process:
        print(f"Final checkpoint (model + optimizer) saved at {final_checkpoint_dir}")
        print(f"Final model (inference) saved at {final_model_path}")
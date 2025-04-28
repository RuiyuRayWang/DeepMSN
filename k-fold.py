import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets.msn_datasets import TestDataset
from models.deepmsn import DeepMSN
from sklearn.model_selection import KFold

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# model = DeepMSN().to(device)
# print(f"Model structure: {model}\n\n")

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

if __name__ == "__main__":
    
    # Hyperparameters
    learning_rate = 1e-3
    batch_size = 32
    n_epochs = 100
    k_folds = 5
    loss_function = nn.BCELoss()
    
    # For fold results
    results = {}
    
    # Set fixed random number see
    torch.manual_seed(42)
    
    # Prepare dataset
    dataset = TestDataset(
       "data/test_data/filt_peaks_catlas_multiome.blacklist_filtered.ATCG.clean.srted.fa",
       "data/test_data/filt_peaks_catlas_multiome.blacklist_filtered.ATCG.clean.srted.bed",
       transform=None,
       target_transform=None
    )
    
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)
    
    # Start print
    print('--------------------------------')
    
    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')
        
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        # Define data loaders for training and testing data in this fold
        trainloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            sampler=train_subsampler,
            num_workers=4
        )
        testloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=test_subsampler,
            num_workers=4
        )
        
        # Init the neural model
        model = DeepMSN().to(device)
        model.apply(reset_weights)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Run the training loop for defined number of epochs
        for epoch in range(0, n_epochs):
            
            # Print epoch
            print(f'Starting epoch {epoch+1}')
            
            # Set current loss value
            current_loss = 0.0
            
            size = len(trainloader)
            
            # Iterate over the DataLoader for training data
            for batch, data in enumerate(trainloader, 0):
                
                # Get data for batch
                X, y = data
                X, y = X.to(device), y.to(device)
                y = y.float()  # Cast y to float
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Perform forward pass
                outputs = model(X)
                
                # Compute loss
                loss = loss_function(outputs, y)
                
                # Perform backward pass
                loss.backward()
                
                # Perform optimization
                optimizer.step()
                
                # Print statistics
                current_loss += loss.item()
                if batch % 100 == 99:
                    print('Loss after mini-batch %5d: %.3f' %
                        (batch + 1, current_loss / 100))
                    current_loss = 0.0
                
                # if batch % 100 == 0:
                #     loss, current = loss.item(), batch * batch_size + len(X)
                #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                
        
        # Process is complete.
        print('Training process has finished. Saving trained model.')
        
        # Print about testing
        print('Starting testing')
        
        # Saving the model
        save_path = f'./models/state_dict/model-fold-{fold}.pth'
        torch.save(model.state_dict(), save_path)
        
        # Evaluationfor this fold
        correct, total = 0, 0
        with torch.no_grad():
            
            # Iterate over the test data and generate predictions
            for batch, data in enumerate(testloader, 0):
                
                # Get data for batch
                X, y = data
                X, y = X.to(device), y.to(device)
                y = torch.argmax(y, dim=1)
                
                # Generate outputs
                outputs = model(X)
                
                # Set total and correct
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            
            # Print accuracy
            print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
            print('--------------------------------')
            results[fold] = 100.0 * (correct / total)
    
    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum/len(results.items())} %')
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets.msn_datasets import TestDataset
from models.deepmsn import DeepMSN
from sklearn.model_selection import KFold


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = DeepMSN().to(device)
print(f"Model structure: {model}\n\n")

X = torch.rand(32, 500, 4, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

# Print model parameters
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

# def reset_weights(m):
#   '''
#     Try resetting model weights to avoid
#     weight leakage.
#   '''
#   for layer in m.children():
#    if hasattr(layer, 'reset_parameters'):
#     print(f'Reset trainable parameters of layer = {layer}')
#     layer.reset_parameters()

# if __name__ == "__main__":
    
#     # Configs
#     k_folds = 9
#     num_epochs = 1
#     loss_function = nn.BCELoss()
    
#     # For fold results
#     results = {}
    
#     # Set fixed random number see
#     torch.manual_seed(42)
    
#     # Prepare dataset
#     dataset = TestDataset(
#        "data/test_data/filt_peaks_catlas_multiome.blacklist_filtered.ATCG.clean.srted.fa",
#        "data/test_data/filt_peaks_catlas_multiome.blacklist_filtered.ATCG.clean.srted.bed",
#        transform=None,
#        target_transform=None
#     )
    
#     # Define the K-fold Cross Validator
#     kfold = KFold(n_splits=k_folds, shuffle=True)
    
#     # Start print
#     print('--------------------------------')
    
#     # K-fold Cross Validation model evaluation
#     for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            
#         # Print
#         print(f'FOLD {fold}')
#         print('--------------------------------')
        
#         # Sample elements randomly from a given list of ids, no replacement.
#         train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
#         test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
#         # Define data loaders for training and testing data in this fold
#         trainloader = DataLoader(
#            dataset, 
#            batch_size=32, sampler=train_subsampler
#         )
#         testloader = DataLoader(
#            dataset,
#            batch_size=32, sampler=test_subsampler
#         )
        
#         # Init the neural network
#         network = DeepMSN()
#         network.apply(reset_weights)
        
#         # Initialize optimizer
#         optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        
#         # Run the training loop for defined number of epochs
#         for epoch in range(0, num_epochs):
            
#             # Print epoch
#             print(f'Starting epoch {epoch+1}')
            
#             # Set current loss value
#             current_loss = 0.0
            
#             # Iterate over the DataLoader for training data
#             for i, data in enumerate(trainloader, 0):
                
#                 # Get inputs
#                 inputs, targets = data
#                 targets = targets.float()  # Cast targets to float
                
#                 # Zero the gradients
#                 optimizer.zero_grad()
                
#                 # Perform forward pass
#                 outputs = network(inputs)
                
#                 # Compute loss
#                 loss = loss_function(outputs, targets)
                
#                 # Perform backward pass
#                 loss.backward()
                
#                 # Perform optimization
#                 optimizer.step()
                
#                 # Print statistics
#                 current_loss += loss.item()
#                 if i % 500 == 499:
#                     print('Loss after mini-batch %5d: %.3f' %
#                         (i + 1, current_loss / 500))
#                     current_loss = 0.0
        
#         # Process is complete.
#         print('Training process has finished. Saving trained model.')
        
#         # Print about testing
#         print('Starting testing')
        
#         # Saving the model
#         save_path = f'./model-fold-{fold}.pth'
#         torch.save(network.state_dict(), save_path)
        
#         # Evaluationfor this fold
#         correct, total = 0, 0
#         with torch.no_grad():
            
#             # Iterate over the test data and generate predictions
#             for i, data in enumerate(testloader, 0):
                
#                 # Get inputs
#                 inputs, targets = data
#                 targets = targets.float()  # Cast targets to float
                
#                 # Generate outputs
#                 outputs = network(inputs)
                
#                 # Set total and correct
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += targets.size(0)
#                 correct += (predicted == targets).sum().item()
            
#             # Print accuracy
#             print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
#             print('--------------------------------')
#             results[fold] = 100.0 * (correct / total)
    
#     # Print fold results
#     print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
#     print('--------------------------------')
#     sum = 0.0
#     for key, value in results.items():
#         print(f'Fold {key}: {value} %')
#         sum += value
#     print(f'Average: {sum/len(results.items())} %')
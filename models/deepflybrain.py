import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import reverse_complement

class DeepFlyBrain(nn.Module):
    def __init__(self, config):
        
        super(DeepFlyBrain, self).__init__()
        
        output_dim = len(config['dataset']['data_path'])
        
        self.conv1d_1 = nn.Conv1d(in_channels=4, out_channels=1024, kernel_size=23, stride=1, padding='same')
        self.max_pooling1d_1 = nn.MaxPool1d(kernel_size=12, stride=12)
        self.dropout_1 = nn.Dropout(0.4)
        self.time_distributed_1 = nn.Linear(1024, 128)
        self.lstm_1 = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        # self.lstm_1 = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True, dropout=0, bidirectional=True)
        self.dropout_2 = nn.Dropout(0.4)
        self.flatten_1 = nn.Flatten()
        self.dense_2 = nn.Linear(128 * 2 * 41, 256)  # Adjust input size based on the output of LSTM
        self.dropout_3 = nn.Dropout(0.4)
        self.dense_3 = nn.Linear(256 * 2, output_dim)  # Adjust input size based on concatenation
        self.sigmoid_1 = nn.Sigmoid()  # Dropped since BCEWithLogitsLoss is used in training

    def forward(self, x):
        x_rc = reverse_complement(x)
        
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1d_1(x))
        x = self.max_pooling1d_1(x)
        x = self.dropout_1(x)
        x = x.permute(0, 2, 1)
        x = F.relu(self.time_distributed_1(x))
        x, _ = self.lstm_1(x)
        x = self.dropout_2(x)
        x = self.flatten_1(x)
        x = F.relu(self.dense_2(x))
        x = self.dropout_3(x)
        
        x_rc = x_rc.permute(0, 2, 1)
        x_rc = F.relu(self.conv1d_1(x_rc))
        x_rc = self.max_pooling1d_1(x_rc)
        x_rc = self.dropout_1(x_rc)
        x_rc = x_rc.permute(0, 2, 1)
        x_rc = F.relu(self.time_distributed_1(x_rc))
        x_rc, _ = self.lstm_1(x_rc)
        x_rc = self.dropout_2(x_rc)
        x_rc = self.flatten_1(x_rc)
        x_rc = F.relu(self.dense_2(x_rc))
        x_rc = self.dropout_3(x_rc)
        
        x = torch.cat((x, x_rc), dim=1)  # Concatenate along the feature dimension
        x = self.dense_3(x)
        x = self.sigmoid_1(x)
        
        return x

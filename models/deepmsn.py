import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import reverse_complement

class DeepMSN(nn.Module):
    def __init__(self, config, 
                 filter_size=512,
                #  embed_dim=128, 
                 internal_emb_dim=64, 
                 early_hidden_dim=64, 
                 num_early_lyr=2, 
                 dropout=0.5,
                 tsfm_hidden_dim=128, 
                 num_tsfm_layers=2, 
                 num_heads=8, 
                 num_reg_tok=1):
        super().__init__()
        
        output_dim = len(config['dataset']['data_path'])
        
        # Replace lines 22-28
        self.cnn_block = nn.Sequential(
            nn.Conv1d(4, filter_size, kernel_size=15, stride=1, padding=7),  # Keep sequence length
            nn.BatchNorm1d(filter_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),  # Local max pooling for feature enhancement
            nn.Dropout1d(0.4)
        )

        # Add a simple residual pathway for the input
        self.input_projection = nn.Conv1d(4, filter_size, kernel_size=1)  # Project input to match CNN output
        
        self.cnn_projection = nn.Linear(2 * filter_size, internal_emb_dim)
        
        cls_tensor = torch.randn(1, num_reg_tok + 1, internal_emb_dim)  # Changed from embed_dim
        cls_tensor = cls_tensor / (float(internal_emb_dim) ** 0.5)
        self.cls_token = nn.Parameter(cls_tensor, requires_grad=True)
        
        # Update early_layers to match CNN output
        self.early_layers = nn.Sequential(
            ResidualBlock(
                feat_in=internal_emb_dim,  # Changed from embed_dim
                feat_out=internal_emb_dim,
                feat_hidden=early_hidden_dim,
                drop_out=dropout, 
                use_norm=True
            ),*(
                ResidualBlock(
                    feat_in=internal_emb_dim,
                    feat_out=internal_emb_dim,
                    feat_hidden=early_hidden_dim,
                    drop_out=dropout,
                    use_norm=True
                ) for _ in range(num_early_lyr - 1)
            )
        )
        
        # self.length_embedding = nn.Parameter(torch.randn(1, 500, embed_dim) / (float(embed_dim) ** 0.5), requires_grad=True)
        # Add positional encoding AFTER CNN processing
        self.pos_encoding = nn.Parameter(torch.zeros(1, 502, internal_emb_dim), requires_grad=True)  # +1 for CLS token
        nn.init.trunc_normal_(self.pos_encoding, std=0.01)
        
        self.input_dropout = nn.Dropout(dropout)
        
        self.transformer = nn.Sequential(
            *(SwigluAttentionBlock(internal_emb_dim, tsfm_hidden_dim, num_heads, dropout=dropout) 
            for _ in range(num_tsfm_layers))
        )
        self.output_layer = nn.Sequential(
            ResidualBlock(
                feat_in=internal_emb_dim,
                feat_out=tsfm_hidden_dim,
                feat_hidden=tsfm_hidden_dim,
                drop_out=dropout, 
                use_norm=False
            ), *(
                ResidualBlock(
                    feat_in=tsfm_hidden_dim,
                    feat_out=tsfm_hidden_dim,
                    feat_hidden=tsfm_hidden_dim,
                    drop_out=dropout,
                    use_norm=False
                ) for _ in range(num_early_lyr - 1)
            ),
            nn.Linear(tsfm_hidden_dim, output_dim, bias=True)
        )
    
    def forward(self, x_in):
        """
        x is a batch of one-hot-encoded sequences, returned is the predicted topic class vectorfor the batch of sequences
        x.shape is (N, L, C) where N is batch size, L is the length of the sequence, C is the number of channels
        """
        
        N, L, C = x_in.shape
    
        # Step 1: Apply reverse complement and transpose
        x = x_in.transpose(1, 2)  # [N, C, L]
        x_rc = reverse_complement(x_in).transpose(1, 2)  # [N, C, L]
        
        # Step 2: Apply CNN with residual connection
        x_cnn = self.cnn_block(x)  # [N, filter_size, L]
        x_cnn_rc = self.cnn_block(x_rc)  # [N, filter_size, L]
        
        # Add residual connection from input
        x_skip = self.input_projection(x)  # [N, filter_size, L]
        x_skip_rc = self.input_projection(x_rc)  # [N, filter_size, L]
        
        x_cnn = x_cnn + x_skip  # Residual connection
        x_cnn_rc = x_cnn_rc + x_skip_rc  # Residual connection
        
        x_cnn = torch.cat([x_cnn, x_cnn_rc], dim=1)  # [N, filter_size * 2, L]
        x = x_cnn.transpose(1, 2)  # [N, L, filter_size * 2]
        
        x = self.cnn_projection(x)  # [N, L, internal_emb_dim]
        
        # Step 3: Add CLS token
        cls_token = self.cls_token.repeat(N, 1, 1)  # [N, 2, internal_emb_dim]
        x = torch.cat([cls_token, x], dim=1)  # [N, 502, internal_emb_dim]
        
        # Step 4: Early processing
        x = self.early_layers(x)  # [N, L+1, internal_emb_dim]
        
        # Step 5: Add positional encoding
        seq_len = x.shape[1]
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Step 6: Transformer
        x = self.input_dropout(x)
        x = self.transformer(x)
        
        # Output prediction
        pred_tok = x[:, 0, :]  # [N, C+1]
        x = self.output_layer(pred_tok)
        
        return x

class SwiGLUFFN(nn.Module):
    '''no auto determined hidden size'''
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()
        
        self.linear1 = nn.Linear(input_dim, 2 * hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_dim, output_dim, bias=True)
    
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.linear1(hidden_state)
        x1, x2 = hidden_state.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.linear2(hidden)

class SwigluAttentionBlock(nn.Module):
    def __init__(self, embed_dim, tsfm_hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.ffn = SwiGLUFFN(embed_dim, tsfm_hidden_dim, embed_dim)
        self.ffn_dropout = nn.Dropout(dropout * 0.5)
    
    def forward(self, x):
        # Attention block
        inp_x = self.layer_norm_1(x)
        attn_output, _ = self.attn(inp_x, inp_x, inp_x, need_weights=False)
        x = x + self.attn_dropout(attn_output)
        
        # FFN block
        ffn_out = self.ffn(self.layer_norm_2(x))
        x = x + self.ffn_dropout(ffn_out)
        
        return x

class ResidualBlock(nn.Module):
    # Follows "Identity Mappings in Deep Residual Networks", uses LayerNorm instead of BatchNorm, and LeakyReLU instead of ReLU
    def __init__(self, feat_in=128, feat_out=128, feat_hidden=256, drop_out=0.0, use_norm=True):
        super().__init__()
        # Define the residual block with or without normalization
        if use_norm:
            self.block = nn.Sequential(
                nn.LayerNorm(feat_in),  # Layer normalization on input features
                nn.LeakyReLU(negative_slope=0.1),  # LeakyReLU activation
                nn.Dropout(p=drop_out),
                nn.Linear(feat_in, feat_hidden),  # Linear layer transforming input to hidden features
                nn.LayerNorm(feat_hidden),  # Layer normalization on hidden features
                nn.LeakyReLU(negative_slope=0.1),  # LeakyReLU activation
                nn.Dropout(p=drop_out),
                nn.Linear(feat_hidden, feat_out)  # Linear layer transforming hidden to output features
            )
        else:
            self.block = nn.Sequential(
                nn.LeakyReLU(negative_slope=0.1),  # LeakyReLU activation
                nn.Dropout(p=drop_out),
                nn.Linear(feat_in, feat_hidden),  # Linear layer transforming input to hidden features
                nn.LeakyReLU(negative_slope=0.1),  # LeakyReLU activation
                nn.Dropout(p=drop_out),
                nn.Linear(feat_hidden, feat_out)  # Linear layer transforming hidden to output features
            )
        
        # Define the bypass connection
        if feat_in != feat_out:
            self.bypass = nn.Linear(feat_in, feat_out)  # Linear layer to match dimensions if they differ
        else:
            self.bypass = nn.Identity()  # Identity layer if input and output dimensions are the same
    
    def forward(self, input_data):
        # Forward pass: apply the block and add the bypass connection
        return self.block(input_data) + self.bypass(input_data)
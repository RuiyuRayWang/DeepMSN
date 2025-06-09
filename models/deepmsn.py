import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DeepMSN(nn.Module):
    def __init__(self, config, embed_dim=128, internal_emb_dim=256, early_hidden_dim=256, num_early_lyr=2, dropout=0.3,
                 tsfm_hidden_dim=512, num_tsfm_layers=6, num_heads=8, num_reg_tok=1):
        
        super().__init__()
        
        output_dim = len(config['dataset']['data_path'])
        self.project = nn.Linear(4, embed_dim, bias=False)
        self.length_embedding = nn.Parameter(torch.randn(1, 500, embed_dim) / (float(embed_dim) ** 0.5), requires_grad=True)
        self.early_layers = nn.Sequential(
            ResidualBlock(
                feat_in=embed_dim,
                feat_out=internal_emb_dim,
                feat_hidden=early_hidden_dim,
                drop_out=dropout, 
                use_norm=True
            ),*(
                ResidualBlock(
                    feat_in=internal_emb_dim,
                    feat_out=internal_emb_dim,
                    feat_hidden=early_hidden_dim,
                    drop_out=dropout, use_norm=True
                ) for _ in range(num_early_lyr - 1)
            )
        )
        cls_tensor = torch.randn(1, num_reg_tok + 1, embed_dim)
        cls_tensor = cls_tensor / (float(embed_dim) ** 0.5)
        self.cls_token = nn.Parameter(cls_tensor, requires_grad=True)
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
                    drop_out=dropout, use_norm=False
                ) for _ in range(num_early_lyr - 1)
            ),
            nn.Linear(tsfm_hidden_dim, output_dim, bias=True)
        )

    def forward(self, x_in):
        """
        x is a batch of image embeddings, returned is the predicted activation for the batch of image in a single voxel
        x.shape is (N, L, C) where N is batch size, L is the length of the sequence, C is the number of channels (4) representing the one-hot encoded DNA sequence.
        
        x_in is the image embeddings for incontext learning: (B, S_ic, E)
        ic_nrn is the neural activation for incontext learning: (B, S_ic)
        unknown_img is the img embedding to predict: (B, S_uk, E)
        """
        
        # print('[DEBUG] self.dropout', self.dropout)
        
        B, S_ic, E = x_in.shape # batch, in context samples, 512
        
        # debug_info()
        # print(f'[DEBUG] N, L, C: {x_in.shape}')   # N, L, C: torch.Size([1, 100, 512])
        
        x = self.project(x_in)  # [B, S, E+1]  batch, in context samples, 512
        
        # # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)  # [B, N, E+1]
        x = torch.cat([cls_token, x], dim=1)  # [B, S+N, E+1]
        x = self.early_layers(x)  # [B, S+N, E+1]

        '''Apply Transformer'''
        # print('[DEBUG] type(x)', x.dtype)       # torch.float32
        x = self.input_dropout(x)
        x = self.transformer(x)

        # Perform hyperweights prediction
        pred_tok = x[:, 0, :]  # [B, E+1]
        
        # print(f'[DEBUG] pred_tok.shape: {pred_tok.shape}')  # [B, E+1]
        # weights = self.weight_pred(pred_tok)  # [B, E+1]
        x = self.output_layer(pred_tok)  # [B, S+N, 18]

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
        """Conventional attention + swiglu and attention residual
        """

        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.ffn = SwiGLUFFN(embed_dim, tsfm_hidden_dim, embed_dim)

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        log_scale = np.log(inp_x.shape[-2])

        attn_output, _ = self.attn(log_scale * inp_x, inp_x, inp_x, need_weights=False)
        x = x + self.attn_dropout(attn_output)  # Apply dropout to the attention output
        x = x + self.ffn(self.layer_norm_2(x))
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
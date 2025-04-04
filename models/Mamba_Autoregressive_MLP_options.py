import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_wo_pos
from mamba_ssm import Mamba

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual

class MambaLayer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.mamba = Mamba(
            d_model = configs.d_model,
            d_state = configs.d_ff,
            d_conv = configs.d_conv,
            expand = configs.expand,
        )
        self.dropout = nn.Dropout(p=configs.dropout)
        self.norm = nn.LayerNorm(configs.d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        x = residual + x
        return x

class MambaLayerWithFFN(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.mamba = Mamba(
            d_model=configs.d_model,
            d_state=configs.d_ff,
            d_conv=configs.d_conv,
            expand=configs.expand,
        )
        self.dropout = nn.Dropout(p=configs.dropout)
        self.norm = nn.LayerNorm(configs.d_model)
        self.ffn = FeedForward(
            d_model=configs.d_model, 
            d_ff=configs.d_ff, 
            dropout=configs.dropout
        )

    def forward(self, x):
        # Mamba block
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        x = residual + x

        # FFN block
        x = self.ffn(x)

        return x

class MambaModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        if configs.mamba_ffn_option:
            self.layers = nn.ModuleList([
                MambaLayerWithFFN(configs) for _ in range(configs.e_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                MambaLayer(configs) for _ in range(configs.e_layers)
            ])
        self.norm = nn.LayerNorm(configs.d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.autoregressive_option = configs.autoregressive_option
        self.embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dropout = nn.Dropout(p=configs.dropout)
        self.mamba = MambaModel(configs)

        self.out_layer = nn.Linear(configs.d_model, configs.c_out, bias=False)
        print("Model initialized with autoregressive option:", configs.autoregressive_option)
        print("Model initialized with Mamba option:", configs.mamba_ffn_option)

    
    def forecast(self, x_enc, x_mark_enc):
        """Non-autoregressive forecasting"""
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        x = self.embedding(x_enc, x_mark_enc)
        x = self.dropout(x)
        x = self.mamba(x)
        x_out = self.out_layer(x)
        x_out = x_out * std_enc + mean_enc
        return x_out
    
    def forecast_AR(self, x_enc, x_mark_enc):
        """Autoregressive forecasting"""
        # Get normalization parameters from input
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc_norm = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc_norm, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc_norm = x_enc_norm / std_enc
        
        batch_size, seq_len, feature_dim = x_enc.shape
        
        # Initialize prediction with the input sequence
        input_seq = x_enc_norm.clone()
        
        # Store predictions
        predictions = []
        
        # Create marker tensor for the entire prediction length
        # Using the last timestep's marker and repeating it
        if x_mark_enc is not None:
            mark_pred = x_mark_enc[:, -1:].repeat(1, self.pred_len, 1)
        else:
            mark_pred = None
        
        # Generate predictions step by step
        for i in range(self.pred_len):
            # Get embedding for the current sequence
            embed = self.embedding(input_seq, x_mark_enc)
            embed = self.dropout(embed)
            
            # Pass through Mamba model
            hidden = self.mamba(embed)
            
            # Get prediction for the last time step
            pred = self.out_layer(hidden[:, -1:])
            
            # Store the prediction
            predictions.append(pred)
            
            # Update input sequence for the next step
            input_seq = torch.cat([input_seq[:, 1:], pred], dim=1)
            
            # Update markers if needed
            if x_mark_enc is not None:
                x_mark_enc = torch.cat([x_mark_enc[:, 1:], mark_pred[:, i:i+1]], dim=1)
        
        # Concatenate predictions and denormalize
        predictions = torch.cat(predictions, dim=1)
        predictions = predictions * std_enc + mean_enc
        
        return predictions

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['short_term_forecast', 'long_term_forecast']:
            if self.autoregressive_option:
                x_out = self.forecast_AR(x_enc, x_mark_enc)
            else:
                x_out = self.forecast(x_enc, x_mark_enc)
                x_out = x_out[:, -self.pred_len:, :]
            return x_out
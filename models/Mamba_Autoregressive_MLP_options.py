import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba
from layers.Embed import DataEmbedding_wo_pos  # note: there's no positional embedding 

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
        self.label_len = configs.label_len
        self.autoregressive_option = configs.autoregressive_option
        self.embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dropout = nn.Dropout(p=configs.dropout)
        self.mamba = MambaModel(configs)

        self.out_layer = nn.Linear(configs.d_model, configs.c_out, bias=False)
        print("Model initialized with autoregressive option:", configs.autoregressive_option)
        print("Model initialized with Mamba option:", configs.mamba_ffn_option)

    def forecast(self, x_enc, x_mark_enc):
        # x_enc = [x_{t-96}, x_{t-95}, ..., x_{t-1}]                    # length 96
        # x_dec = [x_{t-48}, ..., x_{t-1}, placeholder_{t}, ..., placeholder_{t+95}]  # length 144
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        x = self.embedding(x_enc, x_mark_enc)
        x = self.mamba(x)
        x_out = self.out_layer(x)
        x_out = x_out * std_enc + mean_enc
        return x_out

    def autoregressive_forecast(self, x_dec, x_mark_dec):
        # x_enc = [x_{t-96}, x_{t-95}, ..., x_{t-1}]                    # length 96
        # x_dec = [x_{t-48}, ..., x_{t-1}, placeholder_{t}, ..., placeholder_{t+95}]  # length 144
        mean_dec = x_dec[:,:self.label_len,:].mean(1, keepdim=True).detach()
        x_dec[:,:self.label_len,:] = x_dec[:,:self.label_len,:] - mean_dec
        std_dec = torch.sqrt(torch.var(x_dec[:,:self.label_len,:], dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_dec[:,:self.label_len,:] = x_dec[:,:self.label_len,:] / std_dec


        x = self.embedding(x_dec, x_mark_dec)
        x = self.mamba(x)
        x_out = self.out_layer(x)
        x_out = x_out * std_dec + mean_dec
        return x_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.autoregressive_option:
            # Autoregressive branch: use only the decoder input.
            x_out = self.autoregressive_forecast(x_dec, x_mark_dec)
            return x_out[:, -self.pred_len:, :]
        else:
            # Original seq2seq branch: use the encoder input.
            x_out = self.forecast(x_enc, x_mark_enc)
            return x_out[:, -self.pred_len:, :]

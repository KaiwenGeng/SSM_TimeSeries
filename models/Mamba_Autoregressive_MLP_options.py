import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.hydra.modules.hydra import Hydra
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

class HydraLayer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.hydra = Hydra(
            d_model=configs.d_model,
            d_state=configs.d_state if hasattr(configs, 'd_state') else 64,
            d_conv=configs.d_conv,
            expand=configs.expand,
            use_mem_eff_path=True
        )
        self.dropout = nn.Dropout(p=configs.dropout)
        self.norm = nn.LayerNorm(configs.d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.hydra(x)
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

class HydraLayerWithFFN(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.hydra = Hydra(
            d_model=configs.d_model,
            d_state=configs.d_state if hasattr(configs, 'd_state') else 64,
            d_conv=configs.d_conv,
            expand=configs.expand 
        )
        self.dropout = nn.Dropout(p=configs.dropout)
        self.norm = nn.LayerNorm(configs.d_model)
        self.ffn = FeedForward(
            d_model=configs.d_model, 
            d_ff=configs.d_ff, 
            dropout=configs.dropout
        )
    def forwad(self,x):
        residual = x
        x = self.norm(x)
        x = self.hydra(x)
        x = self.dropout(x)
        x = residual + x
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

class HydraEncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        if configs.mamba_ffn_option:
            self.layers = nn.ModuleList([
                HydraLayerWithFFN(configs) for _ in range(configs.e_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                HydraLayer(configs) for _ in range(configs.e_layers)
            ])
        self.norm = nn.LayerNorm(configs.d_model)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

class HydraDecoder(MambaModel):
    pass


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.autoregressive_option = configs.autoregressive_option
        self.embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        if configs.autoregressive_option:
            self.mamba_recurrent = MambaModel(configs)
        else:
            self.hydra_encoder = HydraEncoder(configs)
            self.hydra_decoder = HydraDecoder(configs)

        self.out_layer = nn.Linear(configs.d_model, configs.c_out, bias=False)
        print("Model initialized with autoregressive option:", configs.autoregressive_option)
        print("Model initialized with Mamba option:", configs.mamba_ffn_option)



    def autoregressive_forecast(self, x_dec, x_mark_dec):
        # x_enc = [x_{t-96}, x_{t-95}, ..., x_{t-1}]                    # length 96
        # x_dec = [x_{t-48}, ..., x_{t-1}, placeholder_{t}, ..., placeholder_{t+95}]  # length 144
        mean_dec = x_dec[:,:self.label_len,:].mean(1, keepdim=True).detach()
        x_dec[:,:self.label_len,:] = x_dec[:,:self.label_len,:] - mean_dec
        std_dec = torch.sqrt(torch.var(x_dec[:,:self.label_len,:], dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_dec[:,:self.label_len,:] = x_dec[:,:self.label_len,:] / std_dec


        x = self.embedding(x_dec, x_mark_dec)
        x = self.mamba_recurrent(x)
        x_out = self.out_layer(x)
        x_out = x_out * std_dec + mean_dec
        return x_out
    
    def encoder_decoder_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        embed_out = self.embedding(x_enc, x_mark_enc)
        enc_out = self.hydra_encoder(embed_out)
        dec_out = self.hydra_decoder(enc_out)
        dec_out = self.out_layer(dec_out)
        dec_out = dec_out * std_enc + mean_enc
        return dec_out
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.autoregressive_option:
            x_out = self.autoregressive_forecast(x_dec, x_mark_dec)
            return x_out[:, -self.pred_len:, :]
        else:
            x_out = self.encoder_decoder_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return x_out[:, -self.pred_len:, :]

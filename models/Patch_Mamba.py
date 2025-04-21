import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from hydra.hydra.modules.hydra import Hydra
from layers.Embed import Cross_Mamba_Embedding

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class FiLMFuse(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.to_gamma_beta = nn.Linear(d_model, 2 * d_model)

    def forward(self, time_enc, chan_enc):
        B, V, D, P = time_enc.shape
        gamma_beta = self.to_gamma_beta(chan_enc)     
        gamma, beta = gamma_beta.chunk(2, dim=-1)     

        # broadcast across patches
        gamma = gamma.unsqueeze(-1).expand(B, V, D, P)  
        beta  = beta .unsqueeze(-1).expand(B, V, D, P)

        return gamma * time_enc + beta                
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual



class Model(nn.Module):
    
    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride




        self.embedding = Cross_Mamba_Embedding(configs.enc_in, configs.seq_len, configs.d_model, patch_len, stride, padding, configs.dropout)
        self.timewise_model = Mamba(
            d_model = configs.d_model,
            expand = configs.expand,
        )
        self.timewise_ffn = FeedForward(configs.d_model, configs.d_ff)
        self.channelwise_model = Hydra(
            d_model=configs.d_model,
            d_state=configs.d_state if hasattr(configs, 'd_state') else 16,
            expand=configs.expand,
            headdim=configs.expand * configs.d_model // configs.n_heads,
            use_mem_eff_path=True
        )
        self.channelwise_ffn = FeedForward(configs.d_model, configs.d_ff)
        self.film_fuse = FiLMFuse(configs.d_model)




        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)





    def forecast(self, x_enc, x_mark_enc):

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        timewise_embed, channelwise_embed, n_vars = self.embedding(x_enc)
        timewise_enc = self.timewise_model(timewise_embed)
        timewise_enc = self.timewise_ffn(timewise_enc)

        channelwise_enc = self.channelwise_model(channelwise_embed)
        channelwise_enc = self.channelwise_ffn(channelwise_enc)


        timewise_enc = torch.reshape(
            timewise_enc, (-1, n_vars, timewise_enc.shape[-2], timewise_enc.shape[-1]))
        timewise_enc = timewise_enc.permute(0, 1, 3, 2)
        timewise_enc = self.film_fuse(timewise_enc, channelwise_enc)
        
        timewise_enc = self.head(timewise_enc)
        dec_out = timewise_enc.permute(0, 2, 1)
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out





    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['short_term_forecast', 'long_term_forecast']:
            x_out = self.forecast(x_enc, x_mark_enc)
            return x_out[:, -self.pred_len:, :]

        # other tasks not implemented
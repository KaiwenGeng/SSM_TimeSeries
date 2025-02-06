import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba
from layers.Embed import DataEmbedding

# A simple patch embedding module. This one uses a 1D convolution.
# It assumes that the input has shape [B, 1, L] and produces tokens of dimension d_model.
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super().__init__()
        # Using Conv1d with in_channels=1 so that the same conv is applied per channel
        self.proj = nn.Conv1d(in_channels=1, out_channels=d_model, 
                              kernel_size=patch_len, stride=stride, padding=padding)
        self.dropout = nn.Dropout(dropout)
        
        # For later head, compute the number of patches (this formula assumes L is known)
        # Here we leave it as a parameter that you can compute given a typical input length.
        # Alternatively, you might compute it dynamically from the input.
        self.patch_len = patch_len
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # x: [B, 1, L]
        x = self.proj(x)  # -> [B, d_model, N_patches]
        x = self.dropout(x)
        return x  # return shape: [B, d_model, N_patches]

# The modified model using patching and channel independence
class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in  # number of input channels
        
        # We'll still use your DataEmbedding if desired.
        # (In PatchTST the raw channels are preserved; if your embedding fuses channels,
        #  you might consider a per-channel linear projection instead.)
        self.embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, 
                                       configs.freq, configs.dropout)
        
        # A patch embedding layer that will be applied per channel.
        # We'll assume that after normalization we want to split the time axis into patches.
        # For a given sequence length L, the number of patches is: floor((L + 2*padding - patch_len) / stride) + 1.
        # (You might want to fix padding = stride for simplicity.)
        padding = stride
        self.patch_embedding = PatchEmbedding(configs.d_model, patch_len, stride, padding, configs.dropout)
        
        # Instead of feeding the full (long) sequence, we now feed the patch tokens into Mamba.
        self.mamba = Mamba(
            d_model = configs.d_model,
            d_state = configs.d_ff,
            d_conv = configs.d_conv,
            expand = configs.expand,
        )
        
        # Now, we need a head that will map the (flattened) tokens into the forecast length.
        # For each channel, after patching the sequence becomes N_patches tokens.
        # We will flatten the patch (token) dimension and then project to the desired forecast length.
        # For simplicity, let’s assume N_patches is fixed (or can be computed given a typical input length).
        # For example, if typical seq_len is L, then:
        #   N_patches = floor((L + 2*padding - patch_len) / stride) + 1
        # Here we denote head_nf = d_model * N_patches.
        # You might choose to use a similar strategy as FlattenHead in the PatchTST code.
        # For illustration, we assume:
        self.head_nf = configs.d_model * ( (configs.seq_len + 2*padding - patch_len) // stride + 1 )
        # Here, we want to map the flattened token representation (per channel) to the forecast length.
        # We set up a linear layer that works on each channel independently.
        self.out_layer = nn.Linear(self.head_nf, self.pred_len, bias=False)
    
    def forecast(self, x_enc, x_mark_enc):
        # x_enc assumed shape: [B, L, C]
        # Normalize per sample and per channel:
        # Compute mean and std along time dimension (L) for each channel.
        # mean_enc, std_enc shape: [B, 1, C]
        mean_enc = x_enc.mean(dim=1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        B, L, C = x_enc.shape

        # For channel-independence, we want to process each channel separately.
        # Permute x_enc to [B, C, L]
        x_enc = x_enc.permute(0, 2, 1)  # now: [B, C, L]
        # Reshape so that each channel becomes a separate “batch” element:
        x_enc = x_enc.reshape(B * C, L).unsqueeze(1)  # shape: [B * C, 1, L]

        # Apply the patch embedding
        # The patch embedding will produce tokens of shape: [B * C, d_model, N_patches]
        x_patch = self.patch_embedding(x_enc)
        # Permute to get the tokens in sequence order: [B * C, N_patches, d_model]
        x_patch = x_patch.permute(0, 2, 1)
        
        # Pass patch tokens through Mamba.
        # (Mamba is used in place of a transformer here.)
        x_patch = self.mamba(x_patch)  # shape remains: [B * C, N_patches, d_model]
        
        # Flatten the patch tokens (for each channel) to a single vector:
        # Resulting shape: [B * C, N_patches * d_model]
        x_flat = x_patch.flatten(start_dim=1)
        
        # Apply the prediction head.
        # This head is shared across all channels.
        # x_head: [B * C, pred_len]
        x_head = self.out_layer(x_flat)
        
        # Reshape to get back per-batch and per-channel predictions:
        # x_head: [B, C, pred_len]
        x_head = x_head.reshape(B, C, self.pred_len)
        # Permute to get shape [B, pred_len, C]
        x_out = x_head.permute(0, 2, 1)
        
        # De-normalize: mean_enc and std_enc were of shape [B, 1, C] so broadcast along the time axis.
        x_out = x_out * std_enc + mean_enc  # shape: [B, pred_len, C]
        return x_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['short_term_forecast', 'long_term_forecast']:
            x_out = self.forecast(x_enc, x_mark_enc)
            return x_out  # [B, pred_len, C]
        # other tasks not implemented
        return None

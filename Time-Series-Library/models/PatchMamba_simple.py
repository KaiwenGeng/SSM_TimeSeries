import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba

# A simple patch embedding module using a 1D convolution.
# It assumes the input has shape [B, 1, L] and produces tokens of dimension d_model.
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super().__init__()
        # Using Conv1d with in_channels=1 so that the same conv is applied per channel.
        self.proj = nn.Conv1d(in_channels=1, out_channels=d_model, 
                              kernel_size=patch_len, stride=stride, padding=padding)
        self.dropout = nn.Dropout(dropout)
        self.patch_len = patch_len
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # x: [B, 1, L]
        x = self.proj(x)  # -> [B, d_model, N_patches]
        x = self.dropout(x)
        return x  # returns [B, d_model, N_patches]

# The modified model using patching and channel independence.
# This version supports forecasting, imputation, anomaly detection, and classification.
class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len   # used for forecasting tasks
        self.seq_len = configs.seq_len     # input sequence length (used for imputation/anomaly detection)
        self.enc_in = configs.enc_in       # number of input channels
        self.d_model = configs.d_model
        self.dropout_rate = configs.dropout
        padding = stride

        # Patch embedding is applied per channel.
        self.patch_embedding = PatchEmbedding(configs.d_model, patch_len, stride, padding, configs.dropout)

        # Create a list of Mamba layers (used as the backbone instead of transformers).
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=configs.d_model,
                d_state=configs.d_ff,    # use d_ff as state size
                d_conv=configs.d_conv,
                expand=configs.expand
            )
            for _ in range(configs.e_layers)
        ])

        # Compute the number of patches per channel.
        self.num_patches = ((self.seq_len + 2*padding - patch_len) // stride) + 1
        # head_nf is the flattened feature size per channel.
        self.head_nf = configs.d_model * self.num_patches

        # Create task-specific heads.
        if self.task_name in ['short_term_forecast', 'long_term_forecast']:
            # Forecasting head projects from flattened patch tokens to the forecast length.
            self.out_layer = nn.Linear(self.head_nf, self.pred_len, bias=False)
        elif self.task_name == 'imputation':
            # Imputation head reconstructs the entire input sequence.
            self.out_layer_imp = nn.Linear(self.head_nf, self.seq_len, bias=False)
        elif self.task_name == 'anomaly_detection':
            # Anomaly detection head also reconstructs the entire sequence.
            self.out_layer_anom = nn.Linear(self.head_nf, self.seq_len, bias=False)
        elif self.task_name == 'classification':
            # For classification we flatten the features across channels.
            self.dropout = nn.Dropout(configs.dropout)
            # The input dimension for classification is enc_in * head_nf.
            self.projection = nn.Linear(self.enc_in * self.head_nf, configs.num_class)

    def forecast(self, x_enc, x_mark_enc):
        # x_enc: [B, L, C]
        # Normalize per sample and channel.
        mean_enc = x_enc.mean(dim=1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        B, L, C = x_enc.shape
        # Process each channel independently.
        x_enc = x_enc.permute(0, 2, 1)  # -> [B, C, L]
        x_enc = x_enc.reshape(B * C, L).unsqueeze(1)  # -> [B*C, 1, L]

        # Apply patch embedding.
        x_patch = self.patch_embedding(x_enc)  # -> [B*C, d_model, num_patches]
        x_patch = x_patch.permute(0, 2, 1)        # -> [B*C, num_patches, d_model]

        # Forward through the Mamba backbone.
        for mamba_layer in self.mamba_layers:
            x_patch = mamba_layer(x_patch)
        x_flat = x_patch.flatten(start_dim=1)  # -> [B*C, head_nf]

        # Apply forecasting head.
        x_head = self.out_layer(x_flat)         # -> [B*C, pred_len]
        x_head = x_head.reshape(B, C, self.pred_len)  # -> [B, C, pred_len]
        x_out = x_head.permute(0, 2, 1)           # -> [B, pred_len, C]
        # De-normalize.
        x_out = x_out * std_enc + mean_enc
        return x_out

    def imputation(self, x_enc, x_mark_enc, mask=None):
        # Imputation: reconstruct the entire input sequence.
        # Optionally, if a mask is provided (with 1 for observed, 0 for missing),
        # compute statistics only on the observed values.
        if mask is not None:
            mean_enc = (x_enc * mask).sum(dim=1, keepdim=True) / (mask.sum(dim=1, keepdim=True) + 1e-5)
            var_enc = ((x_enc - mean_enc)**2 * mask).sum(dim=1, keepdim=True) / (mask.sum(dim=1, keepdim=True) + 1e-5)
        else:
            mean_enc = x_enc.mean(dim=1, keepdim=True)
            var_enc = x_enc.var(dim=1, keepdim=True, unbiased=False)
        x_norm = (x_enc - mean_enc) / torch.sqrt(var_enc + 1e-5)

        B, L, C = x_norm.shape
        x_norm = x_norm.permute(0, 2, 1)  # -> [B, C, L]
        x_norm = x_norm.reshape(B * C, L).unsqueeze(1)  # -> [B*C, 1, L]

        x_patch = self.patch_embedding(x_norm)  # -> [B*C, d_model, num_patches]
        x_patch = x_patch.permute(0, 2, 1)        # -> [B*C, num_patches, d_model]
        for mamba_layer in self.mamba_layers:
            x_patch = mamba_layer(x_patch)
        x_flat = x_patch.flatten(start_dim=1)     # -> [B*C, head_nf]

        # Apply imputation head.
        x_head = self.out_layer_imp(x_flat)         # -> [B*C, seq_len]
        x_head = x_head.reshape(B, C, self.seq_len)  # -> [B, C, seq_len]
        x_out = x_head.permute(0, 2, 1)              # -> [B, seq_len, C]
        # Inverse normalization.
        x_out = x_out * torch.sqrt(var_enc + 1e-5) + mean_enc
        return x_out

    def anomaly_detection(self, x_enc, x_mark_enc):
        # Anomaly detection: reconstruct the sequence so that reconstruction error can be used as an anomaly score.
        mean_enc = x_enc.mean(dim=1, keepdim=True)
        x_norm = (x_enc - mean_enc) / torch.sqrt(x_enc.var(dim=1, keepdim=True, unbiased=False) + 1e-5)

        B, L, C = x_norm.shape
        x_norm = x_norm.permute(0, 2, 1)  # -> [B, C, L]
        x_norm = x_norm.reshape(B * C, L).unsqueeze(1)  # -> [B*C, 1, L]

        x_patch = self.patch_embedding(x_norm)  # -> [B*C, d_model, num_patches]
        x_patch = x_patch.permute(0, 2, 1)        # -> [B*C, num_patches, d_model]
        for mamba_layer in self.mamba_layers:
            x_patch = mamba_layer(x_patch)
        x_flat = x_patch.flatten(start_dim=1)     # -> [B*C, head_nf]

        # Apply anomaly detection head.
        x_head = self.out_layer_anom(x_flat)         # -> [B*C, seq_len]
        x_head = x_head.reshape(B, C, self.seq_len)  # -> [B, C, seq_len]
        x_out = x_head.permute(0, 2, 1)              # -> [B, seq_len, C]
        x_out = x_out * torch.sqrt(x_enc.var(dim=1, keepdim=True, unbiased=False) + 1e-5) + mean_enc
        return x_out

    def classification(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        # Classification: produce a single label per input sequence.
        # Normalize the input.
        mean_enc = x_enc.mean(dim=1, keepdim=True).detach()
        x_norm = (x_enc - mean_enc) / torch.sqrt(x_enc.var(dim=1, keepdim=True, unbiased=False) + 1e-5)

        B, L, C = x_norm.shape
        x_norm = x_norm.permute(0, 2, 1)  # -> [B, C, L]
        x_norm = x_norm.reshape(B * C, L).unsqueeze(1)  # -> [B*C, 1, L]

        x_patch = self.patch_embedding(x_norm)  # -> [B*C, d_model, num_patches]
        x_patch = x_patch.permute(0, 2, 1)        # -> [B*C, num_patches, d_model]
        for mamba_layer in self.mamba_layers:
            x_patch = mamba_layer(x_patch)
        x_flat = x_patch.flatten(start_dim=1)     # -> [B*C, head_nf]

        # Reshape to combine channel features.
        x_flat = x_flat.reshape(B, C * self.head_nf)  # -> [B, C * head_nf]
        x_flat = self.dropout(x_flat)
        logits = self.projection(x_flat)              # -> [B, num_class]
        return logits

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ['short_term_forecast', 'long_term_forecast']:
            return self.forecast(x_enc, x_mark_enc)
        elif self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, mask)
        elif self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc, x_mark_enc)
        elif self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        else:
            return None

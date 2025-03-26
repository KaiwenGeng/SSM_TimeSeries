import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba
# Use patch-based embedding instead of DataEmbedding
from layers.Embed import PatchEmbedding

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

class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len

        # Patching parameters: use defaults if not provided in configs.
        self.patch_len = getattr(configs, 'patch_len', 16)
        self.stride = getattr(configs, 'stride', 8)
        padding = self.stride  # using stride as padding

        # Replace the original DataEmbedding with PatchEmbedding.
        self.patch_embedding = PatchEmbedding(configs.d_model, self.patch_len, self.stride, padding, configs.dropout)
        
        # NEW: Variance projection layer to map the local standard deviation (1D) to d_model dimensions.
        self.var_proj = nn.Linear(1, configs.d_model)

        # Your Mamba module (assumed to work on sequences of patches)
        self.mamba = Mamba(
            d_model = configs.d_model,
            d_state = configs.d_ff,
            d_conv = configs.d_conv,
            expand = configs.expand,
        )
        
        # Compute the flattened feature size after patching.
        # Following PatchTST: head_nf = d_model * int((seq_len - patch_len) / stride + 2)
        self.head_nf = configs.d_model * (int((configs.seq_len - self.patch_len) / self.stride) + 2)
        
        # For forecasting tasks, use the FlattenHead to aggregate patch outputs.
        if self.task_name in ['short_term_forecast', 'long_term_forecast']:
            # Here, configs.enc_in (number of channels) is passed as n_vars.
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len, head_dropout=configs.dropout)
        else:
            # For other tasks, retain your original out_layer.
            self.out_layer = nn.Linear(configs.d_model, configs.c_out, bias=False)

    def forecast(self, x_enc, x_mark_enc):
        # Global normalization: compute mean and std along the time dimension.
        # x_enc: [B, seq_len, c]
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        # Permute x_enc to [B, c, seq_len] so that patching is applied per channel.
        x_enc_perm = x_enc.permute(0, 2, 1)

        # --- Variance-aware feature injection ---
        # Compute the local standard deviation (std) for each patch.
        # Use the same padding and unfolding parameters as the patch embedding.
        padding = self.stride  # same as used in patch_embedding
        x_padded = F.pad(x_enc_perm, (0, padding), mode='replicate')
        # Unfold the padded sequence into patches:
        #   shape: [B, c, num_patches, patch_len]
        patches = x_padded.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # Compute the standard deviation within each patch (along patch_len dimension):
        #   shape: [B, c, num_patches]
        local_std = patches.std(dim=-1, unbiased=False)
        # ---------------------------------------

        # Apply patch embedding: returns x_patch of shape [B * n_vars, patch_num, d_model]
        x_patch, n_vars = self.patch_embedding(x_enc_perm)
        
        # Reshape local_std to match x_patch: from [B, c, num_patches] to [B*c, num_patches, 1]
        B = x_enc.size(0)
        num_patches = x_patch.size(1)
        local_std = local_std.view(B * n_vars, num_patches, 1)
        
        # Project the local std (variance feature) to d_model dimensions.
        var_feature = self.var_proj(local_std)  # shape: [B*n_vars, num_patches, d_model]
        
        # Integrate the variance feature with the patch embeddings.
        x_patch = x_patch + var_feature

        # Process the patch sequences with the Mamba module.
        x_patch = self.mamba(x_patch)
        
        # Reshape back: recover the channel dimension.
        # x_patch: [B * n_vars, patch_num, d_model] -> [B, n_vars, patch_num, d_model]
        patch_num = x_patch.size(1)
        x_patch = x_patch.view(B, n_vars, patch_num, -1)
        
        # For the head, permute to [B, n_vars, d_model, patch_num]
        x_patch = x_patch.permute(0, 1, 3, 2)
        
        # Use the FlattenHead to aggregate patches and output a forecast of length pred_len.
        dec_out = self.head(x_patch)  # [B, n_vars, pred_len]
        
        # Permute to [B, pred_len, n_vars] so that each channel (variable) is along the last dimension.
        dec_out = dec_out.permute(0, 2, 1)
        
        # De-normalize the output.
        dec_out = dec_out * std_enc + mean_enc
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['short_term_forecast', 'long_term_forecast']:
            x_out = self.forecast(x_enc, x_mark_enc)
            # Return the last pred_len time steps.
            return x_out[:, -self.pred_len:, :]
        # Other tasks not implemented.

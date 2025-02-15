import math
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from layers.Embed import MambaPatchEmbedding

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super(FlattenHead, self).__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)
    
    def forward(self, x):
        # x shape: [batch_size x n_vars x d_model x patch_num]
        x = self.flatten(x)            # -> [batch_size x n_vars x (d_model * patch_num)]
        x = self.linear(x)            # -> [batch_size x n_vars x target_window]
        x = self.dropout(x)
        return x

class Model(nn.Module):
    """
    PatchTST model with Mamba state-space encoder (no Transformer attention).
    """
    def __init__(self, configs, patch_len=16, stride=None):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # Patch length and stride (use config if available, otherwise default)
        patch_len = getattr(configs, 'patch_len', patch_len)
        if stride is None:
            # If stride not specified, default to patch_len/2 (like original default 8 for patch_len 16)
            stride = getattr(configs, 'stride', patch_len // 2)
        padding = stride  # Replication padding size (same as stride in original PatchTST)
        
        # Patching and embedding layer
        self.patch_embedding = MambaPatchEmbedding(configs.d_model, patch_len, stride, padding, configs.dropout)
        
        # Encoder: replace Transformer layers with Mamba SSM layers
        # Instantiate a stack of Mamba layers (configs.e_layers times)
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=configs.d_model,
                d_state=configs.d_ff,    # use d_ff as state size
                d_conv=configs.d_conv,
                expand=configs.expand
            )
            for _ in range(configs.e_layers)
        ])
        
        # Compute the flattened dimension for the output head
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        
        # Prediction Head initialization for different tasks
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            # For forecasting, target_window = pred_len
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len, head_dropout=configs.dropout)
        elif self.task_name in ['imputation', 'anomaly_detection']:
            # For imputation and anomaly detection, target_window = seq_len (reconstruct the full sequence)
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len, head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            # For classification, use flatten and linear projection to num_class (as in original PatchTST)
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(self.head_nf * configs.enc_in, configs.num_class)
        # (No else needed; assume task_name is one of the above valid options)
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalize input (channel-wise) as in Non-stationary Transformer
        means = x_enc.mean(dim=1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / stdev
        
        # Patch embedding: shape input to [B, C, L] -> [B, C, patch_num, patch_len] -> project to [B*C, patch_num, d_model]
        x_enc = x_enc.permute(0, 2, 1)                           # [B, C, L] -> [B, L, C] -> actually to [B, C, L] because original was [B, L, C]
        # (Note: Original code uses x_enc.permute(0,2,1) to shape [B, enc_in, seq_len] to [B, seq_len, enc_in]. 
        # Here enc_in = number of variables C, so after permute: [B, C, L]. PatchEmbedding expects [B, C, L].)
        enc_out, n_vars = self.patch_embedding(x_enc)            # enc_out: [B*n_vars, patch_num, d_model]
        
        # Pass through Mamba encoder (stack of Mamba layers)
        for mamba_layer in self.mamba_layers:
            enc_out = mamba_layer(enc_out)                      # [B*n_vars, patch_num, d_model] -> [B*n_vars, patch_num, d_model]
        
        # Reshape back to [B, n_vars, patch_num, d_model]
        enc_out = enc_out.reshape(-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        # Permute to [B, n_vars, d_model, patch_num] for the FlattenHead
        enc_out = enc_out.permute(0, 1, 3, 2)
        
        # Decode: use FlattenHead to get final forecasting output [B, n_vars, target_window]
        dec_out = self.head(enc_out)                             # [B, n_vars, target_window]
        dec_out = dec_out.permute(0, 2, 1)                       # [B, target_window, n_vars]
        
        # De-normalize the output
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)) + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out
    
    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalize input with masking (Non-stationary Transformer approach for imputation)
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        # Set missing positions (mask == 0) to zero after mean subtraction
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc / stdev
        
        # Patch embedding
        x_enc = x_enc.permute(0, 2, 1)                           # [B, C, L] -> [B, C, L] (as above reasoning)
        enc_out, n_vars = self.patch_embedding(x_enc)            # [B*n_vars, patch_num, d_model]
        # Mamba encoder
        for mamba_layer in self.mamba_layers:
            enc_out = mamba_layer(enc_out)
        # Reshape and permute to [B, n_vars, d_model, patch_num]
        enc_out = enc_out.reshape(-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        enc_out = enc_out.permute(0, 1, 3, 2)
        
        # Decode through FlattenHead to get imputed sequence [B, n_vars, seq_len]
        dec_out = self.head(enc_out)                            # [B, n_vars, seq_len]
        dec_out = dec_out.permute(0, 2, 1)                       # [B, seq_len, n_vars]
        
        # De-normalize output (restore original scale)
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)) + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out
    
    def anomaly_detection(self, x_enc):
        # Normalize input (Non-stationary Transformer)
        means = x_enc.mean(dim=1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / stdev
        
        # Patch embedding
        x_enc = x_enc.permute(0, 2, 1)                          # [B, C, L] -> [B, C, L]
        enc_out, n_vars = self.patch_embedding(x_enc)           # [B*n_vars, patch_num, d_model]
        # Mamba encoder
        for mamba_layer in self.mamba_layers:
            enc_out = mamba_layer(enc_out)
        # Reshape and permute to [B, n_vars, d_model, patch_num]
        enc_out = enc_out.reshape(-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        enc_out = enc_out.permute(0, 1, 3, 2)
        
        # Decode to reconstructed sequence [B, n_vars, seq_len]
        dec_out = self.head(enc_out)                           # [B, n_vars, seq_len]
        dec_out = dec_out.permute(0, 2, 1)                      # [B, seq_len, n_vars]
        
        # De-normalize output
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)) + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out
    
    def classification(self, x_enc, x_mark_enc):
        # Normalize input (Non-stationary Transformer)
        means = x_enc.mean(dim=1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / stdev
        
        # Patch embedding
        x_enc = x_enc.permute(0, 2, 1)                         # [B, C, L] -> [B, C, L]
        enc_out, n_vars = self.patch_embedding(x_enc)          # [B*n_vars, patch_num, d_model]
        # Mamba encoder
        for mamba_layer in self.mamba_layers:
            enc_out = mamba_layer(enc_out)
        # Reshape and permute to [B, n_vars, d_model, patch_num]
        enc_out = enc_out.reshape(-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        enc_out = enc_out.permute(0, 1, 3, 2)
        
        # Classification head: flatten and project to classes
        output = self.flatten(enc_out)                        # flatten last two dims
        output = self.dropout(output)
        output = output.view(output.size(0), -1)              # [B, n_vars*d_model*patch_num] flattened
        output = self.projection(output)                      # [B, num_class]
        return output
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Dispatch to the appropriate method based on task
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            # Forecasting tasks
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            # Return last pred_len time steps [B, L, D]
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)  # [B, L, D]
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)  # [B, N] where N = num_class
        # If none of the tasks match, return None (should not happen in proper use)
        return None

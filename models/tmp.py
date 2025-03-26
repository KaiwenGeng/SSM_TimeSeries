import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba
from layers.Embed import DataEmbedding


class ConditionalHeteroskedasticityModule(nn.Module):
    """
    Learns a multiplicative scale factor for each time step and channel,
    allowing the network to adapt to time-varying variance in the data.
    """
    def __init__(self, d_model, c_in):
        """
        Args:
            d_model: dimension of the model/embedding
            c_in:    number of input channels
        """
        super().__init__()
        # Projects from [B, L, d_model] to a scale of shape [B, L, c_in].
        # We exponentiate for positivity and unbounded scale.
        self.scale_proj = nn.Linear(d_model, c_in)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: Tensor of shape [B, L, d_model]
        
        Returns:
            scale:         Tensor of shape [B, L, c_in], each > 0
        """
        scale = torch.exp(self.scale_proj(hidden_states))  # shape [B, L, c_in]
        return scale


class Patchify(nn.Module):
    """
    Simple patching along the time dimension, similar in spirit to PatchTST.
    We chunk the sequence into consecutive patches of length `patch_size`.
    """
    def __init__(self, patch_size=16):
        """
        Args:
            patch_size: how many time steps in each patch
        """
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        """
        Args:
            x: [B, L, D] sequence of length L, embedding dimension D

        Returns:
            x_patched: [B, n_patch, patch_size, D]
        """
        B, L, D = x.shape

        # If L is not divisible by patch_size, truncate or pad as needed.
        # Here, we will do simple truncation to keep code clear:
        n_full_patches = L // self.patch_size
        L_trunc = n_full_patches * self.patch_size
        x = x[:, :L_trunc, :]  # shape [B, L_trunc, D]

        # Reshape into patches
        x_patched = x.view(
            B, 
            n_full_patches, 
            self.patch_size, 
            D
        )  # shape [B, n_patch, patch_size, D]

        return x_patched


class GroupMamba(nn.Module):
    """
    Wraps Mamba in a grouped fashion over channels.
    Instead of a single Mamba operating jointly on all channels,
    we split the embedding dimension into G groups and apply Mamba on each group independently.
    
    This channel grouping can help reduce overfitting by limiting 
    parameter interactions across all channels.
    """
    def __init__(self, d_model, d_state, d_conv, expand, n_groups=4):
        """
        Args:
            d_model:  embedding dimension (also treat as 'channels' dimension here)
            d_state:  Mamba's d_state
            d_conv:   Mamba's d_conv
            expand:   Mamba's expand
            n_groups: number of groups to split the d_model dimension
        """
        super().__init__()
        self.d_model = d_model
        self.n_groups = n_groups
        # We'll assume d_model is divisible by n_groups for simplicity
        self.group_size = d_model // n_groups

        # Create one Mamba instance per group
        self.mambas = nn.ModuleList([
            Mamba(
                d_model=self.group_size, 
                d_state=d_state, 
                d_conv=d_conv, 
                expand=expand
            )
            for _ in range(n_groups)
        ])

    def forward(self, x):
        """
        Args:
            x: [B, L, d_model] - we treat d_model as 'channels' dimension, 
               which we now group in self.n_groups chunks.

        Returns:
            out: [B, L, (d_model * 1)] if we let each group Mamba keep output dimension = group_size
                 or we can also expand to something else.
            For simplicity, Mamba is configured to produce output dimension = group_size again,
            so we can just concatenate.
        """
        B, L, D = x.shape
        assert D == self.d_model, "Input dimension must match the defined d_model"

        # Split the embedding dimension into n_groups chunks
        group_outs = []
        for i, mamba_block in enumerate(self.mambas):
            start = i * self.group_size
            end = (i + 1) * self.group_size
            # slice out channels for this group
            x_g = x[:, :, start:end]  # [B, L, group_size]
            out_g = mamba_block(x_g)  # e.g. [B, L, group_size]
            group_outs.append(out_g)

        # Concatenate the group outputs back along the channel dimension
        out = torch.cat(group_outs, dim=-1)  # shape [B, L, d_model]
        return out


class Model(nn.Module):
    """
    New 'Model' that:
      - Uses an additional HeteroskedasticityModule to adapt input scale.
      - Applies patching in the time dimension to reduce overfitting.
      - Applies GroupMamba to encourage channel-independence structure.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len

        self.d_inner = configs.d_model * configs.expand
        self.dt_rank = math.ceil(configs.d_model / 16) # as in your original code

        # Original data embedding
        self.embedding = DataEmbedding(
            configs.enc_in, 
            configs.d_model, 
            configs.embed, 
            configs.freq, 
            configs.dropout
        )

        # 1) Conditional heteroskedasticity module
        self.hetero_module = ConditionalHeteroskedasticityModule(
            d_model=configs.d_model, 
            c_in=configs.enc_in
        )

        # 2) Patching module
        #    Adjust patch_size to your data or treat as a hyperparameter
        self.patchify = Patchify(patch_size=16)

        # 3) Grouped Mamba (channel independence)
        #    We treat d_model as "channels" here.
        self.group_mamba = GroupMamba(
            d_model=configs.d_model, 
            d_state=configs.d_ff,
            d_conv=configs.d_conv,
            expand=configs.expand,
            n_groups=4  # tune this
        )

        # Final linear projection: 
        #   By default, shape after group_mamba remains [B, L, d_model].
        #   We want output shape [B, L, c_out].
        #   c_out is presumably the #channels in the final forecast.
        self.out_layer = nn.Linear(configs.d_model, configs.c_out, bias=False)

    def forecast(self, x_enc, x_mark_enc):
        """
        The original forecast logic with standard normalization.
        We then add the new patching + grouping + heteroskedastic module inside.
        """
        # -------------------------
        # Normalization as before
        # -------------------------
        mean_enc = x_enc.mean(dim=1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x_enc = x_enc / std_enc  # shape [B, L, c_in]

        # ----------------------------------
        # Base embedding in the model space
        # ----------------------------------
        # => shape [B, L, d_model]
        x_embed = self.embedding(x_enc, x_mark_enc)

        # ------------------------------------------------
        # 1) Obtain scale factor from heteroskedasticity
        #    module and rescale original input x_enc.
        # ------------------------------------------------
        # scale has shape [B, L, c_in]; 
        # we multiply the *original un-embedded* x_enc by scale 
        scale = self.hetero_module(x_embed)  # [B, L, c_in]
        x_enc_scaled = x_enc * scale  # still [B, L, c_in]

        # Re-embed the scaled input
        x_embed_scaled = self.embedding(x_enc_scaled, x_mark_enc)  # [B, L, d_model]

        # Combine the two embeddings (original + scaled) 
        # so the model can learn “variance-based gating”
        x_combined = x_embed + x_embed_scaled  # [B, L, d_model]

        # ------------------------------------------------
        # 2) Patchify the time dimension
        # ------------------------------------------------
        # => [B, n_patch, patch_size, d_model]
        x_patched = self.patchify(x_combined)

        B, n_patch, patch_size, D = x_patched.shape
        # Flatten patches into (batch * n_patch) for Mamba
        x_patched = x_patched.view(B * n_patch, patch_size, D)  # => [B*n_patch, patch_size, d_model]

        # ------------------------------------------------
        # 3) Grouped Mamba for channel-independence
        # ------------------------------------------------
        x_mamba = self.group_mamba(x_patched)  # => [B*n_patch, patch_size, d_model]

        # Un-patchify back
        x_mamba = x_mamba.view(B, n_patch, patch_size, D)  # => [B, n_patch, patch_size, d_model]
        # Merge the patch dimension back into the time dimension
        x_mamba = x_mamba.view(B, n_patch * patch_size, D)  # => [B, L, d_model]

        # ------------------------------------------------
        # Final linear to produce forecast
        # ------------------------------------------------
        x_out = self.out_layer(x_mamba)  # => [B, L, c_out]

        # Denormalize
        x_out = x_out * std_enc + mean_enc  # shape [B, L, c_out]

        return x_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Keep your original pipeline: we only return the final predictions,
        cutting off the last self.pred_len points if that's how your pipeline
        handles short/long-term forecasts.
        """
        if self.task_name in ['short_term_forecast', 'long_term_forecast']:
            x_out = self.forecast(x_enc, x_mark_enc)
            # Return only the last pred_len points in time dimension
            return x_out[:, -self.pred_len:, :]
        else:
            # For other tasks, adapt as needed
            return None

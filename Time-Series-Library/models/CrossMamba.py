import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Segment Merging: merges adjacent time segments to form coarser-scale segments
class SegMerging(nn.Module):
    """
    Merges adjacent segments (length = win_size) along the time segment axis to produce a coarser scale.
    This is identical to Crossformer's segment merging: it concatenates each group of `win_size` segments and 
    uses a linear projection back to d_model&#8203;:contentReference[oaicite:7]{index=7}&#8203;:contentReference[oaicite:8]{index=8}.
    """
    def __init__(self, d_model, win_size):
        super(SegMerging, self).__init__()
        self.win_size = win_size
        self.norm = nn.LayerNorm(win_size * d_model)
        self.linear = nn.Linear(win_size * d_model, d_model)
    def forward(self, x):
        """
        Input: x of shape [B, D, seg_num, d_model]
        Output: merged x of shape [B, D, seg_num/win_size, d_model] (if seg_num not divisible, last segments are repeated/padded).
        """
        B, D, seg_num, dm = x.shape
        if seg_num % self.win_size != 0:
            # Pad by repeating the last segment to make segment count divisible by win_size
            pad_count = self.win_size - (seg_num % self.win_size)
            pad_tensor = x[:, :, -1:, :].expand(B, D, pad_count, dm)  # repeat last segment
            x = torch.cat([x, pad_tensor], dim=2)
            seg_num = x.shape[2]
        # Split x into win_size groups along segment axis and concatenate features
        # Example: for win_size=2, take x[:, :, 0::2, :] and x[:, :, 1::2, :] and concat on feature dim
        merged_seq = []
        for i in range(self.win_size):
            merged_seq.append(x[:, :, i::self.win_size, :])  # take every win_size-th segment starting at i
        # Now merged_seq is a list of length win_size, each of shape [B, D, seg_num/win_size, d_model]
        x_merged = torch.cat(merged_seq, dim=-1)  # concat on the feature dimension: shape [B, D, seg_num/win_size, win_size*d_model]
        # Apply layer normalization and linear projection to reduce feature dim back to d_model
        x_merged = self.norm(x_merged)
        x_merged = self.linear(x_merged)
        return x_merged

# Two-Stage Mamba Layer: captures cross-time and cross-dimension dependencies using Mamba
class TwoStageMambaLayer(nn.Module):
    """
    Replaces Crossformer's Two-Stage Attention (TSA) with two Mamba passes&#8203;:contentReference[oaicite:9]{index=9}:
    1. Time-wise Mamba: scans each variable's time segments sequence (captures temporal dependency per variable).
    2. Dimension-wise Mamba: scans each time-segment's across variables (captures inter-variable dependency).
    Both use Mamba's state-space sequence modeling (no attention)&#8203;:contentReference[oaicite:10]{index=10}.
    """
    def __init__(self, d_model, d_state, d_conv, expand):
        super(TwoStageMambaLayer, self).__init__()
        # Mamba modules for time and dimension axes. Using the mamba_ssm library for linear-time sequence modeling.
        from mamba_ssm import Mamba  # ensure mamba_ssm is installed
        # Mamba for sequences of length = seg_num (temporal sequence of segments for one variable)
        self.time_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        # Mamba for sequences of length = number of variables (sequence of variables for one segment position)
        self.dim_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    def forward(self, x):
        """
        x: Tensor of shape [B, D, seg_num, d_model].
        Returns: Tensor of shape [B, D, seg_num, d_model] after two-stage Mamba processing.
        """
        B, D, seg_num, dm = x.shape
        # Stage 1: Time-wise Mamba (process each variable's segment sequence)
        # Reshape to [B*D, seg_num, d_model] to treat each variable in each batch as a separate sequence
        x_time = x.view(B * D, seg_num, dm)  # (batch*variables) x seg_num x d_model
        x_time = self.time_mamba(x_time)     # apply Mamba along the seg_num dimension (time axis for each var)
        # Reshape back to [B, D, seg_num, d_model]
        x_time = x_time.view(B, D, seg_num, dm)
        # Stage 2: Dimension-wise Mamba (process each time segment across variables)
        # Permute to [B, seg_num, D, d_model] to treat variable axis as sequence
        x_dim = x_time.permute(0, 2, 1, 3).contiguous()  # shape [B, seg_num, D, d_model]
        # Reshape to [B*seg_num, D, d_model] to treat each segment index in each batch as a separate sequence of length D
        x_dim = x_dim.view(B * seg_num, D, dm)
        x_dim = self.dim_mamba(x_dim)  # apply Mamba along the D dimension (variable axis for each time segment)
        # Reshape back to [B, seg_num, D, d_model] and then permute to [B, D, seg_num, d_model]
        x_dim = x_dim.view(B, seg_num, D, dm)
        out = x_dim.permute(0, 2, 1, 3).contiguous()  # [B, D, seg_num, d_model]
        return out

# Encoder consisting of multiple scale blocks (with optional segment merging and TwoStageMamba layers)
class MambaEncoder(nn.Module):
    """
    Hierarchical Encoder with Mamba as backbone. Uses multiple scale blocks:
    - The first block operates on the finest time scale (no merging, win_size=1).
    - Subsequent blocks merge segments (win_size > 1) to create coarser scales&#8203;:contentReference[oaicite:11]{index=11}&#8203;:contentReference[oaicite:12]{index=12}.
    Each block applies a TwoStageMambaLayer to capture cross-time and cross-dimension dependencies at that scale.
    """
    def __init__(self, e_layers, win_size, d_model, d_state, d_conv, expand, seg_num):
        super(MambaEncoder, self).__init__()
        self.encode_blocks = nn.ModuleList()
        # First scale block (no merging, win_size=1) operates on initial seg_num
        self.encode_blocks.append(
            MambaScaleBlock(win_size=1, d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, seg_num=seg_num)
        )
        # Subsequent scale blocks with merging factor `win_size`
        for i in range(1, e_layers):
            # After i-th block, segment count reduces by win_size^i (ceil for safety)
            next_seg_num = math.ceil(seg_num / (win_size ** i))
            self.encode_blocks.append(
                MambaScaleBlock(win_size=win_size, d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, seg_num=next_seg_num)
            )
    def forward(self, x):
        """
        x: [B, D, in_seg_num, d_model] (output of PatchEmbedding + optional normalization).
        Returns: List of encoder outputs at each scale (including input as first element), each element shape [B, D, seg_num, d_model].
        """
        encode_outputs = [x]
        for block in self.encode_blocks:
            x = block(x)
            encode_outputs.append(x)
        return encode_outputs

class MambaScaleBlock(nn.Module):
    """
    A single scale processing block: optional segment merging followed by one TwoStageMambaLayer (or more, but depth=1 here).
    """
    def __init__(self, win_size, d_model, d_state, d_conv, expand, seg_num):
        super(MambaScaleBlock, self).__init__()
        # If win_size > 1, include a segment merging layer to reduce time resolution&#8203;:contentReference[oaicite:13]{index=13}&#8203;:contentReference[oaicite:14]{index=14}.
        self.merge = SegMerging(d_model, win_size) if win_size > 1 else None
        # We use a depth of 1 TwoStageMambaLayer per block (similar to Crossformer where block_depth=1&#8203;:contentReference[oaicite:15]{index=15})
        self.tsa_layer = TwoStageMambaLayer(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    def forward(self, x):
        # x shape: [B, D, seg_num, d_model]
        if self.merge is not None:
            x = self.merge(x)  # merge adjacent segments if needed (coarser scale)
        x = self.tsa_layer(x)  # apply two-stage Mamba to capture dependencies at this scale
        return x

# Decoder components
class MambaDecoderLayer(nn.Module):
    """
    Decoder layer with:
    - Mamba-based self-attention (TwoStageMambaLayer) on the target (decoder input) sequence.
    - Cross integration of encoder output (replaces cross-attention with a simple additive fusion of encoder features).
    - A position-wise FeedForward network (MLP) for feature transformation.
    - Linear prediction layer to map decoder output to actual time-series segment values.
    """
    def __init__(self, d_model, seg_len, d_state, d_conv, expand, dropout=0.1):
        super(MambaDecoderLayer, self).__init__()
        self.self_mamba = TwoStageMambaLayer(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        # LayerNorms and Dropouts for residual connections
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        # Position-wise FeedForward (MLP)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        # Linear layer to project features to seg_len values for each segment (predict time-series segment)
        self.linear_pred = nn.Linear(d_model, seg_len)
    def forward(self, x, enc_cross):
        """
        x: [B, D, out_seg_num, d_model] - decoder's running state
        enc_cross: [B, D, enc_seg_num, d_model] - corresponding encoder output (at some scale) for cross integration
        Returns: (dec_output, layer_prediction)
          dec_output: [B, D, out_seg_num, d_model] for next layer input
          layer_prediction: [B, D, out_seg_num, seg_len] partial forecast from this layer
        """
        B, D, out_seg, dm = x.shape
        # Self-attention via Mamba (two-stage): model dependencies within target sequence
        x = self.self_mamba(x)  # shape remains [B, D, out_seg_num, d_model]
        # Cross integration: fuse encoder features. We add encoded features to decoder features (after appropriate reshape) instead of attention.
        if enc_cross is not None:
            # enc_cross: [B, D, enc_seg_num, d_model]. We need to map enc_seg -> out_seg length.
            enc = enc_cross  # shape [B, D, enc_seg_num, d_model]
            # If encoder segments != decoder segments, interpolate or aggregate:
            enc_seg = enc.shape[2]
            if enc_seg != out_seg:
                # Simple interpolation along segment axis to match lengths
                enc = F.interpolate(enc, size=out_seg, mode='linear', align_corners=False)  # [B, D, out_seg, d_model]
            # Now enc shape [B, D, out_seg, d_model]
            # Add encoder features to decoder features (residual cross-connection)
            x = x + self.dropout_attn(enc)
        # Normalize after cross integration
        x = self.norm1(x)
        # Position-wise feed-forward network
        y = self.ffn(x)
        # Residual connection and second normalization
        out = self.norm2(x + self.dropout_ffn(y))
        # Linear projection to actual time-series segment values
        # out shape [B, D, out_seg, d_model] -> linear_pred on last dim -> [B, D, out_seg, seg_len]
        layer_pred = self.linear_pred(out)
        return out, layer_pred

class MambaDecoder(nn.Module):
    """
    Hierarchical Decoder with Mamba. It uses multiple decoder layers, each corresponding to an encoder scale output.
    Each layer produces a prediction for a certain scale of the output, and these are summed to form the final output&#8203;:contentReference[oaicite:16]{index=16}.
    """
    def __init__(self, layers: nn.ModuleList):
        super(MambaDecoder, self).__init__()
        self.layers = layers  # a ModuleList of MambaDecoderLayer
    def forward(self, x, enc_outputs):
        """
        x: [B, D, out_seg_num, d_model] initial decoder query embeddings.
        enc_outputs: list of encoder outputs at each scale (length = e_layers+1, with enc_outputs[0] being input embedding).
        Returns: Tensor of shape [B, total_pred_length, D] which is the final forecast sequence for all variables.
        """
        B, D, out_seg, dm = x.shape
        final_pred = None
        # We assume enc_outputs list length == number of decoder layers
        for i, layer in enumerate(self.layers):
            enc_out_i = enc_outputs[i] if i < len(enc_outputs) else None  # encoder output for this layer's scale
            x, layer_pred = layer(x, enc_out_i)
            # layer_pred: [B, D, out_seg, seg_len]
            # Accumulate predictions from each layer (additive refinement across scales)&#8203;:contentReference[oaicite:17]{index=17}.
            layer_pred_flat = layer_pred.reshape(B, D * out_seg, -1)  # [B, D*out_seg, seg_len]
            if final_pred is None:
                final_pred = layer_pred_flat
            else:
                final_pred = final_pred + layer_pred_flat
        # After all layers, final_pred shape [B, D*out_seg, seg_len]. 
        # Rearrange to [B, (out_seg*seg_len), D] which corresponds to [B, pad_out_len, D] (time × variables)&#8203;:contentReference[oaicite:18]{index=18}.
        # Note: out_seg * seg_len = pad_out_len (the padded prediction length)
        pred_length = final_pred.shape[2]  # seg_len
        final_pred = final_pred.view(B, D, out_seg * pred_length)  # [B, D, pad_out_len]
        final_pred = final_pred.permute(0, 2, 1).contiguous()      # [B, pad_out_len, D]
        return final_pred

# PatchEmbedding for Dimension-Segment-Wise embedding
class PatchEmbedding(nn.Module):
    def __init__(self, d_model: int, patch_len: int, stride: int, padding: int, dropout: float):
        super(PatchEmbedding, self).__init__()
        # Use segment length from config as patch_len (stride is usually equal to patch_len for non-overlapping patches)
        self.patch_len = patch_len  
        self.stride = stride
        self.padding = padding
        # Linear projection from patch_len to d_model
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        # (Optional) positional embedding for patch positions
        self.position_embedding = nn.Parameter(torch.zeros(1, 1, patch_len, d_model))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor):
        """
        x shape: [B, n_vars, L] – batch, number of variables, sequence length
        """
        n_vars = x.shape[1]  # number of variables (channels)
        # 1. Pad the sequence length to be divisible by patch_len
        if self.padding and self.padding > 0:
            x = F.pad(x, (0, self.padding))  # pad last dimension (time) on the right
        # New sequence length after padding
        L_full = x.shape[2]
        seg_num = L_full // self.patch_len  # number of patches per variable
        # 2. Split into patches of length patch_len
        # Using unfold to create a patches: [B, n_vars, seg_num, patch_len]
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)  # shape [B, n_vars, seg_num, patch_len]
        # 3. Reshape to merge batch and variable dimensions for projection
        patches = patches.contiguous().view(-1, seg_num, self.patch_len)        # shape [B * n_vars, seg_num, patch_len]
        # 4. Linear projection for each patch
        patch_embeddings = self.value_embedding(patches)                        # shape [B * n_vars, seg_num, d_model]
        # Add (optional) patch positional encoding (broadcasted across batch and vars)
        if self.position_embedding is not None:
            patch_embeddings = patch_embeddings + self.position_embedding[..., :patch_embeddings.size(1), :]
        patch_embeddings = self.dropout(patch_embeddings)
        # Return patch embeddings and n_vars for reshaping in the caller
        return patch_embeddings, n_vars

# Main Crossformer Model with Mamba backbone
class Model(nn.Module):
    """
    Crossformer model (ICLR'23) re-implemented with Mamba replacing all Transformer-based modules.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name  # e.g., 'long_term_forecast', 'short_term_forecast', 'imputation', 'classification', etc.
        self.enc_in = configs.enc_in        # number of input variables
        self.seq_len = configs.seq_len      # input sequence length
        self.pred_len = configs.pred_len    # prediction length
        self.seg_len = getattr(configs, 'seg_len', 12)  # segment length (using 12 as in Crossformer default)
        self.win_size = getattr(configs, 'win_size', 2)  # window size for segment merging (default 2 in Crossformer)
        self.e_layers = configs.e_layers    # number of encoder layers (and decoder layers = e_layers+1)
        self.d_model = configs.d_model
        # Mamba-specific dimensions: use configs or default values
        self.d_ff = configs.d_ff            # we will use d_ff as Mamba's state dimension
        self.d_conv = getattr(configs, 'd_conv', 1)
        self.expand = getattr(configs, 'expand', 1)
        # Compute padded lengths and segment counts for input and output
        self.pad_in_len = math.ceil(self.seq_len / self.seg_len) * self.seg_len
        self.pad_out_len = math.ceil(self.pred_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        self.out_seg_num = math.ceil(self.in_seg_num / (self.win_size ** (self.e_layers - 1)))  # approximate segments for output scale
        # Embedding 
        self.enc_value_embedding = PatchEmbedding(self.d_model, self.seg_len, self.seg_len, self.pad_in_len - self.seq_len, configs.dropout)
        self.pre_norm = nn.LayerNorm(self.d_model)  # normalize embeddings before encoder (as in original Crossformer)
        # Encoder: Hierarchical with Mamba blocks
        self.encoder = MambaEncoder(e_layers=self.e_layers, win_size=self.win_size, 
                                    d_model=self.d_model, d_state=self.d_ff, d_conv=self.d_conv, expand=self.expand,
                                    seg_num=self.in_seg_num)
        # Decoder: build e_layers+1 decoder layers (to utilize all encoder outputs including input embedding)
        dec_layers = nn.ModuleList()
        for i in range(self.e_layers + 1):
            dec_layers.append(MambaDecoderLayer(d_model=self.d_model, seg_len=self.seg_len,
                                                d_state=self.d_ff, d_conv=self.d_conv, expand=self.expand,
                                                dropout=configs.dropout))
        self.decoder = MambaDecoder(dec_layers)
        # For classification task, define a classification head 
        if self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)  # flatten time and segment dims
            self.dropout = nn.Dropout(configs.dropout)
            # After flatten, shape will be [B, D*seq_len]; project to num_class
            self.projection = nn.Linear(self.enc_in * self.seq_len, configs.num_class)
    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        Perform forecasting. 
        x_enc: [B, seq_len, D] input history
        Returns: [B, pad_out_len, D] predictions (including any padded extra length).
        """
        # 1. DSW embedding on encoder input
        # Rearrange x_enc to [B, D, seq_len] for embedding
        x_enc = x_enc.permute(0, 2, 1).contiguous()  # [B, D, L]
        emb, n_vars = self.enc_value_embedding(x_enc)  # emb: [(B*D), in_seg_num, d_model]
        # Reshape back to [B, D, in_seg_num, d_model]
        enc_emb = emb.view(-1, n_vars, self.in_seg_num, self.d_model)  # here -1 corresponds to B
        # (We assume n_vars == enc_in and -1 ends up as B)
        # 2. (Optional) Normalize embeddings 
        enc_emb = self.pre_norm(enc_emb)
        # 3. Encoder: get multi-scale outputs
        enc_outputs = self.encoder(enc_emb)  # list of [B, D, seg_num, d_model] at scales 0...e_layers
        # 4. Prepare decoder input (query) as zeros (no positional encoding needed). Shape: [B, D, out_seg_num, d_model].
        B = enc_emb.shape[0]
        dec_in = torch.zeros(B, n_vars, self.out_seg_num, self.d_model, device=x_enc.device)
        # 5. Decoder: use encoder outputs to generate predictions
        dec_out = self.decoder(dec_in, enc_outputs)
        # dec_out shape: [B, pad_out_len, D] (padded prediction length, D variables)
        return dec_out
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # Only forecasting is implemented (imputation/anomaly not explicitly handled, similar to Mamba model)&#8203;:contentReference[oaicite:21]{index=21}.
        if self.task_name in ['short_term_forecast', 'long_term_forecast']:
            # Use forecast and then take the last pred_len part of the output (trim padding)
            output = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            # output is [B, pad_out_len, D]; trim to pred_len
            return output[:, -self.pred_len:, :]
        elif self.task_name == 'classification':
            # For classification, we can use the encoder output of the last scale (without decoder)
            # Take the final encoder output at the finest scale (encode_outputs[-1] is coarsest, perhaps use encode_outputs[0] which is input embedding).
            # Here we simply use the raw input or first encoder output for classification, as a placeholder.
            x_input = x_enc.permute(0, 2, 1).contiguous()  # [B, D, L]
            emb, n_vars = self.enc_value_embedding(x_input)
            enc_emb = emb.view(-1, n_vars, self.in_seg_num, self.d_model)
            enc_emb = self.pre_norm(enc_emb)
            enc_outputs = self.encoder(enc_emb)
            # Use the **finest** scale representation (enc_outputs[0] which is just input embedding after norm) for classification
            feat = enc_outputs[0]  # [B, D, in_seg_num, d_model]
            # Flatten and project to classes
            feat_flat = self.flatten(feat)    # [B, D*in_seg_num*d_model] but originally they flattened -2 meaning flatten seg and d_model only
            # Actually, flatten(-2) as in Crossformer classification flattens (D, seg_num) leaving B and d_model separate? 
            # To keep it simple, flatten all except batch:
            feat_flat = feat_flat.view(feat_flat.size(0), -1)
            out = self.dropout(feat_flat)
            out = self.projection(out)
            return out
        else:
            # Placeholder for other tasks (imputation, anomaly detection, etc.)
            # Simply perform forecasting for now (this can be adapted if those tasks require different handling).
            output = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return output[:, -self.pred_len:, :]

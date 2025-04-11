import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer, DecoderLayer  
from layers.Transformer_EncDec import Decoder                       
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np

class ARDecoderLayer(DecoderLayer):
    def __init__(self, self_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__(self_attention, cross_attention=None, d_model=d_model, d_ff=d_ff, dropout=dropout, activation=activation)

    def forward(self, x, x_mask=None, tau=None, delta=None):

        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)
        y = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm3(x + y)

class ARDecoder(Decoder):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__(layers, norm_layer=norm_layer, projection=projection)

    def forward(self, x, x_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, x_mask=x_mask, tau=tau, delta=delta)
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.autoregressive = configs.autoregressive_option
        self.label_len = configs.label_len

        if self.autoregressive:
            # Autoregressive (decoder-only) branch.
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                               configs.dropout)
            # Build the autoregressive decoder using the subclassed ARDecoderLayer.
            self.decoder = ARDecoder(
                [ARDecoderLayer(
                    self_attention=AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads
                    ),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for _ in range(configs.d_layers)],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )
        else:
            # Seq2seq (encoder-decoder) branch.
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                               configs.dropout)
            self.encoder = Encoder(
                [EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)],
                norm_layer=torch.nn.LayerNorm(configs.d_model)
            )
            if self.task_name in ['long_term_forecast', 'short_term_forecast']:
                self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                   configs.dropout)
                self.decoder = Decoder(
                    [DecoderLayer(
                        self_attention=AttentionLayer(
                            FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                            configs.d_model, configs.n_heads
                        ),
                        cross_attention=AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                            configs.d_model, configs.n_heads
                        ),
                        d_model=configs.d_model,
                        d_ff=configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    ) for _ in range(configs.d_layers)],
                    norm_layer=torch.nn.LayerNorm(configs.d_model),
                    projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
                )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc = [x_{t-96}, x_{t-95}, ..., x_{t-1}]                    # length 96
        # x_dec = [x_{t-48}, ..., x_{t-1}, placeholder_{t}, ..., placeholder_{t+95}]  # length 144
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        dec_out = dec_out * std_enc + mean_enc
        return dec_out
    
    def forecast_ar(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc = [x_{t-96}, x_{t-95}, ..., x_{t-1}]                    # length 96
        # x_dec = [x_{t-48}, ..., x_{t-1}, placeholder_{t}, ..., placeholder_{t+95}]  # length 144
        # note: mean and std should only be considered with self.label_len because the rest are placeholder
        mean_dec = x_dec[:,:self.label_len,:].mean(1, keepdim=True).detach()
        x_dec[:,:self.label_len,:] = x_dec[:,:self.label_len,:] - mean_dec
        std_dec = torch.sqrt(torch.var(x_dec[:,:self.label_len,:], dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_dec[:,:self.label_len,:] = x_dec[:,:self.label_len,:] / std_dec


        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, x_mask=None, tau=None, delta=None)
        dec_out = dec_out * std_dec + mean_dec
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        if self.autoregressive:
            dec_out = self.forecast_ar(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        else:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np


class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.autoregressive_option = getattr(configs, 'autoregressive_option', False)
        
        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                               configs.dropout)
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )
        print("Model initialized with autoregressive option:", configs.autoregressive_option)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """Non-autoregressive forecasting"""
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc_standardized = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc_standardized, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc_standardized = x_enc_standardized / std_enc
        
        # Encode
        enc_out = self.enc_embedding(x_enc_standardized, x_mark_enc) 
        enc_out, attns = self.encoder(enc_out, attn_mask=None) 
        
        # Decode
        x_dec_standardized = (x_dec - mean_enc) / std_enc
        dec_out = self.dec_embedding(x_dec_standardized, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        
        # Denormalize
        dec_out = dec_out * std_enc + mean_enc
        return dec_out

    def forecast_AR(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """Autoregressive forecasting"""
        # Normalization parameters
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc_standardized = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc_standardized, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc_standardized = x_enc_standardized / std_enc
        
        # Encode
        enc_out = self.enc_embedding(x_enc_standardized, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        batch_size, _, feature_dim = x_dec.shape
        
        # Initialize with first input token of decoder sequence
        curr_input = x_dec[:, :1].clone()
        curr_input_standardized = (curr_input - mean_enc) / std_enc
        
        # Store predictions
        predictions = []
        
        # Generate predictions step by step
        for i in range(self.pred_len):
            # Get embedding for the current input
            dec_in = self.dec_embedding(curr_input_standardized, x_mark_dec[:, :curr_input.shape[1]])
            
            # Pass through decoder with cross-attention to encoder
            output = self.decoder(dec_in, enc_out, x_mask=None, cross_mask=None)
            
            # Get prediction for the last time step
            pred = output[:, -1:].clone()
            
            # Store the prediction
            predictions.append(pred)
            
            # Update input for next step (standardized)
            curr_input = torch.cat([curr_input, pred * std_enc + mean_enc], dim=1)
            curr_input_standardized = torch.cat([curr_input_standardized, pred], dim=1)
            
        # Concatenate all predictions and denormalize
        predictions = torch.cat(predictions, dim=1)
        predictions = predictions * std_enc + mean_enc
        
        return predictions

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if self.autoregressive_option:
                dec_out = self.forecast_AR(x_enc, x_mark_enc, x_dec, x_mark_dec)
            else:
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                dec_out = dec_out[:, -self.pred_len:, :]  # [B, L, D]
            return dec_out
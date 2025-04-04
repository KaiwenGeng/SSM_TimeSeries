#!/bin/bash

# Define model names
model_name1="Transformer_Autoregressive_MLP_options"
model_name2="Mamba_Autoregressive_MLP_options"

# Loop over prediction lengths
for pred_len in 96 192 336 720; do

    ##############################
    # Transformer: 2 combinations
    ##############################
    # Combination 1: autoregressive_option = False

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_${pred_len}_${pred_len}_Transformer_autoregressiveFalse \
      --model $model_name1 \
      --data ETTh1 \
      --features M \
      --seq_len $pred_len \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 1 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --d_model 64 \
      --des 'Exp' \
      --itr 1 \
      --autoregressive_option False

    # Combination 2: autoregressive_option = True


    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_${pred_len}_${pred_len}_Transformer_autoregressiveTrue \
      --model $model_name1 \
      --data ETTh1 \
      --features M \
      --seq_len $pred_len \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 1 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --d_model 64 \
      --des 'Exp' \
      --itr 1 \
      --autoregressive_option True

    ##############################
    # Mamba: 4 combinations
    ##############################
    # Combination 1: autoregressive_option = False, mamba_ffn_option = False


    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_${pred_len}_${pred_len}_Mamba_autoregressiveFalse_mambaFFNFalse \
      --model $model_name2 \
      --data ETTh1 \
      --features M \
      --seq_len $pred_len \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --enc_in 7 \
      --expand 2 \
      --d_ff 16 \
      --d_conv 4 \
      --c_out 7 \
      --d_model 192 \
      --des 'Exp' \
      --itr 1 \
      --autoregressive_option False \
      --mamba_ffn_option False

    # Combination 2: autoregressive_option = True, mamba_ffn_option = False

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_${pred_len}_${pred_len}_Mamba_autoregressiveTrue_mambaFFNFalse \
      --model $model_name2 \
      --data ETTh1 \
      --features M \
      --seq_len $pred_len \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --enc_in 7 \
      --expand 2 \
      --d_ff 16 \
      --d_conv 4 \
      --c_out 7 \
      --d_model 192 \
      --des 'Exp' \
      --itr 1 \
      --autoregressive_option True \
      --mamba_ffn_option False

    # Combination 3: autoregressive_option = False, mamba_ffn_option = True
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_${pred_len}_${pred_len}_Mamba_autoregressiveFalse_mambaFFNTrue \
      --model $model_name2 \
      --data ETTh1 \
      --features M \
      --seq_len $pred_len \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --enc_in 7 \
      --expand 2 \
      --d_ff 16 \
      --d_conv 4 \
      --c_out 7 \
      --d_model 192 \
      --des 'Exp' \
      --itr 1 \
      --autoregressive_option False \
      --mamba_ffn_option True

    # Combination 4: autoregressive_option = True, mamba_ffn_option = True
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_${pred_len}_${pred_len}_Mamba_autoregressiveTrue_mambaFFNTrue \
      --model $model_name2 \
      --data ETTh1 \
      --features M \
      --seq_len $pred_len \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --enc_in 7 \
      --expand 2 \
      --d_ff 16 \
      --d_conv 4 \
      --c_out 7 \
      --d_model 192 \
      --des 'Exp' \
      --itr 1 \
      --autoregressive_option True \
      --mamba_ffn_option True

done

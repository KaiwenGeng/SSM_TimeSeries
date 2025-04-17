#!/bin/bash

# Define model names
model_name1="Transformer_Autoregressive_MLP_options"
model_name2="Mamba_Autoregressive_MLP_options"

# Loop over prediction lengths
for pred_len in 96 192 336 720; do

    # SSM parameter: 6d^2 (with default hyperparameters) --- 2 layers gives 12d^2
    # SSM + FFN (d_ffn = 3 * d_model): 12d^2
    # Vanilla transformer: encoder-> 12d^2, decoder (self + cross attention) -> 16d^2 so 28d^2 in total
    # Decoder only transformer -> 12d^2

    ##############################
    # Transformer: 2 combinations
    ##############################
    # Combination 1: autoregressive_option = False
    # d_model = 336, d_ff = 1344 as 28 * 336^2 is close to 12 * 512^2

    label_len=$((pred_len / 2)) 
    # python -u run.py \
    #   --task_name long_term_forecast \
    #   --is_training 1 \
    #   --root_path ./dataset/ETT-small/ \
    #   --data_path ETTh1.csv \
    #   --model_id ETTh1_${pred_len}_${pred_len}_Transformer_autoregressiveFalse \
    #   --model $model_name1 \
    #   --data ETTh1 \
    #   --features M \
    #   --seq_len $pred_len \
    #   --label_len $label_len \
    #   --pred_len $pred_len \
    #   --e_layers 1 \
    #   --d_layers 1 \
    #   --factor 3 \
    #   --enc_in 7 \
    #   --dec_in 7 \
    #   --c_out 7 \
    #   --d_model 336 \
    #   --d_ff 1344 \
    #   --des 'Exp' \
    #   --itr 1 \
    #   --autoregressive_option False

    # Combination 2: autoregressive_option = True


    # python -u run.py \
    #   --task_name long_term_forecast \
    #   --is_training 1 \
    #   --root_path ./dataset/ETT-small/ \
    #   --data_path ETTh1.csv \
    #   --model_id ETTh1_${pred_len}_${pred_len}_Transformer_autoregressiveTrue \
    #   --model $model_name1 \
    #   --data ETTh1 \
    #   --features M \
    #   --seq_len $pred_len \
    #   --label_len $pred_len \
    #   --pred_len $pred_len \
    #   --e_layers 1 \
    #   --d_layers 1 \
    #   --factor 3 \
    #   --enc_in 7 \
    #   --dec_in 7 \
    #   --c_out 7 \
    #   --d_model 512 \
    #   --d_ff 2048 \
    #   --des 'Exp' \
    #   --itr 1 \
    #   --autoregressive_option True

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
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers 1 \
      --d_layers 1 \
      --enc_in 7 \
      --expand 2 \
      --d_conv 7 \
      --c_out 7 \
      --d_model 512 \
      --d_ff 1536 \
      --des 'Exp' \
      --itr 1 \
      --autoregressive_option False \
      --mamba_ffn_option False

    # Combination 2: autoregressive_option = True, mamba_ffn_option = False   # mamba 6d^2 ---> FFN another 6d^2 factor of 3 to match 12d^2 

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
      --label_len $pred_len \
      --pred_len $pred_len \
      --e_layers 1 \
      --d_layers 1 \
      --enc_in 7 \
      --expand 2 \
      --d_conv 7 \
      --c_out 7 \
      --d_model 512 \
      --d_ff 1536 \
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
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers 1 \
      --d_layers 1 \
      --enc_in 7 \
      --expand 2 \
      --d_conv 7 \
      --c_out 7 \
      --d_model 512 \
      --d_ff 1536 \
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
      --label_len $pred_len \
      --pred_len $pred_len \
      --e_layers 1 \
      --d_layers 1 \
      --enc_in 7 \
      --expand 2 \
      --d_conv 7 \
      --c_out 7 \
      --d_model 512 \
      --d_ff 1536 \
      --des 'Exp' \
      --itr 1 \
      --autoregressive_option True \
      --mamba_ffn_option True

done

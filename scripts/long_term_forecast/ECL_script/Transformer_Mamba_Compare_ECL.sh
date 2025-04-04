
model_name1=Transformer_Autoregressive_MLP_options
model_name2=Mamba_Autoregressive_MLP_options

for pred_len in 96 192 336 720; do

  for autoregressive_option in false true; do

    for patch_embedding in true false; do

      transformer_model_id_suffix="AR_${autoregressive_option}_PE_${patch_embedding}"

      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/electricity/ \
        --data_path electricity.csv \
        --model_id ECL_${pred_len}_${transformer_model_id_suffix} \
        --model $model_name1 \
        --data custom \
        --features M \
        --seq_len $pred_len \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 1 \
        --d_layers 1 \
        --enc_in 321 \
        --expand 2 \
        --d_ff 16 \
        --d_conv 4 \
        --c_out 321 \
        --d_model 128 \
        --des 'Exp' \
        --itr 1 \
        --autoregressive_option $autoregressive_option \
        --patch_embedding $patch_embedding

      # Now iterate separately for mamba_ffn_option (Mamba-specific)
      for mamba_ffn_option in true false; do

        # Mamba model_id (includes mamba_ffn_option)
        mamba_model_id_suffix="AR_${autoregressive_option}_FFN_${mamba_ffn_option}_PE_${patch_embedding}"

        python -u run.py \
          --task_name long_term_forecast \
          --is_training 1 \
          --root_path ./dataset/electricity/ \
          --data_path electricity.csv \
          --model_id ECL_${pred_len}_${mamba_model_id_suffix} \
          --model $model_name2 \
          --data custom \
          --features S \
          --seq_len $pred_len \
          --label_len 48 \
          --pred_len $pred_len \
          --e_layers 2 \
          --d_layers 1 \
          --factor 3 \
          --enc_in 1 \
          --dec_in 1 \
          --c_out 1 \
          --des 'Exp' \
          --itr 1 \
          --autoregressive_option $autoregressive_option \
          --patch_embedding $patch_embedding \
          --mamba_ffn_option $mamba_ffn_option

      done

    done
  done
done

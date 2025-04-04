model_name1=Transformer_Autoregressive_MLP_options
model_name2=Mamba_Autoregressive_MLP_options


python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Monthly' \
  --model_id m4_Monthly_ar \
  --model $model_name1 \
  --data m4 \
  --features M \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 128 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE'\
  --autoregressive_option 'True'

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Monthly' \
  --model_id m4_Monthly_ar \
  --model $model_name2 \
  --data m4 \
  --features M \
  --e_layers 2 \
  --enc_in 1 \
  --expand 2 \
  --d_ff 16 \
  --d_conv 4 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 300 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE'\
  --autoregressive_option 'True'

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Yearly' \
  --model_id m4_Yearly_ar \
  --model $model_name1 \
  --data m4 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 512 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE' \
  --autoregressive_option 'True'

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Yearly' \
  --model_id m4_Yearly_ar \
  --model $model_name2 \
  --data m4 \
  --features M \
  --e_layers 2 \
  --enc_in 1 \
  --expand 2 \
  --d_ff 16 \
  --d_conv 4 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 128 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE' \
  --autoregressive_option 'True'

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Quarterly' \
  --model_id m4_Quarterly_ar \
  --model $model_name1 \
  --data m4 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 512 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE' \
  --autoregressive_option 'True'

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Quarterly' \
  --model_id m4_Quarterly_ar \
  --model $model_name2 \
  --data m4 \
  --features M \
  --e_layers 2 \
  --enc_in 1 \
  --expand 2 \
  --d_ff 16 \
  --d_conv 4 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 128 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE' \
  --autoregressive_option 'True'

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Weekly' \
  --model_id m4_Weekly_ar \
  --model $model_name1 \
  --data m4 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 512 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE' \
  --autoregressive_option 'True'

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Weekly' \
  --model_id m4_Weekly_ar \
  --model $model_name2 \
  --data m4 \
  --features M \
  --e_layers 2 \
  --enc_in 1 \
  --expand 2 \
  --d_ff 16 \
  --d_conv 4 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 128 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE' \
  --autoregressive_option 'True'

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Daily' \
  --model_id m4_Daily_ar \
  --model $model_name1 \
  --data m4 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 512 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE' \
  --autoregressive_option 'True'

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Daily' \
  --model_id m4_Daily_ar \
  --model $model_name2 \
  --data m4 \
  --features M \
  --e_layers 2 \
  --enc_in 1 \
  --expand 2 \
  --d_ff 16 \
  --d_conv 4 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 128 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE' \
  --autoregressive_option 'True'

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Hourly' \
  --model_id m4_Hourly_ar \
  --model $model_name1 \
  --data m4 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 512 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE' \
  --autoregressive_option 'True'

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Hourly' \
  --model_id m4_Hourly_ar \
  --model $model_name2 \
  --data m4 \
  --features M \
  --e_layers 2 \
  --enc_in 1 \
  --expand 2 \
  --d_ff 16 \
  --d_conv 4 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 128 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE' \
  --autoregressive_option 'True'



























python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Monthly' \
  --model_id m4_Monthly \
  --model $model_name1 \
  --data m4 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 512 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE'

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Monthly' \
  --model_id m4_Monthly \
  --model $model_name2 \
  --data m4 \
  --features M \
  --e_layers 2 \
  --enc_in 1 \
  --expand 2 \
  --d_ff 16 \
  --d_conv 4 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 128 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE'

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Yearly' \
  --model_id m4_Yearly \
  --model $model_name1 \
  --data m4 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 512 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE' 

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Yearly' \
  --model_id m4_Yearly \
  --model $model_name2 \
  --data m4 \
  --features M \
  --e_layers 2 \
  --enc_in 1 \
  --expand 2 \
  --d_ff 16 \
  --d_conv 4 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 128 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE' 

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Quarterly' \
  --model_id m4_Quarterly \
  --model $model_name1 \
  --data m4 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 512 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
#   --loss 'SMAPE' 

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Quarterly' \
  --model_id m4_Quarterly \
  --model $model_name2 \
  --data m4 \
  --features M \
  --e_layers 2 \
  --enc_in 1 \
  --expand 2 \
  --d_ff 16 \
  --d_conv 4 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 128 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE' 

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Weekly' \
  --model_id m4_Weekly \
  --model $model_name1 \
  --data m4 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 512 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE' 

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Weekly' \
  --model_id m4_Weekly \
  --model $model_name2 \
  --data m4 \
  --features M \
  --e_layers 2 \
  --enc_in 1 \
  --expand 2 \
  --d_ff 16 \
  --d_conv 4 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 128 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE' 

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Daily' \
  --model_id m4_Daily \
  --model $model_name1 \
  --data m4 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 512 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE' 

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Daily' \
  --model_id m4_Daily \
  --model $model_name2 \
  --data m4 \
  --features M \
  --e_layers 2 \
  --enc_in 1 \
  --expand 2 \
  --d_ff 16 \
  --d_conv 4 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 128 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE' 

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Hourly' \
  --model_id m4_Hourly \
  --model $model_name1 \
  --data m4 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 512 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE' 

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Hourly' \
  --model_id m4_Hourly \
  --model $model_name2 \
  --data m4 \
  --features M \
  --e_layers 2 \
  --enc_in 1 \
  --expand 2 \
  --d_ff 16 \
  --d_conv 4 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 128 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE' 
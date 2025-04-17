# Update on Apr.9 2025
models/Transformer_Autoregressive_MLP_options.py: support autoregressive option (configs.autoregressive_option)
models/Mamba_Autoregressive_MLP_options.py: support autoregressive option (configs.autoregressive_option); support FFN option (configs.mamba_ffn_option)
# Example usage
bash scripts/long_term_forecast/ETT_script/Transformer_Mamba_Compare_ETTh1.sh 


# Update on Apr.17 2025
Incooporated Hydra as the encoder
Now using Mamba2.2.2

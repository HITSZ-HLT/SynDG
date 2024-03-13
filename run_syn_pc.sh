export CUDA_VISIBLE_DEVICES=3
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=1

python \
    synthesize.py \
    exp_config/syn_config_pc.json

export CUDA_VISIBLE_DEVICES=3
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=1

python \
    train.py \
    exp_config/dial_conf.json

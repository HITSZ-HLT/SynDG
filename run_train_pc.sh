export CUDA_VISIBLE_DEVICES=3
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=1

conf_file="exp_config/base_conf_pc.json"

python \
    train.py \
        $conf_file


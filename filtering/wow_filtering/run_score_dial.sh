set -e

export CUDA_VISIBLE_DEVICES=2
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=1

sampled_flow_dir=../../wizard_of_wikipedia/synthetic_data

python get_pred_scorer_data.py \
    $sampled_flow_dir/generated_predictions.json \
    $sampled_flow_dir/for_filter_input_dial.json \
    dial

# python -m debugpy --listen 5678 --wait-for-client \
python \
    pred_filter_score.py \
    --model_name_or_path outputs/dial_filter/checkpoint-41530 \
    --test_file $sampled_flow_dir/for_filter_input_dial.json \
    --output_dir $sampled_flow_dir \
    --per_device_eval_batch_size 16 \
    --cache_dir cache \
    --evaluation_strategy no \
    --do_predict true \
    --predict_with_generate true \
    --overwrite_output_dir true \
    --max_source_length 1000 \
    --max_target_length 500 \
    --generation_max_length 500 \
    --special_tokens '[prompt] [user-1] [user-2] [mask] [/mask] [grounding] [/grounding] [none] [session] [no_query]' \
    --generation_num_beams 1 \
    --log_level error
    # --max_predict_samples 100

python merge_ppl_score.py \
    $sampled_flow_dir/generated_predictions.json \
    $sampled_flow_dir/dial_scores.json \
    $sampled_flow_dir/generated_dials_w_dial_score.json
    


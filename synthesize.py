import logging
import os
import sys
import datasets
import numpy as np
from datasets import load_dataset
import json

import evaluate
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    M2M100Tokenizer,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
sys.path.append('./')
from utils import get_preprocess_function_inp_first_turn, get_preprocess_function_inp_each_turn

logger = logging.getLogger(__name__)

# A list of all multilingual tokenizer which require src_lang and tgt_lang attributes.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast, M2M100Tokenizer]

from train_config import ModelArguments, DataTrainingArguments

def main():
    dataset_name = os.getenv('DATASET_NAME')
    if not dataset_name:
        raise ValueError("Environment variable DATASET_NAME not set")
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    all_args = {}
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        import json
        with open(os.path.abspath(sys.argv[1]), 'r') as f:
            all_args = json.load(f)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        all_args.update(model_args.__dict__)
        all_args.update(data_args.__dict__)
        all_args.update(training_args.__dict__)
        all_args = str(all_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    data_files = {}
    data_files["test"] = data_args.test_file
    extension = data_args.test_file.split(".")[-1]
    extension = 'json' if extension == 'jsonl' else extension
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    special_tokens = data_args.special_tokens.split(' ')
    tokenizer.add_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False
    preprocess_function = get_preprocess_function_inp_first_turn(tokenizer, data_args, max_target_length, padding)

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                # remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # Metric
    metric = evaluate.load("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams

    def postprocess(text):

        def clean_text(text):
            for st in data_args.special_tokens.split(' '):
                text = text.replace(st, '').strip()
            return text

        text = clean_text(text)
        if text == '':
            text = '[no_query]'

        return text

    if training_args.do_predict:
        logger.info("*** Predict ***")

        gen_kwargs = {
            "max_length": max_length,
            "num_beams": num_beams,
            "top_k": data_args.top_k,
            "temperature": data_args.temperature,
            "do_sample": data_args.do_sample,
            "encoder_no_repeat_ngram_size": data_args.encoder_no_repeat_ngram_size
        }
        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", **gen_kwargs
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        predictions = tokenizer.batch_decode(
            predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        predictions = [pred.strip() for pred in predictions]
        if dataset_name == 'wow':
            predictions = list(map(postprocess, predictions))

        if dataset_name == 'wow':
            for turn in range(1, data_args.num_turns*2):
                logger.info(f"*** Inpainting turn {turn} ... ***")
                preprocess_function = get_preprocess_function_inp_each_turn(tokenizer, data_args, turn, predictions, padding)
                predict_dataset = predict_dataset.map(
                    preprocess_function,
                    with_indices=True,
                    batched=False,
                    # num_proc=data_args.preprocessing_num_workers,
                    # remove_columns=column_names,
                    load_from_cache_file=False,
                    # desc="Running tokenizer on inpainting dataset",
                )
                predict_results = trainer.predict(
                    predict_dataset, metric_key_prefix="predict", **gen_kwargs
                )
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                predictions = list(map(postprocess, predictions))
            
            preprocess_function = get_preprocess_function_inp_each_turn(tokenizer, data_args, data_args.num_turns*2, predictions, padding)
        elif dataset_name == 'pc':
            for turn in range(1, data_args.num_turns):
                logger.info(f"*** Inpainting turn {turn} ... ***")
                preprocess_function = get_preprocess_function_inp_each_turn(tokenizer, data_args, turn, predictions, padding, data_args.m)
                predict_dataset = predict_dataset.map(
                    preprocess_function,
                    with_indices=True,
                    batched=False,
                    # num_proc=data_args.preprocessing_num_workers,
                    # remove_columns=column_names,
                    load_from_cache_file=False,
                    # desc="Running tokenizer on inpainting dataset",
                )
                predict_results = trainer.predict(
                    predict_dataset, metric_key_prefix="predict", **gen_kwargs
                )
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]

            preprocess_function = get_preprocess_function_inp_each_turn(tokenizer, data_args, turn+1, predictions, padding, data_args.m)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")
        predict_dataset = predict_dataset.map(
            preprocess_function,
            with_indices=True,
            batched=False,
            # num_proc=data_args.preprocessing_num_workers,
            # remove_columns=column_names,
            load_from_cache_file=False,
            # desc="Running tokenizer on inpainting dataset",
        )
        gen_dial_strs = predict_dataset['source']
        if dataset_name == 'wow':
            gen_dial_strs = [dial[:dial.rfind('[user-1] [mask] [grounding]')].strip() for dial in gen_dial_strs]

            with open(data_args.test_file, 'r') as f:
                inp_json = json.load(f)
            for idx, d in enumerate(inp_json):
                gen_dial_str = gen_dial_strs[idx]
                sessions = gen_dial_str.split(' [user-1] ')[1:]
                u1_utts = [sess.split(' [user-2] ')[0].strip() for sess in sessions]
                u2_utts = [sess.split(' [user-2] ')[1].strip() for sess in sessions]
                d['u1_utts'] = u1_utts
                d['u2_utts'] = u2_utts
            output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.json")
            with open(output_prediction_file, "w") as f:
                json.dump(inp_json, f, indent=4)
        elif dataset_name == 'pc':
            gen_dial_strs = ['[user-1]'.join(dial.split('[user-1]')[:data_args.num_turns//2+1]) for dial in gen_dial_strs]

            sampled_flows = zip(predict_dataset['user_flow'], predict_dataset['agent_flow'])
            sampled_personas = zip(predict_dataset['user_personas'], predict_dataset['agent_personas'])
            # merge two flow to one flow (two users take turns)
            # sampled_flows = [[item for tup in zip(*flow_pair) for item in tup] for flow_pair in sampled_flows]

            pred_dialogues = []
            for idx, (s, flows, personas) in enumerate(zip(gen_dial_strs, sampled_flows, sampled_personas)):
                turns = s.split(' [user-1] ')
                turns = turns[1:]
                assert len(turns) == 8
                user_utterances = [turn.split(' [user-2] ')[0] for turn in turns]
                agent_utterances = [turn.split(' [user-2] ')[1] for turn in turns]
                dial_dict = {
                    "u1_utts": user_utterances,
                    "u2_utts": agent_utterances,
                    "u1_flows": flows[0],
                    "u2_flows": flows[1],
                    "u1_personas": personas[0],
                    "u2_personas": personas[1],
                }
                pred_dialogues.append(dial_dict)
            output_prediction_file = os.path.join(training_args.output_dir, "generated_dials.json")
            with open(output_prediction_file, 'w') as f:
                json.dump(pred_dialogues, f, indent=2)
            
            import nltk
            MAX_DIAL_NUM = -1
            output_formatted_file = os.path.join(training_args.output_dir, "gen_train_self_original.txt")
            with open(output_formatted_file, 'w') as fp:
                for i, d in enumerate(pred_dialogues):
                    idx = 1
                    for persona in d['u2_personas'].split(' | '):
                        fp.write(f'{idx} your persona: {persona}\n')
                        idx += 1
                    for user_utt, agent_utt in zip(d['u1_utts'], d['u2_utts']):
                        user_utt = ' '.join(nltk.word_tokenize(user_utt))
                        agent_utt = ' '.join(nltk.word_tokenize(agent_utt))
                        fp.write(f'{idx} {user_utt}\t{agent_utt}\t\tNone\n')
                        idx += 1
                    if i == MAX_DIAL_NUM-1:
                        break
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

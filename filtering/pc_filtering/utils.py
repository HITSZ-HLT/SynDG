

def get_preprocess_function(tokenizer, data_args, max_target_length, padding):
    def preprocess_function(examples):
        inputs = examples["source"]
        targets = examples["target"]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        # if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        #     labels["input_ids"] = [
        #         [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        #     ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return preprocess_function


def get_preprocess_function_inp_first_turn(tokenizer, data_args, max_target_length, padding):
    def preprocess_function(examples):
        inputs = examples["source"]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        return model_inputs
    return preprocess_function

def get_preprocess_function_inp_each_turn(tokenizer, data_args, turn, predictions, padding):
    def preprocess_function(examples, idx):
        tmp_user2flow = {'[user-1]': 'user_flow', '[user-2]': 'agent_flow'}
        source = examples['source']

        # get current grounding
        cur_user = '[user-1]' if turn%2==0 else '[user-2]'
        flow_idx = (turn)//2
        cur_flow = tmp_user2flow[cur_user]
        cur_grd = '[none]'
        if flow_idx < len(examples[cur_flow]):
            if len(examples[cur_flow][flow_idx]) != 0:
                cur_grd = ' </s> '.join(examples[cur_flow][flow_idx])

        next_user = '[user-1]' if turn%2==1 else '[user-2]'
        flow_idx = (turn+1)//2
        next_flow = tmp_user2flow[next_user]
        next_grd = '[none]'
        if flow_idx < len(examples[next_flow]):
            if len(examples[next_flow][flow_idx]) != 0:
                next_grd = ' </s> '.join(examples[next_flow][flow_idx]) 
        
        if cur_grd == '[none]':
            cur_grd = examples['agent_personas'] if cur_flow == 'agent_flow' else examples['user_personas']

        pre, tmp = source.split(' [mask] [grounding] ',)
        source = f'{pre} {predictions[idx]}'
        source += f' {cur_user} [mask] [grounding] {cur_grd} [/grounding] [/mask]'
        source += f' {next_user} [grounding] {next_grd} [/grounding]'

        # batch_encoding = tokenizer.prepare_seq2seq_batch(
        #     source,
        #     source,
        #     max_length=data_args.max_source_length,
        #     max_target_length=data_args.val_max_target_length,
        #     padding="longest",
        #     return_tensors="pt"
        # )
        return_dict = tokenizer(source, max_length=data_args.max_source_length, padding=padding, truncation=True)
        return_dict['source'] = source
        return return_dict
    return preprocess_function
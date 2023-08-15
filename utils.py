

def get_preprocess_function(tokenizer, data_args, max_target_length, padding):
    def preprocess_function(examples):
        inputs = examples["source"]
        targets = examples["target"]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return preprocess_function

# get the input data for the first turn
def get_preprocess_function_inp_first_turn(tokenizer, data_args, max_target_length, padding):
    def preprocess_function(examples):
        inputs = examples["source"]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        return model_inputs
    return preprocess_function

# get the input data for the following turns
def get_preprocess_function_inp_each_turn(tokenizer, data_args, turn, predictions, padding):
    def preprocess_function(examples, idx):
        source = examples['source']
        chosen_topic = examples['flow'][0].split(' | ')[0]

        max_following_turns = 3
        grds = []
        for i in range(turn, turn+max_following_turns):
            grd = None
            if i%2 == 0:
                grd = f'{chosen_topic} | [none] | [none]'
            else:
                t = i//2
                if t < len(examples['flow']):
                    grd = examples['flow'][t]
                else:
                    grd = f'{chosen_topic} | [none] | [none]'
            grds.append(grd)

        pre, tmp = source.split(' [mask] [grounding] ')
        source = f'{pre} {predictions[idx]}'
        for i in range(turn, turn+max_following_turns):
            user_token = '[user-1]' if i%2 == 0 else '[user-2]'
            if i == turn:
                source = f'{source} {user_token} [mask] [grounding] {grds[i-turn]} [/grounding] [/mask]'
            else:
                source = f'{source} {user_token} [grounding] {grds[i-turn]} [/grounding]'

        return_dict = tokenizer(source, max_length=data_args.max_source_length, padding=padding, truncation=True)
        return_dict['source'] = source
        return return_dict
    return preprocess_function
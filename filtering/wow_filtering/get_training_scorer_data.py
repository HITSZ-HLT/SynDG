
import json
from copy import deepcopy
import numpy as np
import os

def get_statistics(len_list):
    len_list = np.array(len_list)
    print(f'mean: {np.mean(len_list)}')
    print(f'median: {np.median(len_list)}')
    print(f'max: {np.max(len_list)}')
    print(f'min: {np.min(len_list)}')
    print(f'25%: {np.percentile(len_list, 25)}')
    print(f'50%: {np.percentile(len_list, 50)}')
    print(f'75%: {np.percentile(len_list, 75)}')
    print(f'90%: {np.percentile(len_list, 90)}')
    print(f'95%: {np.percentile(len_list, 95)}')
    print(f'99%: {np.percentile(len_list, 99)}')

def get_dial_data_wow(data_num=None, is_train=True):
    if is_train:
        data_path = '../../wizard_of_wikipedia/train_shuffled.json'
    else:
        data_path = '../../wizard_of_wikipedia/valid_topic_split.json'
    with open(data_path, 'r') as f:
        data = json.load(f)
    if data_num:
        data = data[:data_num]
    
    sources = []
    targets = []
    for d in data:
        utts = []
        for turn in d['dialog']:
            utt = turn['text']
            if 'Apprentice' in turn['speaker']:
                utt = f'[user-1] {utt}'
            else:
                utt = f'[user-2] {utt}'
            utts.append(utt)
        for idx in range(len(utts)):
            masked_utts = deepcopy(utts)
            target = ' '.join(masked_utts[idx].split(' ')[1:])
            user_token = masked_utts[idx].split(' ')[0]
            masked_utts[idx] = f'{user_token} [mask]'
            source = ' '.join(masked_utts)
            sources.append(source)
            targets.append(target)

    source_len_list = list(map(lambda x: len(x.split(' ')), sources))
    get_statistics(source_len_list)
    target_len_list = list(map(lambda x: len(x.split(' ')), targets))
    get_statistics(target_len_list)

    sample_list = []
    for pair in zip(sources, targets):
        if is_train:
            if len(pair[0].split(' ')) > 300:
                continue
            if len(pair[1].split(' ')) > 100:
                continue
        sample_list.append({
            'source': pair[0],
            'target': pair[1]
        })
    save_path = 'training_dial_data'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if is_train:
        with open(os.path.join(save_path, 'train.json'), 'w') as f:
            json.dump(sample_list, f, indent=2)
    else:
        with open(os.path.join(save_path, 'valid.json'), 'w') as f:
            json.dump(sample_list, f, indent=2)

def get_flow_data_wow(data_num=None, is_train=True):
    if is_train:
        data_path = '../../wizard_of_wikipedia/train_shuffled.json'
    else:
        data_path = '../../wizard_of_wikipedia/valid_topic_split.json'
    with open(data_path, 'r') as f:
        data = json.load(f)
    if data_num:
        data = data[:data_num]
    
    sources = []
    targets = []
    for d in data:
        grounding_list = []
        chosen_topic = d['chosen_topic']
        for turn in d['dialog']:
            if 'Apprentice' in turn['speaker']:
                grounding = None
            else:
                grounding_dict = turn['checked_sentence']
                if len(turn['checked_passage']) != 0:
                    pass_topic = list(turn['checked_passage'].values())[0]
                else:
                    pass_topic = ' '.join(list(turn['checked_sentence'].keys())[0].split('_')[1:-1])
                if len(turn['checked_sentence']) == 0:
                    grounding = pass_topic
                else:
                    grounding = list(turn['checked_sentence'].values())[0]
                if grounding == 'no_passages_used':
                    grounding = '[none]'
                if pass_topic == 'no_passages_used':
                    pass_topic = '[none]'
                grounding = '[session] ' + chosen_topic + ' | ' + pass_topic + ' | ' + grounding
                grounding_list.append(grounding)

        for idx in range(len(grounding_list)):
            masked_utts = deepcopy(grounding_list)
            target = ' '.join(masked_utts[idx].split(' ')[1:])
            user_token = masked_utts[idx].split(' ')[0]
            masked_utts[idx] = f'{user_token} [mask]'
            source = ' '.join(masked_utts)
            sources.append(source)
            targets.append(target)
    
    source_len_list = list(map(lambda x: len(x.split(' ')), sources))
    get_statistics(source_len_list)
    target_len_list = list(map(lambda x: len(x.split(' ')), targets))
    get_statistics(target_len_list)

    sample_list = []
    for pair in zip(sources, targets):
        if is_train:
            if len(pair[0].split(' ')) > 300:
                continue
            if len(pair[1].split(' ')) > 100:
                continue
        sample_list.append({
            'source': pair[0],
            'target': pair[1]
        })
    save_path = 'training_flow_data'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if is_train:
        with open(os.path.join(save_path, 'train.json'), 'w') as f:
            json.dump(sample_list, f, indent=2)
    else:
        with open(os.path.join(save_path, 'valid.json'), 'w') as f:
            json.dump(sample_list, f, indent=2)
    

if __name__ == "__main__":

    get_flow_data_wow(None, is_train=True)
    get_dial_data_wow(None, is_train=True)
    get_flow_data_wow(None, is_train=False)
    get_dial_data_wow(None, is_train=False)
    # get_pred_flow_data()

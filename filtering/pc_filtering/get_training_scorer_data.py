
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

def get_flow_data(data_num=None, is_train=True):
    if is_train:
        with open('../../persona_chat/train_both_original_grounded.json', 'r') as f:
            train_data = json.load(f)
    else:
        with open('../../persona_chat/valid_both_original_grounded.json', 'r') as f:
            train_data = json.load(f)
    if data_num:
        train_data = train_data[:data_num]
    sources = []
    targets = []
    for d in train_data:
        f = lambda x: ' | '.join(x) if x else '[none]'
        u1_flows = [f(_) for _ in d['user_grounded_personas']]
        u1_flows = map(lambda x: '[user-1] ' + x, u1_flows)
        u2_flows = [f(_) for _ in d['agent_grounded_personas']]
        u2_flows = map(lambda x: '[user-2] ' + x, u2_flows)
        flows = [item for tup in zip(u1_flows, u2_flows) for item in tup]
        for idx in range(len(flows)):
            masked_flows = deepcopy(flows)
            target = ' '.join(masked_flows[idx].split(' ')[1:])
            user_token = masked_flows[idx].split(' ')[0]
            masked_flows[idx] = f'{user_token} [mask]'
            source = ' '.join(masked_flows)
            sources.append(source)
            targets.append(target)
    sample_list = []
    source_lens = []
    target_lens = []
    for pair in zip(sources, targets):
        if len(pair[0].split(' ')) > 120 or len(pair[1].split(' ')) > 20:
            continue
        source_lens.append(len(pair[0].split(' ')))
        target_lens.append(len(pair[1].split(' ')))
        sample_list.append({
            'source': pair[0],
            'target': pair[1]
        })
    print('**********flow data**********')
    print('source:')
    get_statistics(source_lens)
    print('target:')
    get_statistics(target_lens)

    save_path = 'training_flow_data'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if is_train:
        with open(os.path.join(save_path, 'train.json'), 'w') as f:
            json.dump(sample_list, f, indent=4)
    else:
        with open(os.path.join(save_path, 'valid.json'), 'w') as f:
            json.dump(sample_list, f, indent=4)


def get_dial_data(data_num=None, is_train=True):
    if is_train:
        with open('../../persona_chat/train_both_original_grounded.json', 'r') as f:
            train_data = json.load(f)
    else:
        with open('../../persona_chat/valid_both_original_grounded.json', 'r') as f:
            train_data = json.load(f)
    if data_num:
        train_data = train_data[:data_num]
    sources = []
    targets = []
    for d in train_data:
        u1_flows = map(lambda x: '[user-1] ' + x, d['user_utterances'])
        u2_flows = map(lambda x: '[user-2] ' + x, d['agent_utterances'])
        flows = [item for tup in zip(u1_flows, u2_flows) for item in tup]
        for idx in range(len(flows)):
            masked_flows = deepcopy(flows)
            target = ' '.join(masked_flows[idx].split(' ')[1:])
            user_token = masked_flows[idx].split(' ')[0]
            masked_flows[idx] = f'{user_token} [mask]'
            source = ' '.join(masked_flows)
            sources.append(source)
            targets.append(target)
    sample_list = []
    source_lens = []
    target_lens = []
    for pair in zip(sources, targets):
        if len(pair[0].split(' ')) > 260 or len(pair[1].split(' ')) > 20:
            continue
        source_lens.append(len(pair[0].split(' ')))
        target_lens.append(len(pair[1].split(' ')))
        sample_list.append({
            'source': pair[0],
            'target': pair[1]
        })
    print('**********dial data**********')
    print('source:')
    get_statistics(source_lens)
    print('target:')
    get_statistics(target_lens)
    save_path = 'training_dial_data'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if is_train:
        with open(os.path.join(save_path, 'train.json'), 'w') as f:
            json.dump(sample_list, f, indent=4)
    else:
        with open(os.path.join(save_path, 'valid.json'), 'w') as f:
            json.dump(sample_list, f, indent=4)

def get_pred_flow_data(data_num=None):
    with open('../ddg-persona-3.0/persona_chat/sampled_dialogues_2/generated_dials.json', 'r') as f:
        train_data = json.load(f)
    if data_num:
        train_data = train_data[:data_num]
    sources = []
    targets = []
    for d in train_data:
        f = lambda x: ' | '.join(x) if x else '[none]'
        u1_flows = [f(_) for _ in d['u1_flows']]
        u1_flows = map(lambda x: '[user-1] ' + x, u1_flows)
        u2_flows = [f(_) for _ in d['u2_flows']]
        u2_flows = map(lambda x: '[user-2] ' + x, u2_flows)
        flows = [item for tup in zip(u1_flows, u2_flows) for item in tup]
        for idx in range(len(flows)):
            masked_flows = deepcopy(flows)
            target = ' '.join(masked_flows[idx].split(' ')[1:])
            user_token = masked_flows[idx].split(' ')[0]
            masked_flows[idx] = f'{user_token} [mask]'
            source = ' '.join(masked_flows)
            sources.append(source)
            targets.append(target)
    sample_list = []
    for pair in zip(sources, targets):
        sample_list.append({
            'source': pair[0],
            'target': pair[1]
        })
    with open('../ddg-persona-3.0/persona_chat/sampled_dialogues_2/for_filter_input.json', 'w') as f:
        json.dump(sample_list, f, indent=2)

if __name__ == "__main__":

    # full_data: 8939
    # 1_4: 2234
    # 1_8: 1117
    # 1_16: 558
    # 1_32: 279
    
    get_flow_data(None, is_train=True)
    get_flow_data(None, is_train=False)
    get_dial_data(None, is_train=True)
    get_dial_data(None, is_train=False)
    # get_pred_flow_data()

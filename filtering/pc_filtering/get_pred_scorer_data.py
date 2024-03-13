
import json
from copy import deepcopy
import sys



def get_pred_flow_data(input_file, output_file, data_num=None):
    with open(input_file, 'r') as f:
        train_data = json.load(f)
    if data_num:
        train_data = train_data[:data_num]
    sources = []
    targets = []
    for d in train_data:
        f = lambda x: ' </s> '.join(x) if x else '[none]'
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
    with open(output_file, 'w') as f:
        json.dump(sample_list, f, indent=2)

def get_pred_dial_data(input_file, output_file, data_num=None):
    with open(input_file, 'r') as f:
        train_data = json.load(f)
    if data_num:
        train_data = train_data[:data_num]
    sources = []
    targets = []
    for d in train_data:
        u1_flows = map(lambda x: '[user-1] ' + x, d['u1_utts'])
        u2_flows = map(lambda x: '[user-2] ' + x, d['u2_utts'])
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
    with open(output_file, 'w') as f:
        json.dump(sample_list, f, indent=2)

if __name__ == "__main__":
    # ../ddg-persona-3.0/persona_chat/sampled_dialogues_2/generated_dials.json
    input_file = sys.argv[1]
    # ../ddg-persona-3.0/persona_chat/sampled_dialogues_2/for_filter_input.json
    output_file = sys.argv[2]
    # flow or dial
    mode = sys.argv[3]
    if mode == 'flow':
        get_pred_flow_data(input_file, output_file)
    else:
        get_pred_dial_data(input_file, output_file)
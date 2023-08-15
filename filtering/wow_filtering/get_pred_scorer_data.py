
import json
from copy import deepcopy
import sys



def get_pred_flow_data_wow(input_file, output_file, data_num=None):
    with open(input_file, 'r') as f:
        train_data = json.load(f)
    if data_num:
        train_data = train_data[:data_num]
    sources = []
    targets = []
    for d in train_data:
        flows = d['flow']
        for idx in range(len(flows)):
            masked_flows = deepcopy(flows)
            target = masked_flows[idx]
            masked_flows[idx] = '[mask]'
            source = '[session] ' + ' [session] '.join(masked_flows)
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

def get_pred_dial_data_wow(input_file, output_file, data_num=None):
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
        if flows[0] == '[user-1] [no_query]':
            flows = flows[1:]
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
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    mode = sys.argv[3]
    if mode == 'flow':
        get_pred_flow_data_wow(input_file, output_file)
    else:
        get_pred_dial_data_wow(input_file, output_file)
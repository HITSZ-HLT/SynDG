import json
from copy import deepcopy
import sys
import numpy as np
import os

def merge_ppl_score(dial_file, score_file, output_file):
    with open(dial_file, 'r') as f:
        dial_list = json.load(f)
    with open(score_file, 'r') as f:
        score_list = json.load(f)
    
    turn_per_dial = 8
    step = turn_per_dial * 2
    dial_scores = []
    for i in range(0, len(score_list), step):
        single_turn_scores = score_list[i:i+step]
        dial_score = float(np.array([d['ppl'] for d in single_turn_scores]).mean())
        dial_scores.append(dial_score)

    assert len(dial_list) == len(dial_scores)

    if 'dial_scores' in score_file:
        score_type = 'dial_score'
    else:
        score_type = 'flow_score'
    
    for d, s in zip(dial_list, dial_scores):
        d[score_type] = s

    with open(output_file, 'w') as f:
        json.dump(dial_list, f, indent=2)

def merge_total_score(dial_score_fp, flow_score_fp, output_fp):
    with open(dial_score_fp, 'r') as f:
        dial_score_list = json.load(f)
    with open(flow_score_fp, 'r') as f:
        flow_score_list = json.load(f)
    
    assert len(dial_score_list) == len(flow_score_list)
    
    for d, s in zip(dial_score_list, flow_score_list):
        # d['dial_score'] = d['flow_score']
        d['flow_score'] = s['flow_score']
        d['total_score'] = d['dial_score'] + d['flow_score']
    
    with open(output_fp, 'w') as f:
        json.dump(dial_score_list, f, indent=2)
    
    sorted_dial_list = sorted(dial_score_list, key=lambda _: _['total_score'])
    with open(os.path.join(os.path.dirname(output_fp), 'generated_dials-total_sorted.json'), 'w') as f:
        json.dump(sorted_dial_list, f, indent=2)
    sorted_dial_list = sorted(dial_score_list, key=lambda _: _['dial_score'])
    with open(os.path.join(os.path.dirname(output_fp), 'generated_dials-dial_sorted.json'), 'w') as f:
        json.dump(sorted_dial_list, f, indent=2)
    sorted_dial_list = sorted(dial_score_list, key=lambda _: _['flow_score'])
    with open(os.path.join(os.path.dirname(output_fp), 'generated_dials-flow_sorted.json'), 'w') as f:
        json.dump(sorted_dial_list, f, indent=2)

if __name__ == "__main__":
    # '../ddg-persona-3.0/persona_chat/sampled_dialogues_2/generated_dials.json'
    dial_file = sys.argv[1]
    # '../ddg-persona-3.0/persona_chat/sampled_dialogues_2/flow_score.json'
    score_file = sys.argv[2]
    # '../ddg-persona-3.0/persona_chat/sampled_dialogues_2/generated_dials_w_score.json'
    output_file = sys.argv[3]
    # dial_file = '../ddg-persona-3.0/persona_chat/sampled_dialogues_2/generated_dials.json'
    # score_file = '../ddg-persona-3.0/persona_chat/sampled_dialogues_2/flow_scores.json'
    # output_file = '../ddg-persona-3.0/persona_chat/sampled_dialogues_2/generated_dials_w_flow_score.json'
    merge_ppl_score(dial_file, score_file, output_file)


    # get total score
    # dial_score_fp = '../ddg-persona-3.0/persona_chat/sampled_dialogues_4/generated_dials_w_dial_score.json'
    # flow_score_fp = '../ddg-persona-3.0/persona_chat/sampled_dialogues_4/generated_dials_w_flow_score.json'
    # output_fp = '../ddg-persona-3.0/persona_chat/sampled_dialogues_4/generated_dials_w_total_score.json'
    # merge_total_score(dial_score_fp, flow_score_fp, output_fp)
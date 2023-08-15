import json
from copy import deepcopy
import sys
import numpy as np

def merge_ppl_score(dial_file, score_file, output_file):
    with open(dial_file, 'r') as f:
        dial_list = json.load(f)
    with open(score_file, 'r') as f:
        score_list = json.load(f)
    
    full_dial_scores = []
    if 'flow' in score_file:
        turn_per_dial = 5
        step = turn_per_dial
        for i in range(0, len(score_list), step):
            single_turn_scores = score_list[i:i+step]
            dial_score = float(np.array([d['ppl'] for d in single_turn_scores]).mean())
            full_dial_scores.append(dial_score)
    else:
        score_idx = 0
        for d in dial_list:
            u1_utts = d['u1_utts'] if d['u1_utts'][0] != '[no_query]' else d['u1_utts'][1:]
            u2_utts = d['u2_utts'] if d['u2_utts'][0] != '[no_query]' else d['u2_utts'][1:]
            utt_num = len(u1_utts) + len(u2_utts)
            scores = score_list[score_idx:score_idx+utt_num]
            dial_score = float(np.array([d['ppl'] for d in scores]).mean())
            full_dial_scores.append(dial_score)
            score_idx += utt_num


    assert len(dial_list) == len(full_dial_scores)

    
    for d, s in zip(dial_list, full_dial_scores):
        d['score'] = s
    
    # sorted_dial_list = sorted(dial_list, key=lambda _: _['score'])

    with open(output_file, 'w') as f:
        json.dump(dial_list, f, indent=2)


if __name__ == "__main__":
    dial_file = sys.argv[1]
    score_file = sys.argv[2]
    output_file = sys.argv[3]
    merge_ppl_score(dial_file, score_file, output_file)
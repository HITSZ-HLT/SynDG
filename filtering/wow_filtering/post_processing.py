import json
from copy import deepcopy

def get_total_score():
    with open(f'../../wizard_of_wikipedia/synthetic_data/generated_dials_w_dial_score.json', 'r') as f:
        dial_score_list = json.load(f)

    with open(f'../../wizard_of_wikipedia/synthetic_data/generated_dials_w_flow_score.json', 'r') as f:
        flow_score_list = json.load(f)
    
    total_score_list = deepcopy(dial_score_list)
    for total_d, dial_d, flow_d in zip(total_score_list, dial_score_list, flow_score_list):
        total_d['total_score'] = dial_d['score'] + flow_d['score']
        total_d['dial_score'] = dial_d['score']
        total_d['flow_score'] = flow_d['score']
        del total_d['score']
    
    # sorted_score_list = sorted(total_score_list, key=lambda _: _['total_score'], reverse=True)
    with open(f'../../wizard_of_wikipedia/synthetic_data/generated_dials_w_total_score.json', 'w') as f:
        json.dump(total_score_list, f, indent=4)


get_total_score()
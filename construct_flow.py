
import json
from pathlib import Path
from copy import deepcopy
import random
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import os

prompt = '[prompt] The following is a knowledge-grounded dialogue. Two users chat according to the given knowledge.'


def sample_flow(orig_data_path, num_sampled_flows=None):

    with open(orig_data_path, 'r') as fp:
        wow_data = json.load(fp)
    
    if num_sampled_flows is not None:
        wow_data = wow_data[-num_sampled_flows:]
    
    sampled_flows = []
    for dial_dict in tqdm(wow_data):

        chosen_topic = dial_dict['chosen_topic']
        chosen_doc = dial_dict['chosen_topic_passage']

        doc_pool_dict = {}
        # only the first turn doc
        first_turn_topic_list = []
        for turn_i, turn_dict in enumerate(dial_dict['dialog']):
            for dict_item in turn_dict['retrieved_passages']:
                k = list(dict_item.keys())[0]
                v = list(dict_item.values())[0]
                k = k.replace('amp;', '')
                k = k.replace('&quot;', '"')
                if k == 'No Passages Retrieved':
                    continue
                doc_pool_dict[k] = v
                if turn_i<2:
                    first_turn_topic_list.append(k)
                if k == chosen_topic:
                    for sent in v:
                        if sent not in chosen_doc:
                            chosen_doc.append(sent)

        
        if chosen_topic in first_turn_topic_list:
            first_turn_topic_list.remove(chosen_topic)
        chosen_doc += ['[none]']
        topic_pool = list(doc_pool_dict.keys())

        MAX_TURN_NUM = 5
        FIRST_SENT_PROB = 0.9
        OTHER_TOPIC_PROB = 0.1
        chosen_doc_probs = [FIRST_SENT_PROB] + [(1-FIRST_SENT_PROB)/(len(chosen_doc)-1)] * (len(chosen_doc)-1)
        # chosen_doc_w_prob = list(zip(chosen_doc, chosen_doc_probs))
        def normalize(probs):
            prob_factor = 1 / sum(probs)
            return [prob_factor * p for p in probs]
        sampled_grounding_list = []
        for turn_id in range(MAX_TURN_NUM):
            is_other_topic = True if random.random() < OTHER_TOPIC_PROB else False
            if turn_id == 0:
                is_other_topic = False

            grounding = None
            if (not is_other_topic) and len(chosen_doc) != 0:
                topic = chosen_topic
                chosen_doc_probs = normalize(chosen_doc_probs)
                grounding = random.choices(
                                population=chosen_doc,
                                weights=chosen_doc_probs, 
                                k=1
                            )[0]
                grounding_idx = chosen_doc.index(grounding)
                chosen_doc.pop(grounding_idx)
                chosen_doc_probs.pop(grounding_idx)
            else:
                grounding = '[none]'
                topic = '[none]'
                if len(first_turn_topic_list) != 0:
                    topic_probs = list(reversed(normalize(range(1, 1+len(first_turn_topic_list)))))
                    # topic_list_w_prob = list(zip(first_turn_topic_list, topic_probs))
                    topic = random.choices(
                                    population=first_turn_topic_list,
                                    weights=topic_probs, 
                                    k=1
                                )[0]
                    topic_idx = first_turn_topic_list.index(topic)
                    first_turn_topic_list.pop(topic_idx)
                    topic_probs.pop(topic_idx)
                    grounding = doc_pool_dict[topic][0] # only the first of other topic
            
            grounding =  f'{chosen_topic} | {topic} | {grounding}'

            sampled_grounding_list.append(grounding)
        sampled_flows.append(sampled_grounding_list)

    return sampled_flows

def format_sampled_flow_for_inp(sampled_flows_file, save_inp_source_file, m=2):
    with open(sampled_flows_file, 'r') as fp:
        flows = json.load(fp)
    
    inp_flows = []
    for flow in flows:
        chosen_topic = flow[0].split(' | ')[0]
        source_list = []
        first_turn = f'[user-1] [mask] [grounding] {chosen_topic} | [none] | [none] [/grounding] [/mask]'
        source_list.append(first_turn)
        second_turn = f'[user-2] [grounding] {flow[0]} [/grounding]'
        source_list.append(second_turn)

        # source = f'{prompt} [user-1] [mask] [grounding] {chosen_topic} | [none] | [none] [/grounding] [/mask] [user-2] [grounding] {flow[0]} [/grounding]'
        for grounding in flow[1:m//2+1]:
            tmp_turn_1 = f'[user-1] [grounding] {chosen_topic} | [none] | [none] [/grounding]'
            tmp_turn_2 = f'[user-2] [grounding] {grounding} [/grounding]'
            source_list.append(tmp_turn_1)
            source_list.append(tmp_turn_2)
        
        source = f'{prompt} ' + ' '.join(source_list[:m+1])

        inp_flow = {
            'source': source,
            'flow': flow
        }
        inp_flows.append(inp_flow)

    with open(save_inp_source_file, 'w') as f:
        json.dump(inp_flows, f, indent=4)


def format_gen_dial_for_new_training_data(num_sampled_flows=None):
    with open('wizard_of_wikipedia/train_shuffled.json', 'r') as f:
        orig_train_data = json.load(f)
    if num_sampled_flows is not None:
        orig_train_data = orig_train_data[-num_sampled_flows:]
    
    orig_train_data = orig_train_data + orig_train_data

    with open('wizard_of_wikipedia/sampled_1.3/generated_predictions.json', 'r') as f:
        gen_dials = json.load(f)

    assert len(orig_train_data) == len(gen_dials)

    formatted_gen_dials = []
    for orig_dial, dial in tqdm(zip(orig_train_data, gen_dials), total=len(orig_train_data)):
        orig_first_turn = orig_dial['dialog'][0]
        new_dial_dict = deepcopy(orig_dial)
        new_dialog = []
        for kw, u1_utt, u2_utt in zip(dial['flow'], dial['u1_utts'], dial['u2_utts']):
            u1_d_dic = {
                'speaker': 'Apprentice',
                'text': u1_utt,
                'retrieved_passages': orig_first_turn['retrieved_passages'],
                'retrieved_topics': orig_first_turn['retrieved_topics']
            }

            u2_d_dic = {
                'speaker': 'Wizard',
                'text': u2_utt,
                'retrieved_passages': orig_first_turn['retrieved_passages'],
                'retrieved_topics': orig_first_turn['retrieved_topics']
            }
            topic = kw.split(' | ')[1]
            grounding = kw.split(' | ')[2]
            assert len(kw.split(' | ')) == 3
            if topic == '[none]':
                u2_d_dic['checked_sentence'] = {"no_passages_used": "no_passages_used"}
                u2_d_dic['checked_passage'] = {"no_passages_used": "no_passages_used"}
            else:
                u2_d_dic['checked_sentence'] = {'_'.join(topic.split(' ')): grounding}
                u2_d_dic['checked_passage'] = {'_'.join(topic.split(' ')): topic}
            
            new_dialog.append(u1_d_dic)
            new_dialog.append(u2_d_dic)
        
        if new_dialog[0]['text'] == '[no_query]':
            new_dialog = new_dialog[1:]
        for i, d in enumerate(new_dialog):
            turn_id = i % 2
            speaker = d['speaker']
            d['speaker'] = f'{turn_id}_{speaker}'
        new_dial_dict['dialog'] = new_dialog

        formatted_gen_dials.append(new_dial_dict)

    with open('wizard_of_wikipedia/sampled_1.3/formatted_gen_train.json', 'w') as f:
        json.dump(formatted_gen_dials, f, indent=4)


def format_gen_dial_for_new_training_data_with_score(num_sampled_flows=None):
    m = "6"
    with open('wizard_of_wikipedia/train_shuffled.json', 'r') as f:
        orig_train_data = json.load(f)
    
    train_data = deepcopy(orig_train_data)
    train_data += orig_train_data
    
    if num_sampled_flows is not None:
        train_data = train_data[-num_sampled_flows:]
    
    with open(f'wizard_of_wikipedia/sampled_1.3_m_{m}/generated_dials_w_total_score.json', 'r') as f:
        gen_dials_w_score = json.load(f)
    
    assert len(train_data) == len(gen_dials_w_score)

    formatted_gen_dials = []
    for orig_dial, dial in tqdm(zip(train_data, gen_dials_w_score), total=len(train_data)):
        orig_first_turn = orig_dial['dialog'][0]
        new_dial_dict = deepcopy(orig_dial)
        new_dialog = []
        for kw, u1_utt, u2_utt in zip(dial['flow'], dial['u1_utts'], dial['u2_utts']):
            u1_d_dic = {
                'speaker': 'Apprentice',
                'text': u1_utt,
                'retrieved_passages': orig_first_turn['retrieved_passages'],
                'retrieved_topics': orig_first_turn['retrieved_topics']
            }

            u2_d_dic = {
                'speaker': 'Wizard',
                'text': u2_utt,
                'retrieved_passages': orig_first_turn['retrieved_passages'],
                'retrieved_topics': orig_first_turn['retrieved_topics']
            }
            topic = kw.split(' | ')[1]
            grounding = kw.split(' | ')[2]
            assert len(kw.split(' | ')) == 3
            if topic == '[none]':
                u2_d_dic['checked_sentence'] = {"no_passages_used": "no_passages_used"}
                u2_d_dic['checked_passage'] = {"no_passages_used": "no_passages_used"}
            else:
                u2_d_dic['checked_sentence'] = {'_'.join(topic.split(' ')): grounding}
                u2_d_dic['checked_passage'] = {'_'.join(topic.split(' ')): topic}
            
            new_dialog.append(u1_d_dic)
            new_dialog.append(u2_d_dic)
        
        if new_dialog[0]['text'] == '[no_query]':
            new_dialog = new_dialog[1:]
        for i, d in enumerate(new_dialog):
            turn_id = i % 2
            speaker = d['speaker']
            d['speaker'] = f'{turn_id}_{speaker}'
        new_dial_dict['dialog'] = new_dialog
        new_dial_dict['total_score'] = dial['total_score']
        new_dial_dict['dial_score'] = dial['dial_score']
        new_dial_dict['flow_score'] = dial['flow_score']

        formatted_gen_dials.append(new_dial_dict)

    with open(f'wizard_of_wikipedia/sampled_1.3_m_{m}/formatted_gen_train_w_score.json', 'w') as f:
        json.dump(formatted_gen_dials, f, indent=4)
    
    for score_type in ['total_score', 'flow_score', 'dial_score']:
        formatted_gen_dials = sorted(formatted_gen_dials, key=lambda x: x[score_type], reverse=True)

        output_file = os.path.join(f'wizard_of_wikipedia/sampled_1.3_m_{m}/formatted_gen_train-{score_type}_sorted.json')
        with open(output_file, 'w') as f:
            json.dump(formatted_gen_dials, f, indent=4)

if __name__ == '__main__':

    # preprocess for inpainting
    # NUM_SAMPLED_FLOWS = 2000
    NUM_SAMPLED_FLOWS = None
    orig_data_path = f'wizard_of_wikipedia/train_shuffled.json'
    save_path = f'wizard_of_wikipedia/synthetic_data/sampled_flows.json'
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    sampled_flows_1 = sample_flow(orig_data_path, NUM_SAMPLED_FLOWS)
    sampled_flows_2 = sample_flow(orig_data_path, NUM_SAMPLED_FLOWS)
    sampled_flows = sampled_flows_1 + sampled_flows_2
    print(len(sampled_flows))
    with open(save_path, 'w') as fp:
        json.dump(sampled_flows, fp, indent=4)
    
    sampled_flows_file = save_path
    save_inp_source_file = f'wizard_of_wikipedia/synthetic_data/inpainting_source.json'
    format_sampled_flow_for_inp(sampled_flows_file, save_inp_source_file)

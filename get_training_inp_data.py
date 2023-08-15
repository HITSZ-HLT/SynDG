
import json
from copy import deepcopy
from tqdm import tqdm
import os
import random
import argparse


def get_inp_data_for_training(input_data_dir, output_data_dir, mode='train', num_samples=None, following_kw_num=1):

    with open(input_data_dir, 'r') as fp:
        wow_data = json.load(fp)

    wow_data = wow_data[:num_samples]

    prompt = '[prompt] The following is a knowledge-grounded dialogue. Two users chat according to the given knowledge.'

    masked_inputs = []
    targets = []

    too_long_num = 0

    for dial_dict in tqdm(wow_data):
        utterance_list = []
        if 'Apprentice' not in dial_dict['dialog'][0]['speaker']:
            user_token = '[user-1]'
            utt = '[no_query]'
            utterance_list.append(f'{user_token} {utt}')
        for turn_dict in dial_dict['dialog']:
            user_token = '[user-1]' if 'Apprentice' in turn_dict['speaker'] else '[user-2]'
            utt = turn_dict['text']
            utterance_list.append(f'{user_token} {utt}')

        chosen_topic = dial_dict['chosen_topic']

        grounding_list = []
        for turn_dict in dial_dict['dialog']:
            if 'Apprentice' in turn_dict['speaker']:
                grounding = None
            else:
                grounding_dict = turn_dict['checked_sentence']
                if len(turn_dict['checked_passage']) != 0:
                    pass_topic = list(turn_dict['checked_passage'].values())[0]
                else:
                    pass_topic = ' '.join(list(turn_dict['checked_sentence'].keys())[0].split('_')[1:-1])
                if len(turn_dict['checked_sentence']) == 0:
                    grounding = pass_topic
                else:
                    grounding = list(turn_dict['checked_sentence'].values())[0]
                if grounding == 'no_passages_used':
                    grounding = '[none]'
                if pass_topic == 'no_passages_used':
                    pass_topic = '[none]'
                grounding = chosen_topic + ' | ' + pass_topic + ' | ' + grounding
                grounding_list.append(chosen_topic + ' | ' + '[none] | [none]') # for [user-1]
                grounding_list.append(grounding)

        if len(grounding_list) != len(utterance_list):
            grounding_list.append(chosen_topic + ' | ' + '[none] | [none]') # the last [user-1]

        assert len(grounding_list) == len(utterance_list)

        for mask_id in range(len(utterance_list)):
            masked_utt_list = deepcopy(utterance_list)
            target = ' '.join(masked_utt_list[mask_id].split(' ')[1:])
        
            user_token = masked_utt_list[mask_id].split(' ')[0]

            grounded_kw = grounding_list[mask_id]
            masked_utt_list[mask_id] = f'{user_token} [mask] [grounding] {grounded_kw} [/grounding] [/mask]'
            
            # replace all the following utterances with kw
            for j in range(mask_id+1, len(utterance_list)):
                user_token = masked_utt_list[j].split(' ')[0]
                grounded_kw = grounding_list[j]
                masked_utt_list[j] = f'{user_token} [grounding] {grounded_kw} [/grounding]'
            
            # delete the following utterance
            masked_utt_list = masked_utt_list[:mask_id+following_kw_num+1]
            
            masked_utterance_str = prompt + ' ' + ' '.join(masked_utt_list)
            if len(masked_utterance_str.split(' ')) > 300:
                too_long_num += 1
                continue
            masked_inputs.append(masked_utterance_str)
            targets.append(target)  

    output_list = [{'source': source, 'target': target} for source, target in zip(masked_inputs, targets)]
    with open(output_data_dir, 'w') as f:
        json.dump(output_list, f, indent=4)
    
    print(f'Num of {mode} samples: {len(output_list)}')
    print(f'Num of too long samples: {too_long_num}')



def get_inp_data_for_training_pc(persona_data_dir, output_data_dir, mode='train', num_samples=None, m=1):
    with open(f'{persona_data_dir}/{mode}_both_original_grounded.json', 'r') as fp:
        persona_chat_grounded_data = json.load(fp)
    if num_samples:
        persona_chat_grounded_data = persona_chat_grounded_data[:num_samples]
    
    prompt = '[prompt] The following is a persona-based dialogue. Two users chat according to their personas or profile information.'

    masked_inputs = []
    targets = []

    for dialogue in persona_chat_grounded_data:
        # get dialogue
        utterance_list = []
        for u1_utterance, u2_utterance in zip(dialogue['user_utterances'], \
                                                dialogue['agent_utterances']):
            utterance_list.append(f'[user-1] {u1_utterance}')
            utterance_list.append(f'[user-2] {u2_utterance}')
        
        # get grounded personas
        grounded_personas_list = []
        for u1_grounded_persona, u2_grounded_persona in zip(dialogue['user_grounded_personas'], \
                                                dialogue['agent_grounded_personas']):
            grounded_personas_list.append(u1_grounded_persona)
            grounded_personas_list.append(u2_grounded_persona)
        
        # mask each utterance
        for mask_id in range(len(utterance_list)):
            masked_utterance_list = deepcopy(utterance_list)
            target = ' '.join(masked_utterance_list[mask_id].split(' ')[1:])

            # mask with [none] or persona
            user_token = masked_utterance_list[mask_id].split(' ')[0]

            if user_token == '[user-1]':
                all_personas = ' | '.join(dialogue['user_personas'])
            else:
                all_personas = ' | '.join(dialogue['agent_personas'])

            if grounded_personas_list[mask_id] == []:
                masked_utterance_list[mask_id] = f'{user_token} [mask] [grounding] {all_personas} [/grounding] [/mask]'
            else:
                grounded_personas_str = ' | '.join(grounded_personas_list[mask_id])
                masked_utterance_list[mask_id] = f'{user_token} [mask] [grounding] {grounded_personas_str} [/grounding] [/mask]'
            
            # replace the following utterance with persona
            for i in range(mask_id+1, len(masked_utterance_list)):
                user_token = masked_utterance_list[i].split(' ')[0]
                if grounded_personas_list[i] == []:
                    # masked_utterance_list[i] = f'{user_token} {str_for_none_persona}'
                    masked_utterance_list[i] = f'{user_token} [none]'
                else:
                    grounded_personas_str = ' | '.join(grounded_personas_list[i])
                    masked_utterance_list[i] = f'{user_token} [grounding] {grounded_personas_str} [/grounding]'


            masked_utterance_list = masked_utterance_list[:mask_id+1+m]
            
            masked_utterance_str = prompt + ' ' + ' '.join(masked_utterance_list)
            masked_inputs.append(masked_utterance_str)
            targets.append(target)
    
    if not os.path.exists(output_data_dir):
        os.mkdir(output_data_dir)
    output_list = [{'source': source, 'target': target} for source, target in zip(masked_inputs, targets)]
    with open(f'{output_data_dir}/{mode}.json', 'w') as f:
        json.dump(output_list, f, indent=4)
    
    print(f'Num of {mode} samples: {len(output_list)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['wow', 'pc'], required=True)
    args = parser.parse_args()


    if args.dataset == 'wow':
        input_data_dir = 'wizard_of_wikipedia/train.json'
        with open(input_data_dir, 'r') as fp:
            wow_data = json.load(fp)
        random.shuffle(wow_data)
        with open('wizard_of_wikipedia/train_shuffled.json', 'w') as fp:
            json.dump(wow_data, fp, indent=4)

        input_data_dir = 'wizard_of_wikipedia/train_shuffled.json'
        output_data_dir = 'full_training_inp_data/train.json'
        if os.path.exists('full_training_inp_data') == False:
            os.mkdir('full_training_inp_data')
        num_samples = None
        mode = 'train'
        following_kw_num = 2
        get_inp_data_for_training(input_data_dir, output_data_dir, mode, num_samples, following_kw_num)


        input_data_dir = 'wizard_of_wikipedia/valid_topic_split.json'
        output_data_dir = 'full_training_inp_data/valid.json'
        num_samples = None
        mode = 'valid'
        following_kw_num = 2
        get_inp_data_for_training(input_data_dir, output_data_dir, mode, num_samples, following_kw_num)
    else:
        persona_data_dir = "persona_chat"
        output_data_dir = "pc_full_training_inp_data"
        num_train_samples = None
        get_inp_data_for_training_pc(persona_data_dir, output_data_dir, 'train', num_train_samples)
        get_inp_data_for_training_pc(persona_data_dir, output_data_dir, 'valid')
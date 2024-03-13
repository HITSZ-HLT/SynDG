Code for our paper:

Jianzhu Bao, Rui Wang, Yasheng Wang, Aixin Sun, Yitong Li, Fei Mi, and Ruifeng Xu. 2023. A Synthetic Data Generation Framework for Grounded Dialogues. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10866–10882, Toronto, Canada. Association for Computational Linguistics.

### Installation
* Python version >= 3.8.10
* PyTorch version == 1.10.0
* CUDA version == 11.1

```sh
pip install -r requirements.txt
```

### Usage

#### WoW

1. Prepare the data for training the dialogue content realization model. The resulting files are saved in 'full_training_inp_data'. (Please first download the original WoW dataset, and save it in ./wizard_of_wikipedia)

```sh
python get_training_inp_data.py --dataset wow
```

2. Train the dialogue content realization model.

```sh
bash run_train_wow.sh
```

Choose the model with the lowest eval_loss. For example, 'outputs/full/trial_1/checkpoint-43842'

3. Sample the dialogue flow and get the source input for the dialogue content realization model.

```sh
python construct_flow.py --dataset wow
```

4. Transform the dialogue flows into synthetic dialogues by the dialogue content realization model. The synthetic dialogues are saved in 'wizard_of_wikipedia/synthetic_data/generated_predictions.json'. Remeber to specify the path to the trained dialogue content realization model in `exp_config/syn_config_wow.json`.

```sh
DATASET_NAME=wow bash run_syn_wow.sh
```

5. Enter the filtering/wow_filtering directory and run the following command to train the two filtering models.

```sh
python get_training_scorer_data.py
bash run_train_dial.sh
bash run_train_flow.sh
```

6. In the filtering/wow_filtering directory, run the following command to get the two scores of the synthetic dialogues. Note that we need to specify the path to the filtering model. 

After running the following command, we will get 'wizard_of_wikipedia/synthetic_data/generated_dials_w_total_score.json'. This file contains the synthetic dialogues and their scores. We can choose the final synthetic dialogues according to the scores. Finally, we can view the final synthetic dialogues as new training data for other grounded dialogue generation models.

```sh
bash run_score_dial.sh
bash run_score_flow.sh
python post_processing.py
```


#### PersonaChat

1. Prerocess the original PersonaChat dataset with grounding, get the 'persona_chat/train_both_original_grounded.json' and 'persona_chat/valid_both_original_grounded.json' files. We can get the knowledge grounding by the method mentioned here: https://github.com/caoyu-noob/D3 . We have provided the processed files in the 'persona_chat' directory.

2. Prepare the data for training the dialogue content realization model. The resulting files are saved in 'pc_full_training_inp_data'.
    
```sh
python get_training_inp_data.py --dataset persona_chat
```

3. Train the dialogue content realization model.

```sh
bash run_train_pc.sh
```

Choose the model with the lowest eval_loss. For example, 'outputs/full/pc/checkpoint-32858'

4. Sample the dialogue flow and get the source input for the dialogue content realization model.

```sh
python construct_flow.py --dataset pc
```

5. Transform the dialogue flows into synthetic dialogues by the dialogue content realization model. The synthetic dialogues are saved in 'persona_chat/synthetic_data/generated_predictions.json'. Remeber to specify the path to the trained dialogue content realization model in `exp_config/syn_config_pc.json`.

```sh
DATASET_NAME=pc bash run_syn_pc.sh
```

6. Enter the filtering/pc_filtering directory and run the following command to train the two filtering models.

```sh
python get_training_scorer_data.py
bash run_train_dial.sh
bash run_train_flow.sh
```

7. In the the filtering/pc_filtering directory, run the following command to get the two scores of the synthetic dialogues. Note that we need to specify the path to the filtering model. 

After running the following command, we will get 'persona_chat/synthetic_data/generated_dials_w_total_score.json'.

```sh
bash run_score_dial.sh
bash run_score_flow.sh
python post_processing.py
```






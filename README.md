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

1. Prepare the data for training the dialogue content realization model. The resulting files are saved in 'full_training_inp_data'.

```sh
python get_training_inp_data.py --dataset wow
```

2. Train the dialogue content realization model

```sh
bash run_train_wow.sh
```

Choose the model with the lowest eval_loss. For example, 'outputs/full/trial_1/checkpoint-43842'

3. Sample the dialogue flow and get the source input for the dialogue content realization model

```sh
python construct_flow.py
```

4. Transform the dialogue flows into synthetic dialogues by the dialogue content realization model. The synthetic dialogues are saved in 'wizard_of_wikipedia/synthetic_data/generated_predictions.json'

```sh
bash run_syn.sh
```

5. Enter the filtering/wow_filtering directory and run the following command to train the two filtering models

```sh
python get_training_scorer_data.py
bash run_train_dial.sh
bash run_train_flow.sh
```

6. Run the following command to get the two scores of the synthetic dialogues. Note that you need to specify the path to the filtering model. 

After running the following command, you will get 'wizard_of_wikipedia/synthetic_data/generated_dials_w_total_score.json'. This file contains the synthetic dialogues and their scores. We can choose the final synthetic dialogues according to the scores. Finally, we can view the final synthetic dialogues as new training data for other grounded dialogue generation models.

```sh
bash run_score_dial.sh
bash run_score_flow.sh
python post_processing.py
```


#### PersonaChat

Code about PersonaChat is coming soon.






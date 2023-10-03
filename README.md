# fc_oracles
This repository contains the code for our submission to the 2023 Soccer Prediction Challenge: https://sites.google.com/view/2023soccerpredictionchallenge. The 2023 Soccer Prediction Challenge is an international machine learning competition that invited the machine learning community to predict the outcomes of a set of soccer matches from leagues worldwide played at the beginning of April 2023. The Challenge consisted of two tasks:
1. Predict the exact scores (Goals scored by home team: Goals scored by away team).
2. Predict the win / draw / loss probabilities of the home team.

Our submission was placed 2nd in Task 1 and 4th in Task 2. Task 1 is evaluated using the Root Mean Squared Error (RMSE) metric, while Task 2 is evaluated using Ranked Probability Score (RPS). We run a total of three experiments per task:
1. Experiment 1: Train a global neural network model over all the leagues, without using the league identifier feature. (We had forgotten to add the league identifiers as features for our submission!)
2. Exeriment 2: Train a global neural network model over all the leagues, with the league identifier as an additional feature.
3. Experiment 3: Train a local neural network model for each league.

Before running the experiments, you need to create a 'data' folder and place in it the following files (to be obtained from the organizers):
1. TrainingSet-FINAL.xlsx (training set)
2. Real_outcomes.xlsx (test set)

Steps to run the experiments:
# Task 1: 
# Experiment 1:
python -m run_rmse_global --preprocess_data 1 --run_hp_search 1 --eval_on_test 1 --reproduce_submission 1
# Experiment 2:
python -m run_rmse_global --preprocess_data 1 --run_hp_search 1 --eval_on_test 1 --reproduce_submission 0
# Experiment 3:
python -m run_rmse_non_global --preprocess_data 1 --run_hp_search 1 --eval_on_test 1

# Task 2: 
# Experiment 1:
python -m run_rps_global --preprocess_data 1 --run_hp_search 1 --eval_on_test 1 --reproduce_submission 1
# Experiment 2:
python -m run_rps_global --preprocess_data 1 --run_hp_search 1 --eval_on_test 1 --reproduce_submission 0
# Experiment 3:
python -m run_rps_non_global --preprocess_data 1 --run_hp_search 1 --eval_on_test 1

We have also provided the optimal hyperparameters, by setting --run_hp_search 0, the hyperparameter search is skipped and the optimal hyperparameters are used to evaluate on the test set. If you have already preprocessed the data the first time, you can set --preprocess_data 0 subsequently.




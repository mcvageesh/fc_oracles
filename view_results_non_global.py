import pickle
import numpy as np
from utils import avg_rps
import pandas as pd
from sklearn.metrics import mean_squared_error

random_seed = 4321
use_random_training = False
mode = 'rps'
lge = 'FRA3'

hp_search_save_path = 'results/hp_search_' + mode + '_non_global_' + lge + '_' + str(random_seed) + '_' + str(use_random_training) + '.pickle'
predictions_save_path = 'results/predictions_' + mode + '_non_global_' + str(random_seed) + '_' + str(use_random_training) + '.csv'

with open(hp_search_save_path, 'rb') as file:
    trials = pickle.load(file)

best_hp = trials.results[np.argmin([r['loss'] for r in
                                    trials.results])]['hp_dict']
num_epochs = trials.results[np.argmin([r['loss'] for r in
                                       trials.results])]['num_epochs']
loss = trials.results[np.argmin([r['loss'] for r in
                                       trials.results])]['loss']

print('val loss', loss)
print('best hp', best_hp)
print('num epochs', num_epochs)

test_data_df = pd.read_csv(filepath_or_buffer=predictions_save_path,
                              sep=',',
                              index_col=0,
                              infer_datetime_format=True,
                              parse_dates=True
                              )

if mode == 'rps':
    test_data_df = pd.concat([test_data_df, pd.get_dummies(test_data_df['WDL'], prefix='outcome')], axis=1)

    true = test_data_df.loc[:, ['outcome_W', 'outcome_D', 'outcome_L']].values.astype('float32')
    pred = test_data_df.loc[:, ['prd_W', 'prd_D', 'prd_L']].values.astype('float32')

    rps = avg_rps(true, pred)

    print('RPS', rps)
else:
    true = test_data_df.loc[:, ['HS', 'AS']].values.astype('float32')
    pred = test_data_df.loc[:, ['prd_HS', 'prd_AS']].values.astype('float32')

    rmse = np.sqrt(mean_squared_error(true, pred) * 2)
    print('RMSE', rmse)
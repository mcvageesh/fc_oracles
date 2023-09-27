from hyperopt import hp
from hyperopt import fmin, tpe, Trials, STATUS_OK
import pandas as pd
import numpy as np
from utils import parse_boolean
import tensorflow as tf
from keras.models import load_model
import math
import shutil
from sklearn.model_selection import train_test_split
import pickle
from models import MyModel2
import argparse
import os
import random
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument("--preprocess_data",
                    type=parse_boolean,
                    help="Run preprocess data if you have not done it before",
                    default=False)
parser.add_argument("--run_hp_search",
                    type=parse_boolean,
                    help="Whether to run hyperparameter search",
                    default=False)
parser.add_argument("--max_trials_hp_search",
                    help="Maximum number of trials for hyperparameter search",
                    type=int,
                    default=10)
parser.add_argument("--eval_on_test",
                    help="Whether to evaluate model on test set",
                    type=parse_boolean,
                    default=False)
parser.add_argument("--random_seed",
                    help="Random seed value",
                    type=int,
                    default=4321)
parser.add_argument("--use_random_training",
                    help="Whether to use random validation or last 25 percent for hyperparameter selection",
                    type=parse_boolean,
                    default=False)
parser.add_argument("--reproduce_submission",
                    help="Whether to reproduce submission to competition",
                    type=parse_boolean,
                    default=True)

args = parser.parse_args()

preprocess_data = args.preprocess_data
run_hp_search = args.run_hp_search
max_trials_hp_search = args.max_trials_hp_search
eval_on_test = args.eval_on_test
random_seed = args.random_seed
use_random_training = args.use_random_training
reproduce_submission = args.reproduce_submission

max_epochs = 1000
patience = 20
num_random_seeds = 3
lr_decay_rate = 0.03
val_percent = 0.25

# file paths
data_file_path = 'data/TrainingSet-FINAL.xlsx'
test_data_file_path = 'data/Real_outcomes.xlsx'

train_start_date = '2021-04-13'
train_end_date = '2023-04-13'

hp_search_save_path = 'results/hp_search_rmse_global_' + str(random_seed) + '_' + str(use_random_training) + '_' \
                      + str(reproduce_submission) + '.pickle'
predictions_save_path = 'results/predictions_rmse_global_' + str(random_seed) + '_' + str(use_random_training) + '_' \
                        + str(reproduce_submission) + '.csv'

if not os.path.exists('processed'):
    os.makedirs('processed')

if not os.path.exists('results'):
    os.makedirs('results')

if preprocess_data:
    data_df = pd.read_excel(io=data_file_path)
    data_date = pd.to_datetime(data_df['Date'], dayfirst=True, infer_datetime_format=True, format="%d/%m/%y")
    data_df.loc[:, 'Date'] = data_date
    data_df['Sea'] = data_df['Sea'].str.split('-').str[0].astype(int)

    for league in data_df['Lge'].unique():
        data_df[data_df['Lge'] == league].to_csv('processed/raw_file_' + league + '.csv')

    data_df = pd.read_excel(io=test_data_file_path)
    data_date = pd.to_datetime(data_df['Date'], dayfirst=True, infer_datetime_format=True, format="%d/%m/%y")
    data_df.loc[:, 'Date'] = data_date
    data_df.loc[data_df['Lge'] == 44986, 'Lge'] = 'MAR1'

    data_df.to_csv('processed/test_data.csv')

test_data_file_path = 'processed/test_data.csv'
test_data_df = pd.read_csv(filepath_or_buffer=test_data_file_path,
                              sep=',',
                              index_col=0,
                              infer_datetime_format=True,
                              parse_dates=True
                              )
test_data_df_copy = test_data_df.copy(deep=True)
test_leagues = [lge for lge in test_data_df['Lge'].unique()]

# DZA1 is missing, add it
test_leagues += ['DZA1']

test_leagues_dict = {}
for count, lge in enumerate(test_leagues):
    # test_leagues_dict[lge] = -0.5 + count / len(test_leagues)
    test_leagues_dict[lge] = count


def my_function(row, hp, df):
    HT = row['HT']
    AT = row['AT']
    lge = row['Lge']
    date = row['Date']

    gd_limiter = hp['gd_limiter']
    num_past_matches1 = hp['num_past_matches1']

    tmp_df = df[(df['Date'] < date)]
    tmp_df.loc[tmp_df['GD'] > gd_limiter, 'GD'] = gd_limiter
    tmp_df.loc[tmp_df['GD'] < -gd_limiter, 'GD'] = -gd_limiter

    data = tmp_df.loc[(tmp_df['HT'] == HT), :].sort_values(
        'Date').tail(int(num_past_matches1))

    if len(data) == 0:
        gd_ht = np.nan
    else:
        gd_ht = data['GD'].mean()

    data = tmp_df.loc[(tmp_df['AT'] == HT), :].sort_values(
        'Date').tail(int(num_past_matches1))

    if len(data) == 0:
        gd_ht2 = np.nan
    else:
        gd_ht2 = data['GD'].mean()

    data = tmp_df.loc[(tmp_df['AT'] == AT), :].sort_values(
        'Date').tail(int(num_past_matches1))

    if len(data) == 0:
        gd_at = np.nan
    else:
        gd_at = data['GD'].mean()

    data = tmp_df.loc[(tmp_df['HT'] == AT), :].sort_values(
        'Date').tail(int(num_past_matches1))

    if len(data) == 0:
        gd_at2 = np.nan
    else:
        gd_at2 = data['GD'].mean()

    data = tmp_df.loc[((tmp_df['HT'] == HT) | (tmp_df['AT'] == HT)), :].sort_values(
        'Date').tail(int(num_past_matches1 * 2))
    if len(data) == 0:
        gd_ht3 = np.nan
    else:
        data.loc[data['AT'] == HT, 'GD'] = -data.loc[data['AT'] == HT, 'GD']
        gd_ht3 = data['GD'].mean()

    data = tmp_df.loc[((tmp_df['HT'] == AT) | (tmp_df['AT'] == AT)), :].sort_values(
        'Date').tail(int(num_past_matches1 * 2))
    if len(data) == 0:
        gd_at3 = np.nan
    else:
        data.loc[data['HT'] == AT, 'GD'] = -data.loc[data['HT'] == AT, 'GD']
        gd_at3 = data['GD'].mean()

    return pd.Series([gd_ht / gd_limiter, gd_at / gd_limiter, gd_ht2 / gd_limiter,
                      gd_at2 / gd_limiter, gd_ht3 / gd_limiter, gd_at3 / gd_limiter,
                      test_leagues_dict[lge]])


if run_hp_search:
    space = {
        'num_units1': hp.quniform("num_units1", 8, 80, 8),
        'initial_learning_rate': hp.qloguniform('initial_learning_rate', np.log(5e-4), np.log(1e-2), 5e-4),
        'gd_limiter': hp.quniform('gd_limiter', 2, 5, 1),
        'num_past_matches1': hp.quniform('num_past_matches1', 20, 80, 20),
        'batch_size': hp.choice('batch_size', [128, 256])
    }

    def objective(space):

        for count, lge in enumerate(test_leagues):
            data_file_path = 'processed/raw_file_' + lge + '.csv'
            data_df = pd.read_csv(filepath_or_buffer=data_file_path,
                                  sep=',',
                                  index_col=0,
                                  infer_datetime_format=True,
                                  parse_dates=True
                                  )
            data_df = data_df.dropna()
            data_df = pd.concat([data_df, pd.get_dummies(data_df['WDL'], prefix='outcome')], axis=1)

            train_df = data_df[(data_df['Date'] >= train_start_date) & (data_df['Date'] <= train_end_date)]

            result = train_df.apply(my_function, axis=1, args=(space, data_df))
            train_df[['gd_ht', 'gd_at', 'gd_ht2', 'gd_at2', 'gd_ht3', 'gd_at3', 'lge']] = result

            train_df.dropna(inplace=True)

            x = train_df.loc[:, ['gd_ht', 'gd_at', 'gd_ht2', 'gd_at2', 'gd_ht3', 'gd_at3', 'lge']].values
            y = train_df.loc[:, ['HS', 'AS']].values.astype('float32') / 10

            if use_random_training:
                train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=val_percent, random_state=random_seed)
            else:
                train_x = x[:int((1 - val_percent) * x.shape[0]), :]
                train_y = y[:int((1 - val_percent) * y.shape[0]), :]

                val_x = x[int((1 - val_percent) * x.shape[0]):, :]
                val_y = y[int((1 - val_percent) * y.shape[0]):, :]

            if count == 0:
                TRAIN_X = train_x
                TRAIN_Y = train_y
                VAL_X = val_x
                VAL_Y = val_y
            else:
                TRAIN_X = np.concatenate((TRAIN_X, train_x), axis=0)
                TRAIN_Y = np.concatenate((TRAIN_Y, train_y), axis=0)
                VAL_X = np.concatenate((VAL_X, val_x), axis=0)
                VAL_Y = np.concatenate((VAL_Y, val_y), axis=0)

        epoch_len = []
        pred_y = np.zeros(shape=VAL_Y.shape)

        #####################
        # Train & eval model
        for k in range(num_random_seeds):
            np.random.default_rng(random_seed + k)
            tf.random.set_seed(random_seed + k)
            random.seed(random_seed + k)
            np.random.seed(random_seed + k)

            if reproduce_submission:
                model = MyModel2(space, TRAIN_X.shape[1], embedding_input_dim=None)
            else:
                model = MyModel2(space, TRAIN_X.shape[1], embedding_input_dim=len(test_leagues))

            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath='tmp/checkpoint_{epoch:02d}.h5',
                save_weights_only=False,
                monitor='val_loss',
                mode='min',
                save_best_only=False)
            # _{epoch: 02d}

            # compile the model with the ranked probability score loss function
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=space['initial_learning_rate']),
                          loss='mean_squared_error'
                          )

            def scheduler(epoch, lr,
                          initial_learning_rate=space['initial_learning_rate'],
                          k=lr_decay_rate):
                return initial_learning_rate * math.exp(-k * epoch)

            lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                              patience=patience,
                                                              min_delta=0.001,
                                                              mode='min')

            # train the model
            history = model.fit(TRAIN_X, TRAIN_Y, epochs=max_epochs, validation_data=(VAL_X, VAL_Y),
                                callbacks=[model_checkpoint_callback, lr_scheduler, early_stopping],
                                batch_size=space['batch_size'])

            # make predictions
            epoch_len += [len(history.history['val_loss']) - patience]

            for i in range(5):
                tmp = load_model(
                    'tmp/checkpoint_{0:02d}.h5'.format(len(history.history['val_loss']) - patience + i))
                pred_y += tmp.predict(VAL_X)

        pred_y = pred_y / (num_random_seeds * 5)
        val_score = np.sqrt(2 * mean_squared_error(10 * VAL_Y, np.round(10 * pred_y)))
        shutil.rmtree('tmp')

        return {'loss': val_score, 'hp_dict': space, 'status': STATUS_OK, 'num_epochs': np.max(epoch_len)}

    trials = Trials()
    fmin(objective, space, algo=tpe.suggest, max_evals=max_trials_hp_search,
         verbose=True, trials=trials, rstate=np.random.default_rng(random_seed),
         trials_save_file=hp_search_save_path)

if eval_on_test:
    val_rps = []
    with open(hp_search_save_path, 'rb') as file:
        trials = pickle.load(file)

    idx = np.argmin([r['loss'] for r in trials.results])
    best_hp = trials.results[idx]['hp_dict']
    num_epochs = trials.results[idx]['num_epochs']
    best_hp['num_epochs'] = num_epochs

    for count, lge in enumerate(test_leagues):
        data_file_path = 'processed/raw_file_' + lge + '.csv'
        data_df = pd.read_csv(filepath_or_buffer=data_file_path,
                              sep=',',
                              index_col=0,
                              infer_datetime_format=True,
                              parse_dates=True
                              )
        data_df = data_df.dropna()
        data_df = pd.concat([data_df, pd.get_dummies(data_df['WDL'], prefix='outcome')], axis=1)

        train_df = data_df[(data_df['Date'] >= train_start_date) & (data_df['Date'] <= train_end_date)]

        result = train_df.apply(my_function, axis=1, args=(best_hp, data_df))
        train_df[['gd_ht', 'gd_at', 'gd_ht2', 'gd_at2', 'gd_ht3', 'gd_at3', 'lge']] = result

        train_df.dropna(inplace=True)

        x = train_df.loc[:, ['gd_ht', 'gd_at', 'gd_ht2', 'gd_at2', 'gd_ht3', 'gd_at3', 'lge']].values
        y = train_df.loc[:, ['HS', 'AS']].values.astype('float32') / 10

        if use_random_training:
            train_x, _, train_y, __ = train_test_split(x, y, test_size=val_percent, random_state=random_seed)
        else:
            train_x = x[int(val_percent * x.shape[0]):, :]
            train_y = y[int(val_percent * y.shape[0]):, :]

        if count == 0:
            TRAIN_X = train_x
            TRAIN_Y = train_y
        else:
            TRAIN_X = np.concatenate((TRAIN_X, train_x), axis=0)
            TRAIN_Y = np.concatenate((TRAIN_Y, train_y), axis=0)

    ##########################
    for k in range(num_random_seeds):
        np.random.default_rng(random_seed + k)
        tf.random.set_seed(random_seed + k)
        random.seed(random_seed + k)
        np.random.seed(random_seed + k)

        if reproduce_submission:
            model = MyModel2(best_hp, TRAIN_X.shape[1], embedding_input_dim=None)
        else:
            model = MyModel2(best_hp, TRAIN_X.shape[1], embedding_input_dim=len(test_leagues))

        # compile the model with the MSE loss function
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_hp['initial_learning_rate']),
                      loss='mean_squared_error'
                      )

        def scheduler(epoch, lr,
                      initial_learning_rate=best_hp['initial_learning_rate'],
                      k=lr_decay_rate):
            return initial_learning_rate * math.exp(-k * epoch)


        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='tmp_' + str(k) + '/checkpoint_{epoch:02d}.h5',
            save_weights_only=False,
            monitor='loss',
            mode='min',
            save_best_only=False)

        # train the model
        history = model.fit(TRAIN_X, TRAIN_Y, epochs=best_hp['num_epochs'] + patience,
                            callbacks=[lr_scheduler, model_checkpoint_callback], batch_size=best_hp['batch_size'])

    for count, lge in enumerate(test_leagues):
        if lge == 'DZA1':
            # not present in final challenge test set
            continue

        data_file_path = 'processed/raw_file_' + lge + '.csv'
        data_df = pd.read_csv(filepath_or_buffer=data_file_path,
                              sep=',',
                              index_col=0,
                              infer_datetime_format=True,
                              parse_dates=True
                              )
        data_df = data_df.dropna()
        data_df = pd.concat([data_df, pd.get_dummies(data_df['WDL'], prefix='outcome')], axis=1)

        val_df = test_data_df[test_data_df['Lge'] == lge]
        result = val_df.apply(my_function, axis=1, args=(best_hp, data_df))
        val_df[['gd_ht', 'gd_at', 'gd_ht2', 'gd_at2', 'gd_ht3', 'gd_at3', 'lge']] = result
        val_df.dropna(inplace=True)

        val_x = val_df.loc[:, ['gd_ht', 'gd_at', 'gd_ht2', 'gd_at2', 'gd_ht3', 'gd_at3', 'lge']].values

        pred_y = np.zeros(shape=(val_x.shape[0], 2))
        for k in range(num_random_seeds):
            for i in range(5):
                tmp = load_model('tmp_' + str(k) + '/checkpoint_{0:02d}.h5'.format(best_hp['num_epochs'] + i))
                pred_y += tmp.predict(val_x)
        pred_y = pred_y / (num_random_seeds * 5)
        test_data_df_copy.loc[test_data_df['Lge'] == lge, ['prd_HS', 'prd_AS']] = np.round(10 * pred_y)

    for k in range(num_random_seeds):
        shutil.rmtree('tmp_{}'.format(k))
    test_data_df_copy.to_csv(predictions_save_path)


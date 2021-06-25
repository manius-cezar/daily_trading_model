import os, sys

from tensorflow.python.framework.errors_impl import ResourceExhaustedError
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import pandas as pd
from sklearn import preprocessing
from shutil import copyfile
import seaborn as sns
import json
import optuna
from importlib import import_module, reload
from shutil import copyfile
import time
from pickle import dump

EXISTING_OPTUNA_RUN_ID = 'optuna_2021_06_08-18_15_27'

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
plt.rcParams["figure.figsize"] = (20,3)

np.random.seed(42)
tf.random.set_seed(42)
physical_devices = tf.config.list_physical_devices('GPU')
physical_devices
tf.config.set_visible_devices(
    [physical_devices[1], physical_devices[2], 
    physical_devices[3], physical_devices[4], 
    physical_devices[5]], 'GPU')

historic_data_in_minibatch = 250

ema_lengths = [3, 5, 8, 9, 10, 13, 20, 21, 26, 34, 50, 55, 89, 100, 144, 200, 233, historic_data_in_minibatch]

# Loading data from already prepared datasets
X_train, y_train = (
    np.load('../data/X_train_shuffled_with_mean_model.npz'), 
    np.load('../data/y_train_shuffled_with_mean_model.npz'))
X_train, y_train = X_train['arr_0'], y_train['arr_0']

X_valid, y_valid = (
    np.load('../data/X_valid_shuffled_with_mean_model.npz'), 
    np.load('../data/y_valid_shuffled_with_mean_model.npz'))
X_valid, y_valid = X_valid['arr_0'], y_valid['arr_0']

X_test, y_test = (
    np.load('../data/X_test_shuffled_with_mean_model.npz'), 
    np.load('../data/y_test_shuffled_with_mean_model.npz'))
X_test, y_test = X_test['arr_0'], y_test['arr_0']

X_train = X_train.reshape((X_train.shape[0]), X_train.shape[1])
X_test = X_test.reshape((X_test.shape[0]), X_test.shape[1])
X_valid = X_valid.reshape((X_valid.shape[0]), X_valid.shape[1])

# Data scaling

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

X_train = X_train.reshape((X_train.shape[0]), X_train.shape[1], 1)
X_test = X_test.reshape((X_test.shape[0]), X_test.shape[1], 1)
X_valid = X_valid.reshape((X_valid.shape[0]), X_valid.shape[1], 1)

features_number = X_train.shape[1]

# Split for two model inputs
X_train_CNN, X_train_Dense = X_train[:, :historic_data_in_minibatch, :], X_train[:, historic_data_in_minibatch:, :] 
X_test_CNN, X_test_Dense = X_test[:, :historic_data_in_minibatch, :], X_test[:, historic_data_in_minibatch:, :] 
X_valid_CNN, X_valid_Dense = X_valid[:, :historic_data_in_minibatch, :], X_valid[:, historic_data_in_minibatch:, :] 

# y categorization

one_ids = np.where(y_train >= 0.9)
zero_ids = np.where(y_train < 0.9)

y_train[one_ids[0], one_ids[1]] = 1
y_train[zero_ids[0], zero_ids[1]] = 0

ones, zeroes = len(one_ids[0]), len(zero_ids[0])
print('Train:')
print(f'Positive %: {ones / (ones + zeroes)}')
print(f'Negative %: {1 - (ones / (ones + zeroes))}')

y_train = y_train.astype(np.int)


one_ids = np.where(y_valid >= 0.9)
zero_ids = np.where(y_valid < 0.9)

y_valid[one_ids[0], one_ids[1]] = 1
y_valid[zero_ids[0], zero_ids[1]] = 0

ones, zeroes = len(one_ids[0]), len(zero_ids[0])
print('Valid:')
print(f'Positive %: {ones / (ones + zeroes)}')
print(f'Negative %: {1 - (ones / (ones + zeroes))}')

y_valid = y_valid.astype(np.int)



one_ids = np.where(y_test >= 0.9)
zero_ids = np.where(y_test < 0.9)

y_test[one_ids[0], one_ids[1]] = 1
y_test[zero_ids[0], zero_ids[1]] = 0

ones, zeroes = len(one_ids[0]), len(zero_ids[0])
print('Test:')
print(f'Positive %: {ones / (ones + zeroes)}')
print(f'Negative %: {1 - (ones / (ones + zeroes))}')

y_test = y_test.astype(np.int)


y_all = np.concatenate((y_train, y_valid, y_test))
temp = np.copy(y_all)

one_ids = np.where(temp >= 0.9)
zero_ids = np.where(temp < 0.9)

temp[one_ids[0], one_ids[1]] = 1
temp[zero_ids[0], zero_ids[1]] = 0

ones, zeroes = len(one_ids[0]), len(zero_ids[0])
print('Whole dataset:')
print(f'Positive %: {ones / (ones + zeroes)}')
print(f'Negative %: {1 - (ones / (ones + zeroes))}')

unique, counts = np.unique(y_all, return_counts=True)
for val, qty in zip(unique, counts):
    print(f'Label {val}: {qty}')

# Weights calculation
neg, pos = np.unique(y_train, return_counts=True)[1]
total = neg + pos

weight_for_0 = (1 / neg)*(total)/2.0 
weight_for_1 = (1 / pos)*(total)/2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))

class_weight = {0: weight_for_0, 1: weight_for_1}

# Optuna optimization
tf.keras.backend.clear_session()

config = {
    'cnn_input_shape': [X_train_CNN.shape[1], 1],
    'kernel_initializer': 'he_normal',
    'statistical_input_shape': [X_train_Dense.shape[1], 1],
    'statistical_dense1_neurons_qty': X_train_Dense.shape[1],
    'batch_size': 6000
}

optuna_run_id = EXISTING_OPTUNA_RUN_ID or time.strftime('optuna_%Y_%m_%d-%H_%M_%S')

with open(f'{optuna_run_id}_config.json', 'w') as config_json:
    json.dump(config, config_json)

model_creation = import_module('model_optuna')
reload(model_creation)

devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4"]
copyfile('model_optuna.py', f'model_optuna-{optuna_run_id}.py')


def create_tensorboard_cb():
    root_logdir = os.path.join(os.curdir, 'my_dictionaries')

    def get_run_logdir():
        import time
        run_id = time.strftime('optuna_%Y_%m_%d-%H_%M_%S')
        return os.path.join(root_logdir, run_id)

    run_logdir = get_run_logdir()
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    file_writer = tf.summary.create_file_writer(run_logdir + "/metrics")
    file_writer.set_as_default()

    dump(scaler, open(os.path.join(run_logdir, 'input_scaler_with_ema_shuffled.pkl'), 'wb'))

    return tensorboard_cb, run_logdir


def get_momentum_callback(n: int, m: int, momentum_max: float, momentum_min: float):
    # 1cycle Momentum scheduler
    _momentum_epochs1, _momentums1 = np.array([1, n]).reshape((-1, 1)), np.array([momentum_max, momentum_min]).reshape((-1, 1))
    _momentum_epochs2, _momentums2 = np.array([n, 2*n]).reshape((-1, 1)), np.array([momentum_min, momentum_max]).reshape((-1, 1))

    momentum_reg1 = LinearRegression().fit(_momentum_epochs1, _momentums1)
    momentum_reg2 = LinearRegression().fit(_momentum_epochs2, _momentums2)

    class MomentumCallback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            epoch = epoch + 1
            if epoch <= n: 
                momentum = momentum_reg1.predict(np.array([epoch]).reshape((-1,1)))[0][0]
            elif epoch > n and epoch <= 2 * n:
                momentum = momentum_reg2.predict(np.array([epoch]).reshape((-1,1)))[0][0]
            else:
                momentum = momentum_max

            self.model.optimizer.momentum = momentum
            tf.summary.scalar('momentum', data=momentum, step=epoch-1)
            
    return MomentumCallback()


def get_lr_callback(n: int, m: int, lr_in: float, lr_max: float, lr_min: float):
    # 1cycle Learning Rate Scheduler
    _epochs1, _lrs1 = np.array([1, n]).reshape((-1, 1)), np.array([lr_in, lr_max]).reshape((-1, 1))
    _epochs2, _lrs2 = np.array([n, 2*n]).reshape((-1, 1)), np.array([lr_max, lr_in]).reshape((-1, 1))
    _epochs3, _lrs3 = np.array([2*n, 2*n + m]).reshape((-1, 1)), np.array([lr_in, lr_min]).reshape((-1, 1))

    reg1 = LinearRegression().fit(_epochs1, _lrs1)
    reg2 = LinearRegression().fit(_epochs2, _lrs2)
    reg3 = LinearRegression().fit(_epochs3, _lrs3)

    def one_cycle_lr_scheduler(epoch, curr_lr):
        epoch = epoch + 1
        if epoch <= n:
            lr = reg1.predict(np.array([epoch]).reshape((-1,1)))[0][0]
        elif epoch > n and epoch <= 2 * n:
            lr = reg2.predict(np.array([epoch]).reshape((-1,1)))[0][0]
        else:
            lr = reg3.predict(np.array([epoch]).reshape((-1,1)))[0][0]

        tf.summary.scalar('learning rate', data=lr, step=epoch-1)
        return lr

    one_cycle_scheduler = tf.keras.callbacks.LearningRateScheduler(one_cycle_lr_scheduler)
    return one_cycle_scheduler


def objective(trial: optuna.Trial):
    model = model_creation.build_model(trial, config, devices=devices, cust_lr_scheduler=None)
    
    tensorboard_cb, run_logdir = create_tensorboard_cb()
    tf_val_prc_pruning_cb = optuna.integration.TFKerasPruningCallback(trial, 'val_prc')
    
    n = trial.suggest_int('n', 120, 140, 10)
    m = trial.suggest_int('m', 40, 60, 10)
    
    epochs_qty = 2*n + m
    
    momentum_max = trial.suggest_float('momentum_max', 0.9, 1.1, step=0.05)
    momentum_min = trial.suggest_float('momentum_min', 0.5, 0.8, step=0.05)
    
    lr_in = trial.suggest_loguniform('lr_in', 0.005, 0.05)
    lr_max = 0.1
    lr_min = trial.suggest_loguniform('lr_min', 1e-6, 1e-4)
    
    learning_rate_callback = get_lr_callback(n, m, lr_in, lr_max, lr_min)
    momentum_callback = get_momentum_callback(n, m, momentum_max, momentum_min)
    
    history = model.fit(
        (X_train_CNN, X_train_Dense), y_train, 
        epochs=epochs_qty,
        batch_size = config['batch_size'] * len(devices), 
        validation_data=((X_valid_CNN, X_valid_Dense), y_valid), 
        verbose=0,
        class_weight=class_weight,
        callbacks=[
            tensorboard_cb, 
            learning_rate_callback,
            momentum_callback,
            tf_val_prc_pruning_cb,
        ]
    )
    
    experiment_run_id = run_logdir.split(sep='/')[-1]
    model.save(os.path.join(run_logdir, f'optuna-{experiment_run_id}'))
    
    eval_metrics = model.evaluate((X_test_CNN, X_test_Dense), y_test, return_dict=True, verbose=0)

    optuna_metric = eval_metrics['prc']
    
    return optuna_metric
    

study_name = optuna_run_id
db_url = f'sqlite:////optuna-studyDB.db'

study = optuna.create_study(direction='maximize', study_name=study_name,
                            storage=db_url, 
                            load_if_exists=True,
                            sampler=optuna.samplers.TPESampler(), 
                            pruner=optuna.pruners.HyperbandPruner())

study.optimize(objective, gc_after_trial=True,
            catch = (ValueError,tf.errors.InvalidArgumentError,ResourceExhaustedError,))
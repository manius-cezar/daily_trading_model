import tensorflow as tf
from tensorflow import keras

import optuna

tf.random.set_seed(42)


def add_dense_layer(prev_layer, neurons_qty, config: dict, dropout_rate = None, add_batch_norm: bool = False):
    if dropout_rate is not None:
        prev_layer = keras.layers.Dropout(rate=dropout_rate) (prev_layer)
    if add_batch_norm:
        prev_layer = keras.layers.BatchNormalization() (prev_layer)
    prev_layer = keras.layers.Dense(neurons_qty, kernel_initializer=config["kernel_initializer"],
                                    activation='relu') (prev_layer)
    return prev_layer


def build_model(trial: optuna.Trial, config, devices: list = ["/gpu:0", "/gpu:1", "/gpu:2"], 
                cust_lr_scheduler = None, output_bias = None):
    strategy = tf.distribute.MirroredStrategy(devices=devices)
    with strategy.scope():
        # Sliding window input
        input_CNN = keras.Input(shape=config['cnn_input_shape'])
        batch_norm_CNN_1 = keras.layers.BatchNormalization() (input_CNN)
        conv1d_1 = keras.layers.Conv1D(filters=trial.suggest_int(f'conv1d_1_filters', 128, 256, 32),
                                       kernel_size=trial.suggest_int(f'conv1d_1_kernel_size', 100, 150, 10), 
                                       activation='relu', padding='same', 
                                       kernel_initializer=config["kernel_initializer"],
                                      ) (batch_norm_CNN_1)
        max_pooling1d_1 = keras.layers.MaxPooling1D(
            pool_size=trial.suggest_int(f'max_pooling1d_1_pool_size', 2, 3)) (conv1d_1)
        dropout_cnn_1 = keras.layers.Dropout(
            rate=trial.suggest_float(f'dropout_cnn_1_rate', 0.0, 0.4, step=0.1)) (max_pooling1d_1)
        
        flatten_cnn = keras.layers.Flatten() (dropout_cnn_1)
        previous_layer = flatten_cnn
        previous_layer = add_dense_layer(
            previous_layer, trial.suggest_int(f'previous_layer_1_neurons', 64, 160, 32), 
            config, dropout_rate=trial.suggest_float(f'previous_layer_1_dropout_rate', 0.0, 0.1, step=0.1), 
            add_batch_norm=True)
        previous_layer = add_dense_layer(
            previous_layer, trial.suggest_int(f'previous_layer_2_neurons', 64, 160, 32), 
            config, dropout_rate=trial.suggest_float(f'previous_layer_2_dropout_rate', 0.0, 0.1, step=0.1), 
            add_batch_norm=True)
        previous_layer = add_dense_layer(
            previous_layer, trial.suggest_int(f'previous_layer_3_neurons', 64, 160, 32), 
            config, dropout_rate=trial.suggest_float(f'previous_layer_3_dropout_rate', 0.0, 0.1, step=0.1), 
            add_batch_norm=True)
        previous_layer = add_dense_layer(
            previous_layer, trial.suggest_int(f'previous_layer_4_neurons', 64, 160, 32), 
            config, dropout_rate=trial.suggest_float(f'previous_layer_4_dropout_rate', 0.0, 0.1, step=0.1), 
            add_batch_norm=True)
        previous_layer = add_dense_layer(
            previous_layer, trial.suggest_int(f'previous_layer_5_neurons', 64, 256, 32), 
            config, dropout_rate=trial.suggest_float(f'previous_layer_5_dropout_rate', 0.0, 0.1, step=0.1), 
            add_batch_norm=True)
        previous_layer = add_dense_layer(
            previous_layer, trial.suggest_int(f'previous_layer_6_neurons', 64, 256, 32), 
            config, dropout_rate=trial.suggest_float(f'previous_layer_6_dropout_rate', 0.0, 0.1, step=0.1), 
            add_batch_norm=True)
        previous_layer = add_dense_layer(
            previous_layer, trial.suggest_int(f'previous_layer_7_neurons', 64, 256, 32), 
            config, dropout_rate=trial.suggest_float(f'previous_layer_7_dropout_rate', 0.0, 0.1, step=0.1), 
            add_batch_norm=True)
        previous_layer = add_dense_layer(
            previous_layer, trial.suggest_int(f'previous_layer_8_neurons', 64, 256, 32), 
            config, dropout_rate=trial.suggest_float(f'previous_layer_8_dropout_rate', 0.0, 0.1, step=0.1), 
            add_batch_norm=True)
        batch_norm_CNN_5 = keras.layers.BatchNormalization() (previous_layer)
        flatten_cnn = keras.layers.Flatten() (batch_norm_CNN_5)

        # Statistical features input
        input_dense = keras.Input(shape=config['statistical_input_shape'])
        batch_norm_dense_1 = keras.layers.BatchNormalization() (input_dense)
        dense_1 = keras.layers.Dense(trial.suggest_int(f'dense_1_neurons', 64, 160, 32), activation='relu', 
                                     kernel_initializer=config["kernel_initializer"],
                                    ) (batch_norm_dense_1)
        dropout_1 = keras.layers.Dropout(rate=trial.suggest_float(f'dropout_1_rate', 0.0, 0.1, step=0.1)) (dense_1)
        batch_norm_dense_3 = keras.layers.BatchNormalization() (dropout_1)
        flatten_dense = keras.layers.Flatten() (batch_norm_dense_3)


        # concatenation layer
        concat = keras.layers.Concatenate() ([flatten_cnn, flatten_dense])
        pre_output_dense = add_dense_layer(
            concat, trial.suggest_int(f'pre_output_dense_1_neurons', 64, 160, 32), config, add_batch_norm=True
        )
        pre_output_dense = add_dense_layer(
            pre_output_dense, trial.suggest_int(f'pre_output_dense_2_neurons', 64, 256, 32), 
            config, dropout_rate=trial.suggest_float(f'pre_output_dense_2_dropout_rate', 0.0, 0.2, step=0.1), 
            add_batch_norm=True)
        pre_output_dense = keras.layers.BatchNormalization() (pre_output_dense)

        
        # output
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
        output = keras.layers.Dense(1,
                                    activation='sigmoid',
                                    bias_initializer=output_bias
                                   ) (pre_output_dense)
        
        # model
        model = keras.Model(inputs=[input_CNN, input_dense], outputs=[output])      
        metrics = [
              keras.metrics.TruePositives(name='tp', thresholds=0.9),
              keras.metrics.FalsePositives(name='fp', thresholds=0.9),
              keras.metrics.TrueNegatives(name='tn', thresholds=0.9),
              keras.metrics.FalseNegatives(name='fn', thresholds=0.9), 
              keras.metrics.BinaryAccuracy(name='binary_accuracy', threshold=0.9),
              keras.metrics.Precision(name='precision', thresholds=0.9),
              keras.metrics.Recall(name='recall', thresholds=0.9),
              keras.metrics.AUC(name='auc', thresholds=[0.9]),
              keras.metrics.AUC(name='prc', curve='PR', thresholds=[0.9]), # precision-recall curve
        ]

        if cust_lr_scheduler is None:
            momentum = 0.1
            optimizer = keras.optimizers.SGD(learning_rate=1e-1, momentum = momentum, nesterov=True)
        else:
            optimizer = keras.optimizers.SGD(nesterov=True, learning_rate = cust_lr_scheduler)
            
        model.compile(loss=keras.losses.BinaryCrossentropy(), 
                      optimizer=optimizer, 
                      metrics=metrics)    
    return model
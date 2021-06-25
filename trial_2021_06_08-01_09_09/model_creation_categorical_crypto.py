import tensorflow as tf
from tensorflow import keras


tf.random.set_seed(42)


def add_dense_layer(prev_layer, neurons_qty, config: dict, dropout_rate = None, add_batch_norm: bool = False):
    if dropout_rate is not None:
        prev_layer = keras.layers.Dropout(rate=dropout_rate) (prev_layer)
    if add_batch_norm:
        prev_layer = keras.layers.BatchNormalization() (prev_layer)
    prev_layer = keras.layers.Dense(neurons_qty, kernel_initializer=config["kernel_initializer"],
                                    activation='relu') (prev_layer)
    return prev_layer


def build_model(config, devices: list = ["/gpu:0"], cust_lr_scheduler = None, output_bias = None):
    strategy = tf.distribute.MirroredStrategy(devices=devices)
    with strategy.scope():
        # Sliding window input
        input_CNN = keras.Input(shape=config['cnn_input_shape'])
        batch_norm_CNN_1 = keras.layers.BatchNormalization() (input_CNN)
        conv1d_1 = keras.layers.Conv1D(filters=128, kernel_size=40, 
                                       activation='relu', padding='same', 
                                       kernel_initializer=config["kernel_initializer"],
                                      ) (batch_norm_CNN_1)
        max_pooling1d_1 = keras.layers.MaxPooling1D(pool_size=2) (conv1d_1)
        dropout_cnn_1 = keras.layers.Dropout(rate=0.0) (max_pooling1d_1)
        batch_norm_CNN_2 = keras.layers.BatchNormalization() (dropout_cnn_1)
        conv1d_2 = keras.layers.Conv1D(filters=64, kernel_size=40, 
                                       activation='relu', padding='same', 
                                       kernel_initializer=config["kernel_initializer"],
                                      ) (batch_norm_CNN_2)
        max_pooling1d_2 = keras.layers.MaxPooling1D(pool_size=2) (conv1d_2)
        dropout_cnn_2 = keras.layers.Dropout(rate=0.0) (max_pooling1d_2)
        batch_norm_CNN_3 = keras.layers.BatchNormalization() (dropout_cnn_2)
        conv1d_3 = keras.layers.Conv1D(filters=32, kernel_size=40, 
                                       activation='relu', padding='same', 
                                       kernel_initializer=config["kernel_initializer"],
                                      ) (batch_norm_CNN_3)
        max_pooling1d_3 = keras.layers.MaxPooling1D(pool_size=2) (conv1d_3)
        dropout_cnn_3 = keras.layers.Dropout(rate=0.2) (max_pooling1d_3)
        batch_norm_CNN_4 = keras.layers.BatchNormalization() (dropout_cnn_3)
        conv1d_4 = keras.layers.Conv1D(filters=16, kernel_size=40, 
                                       activation='relu', padding='same', 
                                       kernel_initializer=config["kernel_initializer"],
                                      ) (batch_norm_CNN_4)
        max_pooling1d_4 = keras.layers.MaxPooling1D(pool_size=2) (conv1d_4)
        dropout_cnn_4 = keras.layers.Dropout(rate=0.2) (max_pooling1d_4)

        flatten_cnn = keras.layers.Flatten() (dropout_cnn_4)
        previous_layer = flatten_cnn
        previous_layer = add_dense_layer(
            previous_layer, 64, config, dropout_rate=0.0, add_batch_norm=True)
        previous_layer = add_dense_layer(
            previous_layer, 48, config, dropout_rate=0.1, add_batch_norm=True)
        batch_norm_CNN_5 = keras.layers.BatchNormalization() (previous_layer)
        flatten_cnn = keras.layers.Flatten() (batch_norm_CNN_5)

        # Statistical features input
        input_dense = keras.Input(shape=config['statistical_input_shape'])
        batch_norm_dense_1 = keras.layers.BatchNormalization() (input_dense)
        dense_1 = keras.layers.Dense(48, activation='relu', 
                                     kernel_initializer=config["kernel_initializer"],
                                    ) (batch_norm_dense_1)
        dropout_1 = keras.layers.Dropout(rate=0.0) (dense_1)
        batch_norm_dense_2 = keras.layers.BatchNormalization() (dropout_1)
        dense_2 = keras.layers.Dense(32, activation='relu', 
                                     kernel_initializer=config["kernel_initializer"],
                                    ) (batch_norm_dense_2)
        dropout_2 = keras.layers.Dropout(rate=0.1) (dense_2)
        batch_norm_dense_3 = keras.layers.BatchNormalization() (dropout_2)
        flatten_dense = keras.layers.Flatten() (batch_norm_dense_3)


        # concatenation layer
        concat = keras.layers.Concatenate() ([flatten_cnn, flatten_dense])
        pre_output_dense = add_dense_layer(
            concat, 64, config, add_batch_norm=True)
        pre_output_dense = add_dense_layer(
            pre_output_dense, 32, config, dropout_rate=0.0, add_batch_norm=True)
        pre_output_dense = keras.layers.BatchNormalization() (pre_output_dense)

        
        # output
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
        output = keras.layers.Dense(1, activation='sigmoid',
                                    bias_initializer=output_bias) (pre_output_dense)
        
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
            optimizer = keras.optimizers.SGD(learning_rate=1e-1, momentum = 0.1, nesterov=True)
        else:
            optimizer = keras.optimizers.SGD(momentum = 0.1, nesterov=True, learning_rate = cust_lr_scheduler)
            
        model.compile(loss=keras.losses.BinaryCrossentropy(), 
                      optimizer=optimizer, 
                      metrics=metrics)    
    return model
import tensorflow as tf
from tensorflow import keras


tf.random.set_seed(42)


def add_dense_layer(prev_layer, neurons_qty, dropout_rate, l2_rate, config: dict, add_dropout: bool = False, add_batch_norm: bool = False):
    if add_dropout:
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
        conv1d_1 = keras.layers.Conv1D(256, 150, 
                                       activation='relu', padding='same', 
                                       kernel_initializer=config["kernel_initializer"],
                                      ) (batch_norm_CNN_1)
        max_pooling1d_1 = keras.layers.MaxPooling1D(pool_size=2) (conv1d_1)
        flatten_cnn = keras.layers.Flatten() (max_pooling1d_1)

        previous_layer = flatten_cnn
        previous_layer = add_dense_layer(
            previous_layer, 128, 0.1, 1e-4, config, add_dropout=True, add_batch_norm=True)
        previous_layer = add_dense_layer(
            previous_layer, 128, 0.1, 1e-4, config, add_dropout=True, add_batch_norm=True)
        previous_layer = add_dense_layer(
            previous_layer, 128, 0.1, 1e-4, config, add_dropout=True, add_batch_norm=True)
        previous_layer = add_dense_layer(
            previous_layer, 128, 0.1, 1e-4, config, add_dropout=True, add_batch_norm=True)
        previous_layer = add_dense_layer(
            previous_layer, 256, 0.1, 1e-4, config, add_dropout=True, add_batch_norm=True)
        previous_layer = add_dense_layer(
            previous_layer, 256, 0.1, 1e-4, config, add_dropout=True, add_batch_norm=True)
        previous_layer = add_dense_layer(
            previous_layer, 256, 0.1, 1e-4, config, add_dropout=True, add_batch_norm=True)
        previous_layer = add_dense_layer(
            previous_layer, 256, 0.1, 1e-4, config, add_dropout=True, add_batch_norm=True)
        batch_norm_CNN_5 = keras.layers.BatchNormalization() (previous_layer)
        flatten_cnn = keras.layers.Flatten() (batch_norm_CNN_5)

        # Statistical features input
        input_dense = keras.Input(shape=config['statistical_input_shape'])
        batch_norm_dense_1 = keras.layers.BatchNormalization() (input_dense)
        dense_1 = keras.layers.Dense(128, activation='relu', 
                                     kernel_initializer=config["kernel_initializer"],
                                    ) (batch_norm_dense_1)
        batch_norm_dense_2 = keras.layers.BatchNormalization() (dense_1)
        flatten_dense = keras.layers.Flatten() (batch_norm_dense_2)


        # concatenation layer
        concat = keras.layers.Concatenate() ([flatten_cnn, flatten_dense])
        pre_output_dense = add_dense_layer(
            concat, 128, 0.0, 1e-4, config, add_dropout=True, add_batch_norm=True)
        pre_output_dense = add_dense_layer(
            pre_output_dense, 256, 0.2, 1e-4, config, add_dropout=True, add_batch_norm=True)
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
            model.compile(loss=keras.losses.CategoricalCrossentropy(), 
                          optimizer=keras.optimizers.Adam(learning_rate=1e-1), 
                          metrics=metrics)
        else:
            model.compile(loss=keras.losses.BinaryCrossentropy(), 
                          optimizer=keras.optimizers.Adam(learning_rate=cust_lr_scheduler), 
                          metrics=metrics)
    return model
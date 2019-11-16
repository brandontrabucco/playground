"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf


def dense(
    input_size,
    output_size,
    hidden_size=400,
    num_hidden_layers=2,
    output_activation=None
):
    # construct a dense neural network using the keras functional API
    visible = tf.keras.layers.Input(shape=(input_size,))
    hidden = visible

    # build many hidden layers
    for i in range(num_hidden_layers):
        hidden = tf.keras.layers.Dense(
            hidden_size,
            activation="relu",
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=(1.0 / 3.0),
                mode='fan_in',
                distribution='uniform'))(hidden)

    # build an output layer
    outputs = tf.keras.layers.Dense(
        output_size,
        activation=output_activation,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.003, maxval=0.003))(hidden)

    # finally build the model
    return tf.keras.models.Model(inputs=visible, outputs=outputs)
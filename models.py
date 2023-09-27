import keras.backend
import tensorflow as tf


def MyModel(hp_dict, input_shape, embedding_input_dim=None):
    # Build model
    inputs = tf.keras.Input(shape=input_shape)

    x = inputs

    if embedding_input_dim is None:
        x = x[:, :6]
    else:
        rest = x[:, :6]
        league_id = x[:, -1:]
        league_id = tf.keras.layers.Embedding(input_dim=embedding_input_dim, output_dim=1)(league_id)
        x = tf.concat([rest, league_id[:, :, 0]], axis=1)

    x = tf.keras.layers.Dense(hp_dict['num_units1'])(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Dense(3, use_bias=False, activation='softmax')(x)

    outputs = x

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="my_model")
    return model


def MyModel2(hp_dict, input_shape, embedding_input_dim=None):
    # Build model
    inputs = tf.keras.Input(shape=input_shape)

    x = inputs

    if embedding_input_dim is None:
        x = x[:, :6]
    else:
        rest = x[:, :6]
        league_id = x[:, -1:]
        league_id = tf.keras.layers.Embedding(input_dim=embedding_input_dim, output_dim=1)(league_id)
        x = tf.concat([rest, league_id[:, :, 0]], axis=1)

    x = tf.keras.layers.Dense(hp_dict['num_units1'])(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense(2, use_bias=False)(x)

    outputs = x

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="my_model")
    return model
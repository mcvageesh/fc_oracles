import numpy as np
import tensorflow as tf

def rps(outcome, probs):
    cum_probs = np.cumsum(probs)
    cum_outcomes = np.cumsum(outcome)

    sum_rps = 0
    for i in range(outcome.shape[0]):
        sum_rps += (cum_probs[i] - cum_outcomes[i]) ** 2

    return sum_rps / (outcome.shape[0] - 1)


def avg_rps(y_true, y_pred):
    total_rps = 0
    num_samples = y_true.shape[0]
    for i in range(num_samples):
        total_rps += rps(y_true[i], y_pred[i])
    return total_rps / num_samples


def ranked_probability_loss(y_true, y_pred):

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Compute the cumulative sum of predicted probabilities and actual outcomes
    cum_probs = tf.math.cumsum(y_pred, axis=1)
    cum_outcomes = tf.math.cumsum(y_true, axis=1)

    # Compute the RPS for each sample in the batch
    rps_values = tf.reduce_sum(tf.math.squared_difference(cum_probs, cum_outcomes), axis=1) / 2.0

    # Compute the average RPS across the batch
    avg_rps = tf.reduce_mean(rps_values)

    return avg_rps


def parse_boolean(value):
    value = value.lower()

    if value in ["true", "yes", "True", "1", "t"]:
        return True
    elif value in ["false", "no", "False", "0", "f"]:
        return False

    return False

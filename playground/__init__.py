"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import multiprocessing as m
import numpy as np
import tensorflow as tf


PROCESS_IS_INITIALIZED = False


def maybe_initialize_process(use_gpu=True):
    global PROCESS_IS_INITIALIZED
    if not PROCESS_IS_INITIALIZED:
        PROCESS_IS_INITIALIZED = True

        # on startup ensure all processes are started using the spawn method
        # see https://github.com/tensorflow/tensorflow/issues/5448
        m.set_start_method('spawn', force=True)

        if use_gpu:
            # prevent any process from consuming all gpu memory
            for gpu in tf.config.experimental.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(gpu, True)

        else:
            # prevent any process from consuming any gpu memory
            tf.config.experimental.set_visible_devices([], 'GPU')


def nested_apply(
    function,
    *structures
):
    # apply a function to a nested structure of objects
    if (isinstance(structures[0], np.ndarray) or
            isinstance(structures[0], tf.Tensor) or not (
            isinstance(structures[0], list) or
            isinstance(structures[0], tuple) or
            isinstance(structures[0], set) or
            isinstance(structures[0], dict))):
        return function(*structures)

    elif isinstance(structures[0], list):
        return [
            nested_apply(
                function,
                *x,) for x in zip(*structures)]

    elif isinstance(structures[0], tuple):
        return tuple(
            nested_apply(
                function,
                *x,) for x in zip(*structures))

    elif isinstance(structures[0], set):
        return {
            nested_apply(
                function,
                *x,) for x in zip(*structures)}

    elif isinstance(structures[0], dict):
        keys_list = structures[0].keys()
        values_list = [[y[key] for key in keys_list] for y in structures]
        return {
            key: nested_apply(
                function,
                *values) for key, values in zip(keys_list, zip(*values_list))}


def discounted_sum(
    terms,
    discount_factor
):
    # compute discounted sum of rewards across terms using discount_factor
    weights = tf.tile([[discount_factor]], [1, tf.shape(terms)[1]])
    weights = tf.math.cumprod(weights, axis=1, exclusive=True)
    return tf.math.cumsum(
        terms * weights, axis=1, reverse=True) / weights

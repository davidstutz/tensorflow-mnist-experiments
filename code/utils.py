"""
A collection of utilities.
"""

import tensorflow as tf

def count_elements(name, x):
    """
    Count the number of elements in the given tensor.

    :param name: scope name
    :type name: str
    :param x: input tensor
    :type x: tensorflow.Tensor
    :return: batch normalization tensor
    :rtype: tensorflow.Tensor
    """

    with tf.name_scope(name):
        return tf.reduce_sum(tf.ones_like(x))

def moments(name, x, dimensions):
    """
    Compute mean and variance for the given tensor along the given dimensions.

    :param name: scope name
    :type name: str
    :param x: input tensor
    :type x: tensorflow.Tensor
    :param dimensions: list of dimensions to compute moments over
    :type dimensions: [int]
    :return: moments tensors
    :rtype: (tensorflow.Tensor, tensorflow.Tensor)
    """

    with tf.name_scope(name):
        sum = tf.reduce_sum(x, dimensions)
        squared_sum = tf.reduce_sum(tf.mul(x, x), dimensions)
        elements = count_elements('elements', x)/count_elements('sum_elements', sum)

        mean = tf.div(sum, elements)
        variance = tf.sub(tf.div(squared_sum, elements), tf.mul(mean, mean))

        return mean, variance


def get_kwargs(kwargs, key, default):
    """
    Get an element in kwargs or returnt the default.

    :param kwargs: dictionary of keyworded arguments
    :type kwargs: dict
    :param key: key to retrieve
    :type key: str
    :param default: default value to return
    :type default: mixed
    :return: the retrieved value from kwargs or default
    :rtype: mixed
    """

    if kwargs is not None:
        if key in kwargs.keys():
            return kwargs[key]

    return default
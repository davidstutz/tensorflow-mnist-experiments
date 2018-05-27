"""
Set of regularizers to use in addition to the loss.
"""

import tensorflow as tf
import numpy as np

def none(name, weights):
    return 0

def l2(name, weights):
    """
    L2 regularizer on weights.

    :param name: name of variable
    :type name: str
    :param weights: list of weight variables
    :type weights: [tensorflow.Tensor]
    :return: regularizer
    :rtype: tensorflow.Tensor
    """

    with tf.name_scope(name):
        regularizer = np.float32(0.0)
        for weight in weights:
            tf.add(regularizer, tf.nn.l2_loss(weight))

        return regularizer

def l1(name, weights):
    """
    L1 regularizer on weights.

    :param name: name of variable
    :type name: str
    :param weights: list of weight variables
    :type weights: [tensorflow.Tensor]
    :return: regularizer
    :rtype: tensorflow.Tensor
    """

    with tf.name_scope(name):
        regularizer = np.float32(0.0)
        for weight in weights:
            tf.add(regularizer, tf.nn.l1_loss(weight))

        return regularizer

def orthonormal(name, weights):
    """
    Orthonormal regularizer.

    :param name: name of variable
    :type name: str
    :param weights: list of weight variables
    :type weights: [tensorflow.Tensor]
    :return: regularizer
    :rtype: tensorflow.Tensor
    """

    with tf.name_scope(name):
        regularizer = np.float32(0.0)
        for weight in weights:
            shape = weight.get_shape().as_list()
            if len(shape) == 4:
                matrix = tf.reshape(weight, (shape[3], shape[0]*shape[1]*shape[2]))
            elif len(shape) == 2:
                matrix = weight
            else:
                raise NotImplemented("orthonormal regularizer only supports variables of dimension 2 or 4")

            identity = tf.constant(np.identity(matrix.get_shape().as_list()[1]), dtype = tf.float32)
            difference = tf.sub(tf.matmul(tf.transpose(matrix), matrix), identity)
            tf.add(regularizer, tf.sqrt(tf.reduce_sum(tf.mul(difference, difference))))

        return regularizer
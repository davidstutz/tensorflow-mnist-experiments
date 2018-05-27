"""
Set of initializers including Xavier and orthonormal initialization.
"""

import tensorflow as tf
import numpy as np
import utils
import math

def constant(name, shape, **kwargs):
    """
    Initialize a constant variable.

    :param name: name of variable
    :type name: str
    :param shape: variable shape
    :type shape: (int, int, int, int)

    :Keyword Arguments:
        * *value* (float) -- constant value (0.1)

    :return: initialized variable
    :rtype: tensorflow.Variable
    """

    value = utils.get_kwargs(kwargs, 'value', 0.5)

    return tf.Variable(tf.constant(value), name = name)

def truncated_normal(name, shape, **kwargs):
    """
    Initialize a truncated normal variable, i.e. truncated after 2 standard deviations.

    :param name: name of variable
    :type name: str
    :param shape: variable shape
    :type shape: (int, int, int, int)

    :Keyword Arguments:
        * *stddev* (float) -- standard deviation (0.1)
        * *mean* (float) -- mean (0.0)

    :return: initialized variable
    :rtype: tensorflow.Variable
    """

    stddev = utils.get_kwargs(kwargs, 'stddev', 0.1)
    mean = utils.get_kwargs(kwargs, 'mean', 0.0)

    return tf.Variable(tf.truncated_normal(shape, stddev = stddev, mean = mean), name = name)

def uniform_unit_scale(name, shape, **kwargs):
    """
    Initialize using tf.uniform_unit_scaling_initializer.

    :param name: name of variable
    :type name: str
    :param shape: variable shape
    :type shape: (int, int, int, int)

    :Keyword Arguments:
        * *factor* (float) -- factor (0.1)

    :return: initialized variable
    :rtype: tensorflow.Variable
    """

    factor = utils.get_kwargs(kwargs, 'factor', 1.0)

    return tf.get_variable(name, shape = shape, initializer = tf.uniform_unit_scaling_initializer(factor = factor))

def heuristic(name, shape):
    """
    Initialize variables by choosing random numbers with variance :math:`\frac{1}{n_in}`.

    :param name: name of variable
    :type name: str
    :param shape: variable shape
    :type shape: (int, int, int, int)
    :return: initialized variable
    :rtype: tensorflow.Variable
    """

    if len(shape) == 4: # convolutional layer
        std = math.sqrt(1.0/(shape[0]*shape[1]*shape[2]))
    elif len(shape) == 2:
        std = math.sqrt(1.0/(shape[0]))
    else:
        raise NotImplementedError("huristic initialization only supports variables of dimension 2 or 4, %d requested" % len(shape))

    return tf.get_variable(name, shape = shape, initializer = tf.random_normal_initializer(0.0, std));

def xavier(name, shape, **kwargs):
    """
    Initialize variables by choosing random numbers with variance :math:`\frac{2}{n_in + n_out}`.

    :param name: name of variable
    :type name: str
    :param shape: variable shape
    :type shape: (int, int, int, int)
    :return: initialized variable
    :rtype: tensorflow.Variable
    """

    if len(shape) == 4: # convolutional layer
        std = math.sqrt(2.0/(shape[0]*shape[1]*shape[2] + shape[3]))
    elif len(shape) == 2:
        std = math.sqrt(2.0/(shape[0] + shape[1]))
    else:
        raise NotImplementedError("xavier initialization only supports variables of dimension 2 or 4, %d requested" % len(shape))

    return tf.get_variable(name, shape = shape, initializer = tf.random_normal_initializer(0.0, std));

def orthonormal(name, shape, **kwargs):
    """
    Orthonormal initialization following http://hjweide.github.io/orthogonal-initialization-in-convolutional-layers.

    :param name: name of variable
    :type name: str
    :param shape: variable shape
    :type shape: (int, int, int, int)
    :return: initialized variable
    :rtype: tensorflow.Variable
    """

    if len(shape) == 4:
        flat_shape = (shape[3], shape[0]*shape[1]*shape[2])
        X = np.random.random(flat_shape)
        U, _, V = np.linalg.svd(X, full_matrices = False, compute_uv = True)
        Q = U if U.shape == flat_shape else V
        weights = Q.reshape(shape).astype(np.float32)
    elif len(shape) == 2:
        X = np.random.random((shape[1], shape[0]))
        U, _, V = np.linalg.svd(X, full_matrices = False, compute_uv = True)
        np.allclose(np.dot(V, V.T), np.eye(V.shape[0]))
        weights = V.T.astype(np.float32)
    else:
        raise NotImplemented("orthonormal initialization only supports variables of dimension 2 or 4")

    return tf.Variable(weights, name = name)
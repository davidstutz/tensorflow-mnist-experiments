"""
A collection of common (convolutional) neural network layers for MNIST classification.
"""

import tensorflow as tf
import utils

def identity(name, x):
    """
    Identity layer.

    :param name: scope name
    :type name: str
    :param x: input tensor
    :type x: tensorflow.Tensor
    :return: identity product tensor
    :rtype: tensorflow.Tensor
    """

    with tf.name_scope(name):
        return tf.identity(x)

def inner_product(name, x, W, b):
    """
    Inner product, i.e. fully connected, layer.

    :param name: scope name
    :type name: str
    :param x: input tensor
    :type x: tensorflow.Tensor
    :param W: weight tensor
    :type W: tensorflow.Tensor
    :param b: bias tensor
    :type b: tensorflow.Tensor
    :return: inner product tensor
    :rtype: tensorflow.Tensor
    """

    with tf.name_scope(name):
        return tf.add(tf.matmul(x, W), b)

def convolutional(name, x, W, b, **kwargs):
    """
    Convolutional layer.

    :param name: scope name
    :type name: str
    :param x: input tensor
    :type x: tensorflow.Tensor
    :param W: weight tensor
    :type W: tensorflow.Tensor
    :param b: bias tensor
    :type b: tensorflow.Tensor

    :Keyword Arguments:
        * *strides* ([int, int, int, int]) -- strides in each dimension ([1, 1, 1, 1])
        * *padding* (str) -- padding type, 'SAME' or 'VALID' ('SAME')

    :return: convolutional tensor
    :rtype: tensorflow.Tensor
    """

    strides = utils.get_kwargs(kwargs, 'strides', [1, 1, 1, 1])
    padding = utils.get_kwargs(kwargs, 'padding', 'SAME')

    assert len(strides) ==4
    for stride in strides:
        assert stride > 0

    assert padding == 'SAME' or padding == 'VALID'

    with tf.name_scope(name):
        return tf.add(tf.nn.conv2d(x, W, strides = strides, padding = padding), b)

def pooling(name, x, **kwargs):
    """
    Pooling layer.

    :param name: scope name
    :type name: str
    :param x: input tensore
    :type x: tensorflow.Tensor

    :Keyword Arguments:
        * *type* (str) -- pooling type, 'MAX' or 'AVG' ('MAX')
        * *ksize* ([int, int, int, int]) -- sizes for pooling ([1, 2, 2, 1])
        * *strides* ([int, int, int, int]) -- strides in each dimension ([1, 2, 2, 1])
        * *padding* (str) -- padding type, 'SAME' or 'VALID' ('SAME')

    :return: pooling tensor
    :rtype: tensorflow.Tensor
    """

    type = utils.get_kwargs(kwargs, 'type', 'MAX')
    ksize = utils.get_kwargs(kwargs, 'ksize', [1, 2, 2, 1])
    strides = utils.get_kwargs(kwargs, 'strides', [1, 2, 2, 1])
    padding = utils.get_kwargs(kwargs, 'padding', 'SAME')

    assert len(ksize) == 4
    for size in ksize:
        assert size > 0

    assert len(strides) == 4
    for stride in strides:
        assert stride > 0

    assert type == 'MAX' or type == 'AVG'
    assert padding == 'SAME' or padding == 'VALID'

    with tf.name_scope(name):
        return tf.nn.max_pool(x, ksize = ksize, strides = strides, padding = padding)

def batch_normalization_cpu(name, x, **kwargs):
    """
    Batch normalization layer.

    :param name: scope name
    :type name: str
    :param x: input tensor
    :type x: tensorflow.Tensor

    :Keyword Arguments:
        * *variance_epsilon* (float) -- epsilon to add to variance before dividing (0.0)

    :return: batch normalization tensor
    :rtype: tensorflow.Tensor
    """

    variance_epsilon = utils.get_kwargs(kwargs, 'variance_epsilon', 0.0)
    assert variance_epsilon >= 0

    with tf.name_scope(name):
        offset = tf.Variable(tf.constant(0.0, shape = [x.get_shape().as_list()[-1]]), name = 'offset', trainable = True)
        scale = tf.Variable(tf.constant(1.0, shape = [x.get_shape().as_list()[-1]]), name = 'scale', trainable = True)

        # Convolutional layer:
        if len(x.get_shape().as_list()) == 4:
            # this call won't work on GPU
            mean, variance = tf.nn.moments(x, [0, 1, 2], name = 'moments')
        # Fully connected layer:
        else:
            mean, variance = tf.nn.moments(x, [0, 1], name = 'moments')

        return tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon)

def batch_normalization(name, x, **kwargs):
    """
    Batch normalization layer.

    :param name: scope name
    :type name: str
    :param x: input tensor
    :type x: tensorflow.Tensor

    :Keyword Arguments:
        * *variance_epsilon* (float) -- epsilon to add to variance before dividing (0.0)

    :return: batch normalization tensor
    :rtype: tensorflow.Tensor
    """

    variance_epsilon = utils.get_kwargs(kwargs, 'variance_epsilon', 0.0)
    assert variance_epsilon >= 0

    with tf.name_scope(name):
        offset = tf.Variable(tf.constant(0.0, shape = [x.get_shape().as_list()[-1]]), name = 'offset', trainable = True)
        scale = tf.Variable(tf.constant(1.0, shape = [x.get_shape().as_list()[-1]]), name = 'scale', trainable = True)

        # Convolutional layer:
        if len(x.get_shape().as_list()) == 4:
            mean, variance = utils.moments('moments', x, [0, 1, 2])
        # Fully connected layer:
        else:
            mean, variance = utils.moments('moments', x, [0, 1])

        return tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon)

def layer_normalization_cpu(name, x, **kwargs):
    """
    Layer normalization layer.

    :param name: scope name
    :type name: str
    :param x: input tensor
    :type x: tensorflow.Tensor
    :return: layer normalization tensor
    :rtype: tensorflow.Tensor
    """

    with tf.name_scope(name):
        offset = tf.Variable(tf.constant(0.0, shape = [x.get_shape().as_list()[-1]]), name = 'offset', trainable = True)
        scale = tf.Variable(tf.constant(1.0, shape = [x.get_shape().as_list()[-1]]), name = 'scale', trainable = True)

        # Cnvolutional layer:
        if len(x.get_shape().as_list()) == 4:
            mean, variance = tf.nn.moments(x, [1, 2], name = 'moments')
            mean = tf.expand_dims(tf.expand_dims(mean, 1), 2)
            variance = tf.expand_dims(tf.expand_dims(variance, 1), 2)
        # Fully conencted layer:
        else:
            mean, variance = tf.nn.moments(x, [1], name = 'moments')
            mean = tf.expand_dims(mean, 1)
            variance = tf.expand_dims(variance, 1)

        return tf.add(tf.mul(scale, tf.div(tf.sub(x, mean), variance)), offset)

def layer_normalization(name, x, **kwargs):
    """
    Layer normalization layer.

    :param name: scope name
    :type name: str
    :param x: input tensor
    :type x: tensorflow.Tensor
    :return: layer normalization tensor
    :rtype: tensorflow.Tensor
    """

    with tf.name_scope(name):
        offset = tf.Variable(tf.constant(0.0, shape = [x.get_shape().as_list()[-1]]), name = 'offset', trainable = True)
        scale = tf.Variable(tf.constant(1.0, shape = [x.get_shape().as_list()[-1]]), name = 'scale', trainable = True)

        # Cnvolutional layer:
        if len(x.get_shape().as_list()) == 4:
            mean, variance = utils.moments('moments', x, [1, 2]) # add axis 0 here to get basically batch normalization
            mean = tf.expand_dims(tf.expand_dims(mean, 1), 2)
            variance = tf.expand_dims(tf.expand_dims(variance, 1), 2)
        # Fully conencted layer:
        else:
            mean, variance = utils.moments('moments', x, [1])  # add axis 0 here to get basically batch normalization
            mean = tf.expand_dims(mean, 1)
            variance = tf.expand_dims(variance, 1)

        return tf.add(tf.mul(scale, tf.div(tf.sub(x, mean), variance)), offset)

def min_max_normalization_symmetric(name, x, **kwargs):
    """
    Min max normalization to [-1, 1]

    :param name: scope name
    :type name: str
    :param x: input tensor
    :type x: tensorflow.Tensor
    :return: min max normalization tensor
    :rtype: tensorflow.Tensor
    """

    with tf.name_scope(name):
        # Convolutional layer:
        if len(x.get_shape().as_list()) == 4:
            min = tf.reduce_min(x, [1, 2])
            min = tf.expand_dims(tf.expand_dims(min, 1), 2)

            max = tf.reduce_max(x, [1, 2])
            max = tf.expand_dims(tf.expand_dims(max, 1), 2)
        # Fully connected layer:
        else:
            min = tf.reduce_min(x, [1])
            min = tf.expand_dims(min, 1)

            max = tf.reduce_max(x, [1])
            max = tf.expand_dims(max, 1)

        return tf.sub(tf.mul(tf.div(tf.sub(x, min), tf.sub(max, min)), 2), 1)

def min_max_normalization(name, x, **kwargs):
    """
    Min max normalization.

    :param name: scope name
    :type name: str
    :param x: input tensor
    :type x: tensorflow.Tensor
    :return: min max normalization tensor
    :rtype: tensorflow.Tensor
    """

    with tf.name_scope(name):
        # Convolutional layer:
        if len(x.get_shape().as_list()) == 4:
            min = tf.reduce_min(x, [1, 2])
            min = tf.expand_dims(tf.expand_dims(min, 1), 2)

            max = tf.reduce_max(x, [1, 2])
            max = tf.expand_dims(tf.expand_dims(max, 1), 2)
        # Fully connected layer:
        else:
            min = tf.reduce_min(x, [1])
            min = tf.expand_dims(min, 1)

            max = tf.reduce_max(x, [1])
            max = tf.expand_dims(max, 1)

        return tf.div(tf.sub(x, min), tf.sub(max, min))

def max_normalization(name, x, **kwargs):
    """
    Max normalization layer.

    :param name: scope name
    :type name: str
    :param x: input tensor
    :type x: tensorflow.Tensor
    :return: max normalization tensor
    :rtype: tensorflow.Tensor
    """

    with tf.name_scope(name):
        # Convolutional layer:
        if len(x.get_shape().as_list()) == 4:
            max = tf.reduce_max(x, [1, 2])
            max = tf.expand_dims(tf.expand_dims(max, 1), 2)
        # Fully connected layer:
        else:
            max = tf.reduce_max(x, [1])
            max = tf.expand_dims(max, 1)

        return tf.div(x, max)

def dropout(name, x, **kwargs):
    """
    Dropout layer.

    :param name: scope name
    :type name: str
    :param x: input tensor
    :type x: tensorflow.Tensor

    :Keyword Arguments:
        * *keep_prob* (float) -- keeping probability per unit (0.5)

    :return: dropout tensor
    :rtype: tensorflow.Tensor
    """

    keep_prob = utils.get_kwargs(kwargs, 'keep_prob', 0.5)
    assert keep_prob >= 0 and keep_prob <= 1

    with tf.name_scope(name):
        return tf.nn.dropout(x, keep_prob)

def tanh(name, x):
    """
    Hyperbolic tangent layer.

    :param name: scope name
    :type name: str
    :param x: input tensor
    :type x: tensorflow.Tensor
    """

    with tf.name_scope(name):
        return tf.nn.tanh(x)

def relu(name, x):
    """
    ReLU layer.

    :param name: scope name
    :type name: str
    :param x: input tensor
    :type x: tensorflow.Tensor
    """

    with tf.name_scope(name):
        return tf.nn.relu(x)

def sigmoid(name, x):
    """
    Sigmoid layer.

    :param name: scope name
    :type name: str
    :param x: input tensor
    :type x: tensorflow.Tensor
    """

    with tf.name_scope(name):
        return tf.nn.sigmoid(x)

def softplus(name, x):
    """
    Softplus layer.

    :param name: scope name
    :type name: str
    :param x: input tensor
    :type x: tensorflow.Tensor
    """

    with tf.name_scope(name):
        return tf.nn.softplus(x)

def softsign(name, x):
    """
    Softsign layer.

    :param name: scope name
    :type name: str
    :param x: input tensor
    :type x: tensorflow.Tensor
    """

    with tf.name_scope(name):
        return tf.nn.softsign(x)

def softmax(name, x, **kwargs):
    """
    Softmax layer.

    :param name: scope name
    :type name: str
    :param x: input tensor
    :type x: tensorflow.Tensor

    :Keyword Arguments:
        * *epsilon* (float) -- epsilon to add to denominator (1e-12)

    :return: softmax tensor
    :rtype: tensorflow.Tensor
    """

    epsilon = utils.get_kwargs(kwargs, 'epsilon', 1e-12)
    assert epsilon >= 0

    with tf.name_scope(name):
        return tf.div(tf.exp(x), tf.add(tf.reduce_sum(tf.exp(x)), epsilon))
        #return tf.nn.softmax(x)

def cross_entropy(name, x, y):
    """
    Cross entropy loss.

    :param name: scope name
    :type name: str
    :param x: input tensor, usually softmax
    :type: tensorflow.Tensor
    :return: cross entropy tensor
    :rtype: tensorflow.Tensor
    """

    with tf.name_scope(name):
        return -tf.reduce_mean(y * tf.log(tf.clip_by_value(x, 1e-10, 1.0)))
        #return tf.reduce_mean(tf.add(tf.mul(y, tf.log(x)), tf.mul(tf.sub(tf.ones_like(y), y), tf.log(tf.sub(tf.ones_like(x), x)))))

def prediction(name, x):
    """
    Prediction layer, usually based on softmax.

    :param name: scope name
    :type name: str
    :param x: input tensor
    :type x: tensorflow.Tensor
    :return: prediction tensor
    :rtype: tensorflow.Tensor
    """
    with tf.name_scope(name):
        return tf.cast(tf.arg_max(x, 1), tf.float32)

def label(name, x):
    """
    Label from hot-one encoding.

    :param name: scope name
    :type name: str
    :param x: input tensor
    :type x: tensorflow.Tensor
    :return: label tensor
    :rtype: tensorflow.Tensor
    """

    with tf.name_scope(name):
        return tf.cast(tf.arg_max(x, 1), tf.float32)

def accuracy(name, x, y):
    """
    Accuracy layer, assumes y is the actual label, not the one-hot encoding.
    User :func:`layers.prediction` to get the label form the one-hot encoding
    or the network softmax prediction.

    :param name: scope name
    :type name: str
    :param x: input tensor
    :type x: tensorflow.Tensor
    :return: accuracy tensor
    :rtype: tensorflow.Tensor
    """

    with tf.name_scope(name):
        return tf.reduce_mean(tf.cast(tf.equal(y, tf.cast(x, tf.float32)), tf.float32))
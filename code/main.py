"""
Train and test a (convolutional) neural network on MNIST.
"""

import tensorflow as tf
import numpy as np
import argparse
import initializers
import regularizers
import transformer
import dataset
import layers
import mnist
import plots
import os

def get_parser():
    """
    Returns the parser for the command line tool.

    :return: parser
    :rtype: argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser(description = 'MNIST initialization and normalization benchmark.')
    parser.add_argument('-epochs', dest = 'epochs', type = int,
                        help = 'Number of epochs; i.e. number of iterations over the whole dataset.',
                        default = 1)
    parser.add_argument('-layers', dest = 'layers', type = int,
                        help = 'Number of (convolutional) layers (up to 7). The model has layers + 2 layers in total.',
                        default = 4)
    parser.add_argument('-activation', dest = 'activation', type = str,
                        help = 'Activation function to use for all models.',
                        default = 'relu')
    parser.add_argument('-initializer', dest='initializer', type=str,
                        help = 'Initialization to use.',
                        default = 'truncated_normal')
    parser.add_argument('-normalizer', dest = 'normalizer', type = str,
                        help = 'Normalization to use.',
                        default = 'orthonormal')
    parser.add_argument('-cnn', dest = 'cnn', type = bool,
                        help = 'Whether to use CNNs for the benchmark.',
                        default = True)
    parser.add_argument('-bias', dest = 'bias', type = float,
                        help = 'Constant bias value.',
                        default = 0.0)
    parser.add_argument('-regularizer', dest = 'regularizer', type = str,
                        help = 'Regularizer to use.',
                        default = 'orthonormal')
    parser.add_argument('-regularizer_weight', dest = 'regularizer_weight', type = float,
                        help = 'Regularizer\'s weight in loss.',
                        default = 0.005)
    parser.add_argument('-batch_size', dest = 'batch_size', type = int,
                        help = 'Batch size to use.',
                        default = 32)

    return parser

def main():
    """
    Train and evaluate the model with the chosen parameters.
    """

    test_images, test_labels = mnist.load('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')
    train_images, train_labels = mnist.load('train-images.idx3-ubyte', 'train-labels.idx1-ubyte')

    test_images = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2], 1)
    train_images = train_images.reshape(train_images.shape[0], train_images.shape[1], train_images.shape[2], 1)

    train_dataset = dataset.Dataset(train_images, train_labels, args.batch_size)

    # Used for testing on trainng and testing set as GPU does not take the full training/
    # testing sets in memory.
    test_train_dataset = dataset.Dataset(train_images, train_labels, 1000)
    test_dataset = dataset.Dataset(test_images, test_labels, 1000)

    with tf.device('/gpu:0'):
        if args.cnn:
            x = tf.placeholder(tf.float32, shape = [None, 28, 28, 1])
        else:
            x = tf.placeholder(tf.float32, shape = [None, 784])

        y = tf.placeholder(tf.float32, shape = [None, 10])

        # Two convolutional layers if requested, otherwise fully connected layers.
        if args.cnn:

            kernel_size = [3, 3, 3, 3, 3, 3, 3]

            channels = [16]
            for i in range(1, args.layers):
                channels.append(max(channels[-1]*2, 64))

            #if args.layers % 2 == 0:
            #    channels[args.layers//2 - 1] = 128
            #    channels[args.layers//2] = 128;

            #    for i in range(args.layers//2 - 1, -1, -1):
            #        channels[i] = channels[i + 1]//2
            #        channels[args.layers - i - 1] = channels[args.layers - i - 2]//2
            #else:
            #    channels[args.layers//2] = 128

            #    for i in range(args.layers//2 - 1, -1, -1):
            #        channels[i] = channels[i + 1]//2
            #        channels[args.layers - i - 1] = channels[args.layers - i - 2]//2

            # weights and biases
            W = []
            b = []

            # conv, activations, pooling and normalizes
            s = []
            h = []
            p = []
            n = [x]

            for i in range(args.layers):

                channels_in = 1
                if i > 0:
                    channels_in = channels[i - 1]

                W.append(initializer('W_conv' + str(i), [kernel_size[i], kernel_size[i], channels_in, channels[i]]))
                b.append(initializers.constant('b_conv' + str(i), [channels[i]], value = args.bias))

                s.append(layers.convolutional('s_' + str(i), n[-1], W[-1], b[-1]))
                h.append(activation('h_' + str(i), s[-1]))
                p.append(layers.pooling('p_' + str(i), h[-1]))
                n.append(normalizer('n_' + str(i), p[-1]))

            shape = n[-1].get_shape().as_list()
            n[-1] = tf.reshape(n[-1], [-1, shape[1]*shape[2]*shape[3]])
        else:

            units = [1000, 1000, 1000, 1000, 1000]

            # weights and biases
            W = []
            b = []

            # linear, activations and normalized
            s = []
            h = []
            n = [x]

            for i in range(args.layers):
                units_in = 784
                if i > 0:
                    units_in = units[i - 1]

                W.append(initializer('W_fc' + str(i), [units_in, units[i]]))
                b.append(initializers.constant('b_fc' + str(i), [units[i]], value = args.bias))

                s.append(layers.inner_product('s_' + str(i), n[-1], W[-1], b[-1]))
                n.append(activation('h_' + str(i), s[-1]))
                n.append(normalizer('n_' + str(i), h[-1]))

        W.append(initializer('W_fc3', [n[-1].get_shape().as_list()[1], 100]))
        b.append(initializers.constant('b_fc3', [100], value = args.bias))

        s.append(layers.inner_product('s_3', n[-1], W[-1], b[-1]))
        h.append(activation('h_3', s[-1]))
        n.append(normalizer('n_3', h[-1]))

        W.append(initializer('W_fc4', [100, 10]))
        b.append(initializers.constant('b_fc4', [10], value = args.bias))

        s.append(layers.inner_product('s_4', n[-1], W[-1], b[-1]))
        y_ = layers.softmax('y_', s[-1])

        # Loss definition and optimizer.
        cross_entropy = layers.cross_entropy('cross_entropy', y_, y)

        weights = [v for v in tf.all_variables() if v.name.startswith('W')]
        loss = cross_entropy + args.regularizer_weight*regularizer('regularizer', weights)

        prediction = layers.prediction('prediction', y_)
        label = layers.label('label', y)
        accuracy = layers.accuracy('accuracy', prediction, label)
        optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

        tformer = transformer.Transformer()
        tformer.convert(np.float32)
        tformer.scale(1./255.)

        # Reshape input if necessary.
        if not args.cnn:
            tformer.reshape([None, 784])

        with tf.Session(config = tf.ConfigProto(log_device_placement = True)) as sess:
            sess.run(tf.initialize_all_variables())

            # First run to visualize variables and backpropagated gradients.
            batch = train_dataset.next_batch()
            res = sess.run([optimizer] + W + h,
                           feed_dict = {x: tformer.process(batch[0]), y: batch[1]})

            # Get all the weights for the plot.
            weights = {}
            for i in range(args.layers + 1):
                weights['W_' + str(i)] = res[1 + i]

            plots.plot_normalized_histograms(weights, 'images/' + name + '_weights_t0.png')

            # Get all the activations for the plot.
            activations = {}
            for i in range(args.layers + 1):
                activations['h_' + str(i)] = res[1 + args.layers + 1 + i]

            plots.plot_normalized_histograms(activations, 'images/' + name + '_activations_t0.png')

            losses = []
            batch_accuracies = []
            train_accuracies = []
            test_accuracies = []

            means = [[] for i in range(args.layers)]
            stds = [[] for i in range(args.layers)]

            iterations = args.epochs*(train_dataset.count//args.batch_size + 1)
            interval = iterations // 20
            for i in range(iterations):

                # If this is the last batch, we reshuffle after this step.
                was_last_batch = train_dataset.next_batch_is_last()

                batch = train_dataset.next_batch()
                res = sess.run([optimizer, cross_entropy, accuracy] + h,
                                feed_dict = {x: tformer.process(batch[0]), y: batch[1]})

                # Get mean and variance of all layer activations after normalization, i.e. after normalization
                for j in range(args.layers):
                    means[j].append(np.mean(res[3 + j]))
                    stds[j].append(np.std(res[3 + j]))

                losses.append(res[1])
                batch_accuracies.append(res[2])

                print('Loss [%d]: %f' % (i, res[1]))
                print('Batch accuracy [%d]: %f' % (i, res[2]))

                if was_last_batch:
                    train_dataset.shuffle()
                    train_dataset.reset()

                if i % interval == 0:

                    # Accuracy on training set.
                    train_accuracy = 0
                    batches = test_train_dataset.count // test_train_dataset.batch_size

                    for j in range(batches):
                        batch = test_train_dataset.next_batch()
                        train_accuracy += sess.run(accuracy, feed_dict = {x: tformer.process(batch[0]), y: batch[1]})

                    train_accuracy = train_accuracy / batches
                    train_accuracies.append(train_accuracy)

                    print('Train accuracy [%d]: %f' % (i, train_accuracy))
                    test_train_dataset.reset()

                    # Accuracy on testing set.
                    test_accuracy = 0
                    batches = test_dataset.count // test_dataset.batch_size

                    for j in range(batches):
                        batch = test_dataset.next_batch()
                        test_accuracy += sess.run(accuracy, feed_dict = {x: tformer.process(batch[0]), y: batch[1]})

                    test_accuracy = test_accuracy / batches
                    test_accuracies.append(test_accuracy)

                    print('Test accuracy [%d]: %f' % (i, test_accuracy))
                    test_dataset.reset()

                    # Plot loss for each iteration.
                    plots.plot_lines({'loss': (np.arange(0, i + 1), np.array(losses))},
                                     'images/' + name + '_loss.png')


                    statistics = {}
                    for j in range(args.layers):
                        statistics['h_' + str(j) + '_mean'] = (np.arange(0, i + 1), np.array(means[j]))
                        statistics['h_' + str(j) + '_std'] = (np.arange(0, i + 1), np.array(stds[j]))

                    # Plot activations for each iteration.
                    plots.plot_lines(statistics, 'images/' + name + '_activations.png')

                    # Plot measures accuracy every 250th iteration.
                    plots.plot_lines({
                        'batch_accuracy': (np.arange(0, i + 1), np.array(batch_accuracies)),
                        'train_accuracy': (np.arange(0, i + 1, interval), np.array(train_accuracies)),
                        'test_accuracy': (np.arange(0, i + 1, interval), np.array(test_accuracies)),
                    }, 'images/' + name + '_accuracy.png')

            sess.close()

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    activation = layers.tanh
    if args.activation == 'relu':
        activation = layers.relu
    elif args.activation == 'softsign':
        activation = layers.softsign

    initializer = initializers.truncated_normal
    if args.initializer == 'uniform_unit_scale':
        initializer = initializers.uniform_unit_scale
    elif args.initializer == 'heuristic':
        initializer = initializers.heuristic
    elif args.initializer == 'xavier':
        initializer = initializers.xavier

    normalizer = layers.identity
    if args.normalizer == 'batch_normalization':
        normalizer = layers.batch_normalization
    elif args.normalizer == 'layer_normalization':
        normalizer = layers.layer_normalization
    elif args.normalizer == 'min_max_normalization':
        normalizer = layers.min_max_normalization
    elif args.normalizer == 'min_max_normalization_symmetric':
        normalizer = layers.min_max_normalization_symmetric

    regularizer = regularizers.none
    if args.regularizer == 'l2':
        regularizer = regularizers.l2
    elif args.regularizer == 'l1':
        regularizer = regularizers.l1
    elif args.regularizer == 'orthonormal':
        regularizer = regularizers.orthonormal

    name = 'l' + str(args.layers) + 'b' + str(args.batch_size) \
            + '_' + args.activation + '_' + args.initializer
    if normalizer != 'identitiy':
        name += '_' + args.normalizer
    if args.regularizer_weight > 0 and args.regularizer != 'none':
        name += '_' + args.regularizer + str(args.regularizer_weight)
    if args.bias != 0.0:
        name += '_' + str(args.bias)
    if args.cnn:
        name += '_cnn'

    # Create directory for plots.
    if not os.path.exists('images/'):
        os.makedirs('images/')

    if args.layers < 0 or args.layers > 7:
        print('Only up to 5 layers supported!')
        exit(0)

    main()
"""
Benchmark different (convolutional) neural network configurations on MNIST. For each configuration, a
randomly initialized network is trained and tested on MNIST multiple times.
"""

from matplotlib import pyplot
import tensorflow as tf
import numpy as np
import argparse
import initializers
import regularizers
import transformer
import dataset
import layers
import mnist
import time
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
                        default = 1)
    parser.add_argument('-activation', dest = 'activation', type = str,
                        help = 'Activation function to use for all models.',
                        default = 'relu')
    parser.add_argument('-initializer', dest='initializer', type=str,
                        help = 'Initialization to use.',
                        default = 'truncated_normal')
    parser.add_argument('-normalizer', dest = 'normalizer', type = str,
                        help = 'Normalization to use.',
                        default = 'identity')
    parser.add_argument('-normalizer_data', dest = 'normalizer_data', type = bool,
                        help = 'Whether to apply the normalizer on the input data.',
                        default = False)
    parser.add_argument('-cnn', dest = 'cnn', type = bool,
                        help = 'Whether to use CNNs for the benchmark.',
                        default = True)
    parser.add_argument('-bias', dest = 'bias', type = float,
                        help = 'Constant bias value.',
                        default = 0.0)
    parser.add_argument('-regularizer', dest = 'regularizer', type = str,
                        help = 'Regularizer to use.',
                        default = 'none')
    parser.add_argument('-regularizer_weight', dest = 'regularizer_weight', type = float,
                        help = 'Regularizer\'s weight in loss.',
                        default = 0.05)
    parser.add_argument('-batch_size', dest = 'batch_size', type = int,
                        help = 'Batch size to use.',
                        default = 32)
    parser.add_argument('-repetitions', dest = 'repetitions', type = int,
                        help = 'Number of random repetitions of training.',
                        default = 3)
    return parser

def main():
    """
    Train a model with the given parameters.

    :return: number of iterations, interval used to measure accuracy, weight mean and standard
        deviation for each iteration and weight, activations in each layer for each iteration
        as mean and standard deviation, accuracies including batch, train and test accuracy
    :rtype: (int, int, {'W_0_mean': numpy.ndarray, ...}, {'h_0_mean': numpy.ndarray, ...},
        {'batch_accuracies': numpy.ndarray, 'test_accuracies': numpy.ndarray, 'train_accuracies': numpy.ndarray})
    """

    # To be sure ...
    tf.reset_default_graph()

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
        tf.set_random_seed(time.time()*1000)

        if args.cnn:
            x = tf.placeholder(tf.float32, shape = [None, 28, 28, 1])
        else:
            x = tf.placeholder(tf.float32, shape = [None, 784])

        if args.normalizer_data:
            n_x = normalizer('n_x', x)
        else:
            n_x = x

        y = tf.placeholder(tf.float32, shape = [None, 10])

        # Two convolutional layers if requested, otherwise fully connected layers.
        if args.cnn:

            kernel_size = [3, 3, 3, 3, 3, 3, 3]

            channels = [16]
            for i in range(1, args.layers):
                channels.append(max(channels[-1]*2, 64))

            # weights and biases
            W = []
            b = []

            # conv, activations, pooling and normalizes
            s = []
            h = []
            p = []
            n = [n_x]

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
            n = [n_x]

            for i in range(args.layers):
                units_in = 784
                if i > 0:
                    units_in = units[i - 1]

                W.append(initializer('W_fc' + str(i), [units_in, units[i]]))
                b.append(initializers.constant('b_fc' + str(i), [units[i]], value = args.bias))

                s.append(layers.inner_product('s_' + str(i), n[-1], W[-1], b[-1]))
                h.append(activation('h_' + str(i), s[-1]))
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

            losses = []
            batch_accuracies = []
            train_accuracies = []
            test_accuracies = []

            n_W = args.layers + 2
            n_h = args.layers + 1

            weight_means = [[] for i in range(n_W)]
            activation_means = [[] for i in range(n_h)]

            iterations = args.epochs*(train_dataset.count//args.batch_size + 1)
            interval = iterations // 20
            for i in range(iterations):

                # If this is the last batch, we reshuffle after this step.
                was_last_batch = train_dataset.next_batch_is_last()

                batch = train_dataset.next_batch()
                res = sess.run([optimizer, cross_entropy, accuracy] + W + h,
                                feed_dict = {x: tformer.process(batch[0]), y: batch[1]})

                for j in range(n_W):
                    weight_means[j].append(np.mean(res[3 + j]))

                # Get mean and variance of all layer activations after normalization, i.e. after normalization
                for j in range(n_h):
                    activation_means[j].append(np.mean(res[3 + n_W + j]))

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

            sess.close()

            weights = {}
            for j in range(args.layers):
                weights['W_' + str(j)] = np.array(weight_means[j])

            activations = {}
            for j in range(args.layers):
                activations['h_' + str(j)] = np.array(activation_means[j])

            accuracies = {
                'batch_accuracies': batch_accuracies,
                'train_accuracies': train_accuracies,
                'test_accuracies': test_accuracies
            }

            return iterations, interval, weights, activations, accuracies

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    if args.activation == 'sigmoid':
        activation = layers.sigmoid
    elif args.activation == 'tanh':
        activation = layers.tanh
    elif args.activation == 'relu':
        activation = layers.relu
    elif args.activation == 'softsign':
        activation = layers.softsign
    elif args.activation == 'softplus':
        activation = layers.softplus
    else:
        print('Invalid activation!')
        exit(1)

    if args.initializer == 'truncated_normal':
        initializer = initializers.truncated_normal
    elif args.initializer == 'uniform_unit_scale':
        initializer = initializers.uniform_unit_scale
    elif args.initializer == 'heuristic':
        initializer = initializers.heuristic
    elif args.initializer == 'xavier':
        initializer = initializers.xavier
    elif args.initializer == 'orthonormal':
        initializer = initializers.orthonormal
    else:
        print('Invalid initializer!')
        exit(1)

    if args.normalizer == 'identity':
        normalizer = layers.identity
    elif args.normalizer == 'batch_normalization':
        normalizer = layers.batch_normalization
    elif args.normalizer == 'layer_normalization':
        normalizer = layers.layer_normalization
    elif args.normalizer == 'min_max_normalization':
        normalizer = layers.min_max_normalization
    elif args.normalizer == 'min_max_normalization_symmetric':
        normalizer = layers.min_max_normalization_symmetric
    else:
        print('Invalid normalizer!')
        exit(1)

    if args.regularizer == 'none':
        regularizer = regularizers.none
    elif args.regularizer == 'l2':
        regularizer = regularizers.l2
    elif args.regularizer == 'l1':
        regularizer = regularizers.l1
    elif args.regularizer == 'orthonormal':
        regularizer = regularizers.orthonormal
    else:
        print('Invalid normalizer!')
        exit(1)

    name = 'l' + str(args.layers) + 'b' + str(args.batch_size) \
            + '_' + args.activation + '_' + args.initializer + '_' + args.normalizer
    if args.normalizer_data:
        name += '_dn'
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
        print('Only up to 7 layers supported!')
        exit(1)

    benchmark_weights = [None] * args.repetitions
    benchmark_activations = [None] * args.repetitions
    benchmark_accuracies = [None] * args.repetitions

    iterations = 0
    interval = 0

    weights_name = 'images/' + name + '_weights.png'
    activations_name = 'images/' + name + '_activations.png'
    accuracies_name = 'images/' + name + '_accuracies.png'
    text_name = 'images/' + name + '.txt'

    if os.path.exists(weights_name) and os.path.exists(activations_name) and os.path.exists(accuracies_name) and os.path.exists(text_name):
        print('Already evaluated!')
        exit(0)

    for i in range(args.repetitions):
        iterations, interval, benchmark_weights[i], benchmark_activations[i], benchmark_accuracies[i] = main()

    pyplot.clf()
    weight_plots = []
    weight_keys = list(benchmark_weights[0].keys())
    weight_keys.sort()

    for key in weight_keys:
        weights_merged = [weights[key] for weights in benchmark_weights]
        weights_merged = np.array(weights_merged)
        weights_mean = weights_merged.mean(axis = 0)
        weights_std = weights_merged.std(axis = 0)

        plot = pyplot.errorbar(np.arange(0, iterations, 1), weights_mean, weights_std, errorevery = 5)
        weight_plots.append(plot)

    legend = pyplot.legend(weight_plots, weight_keys, loc = 'upper center', bbox_to_anchor = (0.5, -0.1))
    pyplot.grid('on')
    pyplot.savefig(weights_name, bbox_extra_artists=(legend,), bbox_inches='tight')

    pyplot.clf()
    activation_plots = []
    activation_keys = list(benchmark_activations[0].keys())
    activation_keys.sort()

    for key in activation_keys:
        activations_merged = [activations[key] for activations in benchmark_activations]
        activations_merged = np.array(activations_merged)
        activations_mean = activations_merged.mean(axis = 0)
        activations_std = activations_merged.std(axis = 0)

        plot = pyplot.errorbar(np.arange(0, iterations, 1), activations_mean, activations_std, errorevery = 5)
        activation_plots.append(plot)

    legend = pyplot.legend(activation_plots, activation_keys, loc = 'upper center', bbox_to_anchor = (0.5, -0.1))
    pyplot.grid('on')
    pyplot.savefig(activations_name, bbox_extra_artists = (legend,), bbox_inches = 'tight')

    pyplot.clf()
    accuracy_plots = []
    accuracy_keys = ['batch_accuracies', 'train_accuracies', 'test_accuracies']
    accuracy_keys.sort()

    for key in accuracy_keys:
        accuracies_merges = [accuracies[key] for accuracies in benchmark_accuracies]
        accuracies_merges = np.array(accuracies_merges)
        accuracies_mean = accuracies_merges.mean(axis = 0)
        accuracies_std = accuracies_merges.std(axis = 0)

        if key == 'batch_accuracies':
            plot = pyplot.errorbar(np.arange(0, iterations, 1), accuracies_mean, accuracies_std, errorevery = 5)
        else:
            plot = pyplot.errorbar(np.arange(0, iterations, interval), accuracies_mean, accuracies_std)
        accuracy_plots.append(plot)

    legend = pyplot.legend(accuracy_plots, accuracy_keys, loc = 'upper center', bbox_to_anchor = (0.5, -0.1))
    pyplot.grid('on')
    pyplot.savefig(accuracies_name, bbox_extra_artists = (legend,), bbox_inches = 'tight')

    with open(text_name, 'w') as fp:
        steps = np.arange(0, iterations, interval)
        for j in range(len(steps)):
            fp.write(str(steps[j]) + ' ')
            for i in range(len(benchmark_accuracies)):
                fp.write(str(benchmark_accuracies[i]['test_accuracies'][j]) + ' ' + str(benchmark_accuracies[i]['train_accuracies'][j]))
            fp.write('\n')
"""
Dataset tool to provide shuffled batches for training and testing.
"""

import numpy as np

class Dataset(object):
    """
    Idea from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py.
    """

    def __init__(self, images, labels, batch_size = 1):
        """
        Construct a dataset.

        :param images: images
        :type images: numpy.ndarray
        :param labels: labels
        :type labels: numpy.ndarray
        :param batch_size: default batch size
        :type batch_size: int
        """

        assert images.shape[0] > 0
        assert images.shape[0] == labels.shape[0]

        self._index = 0
        """ (int) Current index for batch access. """

        self._indices = np.array(range(0, images.shape[0]))
        """ (numpy.ndarray) Underlying indices, can be used to shuffle data access. """

        self._images = images
        """ (numpy.ndarray) Underlying images of the dataset. """

        self._labels = labels
        """ (numpy.ndarray) Underlying labels of the dataset. """

        self._batch_size = batch_size
        """ (int) Batch size to use. """

    @property
    def images(self):
        """
        Get the images.

        :return: images
        :rtype: numpy.ndarray
        """

        return self._images

    @property
    def labels(self):
        """
        Get the labels.

        :return: labels
        :rtype: numpy.ndarray
        """

        return self._labels

    @property
    def count(self):
        """
        Get size of dataset.

        :return: size
        :rtype: int
        """

        return self._labels.shape[0]

    @property
    def batch_size(self):
        """
        Get batch size.

        :return: batch size
        :rtype: int
        """

        return self._batch_size

    def shuffle(self):
        """
        Shuffles the data (actually shuffles the indices which are used to
        get the next batch).
        """

        self._indices = np.random.permutation(self._indices)

    def reset(self):
        """
        Reset batch access.
        """

        self._index = 0

    def next_batch_is_last(self):
        """
        Whether the next batch is the last one.

        :return: true if next batch is last one
        :rtype: bool
        """

        return self._index + self._batch_size >= self._images.shape[0]

    def next_batch(self):
        """
        Get the next batch of the given size.

        :param batch_size: size of batch
        :type batch_size: int
        :return: images and labels
        :rtype: (numpy.ndarray, numpy.ndarray)
        """

        index = self._index
        self._index = (self._index + self._batch_size) % self._images.shape[0]

        if index + self._batch_size < self._images.shape[0]:
            return self._images[index : min(index + self._batch_size, self._images.shape[0]), :, :, :], \
                self._labels[index : min(index + self._batch_size, self._labels.shape[0])]
        else:
            images = np.concatenate((self._images[index : self._images.shape[0], :, :, :], \
                                   self._images[0: (index + self._batch_size) % self._images.shape[0], :, :, :]), 0)
            labels = np.concatenate((self._labels[index: self._labels.shape[0]], \
                                   self._labels[0: (index + self._batch_size) % self._labels.shape[0]]), 0)
            return images, labels
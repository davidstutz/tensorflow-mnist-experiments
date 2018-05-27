"""
Load the MNIST dataset.
"""

import struct
import array
import numpy as np

def load(images_file, labels_file, digits = np.arange(10)):
    """
    Load the MNIST dataset given the image and label files.

    :param images_file: path to images file
    :type images_file: str
    :param labels_file: path to labels file
    :type labels_file: str
    :return: images and alebls
    :rtype: (numpy.ndarray, numpy.ndarray)
    """

    labels_raw = None
    images_raw = None
    size = 0

    with open(labels_file, 'rb') as f:
        magic_nr, size = struct.unpack(">II", f.read(8))
        labels_raw = array.array("b", f.read())

    with open(images_file, 'rb') as f:
        magic_nr, size, rows, cols = struct.unpack(">IIII", f.read(16))
        images_raw = array.array("B", f.read())

    ind = [k for k in range(size) if labels_raw[k] in digits]
    N = len(ind)

    images = np.zeros((size, rows, cols), dtype = np.uint8)
    labels = np.zeros((size, 10), dtype = np.int8)
    for i in range(size):
        images[i] = np.array(images_raw[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i, labels_raw[ind[i]]] = 1

    return images, labels
"""
Transformer to pre-process network input.
"""

class Transformer:
    """
    A transformer transforms input data according to normalization and pre-processing best practices. The transformer
    bundles several of these best practices.
    """

    def __init__(self):
        """
        Constructor, sets default values which corresponds to the identity transformation.
        """

        self._transformations = {}
        """ ({}) Transformations mapping transformation names to parameter values. """

    def convert(self, type):
        """
        Add conversion to the given type.

        :param type: conversion type
        :type type: type
        """

        self._transformations['convert'] = type

    def scale(self, scale):
        """
        Add scale.

        :param scale: value to multiply by
        :type scale: float
        """

        self._transformations['scale'] = scale

    def reshape(self, shape):
        """
        Add reshape.

        The shape is given as list of ints or None. If the first element is None this corresponds to the
        batch size.

        :param shape: NumPy compatible shape
        :type shape: [int]
        :return:
        """

        self._transformations['reshape'] = shape

    def process(self, image):
        """
        Process the image based on the configured transformations.

        :param image: input image
        :type image: numpy.ndarray
        :return: processed image
        :rtype: numpy.ndarray
        """

        if 'reshape' in self._transformations.keys():

            shape = list(self._transformations['reshape']) # must be copy/clone
            if shape[0] == None:
                shape[0] = image.shape[0]

            image = image.reshape(tuple(shape))
        if 'convert' in self._transformations.keys():
            image = image.astype(self._transformations['convert'])
        if 'scale' in self._transformations.keys():
            image = image*self._transformations['scale']

        return image
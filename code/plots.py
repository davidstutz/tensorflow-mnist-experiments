"""
Utilities for plotting.
"""

from matplotlib import pyplot
import numpy as np
import utils

def plot_lines(values, path):
    """
    Plot lines.

    :param values: object containing the names as keys and the values as numpy.ndarray
    :type values: {string: (numpy.ndarray, numpy.ndarray)}
    :param path: path to file to store the plot
    :type path: str
    """

    for key in values.keys():
        assert len(values[key][0]) == len(values[key][1]), "invalid lengths (%s): %d != %d" % (key, len(values[key][0]), len(values[key][1]))

    pyplot.clf()
    plots = []

    for key in values.keys():
        plot, = pyplot.plot(values[key][0], values[key][1])
        plots.append(plot)

    # http://stackoverflow.com/questions/10101700/moving-matplotlib-legend-outside-of-the-axis-makes-it-cutoff-by-the-figure-box
    legend = pyplot.legend(plots, values.keys(), loc='upper center', bbox_to_anchor=(0.5, -0.1))
    pyplot.grid('on')
    pyplot.savefig(path, bbox_extra_artists=(legend,), bbox_inches='tight')

def plot_normalized_histograms(values, path, **kwargs):
    """
    Plots histograms for all of the provided numpy.ndarrays.

    :param values: object containing the names as keys and the values as numpy.ndarrays
    :type values: {string: numpy.ndarray}
    :param path: path to file to store the plot
    :type path: str

    :Keyword Arguments:
    * *bins* (int) -- number of bins for histogram (1000)
    """

    bins = utils.get_kwargs(kwargs, 'bins', 1000)
    assert bins > 0

    pyplot.clf()
    plots = []

    for key in values.keys():
        y, edges = np.histogram(values[key], bins = bins)
        centers = 0.5*(edges[1:] + edges[:-1])
        y = y / y.sum()
        plot, = pyplot.plot(centers, y, '-')
        plots.append(plot)

    # http://stackoverflow.com/questions/10101700/moving-matplotlib-legend-outside-of-the-axis-makes-it-cutoff-by-the-figure-box
    legend = pyplot.legend(plots, values.keys(), loc = 'upper center', bbox_to_anchor = (0.5, -0.1))
    pyplot.grid('on')
    pyplot.savefig(path, bbox_extra_artists = (legend,), bbox_inches = 'tight')

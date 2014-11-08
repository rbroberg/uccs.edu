# Copyright (c) 2009 Leif Johnson <leif@leifjohnson.net>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''Basic self-organizing map implementation.

This module contains the following Kohonen map implementations :

  - Map. A standard rectangular N-dimensional Kohonen map.

    - Gas. A vector quantizer that does not have a fixed topology. Neurons in a
      gas are sorted for updates based on their distance from the cue, with the
      sort order defining a topology for each cue presentation.

      - GrowingGas. A Gas-based quantizer that can add neurons dynamically to
        explain high-error areas of the input space.

  - Filter. A wrapper over an underlying Map instance that maintains an explicit
    estimate of the likelihood of each neuron.

These are tested using the kohonen_test.py file in this source distribution.

Because they have a grid topology, Map objects have some cool visualization
options, including Map.neuron_colormap and Map.distance_heatmap. These require
the Python Image Library.

There is also a collection of distance metrics :

  - cosine_metric. A callable that calculates the cosine distance between a cue
    and each neuron in a Kohonen Map.

  - euclidean_metric. A callable that calculates the Euclidean distance between
    a cue and each neuron in a Kohonen Map.

  - manhattan_metric. A callable that calculates the Manhattan distance between
    a cue and each neuron in a Kohonen Map.

There are also some small utility classes for modeling time series values :

  - Timeseries. A callable that takes no arguments and returns a value that
    might vary over time. Each call to the function will generally return a
    unique value (though this is not necessary).

    - ExponentialTimeseries. A callable that takes no arguments and returns an
      exponentially decreasing (or increasing) series of values, dependent on
      the parameters passed in at construction time.

    - etc.

These distance functions and time series objects are generally used to regulate
the learning parameters in Kohonen Map objects.
'''

import numpy
from numpy import random as rng


def cosine_metric(x, y):
    #Returns the cosine distance between x and y.
    nx = numpy.sqrt(numpy.sum(x * x, axis=-1))
    ny = numpy.sqrt(numpy.sum(y * y, axis=-1))
    # the cosine metric returns 1 when the args are equal, 0 when they are
    # orthogonal, and -1 when they are opposite. we want the opposite effect,
    # and we want to make sure the results are always nonnegative.
    return 1 - numpy.sum(x * y, axis=-1) / nx / ny

def euclidean_metric(x, y):
    #Returns the euclidean distance (L-2 norm) between x and y.
    d = x - y
    return numpy.sqrt(numpy.sum(d * d, axis=-1))

def manhattan_metric(x, y):
    #Returns the manhattan distance (L-1 norm) between x and y.
    d = x - y
    return numpy.sum(numpy.abs(d), axis=-1)

def weighted_euclidean_metric(weights):
    #Implements a standard euclidean distance with weighted dimensions.
    def calculate(x, y):
        d = x - y
        return numpy.sqrt(numpy.sum(d * d * weights, axis=-1))
    return calculate


class Timeseries(object):
    #Represents some sort of value that changes over time.

    def __init__(self):
        #Set up this timeseries.
        super(Timeseries, self).__init__()
        self.ticks = 0

    def __call__(self):
        #Call this timeseries.
        t = self.ticks
        self.ticks += 1
        return t

    def reset(self):
        #Reset the time for this series.
        self.ticks = 0

class ConstantTimeseries(Timeseries):
    #This timeseries just returns a constant value.

    def __init__(self, k=1):
        #Set up this series with a constant value.
        self.k = k

    def __call__(self):
        #Return the constant.
        return self.k

class ExponentialTimeseries(Timeseries):
    #Represents an exponential decay process.

    def __init__(self, rate=-1, initial=1, final=0):
        #Create a new exponential timeseries object.
        super(ExponentialTimeseries, self).__init__()
        self.initial = initial - final
        self.rate = rate
        self.final = final

    def __call__(self):
        #Return an exponentially-decreasing series of values.
        super(ExponentialTimeseries, self).__call__()
        return self.final + self.initial * numpy.exp(self.rate * self.ticks)


class Parameters(object):
    #We are plain old data holding self-organizing map parameters.

    def __init__(self,
                 dimension=None,
                 shape=None,
                 metric=None,
                 learning_rate=None,
                 neighborhood_size=None,
                 noise_variance=None):
        '''This class holds standard parameters for self-organizing maps.

        dimension: The length of a neuron vector in a Map or a Gas.

        shape: The shape of the neuron topology in whatever Map or Gas we are
          building.

        metric: The distance metric to use when comparing cues to neurons in the
          map. Defaults to euclidean_metric.

        learning_rate: This parameter determines the time course of the learning
          rate for a Map. This parameter should be a callable that takes no
          arguments and returns a floating point value for the learning rate.

          If this parameter is None, a default learning rate series will be
          used, equivalent to ExponentialTimeseries(-1e-3, 1, 0.2).

          If this parameter is a numeric value, it will be used as the
          constant value for the learning rate: ConstantTimeseries(value).

        neighborhood_size: Like the learning rate, this parameter determines the
          time course of the neighborhood size parameter. It should be a
          callable that takes no arguments and returns a neighborhood size for
          storing each cue.

          If this is None, a default neighborhood size series will be used. The
          initial size will be the maximum of the dimensions given in shape, and
          the decay will be -1e-3: ExponentialTimeseries(-1e-3, max(shape), 1).

          If this is a floating point value, it will be used as a constant
          neighborhood size: ConstantTimeseries(value).

        noise_variance: Like the learning rate and neighborhood size, this
          should be a factory for creating a callable that creates noise
          variance values.

          If this is None, no noise will be included in the created Maps.

          If this parameter is a number, it will be used as a constant noise
          variance.
        '''
        assert dimension is not None
        self.dimension = dimension

        assert shape is not None
        self.shape = shape

        self.metric = metric or euclidean_metric

        ET = ExponentialTimeseries
        CT = ConstantTimeseries

        self.learning_rate = learning_rate
        if isinstance(learning_rate, (float, int)):
            self.learning_rate = CT(learning_rate)
        if learning_rate is None:
            self.learning_rate = ET(-1e-3, 1, 0.2)

        self.neighborhood_size = neighborhood_size
        if isinstance(neighborhood_size, (float, int)):
            self.neighborhood_size = CT(neighborhood_size)
        if neighborhood_size is None:
            self.neighborhood_size = ET(-1e-3, max(shape), 1)

        self.noise_variance = noise_variance
        if isinstance(noise_variance, (float, int)):
            self.noise_variance = CT(noise_variance)


def heatmap(raw, axes=(0, 1), lower=None, upper=None):
    #Create a heat map image from the given raw matrix.

    raw: An array of values to use for the image pixels.
    axes: The axes in the array that we want to preserve for the final image.
      All other axes will be summed away.
    lower: If given, clip values in the matrix to this lower limit. If not
      given, raw.min() will be used.
    upper: If given, clip values in the matrix to this upper limit. If not
      given, raw.max() will be used.

    Returns an annotated Image object (as returned from _image).
    #
    assert len(axes) == 2
    for ax in xrange(len(raw.shape) - 1, -1, -1):
        if ax in axes:
            continue
        raw = raw.sum(axis=ax)
    l = lower
    if l is None:
        l = raw.min()
        l *= l < 0 and 1.01 or 0.99
    u = upper
    if u is None:
        u = raw.max() * 1.01
        u *= u > 0 and 1.01 or 0.99
    return _image(raw, l, u, format)


def colormap(raw, axes=(0, 1, 2), layers=(0, 1, 2)):
    '''Create an RGB image using the given layers of a 3D raw values matrix.

    raw: An array of raw values to use for the image.
    axes: The axes in the array that we want to preserve for the final image.
      All other axes will be summed away.
    layers: The indices of the third preserved axis that we should use for the
      red, green, and blue channels in the output image.

    Raw values will be scaled along each layer to lie in [lower, upper], where
    lower (upper) is the global lower (upper) bound of all values in each of the
    raw layers.

    Returns an Image object, as in the heatmap() function.
    '''
    assert len(axes) == len(layers) == 3
    for ax in xrange(len(raw.shape) - 1, -1, -1):
        if ax in axes:
            continue
        raw = raw.sum(axis=ax)
    u = -numpy.inf
    l = numpy.inf
    for i in layers:
        v = raw[:, :, i]
        l = min(l, v.min())
        u = max(u, v.max())
    l *= l < 0 and 1.01 or 0.99
    u *= u > 0 and 1.01 or 0.99
    return _image(raw[:, :, layers], l, u, 'RGB')


def _image(values, lower, upper, format='L'):
    '''Create a PIL image using the given 2D array of values.

    Pixel values in the range [lower, upper] are scaled linearly to [0, 1]
    before creating the image.

    Returns an Image object annotated with the lower and upper bounds that were
    used to scale the values to convert them to pixels.
    '''
    from PIL import Image
    ratios = (values - lower) / (upper - lower)
    img = Image.fromarray(numpy.array(256 * ratios, numpy.uint8), format)
    img.lower_bound = lower
    img.upper_bound = upper
    return img


def _zeros(shape, dtype='d'):
    #Get a blank (all-zero) matrix with a certain shape.
    return numpy.zeros(shape, dtype=dtype)


def itershape(shape):
    #Given a shape tuple, iterate over all indices in that shape.
    if not shape:
        yield ()
        return
    for i in xrange(shape[0]):
        for z in itershape(shape[1:]):
            yield (i, ) + z


def argsample(pdf, n=1):
    #Return n indices drawn proportionally from a discrete mass vector.
    assert (pdf >= 0).all(), 'cannot sample from %r!' % pdf
    cdf = pdf.cumsum()
    return numpy.searchsorted(cdf, rng.uniform(0, cdf[-1], n))


def sample(pdf, n=1):
    #Return n samples drawn proportionally from a discrete mass vector.
    assert len(pdf.shape) == 1
    return pdf[argsample(pdf, n)]


class Map(object):
    '''Basic implementation of a rectangular N-dimensional self-organizing map.

    A Self-Organizing or Kohonen Map (henceforth just Map) is a group of
    lightweight processing units called neurons, which are here implemented as
    vectors of real numbers. Neurons in a Map are arranged in a specific
    topology, so that a given neuron is connected to a small, specific subset of
    the overall neurons in the Map. In addition, the Map uses a distance metric
    (e.g., Euclidean distance) for computing similarity between neurons and cue
    vectors, as described below.

    The Map accepts cues---vectors of real numbers---as inputs. In standard Map
    usage, cues represent some data point of interest. Normally applications of
    Maps use input vectors like the activation patterns for an array of sensors,
    term frequency vectors for a document, etc. Cues are stored in the Map as
    follows : First, a "winner" neuron w is chosen from the Map, and, second,
    the neurons in the Map topologically near w are altered so that they become
    closer to the cue. Each of these steps is described briefly below.

    For the first step, the Map computes the distance between the cue and each
    of the Map neurons using its metric. The neuron closest to the cue under
    this metric is declared the "winner" w. Alternatively, the winner can be
    selected probabilistically based on the overall distance landscape.

    Next, the Map alters the neurons in the neighborhood of w, normally using
    some function of the difference between the cue and the neuron being
    modified. The weight of the alteration decreases exponentially as the
    topological distance from w increases. The learning rule for a neuron n is

    n += eta * exp(-d**2 / sigma**2) * (c - n)

    where eta is the learning rate, sigma is called the neighborhood size, d is
    the topological distance between n and w, and c is the cue vector being
    stored in the map. Eta and sigma normally decrease in value over time, to
    take advantage of the empirical machine learning benefits of simulated
    annealing.

    The storage mechanism in a Map has the effect of grouping cues with similar
    characteristics into similar areas of the Map. Because the winner---and its
    neighborhood---are altered to look more like the cues that they capture, the
    winner for a given cue will tend to win similar inputs in the future. This
    tends to cluster similar Map inputs, and can lead to interesting data
    organization patterns.
    '''

    def __init__(self, params):
        #Initialize this Map.
        self._shape = params.shape
        self.dimension = params.dimension
        self.neurons = _zeros(self.shape + (self.dimension, ))

        self._metric = params.metric

        self._learning_rate = params.learning_rate
        self._neighborhood_size = params.neighborhood_size
        self._noise_variance = params.noise_variance

        # precompute a neighborhood mask for performing fast storage updates.
        # this mask is the same dimensionality as self.shape, but twice the size
        # along each axis. the maximum value in the mask is 1, occurring in the
        # center. values decrease in a gaussian fashion from the center.
        S = tuple(2 * size - 1 for size in self.shape)
        self._neighborhood_mask = _zeros(S)
        for coords in itershape(S):
            z = 0
            for axis, offset in enumerate(coords):
                d = offset + 1 - self.shape[axis]
                z += d * d
            self._neighborhood_mask[coords] = numpy.exp(-z / 2)

    @property
    def shape(self):
        return self._shape

    def neuron(self, coords):
        #Get the current state of a specific neuron.
        return self.neurons[coords]

    def reset(self, f=None):
        #Reset the neurons and timeseries in the Map.

        f: A callable that takes a neuron coordinate and returns a value for
          that neuron. Defaults to random values from the standard normal.
        #
        self._learning_rate.reset()
        self._neighborhood_size.reset()
        if f is None:
            self.neurons = rng.randn(*self.neurons.shape)
        else:
            for z in itershape(self.shape):
                self.neurons[z] = f(z)

    def weights(self, distances):
        #Get an array of learning weights to use for storing a cue.
        i = self.smallest(distances)
        z = []
        for axis, size in enumerate(self.flat_to_coords(i)):
            offset = self.shape[axis] - size - 1
            z.append(slice(offset, offset + self.shape[axis]))
        sigma = self._neighborhood_size()
        return self._neighborhood_mask[z] ** (1.0 / sigma / sigma)

    def distances(self, cue):
        #Get the distance of each neuron in the Map to a particular cue.
        z = numpy.resize(cue, self.neurons.shape)
        return self._metric(z, self.neurons)

    def flat_to_coords(self, i):
        #Given a flattened index, convert it to a coordinate tuple.
        coords = []
        for limit in reversed(self.shape[1:]):
            i, j = divmod(i, limit)
            coords.append(j)
        coords.append(i)
        return tuple(reversed(coords))

    def winner(self, cue):
        #Get the coordinates of the most similar neuron to the given cue.

        Returns a flat index ; use flat_to_coords to convert this to a neuron
        index.
        #
        return self.smallest(self.distances(cue))

    def sample(self, n):
        #Get a sample of n neuron coordinates from the map.

        The returned values will be flat indices ; use flat_to_coords to convert
        them to neuron indices.
        #
        return rng.randint(0, self.neurons.size / self.dimension - 1, n)

    def smallest(self, distances):
        #Get the index of the smallest element in the given distances array.

        Returns a flat index ; use flat_to_coords to convert this to a neuron
        index.
        #
        assert distances.shape == self.shape
        return distances.argmin()

    def learn(self, cue, weights=None, distances=None):
        #Add a new cue vector to the Map, moving neurons as needed.
        if weights is None:
            if distances is None:
                distances = self.distances(cue)
            weights = self.weights(distances)
        assert weights.shape == self.shape
        weights.shape += (1, )
        delta = numpy.resize(cue, self.neurons.shape) - self.neurons
        eta = self._learning_rate()
        self.neurons += eta * weights * delta
        if self._noise_variance:
            self.neurons += rng.normal(
                0, self._noise_variance(), self.neurons.shape)

    def neuron_heatmap(self, axes=(0, 1), lower=None, upper=None):
        #Return an image representation of this Map.
        return heatmap(self.neurons, axes, lower, upper)

    def distance_heatmap(self, cue, axes=(0, 1), lower=None, upper=None):
        #Return an image representation of the distance to a cue.
        return heatmap(self.distances(cue), axes, lower, upper)



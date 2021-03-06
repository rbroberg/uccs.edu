"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer

from lenet_util import load_data_cct, reshape_crosscorr

cases=[ ["Dog_1",     24,  480,  502, 4, 3],
        ["Dog_2",     42,  500, 1000, 8, 5],
        ["Dog_3",     72, 1440,  907, 12, 8],
        ["Dog_4",     97,  804,  990, 16, 11],
        ["Dog_5",     30,  450,  191, 5, 3],
        ["Patient_1", 18,   50,  195, 3, 2],
        ["Patient_2", 18,   42,  150, 3, 2]];

cases = [ ["Dog_1",     24,  480,  502, 4, 3],
          ["Dog_2",     42,  500, 1000, 8, 5]]

# previously extracted features, only partially scaled
# features=["ac","var","ent","ent2","skew","kurt"]
# features=["cc"] # cross correlation has length (n^2-n)/2 where n = number of channels
features=["cct"]

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """
    
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.
        
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        
        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape
        
        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)
        
        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)
        
        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """
        
        assert image_shape[1] == filter_shape[1]
        self.input = input
        
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        
        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        
        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )
        
        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )
        
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        
        # store parameters of this layer
        self.params = [self.W, self.b]


# TODO: batch_size seems ill-suited for variable sized data sets
def evaluate_lenet5(learning_rate=0.1, n_epochs=10,
                    casenum=0,
                    nkerns=[20, 50], batch_size=24):
    """ lenet on UPenn EEG dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type casenum: int
    :param casenum: case number from U Penn data set (via Kaggle.com)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """
    
    '''
	case=cases[0][0]
	npre=cases[0][1]
	ninter=cases[0][2]
	ntest=cases[0][3]
	sn=cases[0][4] # number of preictal series of 6 files
	sm=cases[0][5] # number of series to include in training
	features=["cctime"]
	learning_rate=0.1
	n_epochs=84
	casenum=0
	nkerns=[20, 50]
	batch_size=6
    '''

    rng = numpy.random.RandomState(23455)
    
    datasets = load_data_cct(cases[casenum][0],cases[casenum][1],cases[casenum][2],cases[casenum][3],["cctime"],cases[casenum][4], cases[casenum][5])
    
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value().shape[0] 
    n_valid_batches = valid_set_x.get_value().shape[0]
    n_test_batches = test_set_x.get_value().shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size
    
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    
    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    
    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    d = int(((train_set_x.get_value()).shape[1])**0.5)
    layer0_input = x.reshape((batch_size, 1, d, d))
    
    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    
    # filtering reduces the image size to (16-1+1 , 16-1+1) = (16, 16)
    # maxpooling reduces this further to (16/2, 16/2) = (8, 8)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 8, 8)
    print(d)
    if d==16:
       nb1=1;nb2=1;d2=8;d3=4
    elif d==15:
       nb1=2;nb2=2;d2=7;d3=3
    elif d==24:
       nb1=1;nb2=1;d2=12;d3=6
    elif d==120:
       nb1=25;nb2=9;d2=24;d3=8
        
    print("layer0")
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, d, d),
        filter_shape=(nkerns[0], 1, 1, nb1),
        poolsize=(1, 4)
    )
    
    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (8-1+1, 8-1+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (nkerns[0], nkerns[1], 4, 4)
    print("layer1")
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], d, d2),
        filter_shape=(nkerns[1], nkerns[0], 1, nb2),
        poolsize=(1, 2)
    )
    
    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    print("layer2a")
    layer2_input = layer1.output.flatten(2)
    
    # construct a fully-connected sigmoidal layer
    print("layer2b")
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * d * d3,
        n_out=batch_size,
        activation=T.tanh
    )
    
    # classify the values of the fully-connected sigmoidal layer
    print("layer3")
    layer3 = LogisticRegression(input=layer2.output, n_in=batch_size, n_out=2)
    
    # the cost we minimize during training is the NLL of the model
    print("cost")
    cost = layer3.negative_log_likelihood(y)
    
    # create a function to compute the mistakes that are made by the model
    # works great if you have a vector of "known classes" for test data
    #test_model = theano.function(
    #    [index],
    #    layer3.errors(y),
    #    givens={
    #        x: test_set_x[index * batch_size: (index + 1) * batch_size],
    #        y: test_set_y[index * batch_size: (index + 1) * batch_size]
    #    }
    #)
    
    # create a function to predict labels that are made by the model
    model_predict = theano.function(
        [index], 
        layer3.y_pred,
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params
    
    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)
    
    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]
    
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1
    
    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless (was 10000)
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch
    
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()
    
    epoch = 0
    done_looping = False
    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
        
            iter = (epoch - 1) * n_train_batches + minibatch_index
            
            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)
            
            if (iter + 1) % validation_frequency == 0:
                
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))
                
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    
                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    
                    # test it on the test set
                    #test_losses = [
                    #    test_model(i)
                    #    for i in xrange(n_test_batches)
                    #]
                    #test_score = numpy.mean(test_losses)
                    #print(('     epoch %i, minibatch %i/%i, test error of '
                    #       'best model %f %%') %
                    #      (epoch, minibatch_index + 1, n_train_batches,
                    #       test_score * 100.))
                    
            if patience <= iter:
                done_looping = True
                break
    
    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stdout, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    p=[model_predict(i) for i in xrange(n_test_batches)]
    p=numpy.array(p)
    p=p.reshape((p.shape[0]*p.shape[1]))
    return(p)

if __name__ == '__main__':
    f1=open('testfile.csv', 'w')
    f1.write(",".join(["clip","preictal"])+'\n')
    for c in range(len(cases)):
        print(">>>>>>> "+cases[c][0]+" <<<<<<<<\n");
        bs=int(cases[c][3]/20) # variable batchsize based on data size
        p=evaluate_lenet5(casenum=c,batch_size=bs)
        for n in range(cases[c][3]):
            if n < len(p):
                if p[n]>1:
                    p[n]=1
                f1.write(",".join(["_".join([cases[c][0],"test_segment",str(n+1).zfill(4)+".mat"]),str(p[n])])+'\n')
            else:
                f1.write(",".join(["_".join([cases[c][0],"test_segment",str(n+1).zfill(4)+".mat"]),str(0.00123)])+'\n')
    f1.close()

def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)


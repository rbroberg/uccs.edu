#!/usr/bin/env python
# Author:      Flávio Codeço Coelho
# License:     GPL
# http://code.activestate.com/recipes/414200-metropolis-hastings-sampler/

from math import *
#from RandomArray import *
from matplotlib.pylab import *

def sdnorm(z):
    """
    Standard normal pdf (Probability Density Function)
    """
    return exp(-z*z/2.)/sqrt(2*pi)

# 20160128rb: beta
from scipy.stats import beta
def mybeta(z):
    return beta.pdf(z,0.5,0.5)

n = 10000
alpha = 1
x = 0.
vec = []
vec.append(x)
innov = uniform(-alpha,alpha,n) #random inovation, uniform proposal distribution
for i in xrange(1,n):
    can = x + innov[i] #candidate
    #aprob = min([1.,sdnorm(can)/sdnorm(x)]) #acceptance probability
    aprob = min([1.,mybeta(can)/mybeta(x)]) #acceptance probability
    u = uniform(0.001,.999)
    if u < aprob:
        x = can
        vec.append(x)

#plotting the results:
#theoretical curve
#x = arange(-3,3,.1)
#y = sdnorm(x)
x = arange(0,1,.01)
y = mybeta(x)
subplot(211)
title('Metropolis-Hastings')
plot(vec)
subplot(212)

hist(vec, bins=30,normed=1)
plot(x,y,'ro')
ylabel('Frequency')
xlabel('x')
legend(('PDF','Samples'))
show()

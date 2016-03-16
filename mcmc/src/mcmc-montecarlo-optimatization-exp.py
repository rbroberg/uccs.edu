# MONTE CARLO OPTIMIZATION OF exp(x-4)^2
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pylab

np.random.seed(271828)

def g(x):
    return np.exp(-0.5*(x-4)**2)

# INITIALZIE
N = 100000
step=0.1
x = np.arange(0,6+step,step)
C = np.sqrt(2*np.pi)

y = stats.norm.pdf(4,1,x)
 
# CALCULATE MONTE CARLO APPROXIMATION
#x = normrnd(4,1,1,N);
n = np.random.normal(4,1,size=N)
h=np.histogram(n,100)
counts=h[0]
bins=h[1]
optIdx = np.argmax(counts)
x_hat = bins[optIdx];

'''
# OPTIMA AND ESTIMATED OPTIMA
# ph = plot(x,g(x)/C,'r','Linewidth',3); hold on
# gh = plot(x,g(x),'b','Linewidth',2); hold on;
# oh = plot([4 4],[0,1],'k');
# hh = plot([xHat,xHat],[0,1],'g');
plt.plot(x,g(x),color='blue',linewidth=3,label="g(x)")
plt.plot(x,g(x)/C,color='red',linewidth=3,label="p(x) = g(x)/C")
plt.axvline(x_hat, color='green',linewidth=1,label="x_opt")
plt.axvline(4, color='black',linewidth=1,label="x_hat")
leg=plt.legend(loc='upper left')
ltext  = leg.get_texts()
llines = leg.get_lines()
frame  = leg.get_frame()
plt.title("Example Monte Carlo Optimization")
plt.show()
#plt.savefig('mcmc-monte-carlo-optimization-exp.png')
'''


fig = plt.figure(figsize=(8, 6)) 
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
ax0 = plt.subplot(gs[0])
ax0.plot(x,g(x),color='blue',linewidth=3,label="g(x)")
ax0.plot(x,g(x)/C,color='red',linewidth=3,label="p(x) = g(x)/C")
ax0.axvline(x_hat, color='green',linewidth=1,label="x_opt")
ax0.axvline(4, color='black',linewidth=1,label="x_hat")
leg=ax0.legend(loc='upper left')
ltext  = leg.get_texts()
llines = leg.get_lines()
frame  = leg.get_frame()
fig.suptitle("Example Monte Carlo Optimization")
ax1 = plt.subplot(gs[1])
ax1.plot(x,g(x),color='blue',linewidth=3,label="g(x)")
ax1.plot(x,g(x)/C,color='red',linewidth=3,label="p(x) = g(x)/C")
ax1.axvline(x_hat, color='green',linewidth=3,label="x_opt")
ax1.axvline(4, color='black',linewidth=3,label="x_hat")
ax1.set_xlim([3.9,4.1])
ax1.locator_params(axis='x',nbins=3)
#ax1.xaxis.set_major_locator(pylab.NullLocator())
#plt.show()
plt.savefig('mcmc-monte-carlo-optimization-exp.png')

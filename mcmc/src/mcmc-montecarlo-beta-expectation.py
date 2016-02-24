# MONTE CARLO EXPECTATION
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pylab

np.random.seed(271828)

alpha1 = 2; 
alpha2 = 10;
N = 10000;
#x = betarnd(alpha1,alpha2,1,N);
x = stats.beta.rvs(alpha1,alpha2, size=N)
 
# MONTE CARLO EXPECTATION
expectMC = np.mean(x);
 
# ANALYTIC EXPRESSION FOR BETA MEAN
expectAnalytic = 1.*alpha1/(alpha1 + alpha2);
 
plt.figure(figsize=(8, 6)) 
plt.hist(x[:,0])
plt.show()

# DISPLAY
steps=0.02
bins = np.arange(0,1.+steps,steps)
h=np.histogram(x,bins)
counts=h[0]

probSampled = 1.*counts/sum(counts);
probTheory = stats.beta.pdf(bins,alpha1,alpha2);

fig = plt.figure(figsize=(8, 6)) 
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
ax0 = plt.subplot(gs[0])
ax0.bar(bins[1:],probSampled,width=steps,color='grey',label="samples")
ax0.plot(bins[1:]-0.5*steps,probTheory[1:]/sum(probTheory[1:]),'r',linewidth=2,label="theory")
ax0.axvline(x=expectMC,color='g',linewidth=2,label="sampled expectation")
ax0.axvline(x=expectAnalytic,color='b',linewidth=2,label="theory expectation")
ax0.set_xlim([0.02,1.00])
leg=ax0.legend(loc='upper right')
ltext  = leg.get_texts()
llines = leg.get_lines()
frame  = leg.get_frame()plt.setp(ltext, fontsize='x-small')
fig.suptitle("Example Monte Carlo Expectation for Beta Distribution")
ax1 = plt.subplot(gs[1])
ax1.bar(bins[1:],probSampled,width=steps,color='grey')
ax1.plot(bins[1:]+0.5*steps,probTheory[1:]/sum(probTheory[1:]),'r',linewidth=2)
ax1.axvline(x=expectMC,color='g',linewidth=2)
ax1.axvline(x=expectAnalytic,color='b',linewidth=2)
ax1.set_xlim([0.16,0.18])
ax1.xaxis.set_major_locator(pylab.NullLocator())
#ax1.yaxis.set_major_locator(pylab.NullLocator())
# plt.show()
plt.savefig('mcmc-monte-carlo-beta-expectation.png')
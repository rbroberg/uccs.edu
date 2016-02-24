# METROPOLIS SAMPLING EXAMPLE
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pylab

np.random.seed(271828)

# DEFINE THE TARGET DISTRIBUTION
def p(x):
	return 1./(1.+x**2)

 
# INITIALIZE CONSTANTS
nSamples = 5000;
burnIn = 500;
nDisplay = 30;
sigma = 1;
minn = -20;
maxx = 20;
step=0.1
xx = np.arange(3.*minn,3.*maxx+step,step);
target = p(xx);
pauseDur = .8;
 
# INITIALZE SAMPLER
x = np.zeros((1,nSamples));
x[0,0]= np.random.normal()
 
# RUN SAMPLER
for t in range(nSamples-1):
	
	# SAMPLE FROM PROPOSAL
	xStar = np.random.normal(x[0,t],sigma);
	proposal = stats.norm.pdf(xx,x[0,t],sigma);
	
	# CALCULATE THE ACCEPTANCE PROBABILITY
	alpha = min([1., p(xStar)/p(x[0,t])]);
	
	# ACCEPT OR REJECT?
	u = np.random.uniform()
	if u < alpha:
		x[0,t+1] = xStar;
		str = 'Accepted';
	else:
		x[0,t+1] = x[0,t];
		str = 'Rejected';
	#end


# DISPLAY SAMPLING DYNAMICS
# to do

# DISPLAY RESULTS
# x-axis steps (t)
step=1
a = np.arange(1,nSamples+step,step)
# plot it
fig = plt.figure(figsize=(8, 6)) 
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
# DISPLAY MARKOV CHAIN
ax0 = plt.subplot(gs[0])
ax0.set_ylim([-10,10])
ax0.plot(a, x[0,:])
ax0.axvline(x=burnIn,color='r',linewidth=2)
ax0.text(burnIn+50, 8, r'Burn In')
plt.title('Metropolis Markov Samples for 1/(1 + x^2)')
# DISPLAY SAMPLES
ax1 = plt.subplot(gs[1])
ax1.set_ylim([-10,10])
ax1.xaxis.set_major_locator(pylab.NullLocator())
ax1.yaxis.set_major_locator(pylab.NullLocator())
h=ax1.hist(x[0,burnIn:],bins=200,orientation="horizontal",color='lightgrey',edgecolor = 'grey',label="Samples")
b=np.arange(-10,10,0.1)
n=stats.norm.pdf(b)
t=stats.t.pdf(b,1)
plt.plot(n*nSamples/sum(n),b,color='r',linewidth=2,label="Normal Dist")
plt.plot(t*nSamples/sum(n),b,color='lime',linewidth=2,label="T-Student Dist")
#plt.ylabel('samples')
leg=ax1.legend(loc='upper left')
ltext  = leg.get_texts()
llines = leg.get_lines()
frame  = leg.get_frame()
plt.setp(ltext, fontsize='x-small')

plt.tight_layout()
#plt.show()

plt.savefig('mcmc-metropolis-sampler.png')
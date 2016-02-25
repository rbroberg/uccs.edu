# METROPOLIS-HASTINGS BAYESIAN POSTERIOR
import numpy as np
#import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pylab
from math import gamma

# INITIALIZE
np.random.seed(271828) 
 
# PRIOR OVER SCALE PARAMETERS
B = 1.;
 
# DEFINE LIKELIHOOD
# likelihood = inline('(B.^A/gamma(A)).*y.^(A-1).*exp(-(B.*y))','y','A','B');
def likelihood(y,A,B):
	return (B**A/gamma(A))*y**(A-1.)*np.exp(-(B*y))

# CALCULATE AND VISUALIZE THE LIKELIHOOD SURFACE
# yy = linspace(0,10,100);
# AA = linspace(0.1,5,100);
# avoid infinite edges
yy=np.arange(0.1,10.1,0.1)
AA=np.arange(0.05,5.05,.05)

likeSurf = np.zeros((yy.size,AA.size));
for iA in range(AA.size):
	likeSurf[:,iA]=likelihood(yy[:],AA[iA],B)

from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

#PLOT
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
axs=ax.get_axes()
axs.azim=-35.
axs.elev=20.
#axs.dist=
Ys, As = np.meshgrid(yy, AA)
ax.plot_surface(As.T,Ys.T,likeSurf, rstride=2, cstride=2, alpha=0.6,cmap=cm.hot)
ax.set
# plot A=2
ax.plot(list(As[39]),list(yy),  list(likeSurf[:,39]), color='lime', linewidth=2,label='p(y|A=2)')
leg=plt.legend(loc='lower left')
ltext  = leg.get_texts()
llines = leg.get_lines()
frame  = leg.get_frame()
plt.title("Sampling from a Bayesian posterior with improper prior (A)")
#plt.show()
plt.savefig('mcmc-metropolis-hastings-improper-prior-A.png')

# DEFINE PRIOR OVER SHAPE PARAMETERS
#prior = inline('sin(pi*A).^2','A');
def prior(A):
	return np.sin(np.pi*A)**2

# DEFINE THE POSTERIOR
# p = inline('(B.^A/gamma(A)).*y.^(A-1).*exp(-(B.*y)).*sin(pi*A).^2','y','A','B');
def p(y,A,B):
 	return (B**A/gamma(A))*y**(A-1.)*np.exp(-(B*y))*np.sin(np.pi*A)**2

# CALCULATE AND DISPLAY THE POSTERIOR SURFACE
postSurf = np.zeros(likeSurf.shape);
for iA in range(AA.size):
	postSurf[:,iA]=p(yy[:],AA[iA],B)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
axs=ax.get_axes()
axs.azim=-35.
axs.elev=20.
Ys, As = np.meshgrid(yy, AA)
ax.plot_surface(As.T,Ys.T,postSurf, rstride=2, cstride=2, alpha=1.0,cmap=cm.hot)
ax.set
# prior
ax.plot(AA,np.ones((1,AA.size)).T*10,prior(AA),color="blue",linewidth=3,label="p(A)")
# posterior (not shadowed properly)
ax.plot(As[:,14],np.ones((1,AA.size)).T*1.6,postSurf[14,:],color="lime",alpha=0.6,linewidth=3,label="p(A|y=1.5)")
leg=plt.legend(loc='lower left')
ltext  = leg.get_texts()
llines = leg.get_lines()
frame  = leg.get_frame()
plt.setp(ltext, fontsize='x-small')
plt.title("Sampling from a Bayesian posterior with improper prior (B)")
#plt.show()
plt.savefig('mcmc-metropolis-hastings-improper-prior-B.png')
 
# INITIALIZE THE METROPOLIS-HASTINGS SAMPLER
# DEFINE PROPOSAL DENSITY
# q = inline('exppdf(x,mu)','x','mu');
def exppdf(x,a=1.0):
    return 1./a * np.exp(-x/a)

def q(x,mu):
	return exppdf(x,mu)

# MEAN FOR PROPOSAL DENSITY
mu = 5.;

fig = plt.figure()
plt.plot(AA,postSurf[14,:],color='lime',linewidth=2,label='target p(A|y=1.5)')
plt.plot(AA,q(AA,mu),color='black',linewidth=2,label='proposal q(A)')
leg=plt.legend(loc='upper right')
ltext  = leg.get_texts()
llines = leg.get_lines()
frame  = leg.get_frame()
plt.title("Proposal Density for MH Sampler")
#plt.show()
plt.savefig('mcmc-metropolis-hastings-proposal-density.png')

# DISPLAY TARGET AND PROPOSAL

# SOME CONSTANTS
nSamples = 5000;
burnIn = 500;
minn = 0.1; maxx = 5.;
 
# INTIIALZE SAMPLER
x = np.zeros((1 ,nSamples));
x[0,0] = mu;
t = 0;
y=1.5
 
# RUN METROPOLIS-HASTINGS SAMPLER
for t in range(nSamples-1):
	
    # SAMPLE FROM PROPOSAL
    xStar = np.random.exponential(mu);
	
    # CORRECTION FACTOR
    c = q(x[0,t],mu)/q(xStar,mu);
	
    # CALCULATE THE (CORRECTED) ACCEPTANCE RATIO
    alpha = np.min([1., p(y,xStar,B)/p(y,x[0,t],B)*c]);
	
    # ACCEPT OR REJECT?
    u = np.random.rand();
    if u < alpha:
        x[0,t+1] = xStar;
    else:
        x[0,t+1] = x[0,t];

# xStar = np.random.exponential(mu); c = q(x[0,t],mu)/q(xStar,mu); p(y,xStar,B)/p(y,x[0,t],B)*c;


# DISPLAY RESULTS
# x-axis steps (t)
step=1
a = np.arange(1,nSamples+step,step)
# plot it
fig = plt.figure(figsize=(8, 6)) 
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
# DISPLAY MARKOV CHAIN
ax0 = plt.subplot(gs[0])
ax0.set_ylim([minn,maxx])
ax0.plot(a, x[0,:])
ax0.axvline(x=burnIn,color='r',linewidth=2)
ax0.text(burnIn+50, 0.2, r'Burn In')
plt.title('Metropolis Hastings Posterior')
# DISPLAY SAMPLES
ax1 = plt.subplot(gs[1])
ax1.set_ylim([minn,maxx])
ax1.xaxis.set_major_locator(pylab.NullLocator())
ax1.yaxis.set_major_locator(pylab.NullLocator())
h=ax1.hist(x[0,burnIn:],bins=200,orientation="horizontal",color='lightgrey',edgecolor = 'grey',label="samples")
#b=np.arange(-10,10,0.1)
#n=stats.norm.pdf(b)
#t=stats.t.pdf(b,1)
#plt.plot(n*nSamples/sum(n),b,color='r',linewidth=2,label="Normal Dist")
#plt.plot(t*nSamples/sum(n),b,color='lime',linewidth=2,label="T-Student Dist")
plt.plot(postSurf[14,:]*(nSamples-burnIn)/sum(postSurf[14,:]),AA, color='lime', linewidth=2,label="posterior");
#plt.ylabel('samples')
leg=ax1.legend(loc='upper right')
ltext  = leg.get_texts()
llines = leg.get_lines()
frame  = leg.get_frame()
plt.setp(ltext, fontsize='x-small')

plt.tight_layout()
#plt.show()

plt.savefig('mcmc-metropolis-hastings-posterior.png')



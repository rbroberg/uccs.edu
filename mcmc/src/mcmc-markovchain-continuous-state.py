# example of continuous markov chain
import numpy as np
import matplotlib.pyplot as plt

# initialize
np.random.seed(271828) 
nBurnin = 50; # BURNIN
nChains = 5;  # MARKOV CHAINS
nTransitions = 1000;
# initialize state space
x = np.zeros((nTransitions,nChains));
x[0,:] = np.random.normal(1,1,size=nChains);

# define transition operator p(x)
def P(x,n):
	return np.random.normal(0.5*x,1,size=n)

# run the markov chains
for iT in range(nTransitions-1):
    x[iT+1,:] = P(x[iT],nChains);

# plot
plt.close('all')
fig = plt.figure()

ax1 = plt.subplot(221) # Burn In
ax2 = plt.subplot(223) # Full Chains for 5 runs
ax3 = plt.subplot(122) # Histogram of Full without Burnin

ax1.plot(x[0:100,:])
ax1.axvline(50,color='k',linewidth=2,label="Burn In")
ax1.text(50+5,np.floor(np.min(x[:100,:])) + 0.5, r'Burn In')
ax1.set_title("First 100 Samples")
ax2.plot(x[:,:])
ax2.axvline(50,color='k',linewidth=2,label="Burn In")
ax2.set_title("Entire Chain")
h=ax3.hist(np.ndarray.flatten(x[100:,:]),bins=100)
ax3.set_title("Markov Chain Samples")
plt.tight_layout()
#plt.show()
plt.savefig('mcmc-markov-chain-continuous-state.png')
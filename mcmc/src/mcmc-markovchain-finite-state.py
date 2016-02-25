# FINITE STATE-SPACE MARKOV CHAIN EXAMPLE
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pylab

# TRANSITION OPERATOR
# S = Sunny
# F = Foggy
# R = Rainy
#  S->S F->S R->S
#  S->F F->F R->F
#  S->R F->R R->R

P = np.array([[ 0.8 ,  0.15,  0.05],
              [ 0.4 ,  0.5 ,  0.1 ],
              [ 0.1 ,  0.3 ,  0.6 ]])

nWeeks = 25

# time series of state vectors
X = np.zeros((nWeeks,3)) 

#INITIAL STATE IS RAINY
X[0,:] = [0,0,1];
 
# RUN MARKOV CHAIN
for iB in range(nWeeks-1):
    X[iB+1,:] = np.dot(X[iB,:],P)

# DISPLAY

plt.plot(X[:,0],color='r',linewidth=2,label='Sunny')
plt.plot(X[:,1],color='k',linewidth=2,label='Foggy')
plt.plot(X[:,2],color='b',linewidth=2,label='Rainy')
plt.axvline(15, color='g',ls='dashed',label='Burn In')
leg=plt.legend(loc='upper right')
ltext  = leg.get_texts()
llines = leg.get_lines()
frame  = leg.get_frame()
plt.title("Markov Chain Weather State Transitions")
#plt.show()
plt.savefig('mcmc-markov-chain-finite-state.png')
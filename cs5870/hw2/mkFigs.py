import matplotlib.pyplot as plt
import numpy as np

datadir = "/projects/uccs.edu/cs5870/data/"
ftour=datadir+"ulysses16.tsp.csv"
fsoln=datadir+"ulysses16.opt.tour.csv"

# read tour locations, drop index
tour = np.genfromtxt(ftour, delimiter=',')
tour = tour[:,1:]

# read optimal path, convert to int
pathidx = np.genfromtxt(fsoln, delimiter=',')
pathidx = pathidx.astype(int)

# prepare opt path array
soln=np.zeros((len(pathidx)+1,2))
for i in range(len(pathidx)):
    soln[i,:]=tour[(pathidx[i]-1),:]

soln[i+1,:]=tour[(pathidx[0]-1),:]


plt.plot(tour[:,0],tour[:,1],'ro')
plt.plot(soln[:,0],soln[:,1],'r-')
cmean=tour.mean(axis=0)
plt.plot(cmean[0],cmean[1],'bo')
plt.show()

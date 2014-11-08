import matplotlib.pyplot as plt
import numpy as np
from math import sin, cos, pi, e
import numpy.random as rnd
import sys

def calcDist(narr):
    l=narr.shape[0]-1
    return np.sum(np.sum((narr[0:l]-narr[1:,])**2,axis=1))


class mapNode:
    def __init__(self,x,y):
        self.x = float(x)
        self.y = float(y)
        self.cities = []
        self.inhibited = False
        self.lastwin = 0
    
    def __repr__(self):
        return('('+str(self.x)+','+str(self.y)+')')
    
    def pos(self):
        return(np.array([self.x,self.y]).astype(float))

#x=mapNode(1,2)
#x
#x.pos()

datadir = "/projects/uccs.edu/cs5870/data/"
ftour=datadir+"ulysses16.tsp.csv"
fsoln=datadir+"ulysses16.opt.tour.csv"

# read tour locations, drop index
tour = np.genfromtxt(ftour, delimiter=',')
tour = tour[:,1:]

# read optimal path, convert to int
pathidx = np.genfromtxt(fsoln, delimiter=',')
pathidx = pathidx.astype(int)

# number of cities
ncity=tour.shape[0]

# number of nodes
nnodes=ncity

# number of neighbors, +/-
nnghbr = 1

# 1/sqrt(2)
sqrt2inv = 1/(2.)**0.5

# center of tour
cmean=tour.mean(axis=0)

# manhatten distance of cities from center
cradius=np.max(np.abs(np.sum(tour - cmean, axis=1)))

# normalize data to a -1x1 box, center and scale
tournorm=(tour-cmean)/cradius

# number of runs
runs=20
results=[]
seeds=[]
dists=[]

for run in range(runs):
    # gain and gain factor
    G=1.0
    alpha=0.02
    
    # create randomized list of the cities, same list used each epoch
    rndidx=[i for i in range(ncity)]
    rseed=rnd.randint(65535)
    rnd.seed(rseed) # change seed to get different permutation
    rnd.shuffle(rndidx)
    
    # set nnodes evenly around bounding circle
    # nodepos=cradius*np.array([[cos(2*pi*i/nnodes),sin(2*pi*i/nnodes)] for i in range(nnodes)])
    
    # first node starts at (0,0)
    n=mapNode(0,0)
    
    # insert first node into ring
    ring=[n]
    
    # one epoch is one loop through pre-randomized city index   
    for e in range(800):
        for ri in rndidx:
            # for each node j, compute potential, find nearest node to city
            cpos=tournorm[ri,]
            npos=np.zeros((len(ring),2))
            for j in range(len(ring)):
                npos[j,:]=ring[j].pos()
            
            ni=np.argmin(np.sum((npos-cpos)**2,axis=1))
            
            # flag that node as city winner
            ring[ni].cities.append(ri)
            
            # move nodes towards the city
            for j in range(len(ring)):
                if not ring[j].inhibited:
                    n = (j-ni)%ncity
                    fG = sqrt2inv * e**(-(n/G)**2)
                    ring[j].x=ring[j].x+fG*(cpos[0]-ring[j].x)
                    ring[j].y=ring[j].y+fG*(cpos[1]-ring[j].y)
        
        G=(1.-alpha)*G
        
        # do this in reverse order
        nring=len(ring)
        idxrev=[nring-i-1 for i in range(nring)]
        for ni in idxrev:
            # clear inhibited flag
            ring[ni].inhibited=False
            
            # check number of winning cities
            if len(ring[ni].cities)==0: 
                # if not a winner, decrement the last win counter
                ring[ni].lastwin = ring[ni].lastwin-1
                # if not a winner in 3 epochs, remove node
                if ring[ni].lastwin < -2:
                    a=ring[0:(ni)]
                    c=ring[(ni+1):]
                    ring=a+c
            elif len(ring[ni].cities)>1: 
                # if winner of more than one city, duplicate node
                a=ring[0:(ni+1)]
                c=ring[(ni+1):]
                b=[mapNode(ring[ni].x,ring[ni].y)]
                ring=a+b+c
                # set inhibition to prevent movement next epoch
                #ring[ni].inhibited=True
                ring[ni].cities=[]
                ring[ni+1].inhibited=True
                # reset winning count
                ring[ni].lastwin=0
                ring[ni+1].lastwin=0
            else:
                # clear winning city list
                ring[ni].cities=[]
    
    somnorm=np.zeros((len(ring)+1,2))
    for i in range(len(ring)):
        somnorm[i]=ring[i].pos()
    
    somnorm[i+1]=ring[0].pos()
    dists.append(calcDist(somnorm))
    results.append(somnorm)
    seeds.append(rseed)
    print(dists[-1], G, len(ring), rseed)
    sys.stdout.flush()



# ===============================
# print fig of least path
# ===============================
ni=np.argmin(np.array(dists))
som_d=dists[ni]
somnorm=results[ni]

# read optimal path, convert to int
pathidx = np.genfromtxt(fsoln, delimiter=',')
pathidx = pathidx.astype(int)

# prepare opt path array
solnnorm=np.zeros((len(pathidx)+1,2))
for i in range(len(pathidx)):
    solnnorm[i,:]=tournorm[(pathidx[i]-1),:]

solnnorm[i+1,:]=tournorm[(pathidx[0]-1),:]
soln_d=calcDist(solnnorm)

plt.plot(tournorm[:,0],tournorm[:,1],'ro')
plt.plot(solnnorm[:,0],solnnorm[:,1],'r-',linewidth=3.0)
plt.plot(somnorm[:,0],somnorm[:,1],'bo')
plt.plot(somnorm[:,0],somnorm[:,1],'b-')
plt.text(-0.25, 0.55,"best distance: "+str(soln_d))
plt.text(-0.25, 0.45,"this distance: "+str(som_d))
plt.show()


plt.hist(dists)
plt.axvline(soln_d)
plt.show()




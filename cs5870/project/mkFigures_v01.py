import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def reshape_crosscorr(diag):
    d=len(diag)
    #n=(int(round((d*2.*4.)**0.5))+1)/2 # form of soln for quad eqn
    n=int(round(d**0.5)) # form of soln for quad eqn
    m=np.ones((n,n))
    k=0
    for i in range(n):
        for j in range((i+1),n):
            m[i,j]=diag[k]
            m[j,i]=diag[k]
            k=k+1
    return m


f="/data/www.kaggle.com/c/seizure-prediction/download/Dog_1/cctime.csv"
dat=np.genfromtxt(f, delimiter=',')


for n in range(0,24):
    data=dat[n,:].reshape(120,120)
    #data=np.abs(data) # used for 'c'
    fig = plt.figure()
    x,y = np.mgrid[0:120, 0:120]
    ax = fig.add_subplot(111)
    #ticks_at = [-abs(data).max(), 0, abs(data).max()]
    ticks_at = [0, .33, .66]
    cax = ax.imshow(data, interpolation='nearest', cmap=cm.OrRd,
                    origin='lower', extent=[0, 120, 0, 120],
                    vmin=ticks_at[0], vmax=ticks_at[-1])
    cbar = fig.colorbar(cax,ticks=ticks_at,format='%1.2g')
    #fig.show()
    fig.savefig('img/fig1a_'+str(n+1).zfill(4)+'.png')
    plt.close(fig)

for n in range(100,124):
    data=dat[n,:].reshape(120,120)
    #data=np.abs(data) # used for 'd'
    fig = plt.figure()
    x,y = np.mgrid[0:120, 0:120]
    ax = fig.add_subplot(111)
    #ticks_at = [-abs(data).max(), 0, abs(data).max()]
    ticks_at = [0, .33, .66]
    cax = ax.imshow(data, interpolation='nearest', cmap=cm.OrRd,
                    origin='lower', extent=[0, 120, 0, 120],
                    vmin=ticks_at[0], vmax=ticks_at[-1])
    cbar = fig.colorbar(cax,ticks=ticks_at,format='%1.2g')
    #fig.show()
    fig.savefig('img/fig1b_'+str(n+1).zfill(4)+'.png')
    plt.close(fig)


#http://codereview.stackexchange.com/questions/24276/correct-implementation-of-a-markov-chain
import pandas as pd
import numpy as np

pdf = pd.read_csv("../data/tmp.e",sep=" ",names=["src","page","pass"])
pdf2=pdf.iloc[:,0:2]
pdf2.src = [hash(p) for p in pdf2.src]
pdf2.page = [hash(p) for p in pdf2.page]
pdf2.src = pd.DataFrame(pdf2.src, dtype='int')
pdf2.page = pd.DataFrame(pdf2.page, dtype='int')

#pdf2.to_csv("../data/tmp.p")

bysrc=pdf2.groupby('src')
dist = bysrc['page'].mean()

def getTrigrams(x):
    #print x.iloc[0]
    tg=[]
    for n in range(len(x)-2):
         tg.append([x.iloc[n],x.iloc[n+1],x.iloc[n+2]])
    return(tg)

trigrams=pdf2.groupby('src').page.apply(getTrigrams)
tflat=[item for sublist in list(trigrams) for item in sublist]
bigrams=[':'.join([str(t[0]),str(t[1])]) for t in tflat]
bidict={}

bmarkov = {i:[] for i in bigrams} 
'''
# for 2.6
bmarkov={}
for i in bigrams:
    bmarkov[i]=[]
'''
for t in tflat:
    a=':'.join([str(t[0]),str(t[1])])
    b=[str(t[2])]
    bmarkov[a] += b

def pageProb(newpage,pagelist):
    return (1.0*sum([page==newpage for page in pagelist])/len(pagelist), len(pagelist))

# for each src
# find the pathProb and rankPath
# may be faster to create probdict and rankdict from trigrams first
tuniq = list(set(map(tuple, tflat)))
pdict = {i:[] for i in tuniq} 
rdict = {i:[] for i in tuniq} 
'''
# for 2.6
pdict={}
for i in tuniq:
    pdict[i]=[]

rdict={}
for i in tuniq:
    rdict[i]=[]
'''
for t in tuniq:
    a=':'.join([str(t[0]),str(t[1])])
    b=str(t[2])
    p,r=pageProb(b,bmarkov[a])
    pdict[t]=p
    rdict[t]=r

def statProbs(x):
    tg=[]
    for n in range(len(x)-2):
         tg.append(pdict[(x.iloc[n],x.iloc[n+1],x.iloc[n+2])])
    if len(x)>2:
        return([np.min(tg),np.mean(tg),np.max(tg),])
    else:
        return([0,0,0])

def statRank(x):
    tg=[]
    for n in range(len(x)-2):
         tg.append(rdict[(x.iloc[n],x.iloc[n+1],x.iloc[n+2])])
    if len(x)>2:
        return([np.min(tg),np.mean(tg),np.max(tg),])
    else:
        return([0,0,0])

pstats=pdf2.groupby('src').page.apply(statProbs)
rstats=pdf2.groupby('src').page.apply(statRank)
srclen=pdf2.groupby('src').page.apply(len)

pmin=[p[0] for p in pstats]
pmean=[p[1] for p in pstats]
pmax=[p[2] for p in pstats]
rmin=[r[0] for r in rstats]
rmean=[r[1] for r in rstats]
rmax=[r[2] for r in rstats]

# normalized
npmin=pmin/max(pmin)
npmean=pmean/max(pmean)
npmax=pmax/max(pmax)
nrmin=1.0*np.array(rmin)/max(rmin)
nrmean=rmean/max(rmean)
nrmax=1.0*np.array(rmax)/max(rmax)
nsrclen=srclen/max(srclen)

X=np.vstack((npmin,npmean,npmax,nrmin,nrmean,nrmax,nsrclen.values)).transpose()


'''
srcset=pdf2.groupby('src').src.apply(set)
srcset[Y==-1]
#850279091    set([850279091])
pdf[pdf2.src==850279091]
                        src                          page  pass
694155  cyber03.imaginet.fr  /images/MOSAIC-logosmall.gif  PASS
694156  cyber03.imaginet.fr     /images/USA-logosmall.gif  PASS
694159  cyber03.imaginet.fr   /images/WORLD-logosmall.gif  PASS
'''

# normalized
lpmin=10.0*np.array(pmin)
lpmean=10.0*np.array(pmean)
lpmax=10.0*np.array(pmax)
lrmin=np.log10(np.array(rmin))
lrmean=np.log10(np.array(rmean))
lrmax=np.log10(np.array(rmax))
lsrclen=np.log10(np.array(srclen))

lrmin[lrmin<0]=0
lrmax[lrmax<0]=0
lrmean[lrmean<0]=0

X2=np.vstack((lpmin,lpmean,lpmax,lrmin,lrmean,lrmax,lsrclen)).transpose()

# ===================
#DBSCAN(eps=0.5, min_samples=5, metric='euclidean', algorithm='auto', leaf_size=30, p=None, random_state=None)[source]
# http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#example-cluster-plot-dbscan-py
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

X=np.vstack((pmin,pmean,pmax,rmin,rmean,rmax,srclen.values)).transpose()
#X=np.vstack((npmin,npmean,npmax,nrmin,nrmean,nrmax,nsrclen.values)).transpose()
X = StandardScaler().fit_transform(X)

# Y100 <= eps 10. => 0 cats , 5 outliers, na silhouette
# Y40 <= eps 4. => 2 cats , 14 outliers, 0.918 silhouette
# Y20 <= eps 2. => 4 cats , 31 outliers, 0.601 silhouette
# Y20 <= eps 2. => 3 cats min samples 20, 71 outliers, 0.601 silhouette
# Y10 <= eps 1. => 17 cats , 111 outliers, 0.297 silhouette
# Y7 <= eps 0.7 => 23 cats , 214 outliers, -0.378 silhouette
# Y5 <= eps 0.5 => 41 cats
# Y7 <= eps 0.7, min_samples 20 => 18 cats

Y = DBSCAN(eps=0.7, min_samples=10).fit(X)

y = Y.labels_
set(list(y))
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(y)) - (1 if -1 in y else 0)
sum(y==-1)
print('Estimated number of clusters: %d' % n_clusters_)
#print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X,y))
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X[0:10000,],y[0:10000]))

for i in set(list(y)):
    print i, sum(y==i)


import matplotlib.pyplot as plt
import matplotlib as mpl

yy=np.array(y)
yy.shape = (yy.shape[0],1)
results=pd.DataFrame(np.hstack((X,yy)))
res=results.sort([7])

plt.scatter(res[0],res[3],c=res[7],s=80,cmap=mpl.cm.Reds,alpha=0.1)
plt.show()

plt.scatter(res[1],res[4],c=res[7],s=80,cmap=mpl.cm.Reds,alpha=0.1)
plt.show()

plt.scatter(res[0],res[2],c=res[7],s=80,cmap=mpl.cm.Reds,alpha=0.5)
plt.show()

plt.scatter(res[3],res[5],c=res[7],s=80,cmap=mpl.cm.Reds,alpha=0.5)
plt.show()

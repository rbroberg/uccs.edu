#http://codereview.stackexchange.com/questions/24276/correct-implementation-of-a-markov-chain
import pandas as pd
import numpy as np
#import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

pdf = pd.read_csv("../data/tmp.e",sep=" ",names=["src","page","pass"])
pdf2=pdf.iloc[:,0:2]
pdf2.src = [hash(p) for p in pdf2.src]
pdf2.page = [hash(p) for p in pdf2.page]
pdf2.src = pd.DataFrame(pdf2.src, dtype='int')
pdf2.page = pd.DataFrame(pdf2.page, dtype='int')

#pdf2.to_csv("../data/tmp.p")

bysrc=pdf2.groupby('src')

# number of page hits by source
dist = bysrc['page'].count()
sourcecnt = len(dist) # 137833 number of unique sources
requestcnt = sum(dist) # 3453449 number page requests
#
np.min(dist) # 1
np.max(dist) # 21988
np.mean(dist) # 25
np.median(dist) # 10

#
np.log(np.min(dist)) # 0
np.log(np.max(dist)) #  9.998
np.log(np.mean(dist)) # 3.22
np.log(np.median(dist)) # 2.30

# bins are 0.1 wide
n, bins, patches = plt.hist(np.log(list(dist)), 100, normed=0, facecolor='green', alpha=0.5)
plt.show()

sum(dist==1) # 9255
1.*sum(dist==2) # 6032
1.*sum(dist==3) # 5423
1.*sum(dist==4) # 9878
1.*sum(dist==5) # 7639
1.*sum(dist==6) # 9337
1.*sum(dist==7) # 4848
1.*sum(dist==8) # 6355
1.*sum(dist==9) # 7920
1.*sum(dist==10) # 4687

1.*sum(dist==1)/len(dist) # 0.0671
1.*sum(dist>=10)/len(dist) # 0.51617


bypage=pdf2.groupby('page')
dist = bypage['src'].count()
pagecnt=len(dist) # 30900 number of unique pages
requestcnt2 = sum(dist) # 3453449 number page requests
requestcnt == requestcnt2 # should be true

#
np.min(dist) # 1
np.max(dist) # 208353
np.mean(dist) # 111.76210355987055
np.median(dist) # 1.0

#
np.log(np.min(dist)) # 0
np.log(np.max(dist)) #  12.2469
np.log(np.mean(dist)) # 4.7163
np.log(np.median(dist)) # 0

# bins are 0.1 wide
n, bins, patches = plt.hist(np.log(list(dist)), 100, normed=0, facecolor='green', alpha=0.5)
plt.show()

sum(dist==1) # 16748
sum(dist==2) # 3428
sum(dist==3) # 1688
sum(dist==4) # 1117
sum(dist==5) # 851
sum(dist==6) # 687
sum(dist==7) # 520
sum(dist==8) # 438
sum(dist==9) # 389
sum(dist==10) # 327

1.*sum(dist==1)/len(dist) # 0.542
1.*sum(dist>=10)/len(dist) # 0.1629

1.* bypage['page'].get_group(1011560224).count() / len(bypage)

def pageProb(pageid):
    return(1.* bypage['page'].get_group(pageid).count() / requestcnt)

def entropy(p):
    return(-1.*p*np.log(p))

entropy(pageProb(-2147228971))
# 0.00000436, page count of 1
entropy(pageProb(2144881757))
# 0.0001759, page count of 55
entropy(pageProb(806440580))
# 0.16940552458478625, page count of 208353

def getEntropies(x):
    return(np.product([entropy(pageProb(y)) for y in x]))

src_ents=bysrc.page.apply(getEntropies)

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

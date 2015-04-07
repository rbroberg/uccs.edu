
import pandas as pd
pdf = pd.read_csv("../data/tmp.e",sep=" ",names=["src","page","pass"])
cnt = pd.read_fwf("../data/freq.count", names=["freq","page"])

for i in range(len(cnt)):
    if (i%100==0): print i
    pdf.loc[pdf.page==cnt.iloc[i,1]].page=cnt.iloc[i,0]

dummy=[freqdict[cnt.iloc[i,1]]=cnt.iloc[i,0] for i in range(len(cnt))]

freqdict={}
for i in range(len(cnt)):
    try:
        freqdict[cnt.iloc[i,1]]=cnt.iloc[i,0]
    except:
        print("failed on " + str(i), cnt.iloc[i,:])

for i in range(len(pdf)):
    if (i%1000==0): print i
    try:
        pdf.loc[i].page=freqdict[pdf.loc[i].page]
    except:
        print("failed on " + str(i), pdf.iloc[i,:])


pdf2=pdf[pdf.applymap(np.isreal).page]
pdf2=pdf2.iloc[:,0:2]
pdf2.page=pdf2.page.astype(float)
pdf2.to_csv("../data/tmp.o")

'''
>>> len(pdf2)
3436367
>>> len(pdf)
3453449
>>> len(pdf)-len(pdf2)
17082
>>> (len(pdf)-len(pdf2))/len(pdf)
0
>>> 1.*(len(pdf)-len(pdf2))/len(pdf)
0.004946359422131325
'''

bysrc=pdf2.groupby('src')
dist = bysrc['page'].mean()

'''
>>> dist.describe()
count    137540.000000
mean      55538.201133
std       35080.060349
min           1.000000
25%       27540.264286
50%       53572.410714
75%       75914.000000
max      208798.000000
dtype: float64
>>> sum(dist<100)
675
>>> 1.0*sum(dist<100)/len(dist) 
0.004907663225243566
'''


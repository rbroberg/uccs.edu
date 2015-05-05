#http://codereview.stackexchange.com/questions/24276/correct-implementation-of-a-markov-chain
import pandas as pd
import numpy as np
#import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy import stats

pdf = pd.read_csv("../data/tmp.e",sep=" ",names=["src","page","pass"])
pdf2=pdf.iloc[:,0:2]
pdf2.src = [hash(p) for p in pdf2.src]
pdf2.page = [hash(p) for p in pdf2.page]
pdf2.src = pd.DataFrame(pdf2.src, dtype='int')
pdf2.page = pd.DataFrame(pdf2.page, dtype='int')

bysrc=pdf2.groupby('src')

# number of page hits by source
dist = bysrc['page'].count()
sourcecnt = len(dist) # 137833 number of unique sources
requestcnt = sum(dist) # 3453449 number page requests

bypage=pdf2.groupby('page')

dist = bypage['src'].count()
pagecnt=len(dist) # 30900 number of unique pages
requestcnt2 = sum(dist) # 3453449 number page requests
requestcnt == requestcnt2 # should be true

pgfreq=list(dist)
pgfreq.sort(reverse=True)
logfreq=np.log(pgfreq)
logranks=[np.log(i+1) for i in range(len(pgfreq))]

gradient, intercept, r_value, p_value, std_err = stats.linregress(logranks,logfreq)

z = np.polyfit(logranks[0:1000],logfreq[0:1000], 1)
p = np.poly1d(z)

plt.figure(figsize=(6,5))
plt.plot(logranks,logfreq, label='page requests')
plt.plot(logranks,p(logranks),"r-", label="linear fit first 1000 pages")
plt.title("Zipf's Law in Nasa Apache Access Logs")
plt.xlabel('log(page rank)')
plt.ylabel('log(page frequency)')
leg = plt.legend(loc='upper right',prop={'size':12})
leg.get_frame().set_linewidth(0.0)
#plt.show()
plt.savefig('zipf-apache-log.png')





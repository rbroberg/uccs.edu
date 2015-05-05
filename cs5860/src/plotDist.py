#http://codereview.stackexchange.com/questions/24276/correct-implementation-of-a-markov-chain
import pandas as pd
import numpy as np
#import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
ptbk = 24
n, bins, patches = plt.hist(np.log(list(dist)), 100, normed=0, facecolor='green', alpha=0.6)
z = np.polyfit(bins[ptbk:90],np.log(n[ptbk:90]+1), 1)
p = np.poly1d(z)
lowshelf=np.exp(p(bins[ptbk]))
plt.plot([0,bins[ptbk]],[lowshelf,lowshelf],"r-", label="model fit",linewidth=2.0)
plt.plot(bins[ptbk:90],np.exp(p(bins[ptbk:90])),"r-",linewidth=2.0)
plt.title("Number of Web Page Requests Per Source")
plt.ylabel('number of visitors (sources)')
plt.xlabel('log(number of web page requests)')
#plt.show()
plt.savefig('plot_bysrc.png')

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
ptbk = 0
ptbk2 = 12

n, bins, patches = plt.hist(np.log(list(dist)), 18, normed=0, facecolor='green', alpha=0.6)
z = np.polyfit(bins[ptbk:ptbk2],np.log(n[ptbk:ptbk2]+1), 1)
p = np.poly1d(z)
lowshelf=np.exp(p(bins[ptbk]))
#plt.plot([0,bins[ptbk]],[lowshelf,lowshelf],"r-", label="model fit",linewidth=2.0)
plt.plot(bins[ptbk:ptbk2],np.exp(p(bins[ptbk:ptbk2])),"r-",linewidth=2.0)
plt.title("Number of Visitors Per Unique Web Page ")
plt.ylabel('number of visits to this page')
plt.xlabel('log(number of pages with this many vists)')
#plt.show()
plt.savefig('plot_bypage.png')


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

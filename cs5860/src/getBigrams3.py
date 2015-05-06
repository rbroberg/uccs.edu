#http://codereview.stackexchange.com/questions/24276/correct-implementation-of-a-markov-chain
import pandas as pd
import numpy as np
#import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit
from collections import Counter # 2.7
import pickle
import os.path

def save_obj(obj, name ):
    print("saving "+name)
    with open('../data/e10/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    print("loading "+name)
    with open('../data/e10/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


pdf = pd.read_csv("../data/tmp.e10",sep=" ",names=["src","page","pass"])
# save dict of hashes to rediscover web pages
pageset = set(pdf['page'])
hashpagedict=dict([(str(hash(page)),page) for page in list(pageset)])
pagehashdict=dict([(page,str(hash(page))) for page in list(pageset)])
save_obj(hashpagedict,"tmp.hashpagedict")
save_obj(pagehashdict,"tmp.pagehashdict")

srcset = set(pdf['src'])
hashsrcdict=dict([(str(hash(src)),src) for src in list(srcset)])
srchashdict=dict([(src,str(hash(src))) for src in list(srcset)])
save_obj(hashsrcdict,"tmp.hashsrcdict")
save_obj(srchashdict,"tmp.srchashdict")

# 
badpages=list(set(pdf[pdf['pass']=='FAIL']['page']))
save_obj(badpages,"tmp.badpages")


pdf2=pdf.iloc[:,0:2]
pdf2.src = [str(hash(p)) for p in pdf2.src]
pdf2.page = [str(hash(p)) for p in pdf2.page]
pdf2.src = pd.DataFrame(pdf2.src, dtype='str')
pdf2.page = pd.DataFrame(pdf2.page, dtype='str')

bypage=pdf2.groupby('page')
pagecnt=bypage.count()
unigramdict=pagecnt['src'].to_dict()
for k in unigramdict.keys():
    unigramdict[k]=int(unigramdict[k])

save_obj(unigramdict,"tmp.unigramdict")

def getBigrams(x):
    tg=[]
    for n in range(len(x)-1):
         tg.append([str(x.iloc[n]),str(x.iloc[n+1])])
    return(tg)

def getTrigrams(x):
    tg=[]
    for n in range(len(x)-2):
         tg.append([str(x.iloc[n]),str(x.iloc[n+1]),str(x.iloc[n+2])])
    return(tg)

# -1005790194    [[-863432790, 1535905453], [1535905453, -15011...
bigrams=pdf2.groupby('src').page.apply(getBigrams)
save_obj(bigrams,"tmp.bigrams")

# [['-863432790', '1535905453'], ['1535905453', '-1501131867'], ...
bflat=[item for sublist in list(bigrams) for item in sublist]
save_obj(bflat,"tmp.bflat")

# ['716482353:477775975', '477775975:716482353', '716482353:1862355765', ...]
bigrams2=[':'.join([str(t[0]),str(t[1])]) for t in bflat]
save_obj(bigrams2,"tmp.bigrams2")

# Counter is the program
bigramcnt=Counter(bigrams2)
save_obj(bigramcnt,"tmp.bigramcnt")

# ['544613893:-1825626037', '-1693265907:-170875555', '1527289180:-189339146']
# bigramcnt.keys()[0:3]
# [1, 1, 1]
# bigramcnt.values()[0:3]
sum(bigramcnt.values()) # 3315616 ## 31645
len(bigramcnt.keys()) # 228180 ## 8867

# given A, what are the possible A:B bigrams
bigramdict={}
for i in bigramcnt.keys():
    a=i.split(':')[0]
    try:
        bigramdict[a]=bigramdict[a]+[i]
    except:
        bigramdict[a]=[i]

save_obj(bigramdict,"tmp.bigramdict")

# =========================================================================

# -1005790194    [[-863432790, 1535905453, -1501131867], [15359...
trigrams=pdf2.groupby('src').page.apply(getTrigrams)
save_obj(trigrams,"tmp.trigrams")

# [['-863432790', '1535905453', '-1501131867'], ['1535905453', '-1501131867', ...
tflat=[item for sublist in list(trigrams) for item in sublist]
save_obj(tflat,"tmp.tflat")

# -1005790194    [[-863432790, 1535905453, -1501131867], [15359...
trigrams2=[':'.join([str(t[0]),str(t[1]),str(t[2]),]) for t in tflat]
save_obj(trigrams2,"tmp.trigrams2")

# Counter is the program
trigramcnt=Counter(trigrams2)
save_obj(trigramcnt,"tmp.trigramcnt")

# ['544613893:-1825626037', '-1693265907:-170875555', '1527289180:-189339146']
# trigramcnt.keys()[0:3]
# [1, 1, 1]
# trigramcnt.values()[0:3]
sum(trigramcnt.values()) # xxxx ## 29026
len(trigramcnt.keys()) # xxxx ## 13933

# given A, what are the possible A:B trigrams
trigramdict={}
for i in trigramcnt.keys():
    a=':'.join(i.split(':')[0:2])
    try:
        trigramdict[a]=trigramdict[a]+[i]
    except:
        trigramdict[a]=[i]

save_obj(trigramdict,"tmp.trigramdict")

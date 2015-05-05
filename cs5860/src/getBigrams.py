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
    with open('../data/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('../data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


pdf = pd.read_csv("../data/tmp.e",sep=" ",names=["src","page","pass"])
pdf2=pdf.iloc[:,0:2]
pdf2.src = [hash(p) for p in pdf2.src]
pdf2.page = [hash(p) for p in pdf2.page]
pdf2.src = pd.DataFrame(pdf2.src, dtype='int')
pdf2.page = pd.DataFrame(pdf2.page, dtype='int')

#pdf2.to_csv("../data/tmp.p")

bysrc=pdf2.groupby('src')
bypage=pdf2.groupby('page')
pagecnt=bypage.count()
pagecntdict=pagecnt.to_dict()
unigramdict=pagecnt['src'].to_dict()
#unigramdict.keys()[0:10]
#unigramdict[int('2007143504')] # 47258

def biMI(ab, unigramdict, bigramdict):
    bfreq=sum(bigramdict.values())
    ufreq=sum(unigramdict.values())
    pa=1.*unigramdict[int(ab.split(':')[0])]/ufreq
    pb=1.*unigramdict[int(ab.split(':')[1])]/ufreq
    pab=1.*bigramdict[ab]/bfreq
    return np.log2(pab/(pa*pb))

def getBigrams(x):
    tg=[]
    for n in range(len(x)-1):
         tg.append([x.iloc[n],x.iloc[n+1]])
    return(tg)

def getBIMI(x):
    return [biMI(':'.join([str(b[0]),str(b[1])]),unigramdict, bcntdict) for b in x]

def triMI(abc, unigramdict, trigramdict):
    tfreq=sum(trigramdict.values())
    ufreq=sum(unigramdict.values())
    pa=1.*unigramdict[int(abc.split(':')[0])]/ufreq
    pb=1.*unigramdict[int(abc.split(':')[1])]/ufreq
    pc=1.*unigramdict[int(abc.split(':')[2])]/ufreq
    pab=1.*bigramdict[abc]/tfreq
    return np.log2(pab/(pa*pb*pc))

def getTrigrams(x):
    tg=[]
    for n in range(len(x)-2):
         tg.append([x.iloc[n],x.iloc[n+1],x.iloc[n+2]])
    return(tg)

def getTRIMI(x):
    return [triMI(':'.join([str(b[0]),str(b[1]),str(b[2])]),unigramdict, tcntdict) for b in x]

#''
# for each source, calculate bigrams
if os.path.isfile('../data/tmp.bigrams.pkl'):
    bigrams=load_obj(bigrams)
else:
    bigrams=pdf2.groupby('src').page.apply(getBigrams)
    save_obj(bigrams,"tmp.bigrams")


# for all bigrams, calc prob B given A
# first, flatten src 'sensitive' bigram list
# [[716482353,477775975], [477775975,716482353], [716482353,1862355765], ...]
if os.path.isfile('../data/tmp.bflat.pkl'):
    bflat=load_obj(bflat)
else:
    bflat=[item for sublist in list(bigrams) for item in sublist]
    save_obj(bflat,"tmp.bflat")


# ['716482353:477775975', '477775975:716482353', '716482353:1862355765', ...]
if os.path.isfile('../data/tmp.bigrams2.pkl'):
    bgrams2=load_obj(bgrams2)
else:
    bigrams2=[':'.join([str(t[0]),str(t[1])]) for t in bflat]
    save_obj(bigrams2,"tmp.bgrams2")

# bidict=[':'.join([str(t[0]),str(t[1])]) for t in bflat]

# Counter is the program
if os.path.isfile('../data/tmp.bcntdict.pkl'):
    bcntdict=load_obj(bcntdict)
else:
    bcntdict=Counter(bigrams2)
    save_obj(bcntdict,"tmp.bcntdict")


sum(bcntdict.values()) # 3315616
len(bcntdict.keys()) # 228180

# given A, what are the possible A:B bigrams
if os.path.isfile('../data/tmp.bitdict.pkl'):
    bidict=load_obj(bidict)
else:
    bidict={}
    for i in bcntdict.keys():
        a=i.split(':')[0]
        try:
            bidict[a]=bidict[a]+[i]
        except:
            bidict[a]=[i]
    save_obj(bidict,"tmp.bidict")


# lots of combinations
#[bcntdict[i] for i in bidict['2007143504']]

#biMI('2007143504:-994741562', unigramdict, bcntdict)


#bysrc.get_group(-1061628487)

#bimi=bigrams.apply(getBIMI)
#''

# ===============================

if os.path.isfile('../data/tmp.trigrams.pkl'):
    trigrams=load_obj(trigrams)
else:
    trigrams=pdf2.groupby('src').page.apply(getTrigrams)
    save_obj(trigrams,"tmp.trigrams")

if os.path.isfile('../data/tmp.tflat.pkl'):
    tflat=load_obj(tflat)
else:
    tflat=[item for sublist in list(trigrams) for item in sublist]
    save_obj(bflat,"tmp.tflat")


# ['716482353:477775975', '477775975:716482353', '716482353:1862355765', ...]
if os.path.isfile('../data/tmp.tgrams2.pkl'):
    tgrams2=load_obj(tgrams2)
else:
    tgrams2=[':'.join([str(t[0]),str(t[1])]) for t in tflat]
    save_obj(tgrams2,"tmp.tgrams2")

# tridict=[':'.join([str(t[0]),str(t[1])]) for t in bflat]

# Counter is the program
if os.path.isfile('../data/tmp.tcntdict.pkl'):
    tcntdict=load_obj(bcntdict)
else:
    tcntdict=Counter(tgrams2)
    save_obj(tcntdict,"tmp.tcntdict")


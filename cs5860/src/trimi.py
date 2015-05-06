#http://codereview.stackexchange.com/questions/24276/correct-implementation-of-a-markov-chain
from collections import Counter # 2.7
import pickle
import os.path
import math

def save_obj(obj, name ):
    with open('../data/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('../data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def biMI(ab, unigramcnt, bigramcnt):
    bfreq=sum(bigramcnt.values())
    ufreq=sum(unigramcnt.values())
    pa=1.*unigramcnt[ab.split(':')[0]]/ufreq
    pb=1.*unigramcnt[ab.split(':')[1]]/ufreq
    pab=1.*bigramcnt[ab]/bfreq
    return math.log(pab/(pa*pb),2)

def getBigrams(x):
    tg=[]
    for n in range(len(x)-1):
         tg.append([x.iloc[n],x.iloc[n+1]])
    return(tg)

def getBIMI(x):
    return [biMI(':'.join(b),unigramcnt, bigramcnt) for b in x]

def triMI(abc, unigramcnt, trigramcnt):
    tfreq=sum(trigramcnt.values())
    ufreq=sum(unigramcnt.values())
    pa=1.*unigramcnt[abc.split(':')[0]]/ufreq
    pb=1.*unigramcnt[abc.split(':')[1]]/ufreq
    pc=1.*unigramcnt[abc.split(':')[2]]/ufreq
    pab=1.*trigramcnt[abc]/tfreq
    return math.log(pab/(pa*pb*pc),2)

def getTrigrams(x):
    tg=[]
    for n in range(len(x)-2):
         tg.append([x.iloc[n],x.iloc[n+1],x.iloc[n+2]])
    return(tg)

def getTRIMI(x):
    return [triMI(':'.join(b),unigramcnt, trigramcnt) for b in x]


unigramcnt=load_obj('tmp.udict')
trigramcnt=load_obj('tmp.tcntdict')
tdict=load_obj('tmp.tdict')

trimi=[getTRIMI(tdict[x]) for x in tdict]
save_obj(trimi,"tmp.trimi2")


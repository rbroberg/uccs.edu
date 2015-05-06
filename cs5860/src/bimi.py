#http://codereview.stackexchange.com/questions/24276/correct-implementation-of-a-markov-chain
from collections import Counter # 2.7
import pickle
import os.path
import math

def save_obj(obj, name ):
    with open('../data/e100/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('../data/e10/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

# pab_a == pab?
#def biMI(ab, pab_a, unigramcnt, bigramcnt):
def biMI(ab, unigramcnt, bigramcnt):
    bfreq=sum(bigramcnt.values())
    ufreq=sum(unigramcnt.values())
    pa=1.*unigramcnt[ab.split(':')[0]]/ufreq
    pb=1.*unigramcnt[ab.split(':')[1]]/ufreq
    pab=1.*bigramcnt[ab]/bfreq
    #return math.log((pb*pa)/(pb),2)
    #return math.log(pa,2) # promising
    #return math.log((pab)/(pb),2)
    #return math.log(pab * pa/pb,2)
    #return math.log(pab_a * pa/pb,2)
    return math.log(pab * pa/pb,2)

def getBIMI(x):
    #return [biMI(b,1.*bigramcnt[b]/sum([bigramcnt[y] for y in x]),unigramcnt, bigramcnt) for b in x]
    return [biMI(b,unigramcnt, bigramcnt) for b in x]

def getNBP2(x):
    return([math.log(1.*bigramcnt[b]/sum([bigramcnt[y] for y in x]),2) for b in x])

def triMI(abc, unigramcnt, bigramcnt, trigramcnt):
    tfreq=sum(trigramcnt.values())
    bfreq=sum(bigramcnt.values())
    ufreq=sum(unigramcnt.values())
    a=abc.split(':')[0]
    b=abc.split(':')[1]
    c=abc.split(':')[2]
    ab=":".join([a,b])
    ac=":".join([a,c])
    bc=":".join([b,c])
    pa=1.*unigramcnt[a]/ufreq
    pb=1.*unigramcnt[b]/ufreq
    pc=1.*unigramcnt[c]/ufreq
    #pab=(1.*bigramcnt[ab]/bfreq)
    pbc=(1.*bigramcnt[bc]/bfreq)
    #pa_b=(1.*bigramcnt[ab]/bfreq)/pb
    #pc_b=(1.*bigramcnt[ac]/bfreq)/pb
    pabc=1.*trigramcnt[abc]/tfreq
    #pcab=(1.*trigramcnt[ac]/tfreq)/pb_a
    #return math.log(pab/(pa*pb*pc),2)
    #return math.log(pa * pba * pcab ,2)
    #return math.log(pa*pbc/pbc,2) # promising  
    return math.log(pabc*pa/pbc,2)

def getTRIMI(x):
    return [triMI(b,unigramcnt, bigramcnt, trigramcnt) for b in x]

# unigramcnt[unigramcnt.keys()[0]] ## 27
unigramcnt=load_obj('tmp.unigramdict')

# bigramcnt[bigramcnt.keys()[0]] ## 1
bigramcnt=load_obj('tmp.bigramcnt')

# bdict[bdict.keys()[0]] ## ['106582840:-1977522313', '106582840:-2146217110', ...
bdict=load_obj('tmp.bigramdict')

#
hashpagedict=load_obj('tmp.hashpagedict')
badpages=load_obj('tmp.badpages')

bimi=[getBIMI(bdict[x]) for x in bdict.keys()]
#bimi=[getNBP2(bdict[x]) for x in bdict.keys()]
#save_obj(bimi,"tmp.bimi2")

#x=bdict[bdict.keys()[0]]
#b=x[0]
#biMI(b,unigramcnt, bigramcnt)

# find negative MI bigrams
# bimi[4] has 5 negative bigrams [5,6,14,18,21]
#bdict[bdict.keys()[4]]

# bdict[bdict.keys()[4]][5]
## '1170436880:1663343820'
# bdict[bdict.keys()[4]][6]
## '1170436880:806440580'
# bdict[bdict.keys()[4]][14]
## '1170436880:569593942'
# bdict[bdict.keys()[4]][18]
## '1170436880:1894851142'
# bdict[bdict.keys()[4]][21]
## '1170436880:1369747478'
#hashpagedict['1170436880']

n=0
m=0
#bad=set()
bad=0
for i in range(len(bimi)):
    for j in range(len(bimi[i])):
        m=m+1
        if hashpagedict[bdict[bdict.keys()[i]][j].split(':')[0]] == '/test/me':
            print bimi[i]
        if bimi[i][j] < -25.9:
            n=n+1
            if hashpagedict[bdict[bdict.keys()[i]][j].split(':')[0]] == '/test/me':
                bad=bad+1
            if hashpagedict[bdict[bdict.keys()[i]][j].split(':')[1]] == '/test/me':
                bad=bad+1

print bad,n,m,len(badpages)

'''
            #print(hashpagedict[bdict[bdict.keys()[i]][j].split(':')[0]] in badpages," >>> ",hashpagedict[bdict[bdict.keys()[i]][j].split(':')[1]] in badpages)
            #if hashpagedict[bdict[bdict.keys()[i]][j].split(':')[0]] in badpages:
                #bad.add(hashpagedict[bdict[bdict.keys()[i]][j].split(':')[1]])
                #print(hashpagedict[bdict[bdict.keys()[i]][j].split(':')[0]]+" >>> "+hashpagedict[bdict[bdict.keys()[i]][j].split(':')[1]])
                #print(hashpagedict[bdict[bdict.keys()[i]][j].split(':')[0]] in badpages," >>> ",hashpagedict[bdict[bdict.keys()[i]][j].split(':')[1]] in badpages)





# sum([v==1 for v in bv]) ## 6484 unique bigrams in 8871 bigrams


# unigramcnt[unigramcnt.keys()[0]] ## 27
# unigramcnt=load_obj('tmp.unigramdict')

# trigramcnt[trigramcnt.keys()[0]] ## 1
trigramcnt=load_obj('tmp.trigramcnt')

# tdict[tdict.keys()[0]] ## ['106582840:-1977522313', '106582840:-2146217110', ...
tdict=load_obj('tmp.trigramdict')

trimi=[getTRIMI(tdict[x]) for x in tdict.keys()]

n=0
m=0
#bad=set()
bad=0

for i in range(len(trimi)):
    for j in range(len(trimi[i])):
        m=m+1
        if hashpagedict[tdict[tdict.keys()[i]][j].split(':')[0]] == '/test/me':
            print trimi[i]
        if trimi[i][j] < -20.6:
            n=n+1
            if hashpagedict[tdict[tdict.keys()[i]][j].split(':')[0]] == '/test/me':
                bad=bad+1

print bad,n,m,len(badpages)


'''
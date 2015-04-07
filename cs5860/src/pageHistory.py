#http://codereview.stackexchange.com/questions/24276/correct-implementation-of-a-markov-chain
import pandas as pd
#ph = pd.read_csv("../data/page.hash",sep=" ",names=["hash","page"])
pdf = pd.read_csv("../data/tmp.e",sep=" ",names=["src","page","pass"])

# bad bytes in my window file
#ph.hash[0]=1

'''
pagedict={}
for i in range(len(ph)):
    try:
        pagedict[ph.iloc[i,1]]=int(ph.iloc[i,0])
    except:
        print("failed on " + str(i), ph.iloc[i,:])

for i in range(len(pdf)):
    if (i%1000==0): print i
    try:
        pdf.loc[i].page=str(pagedict[pdf.loc[i].page])
    except:
        pdf.loc[i].page=str(-9999)
        print("failed on " + str(i), pdf.iloc[i,:])

pdf.page = pd.DataFrame(pdf.page, dtype='float')
pdf2=pdf[pdf.page!=-9999]
pdf2.page = pd.DataFrame(pdf2.page, dtype='int')
'''

pdf2=pdf.iloc[:,0:2]
pdf2.src = [hash(p) for p in pdf2.src]
pdf2.page = [hash(p) for p in pdf2.page]
pdf2.src = pd.DataFrame(pdf2.src, dtype='int')
pdf2.page = pd.DataFrame(pdf2.page, dtype='int')

pdf2.to_csv("../data/tmp.p")
#http://codereview.stackexchange.com/questions/24276/correct-implementation-of-a-markov-chain
import pandas as pd
ph = pd.read_csv("../data/page.hash",sep=" ",names=["hash","page"])
pdf = pd.read_csv("../data/tmp.e",sep=" ",names=["src","page","pass"])

# bad bytes in my window file
ph.hash[0]=1

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


pdf2=pdf[pdf.applymap(np.isreal).page]
mypage = pd.DataFrame(pdf.page, dtype='float')

pdf2=pdf[pdf.applymap(int).page]

pdf2=pdf2.iloc[:,0:2]
pdf2.page=pdf2.page.astype(integer)
pdf2.to_csv("../data/tmp.p")
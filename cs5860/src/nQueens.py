import numpy as np
import random as rnd

# assumes square diagonal
def countDiagonals(A):
    n=int(A.shape[0])
    sums=[]
    for i in range(n):
        sums.append(np.trace(A[i:n,0:(n-i)]))
        sums.append(np.trace(A[0:(n-i),i:n]))
        sums.append(antiTrace(A[0:(i+1),0:(i+1)]))
        sums.append(antiTrace(A[i:n,i:n]))
    return sums

def antiTrace(A):
    n=int(A.shape[0])
    sum=0
    for i in range(n):
        sum = sum + A[i,n-i-1]
    return sum

# number of queens
N=8

# create lists of possible rows and cols

# place random queens, checking compliance each placement
ITER = 1000000
iter1 = 0
while iter1 < ITER:
    rows=[i for i in range(N)]
    cols=[i for i in range(N)]
    nq = 0
    occ=[]
    while nq < N:
        # cannot repeat a row
        x=rnd.randint(0,N-nq-1)
        x=rows.pop(x)
        # cannot repeat a col
        y=rnd.randint(0,N-nq-1)
        y=cols.pop(y)
        occ.append([x,y])
        nq = nq + 1
        
    # place proposed solution on board
    B=np.zeros((N,N))
    for o in occ:
        B[o[0],o[1]]=1
    
    # count diagonals; no diagonal can sum more than '1'
    z=countDiagonals(B)
    if sum([i>1 for i in z])<1:
        # this is valid solution!
        print "valid solution"
        print(occ)
        #break
    iter1=iter1+1

print "finished runs: "+str(iter1)
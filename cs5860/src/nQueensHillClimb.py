import numpy as np
import random as rnd
import math

# replace with lookup list for performance
def nCr(n,r):
    f = math.factorial
    try:
        ncr= f(n) / f(r) / f(n-r)
    except:
        ncr = 0
    return ncr

def d1tod2(i,N):
    x=i/N
    y=i-x*N
    return [x,y]

def d2tod1(p,N):
    return p[0]*N+p[1]

def board2list(B):
    l=[]
    (rows,cols)=B.shape
    for r in range(rows):
        for c in range(cols):
            if B[r,c]>0:
                l.append([r,c])
    return l

def list2board(ll):
    n=len(ll) # assumes board dim = number of queens
    B=np.zeros((n,n),dtype=np.int)
    for l in ll:
        p=d1tod2(l,n)
        B[p[0],p[1]]=1
    return B

def points2board(ll):
    n=len(ll) # assumes board dim = number of queens
    B=np.zeros((n,n),dtype=np.int)
    for p in ll:
        B[p[0],p[1]]=1
    return B

# use to initialize random board
def randomList(N):
    return rnd.sample(range(0,N*N), N)

# assumes square diagonal
# double counts main diagonal
def scoreDiagonals(B):
    n=int(B.shape[0])
    sums=[]
    for i in range(1,n):
        sums.append(np.trace(B[i:n,0:(n-i)]))
        sums.append(np.trace(B[0:(n-i),i:n]))
        sums.append(antiTrace(B[0:i,0:i]))
        sums.append(antiTrace(B[i:n,i:n]))
    # add a negative of main trace and anti-trace
    # to correct for double counting
    sums.append(np.trace(B))
    sums.append(antiTrace(B))
    # anti trace of B included above
    return sum([nCr(s,2) for s in sums])

def scoreCols(B):
    return sum([nCr(s,2) for s in np.sum(B,axis=0)])

def scoreRows(B):
    return sum([nCr(s,2) for s in np.sum(B,axis=1)])

def antiTrace(B):
    n=int(B.shape[0])
    sum=0
    for i in range(n):
        sum = sum + B[i,n-i-1]
    return sum

def scoreBoard(B):
    return scoreCols(B)+scoreRows(B)+scoreDiagonals(B)

# the queen can move in eight directions, clockwise
# N, NE, E, SE, S, SW, W, NW
# find how far the can the queen move in each direction

# this is sample in Marsland
# pieces=[[3,3],[4,4],[5,5],[6,6],[4,0],[5,1],[6,2],[5,7]]

#p= some point on board
#vv = eight vectors from pB
#pieces = board2list(B)

def findSolution(B,ITER,nscale):
    #ITER=100
    bests=[[scoreBoard(B),B]]
    bhist=[]
    pieces=board2list(B)
    score_start=scoreBoard(B)
    # call the solution "done" when score is unchanged for N rounds
    stag=[score_start]*nscale
    iters=0;
    for iter in range(ITER):
        iters=iters+1;
        boards=[]
        bhist.append([scoreBoard(B),B])
        for pn in range(N):
            p=pieces[pn]
            Bnew=np.copy(B)
            Bnew[p[0],p[1]]=0
            # North
            for i in range(1,N):
                if p[0]-i < 0:
                    break
                if B[p[0]-i,p[1]]>0:
                    break
                Bthis = np.copy(Bnew)
                Bthis[p[0]-i,p[1]]=1
                boards.append([scoreBoard(Bthis),Bthis])
            
            # North-East
            for i in range(1,N):
                if p[0]-i < 0:
                    break
                if p[1]+i > N-1:
                    break
                if B[p[0]-i,p[1]+i]>0:
                    break
                Bthis = np.copy(Bnew)
                Bthis[p[0]-i,p[1]+i]=1
                boards.append([scoreBoard(Bthis),Bthis])
            
            # East
            for i in range(1,N):
                if p[1]+i > N-1:
                    break
                if B[p[0],p[1]+i]>0:
                    break
                Bthis = np.copy(Bnew)
                Bthis[p[0],p[1]+i]=1
                boards.append([scoreBoard(Bthis),Bthis])
            
            # South-East
            for i in range(1,N):
                if p[0]+i > N-1:
                    break
                if p[1]+i > N-1:
                    break
                if B[p[0]+i,p[1]+i]>0:
                    break
                Bthis = np.copy(Bnew)
                Bthis[p[0]+i,p[1]+i]=1
                boards.append([scoreBoard(Bthis),Bthis])
            
            # South
            for i in range(1,N):
                if p[0]+i > N-1:
                    break
                if B[p[0]+i,p[1]]>0:
                    break
                Bthis = np.copy(Bnew)
                Bthis[p[0]+i,p[1]]=1
                boards.append([scoreBoard(Bthis),Bthis])
            
            # South-West
            for i in range(1,N):
                if p[0]+i > N-1:
                    break
                if p[1]-i < 0:
                    break
                if B[p[0]+i,p[1]-i]>0:
                    break
                Bthis = np.copy(Bnew)
                Bthis[p[0]+i,p[1]-i]=1
                boards.append([scoreBoard(Bthis),Bthis])
            
            # West
            for i in range(1,N):
                if p[1]-i < 0:
                    break
                if B[p[0],p[1]-i]>0:
                    break
                Bthis = np.copy(Bnew)
                Bthis[p[0],p[1]-i]=1
                boards.append([scoreBoard(Bthis),Bthis])
            
            # North-West
            for i in range(1,N):
                if p[0]-i < 0:
                    break
                if p[1]-i < 0:
                    break
                if B[p[0]-i,p[1]-i]>0:
                    break
                Bthis = np.copy(Bnew)
                Bthis[p[0]-i,p[1]-i]=1
                boards.append([scoreBoard(Bthis),Bthis])
        
        # find a set of best boards (minimal threat pairs)
        for b in boards:
            if b[0]<bests[0][0]:
                bests = [b]
            elif b[0]==bests[0][0]:
                bests.append(b)
               
        # did we "win" - no threats on board?
        if bests[0][0] == 0:
            '''
            for b in bests:
                print "#==========================#"
                print b[1]
                print "Score: "+str(b[0])
                print "Iterations: " +str(iter)
                print "#==========================#"
                print
            '''
            break
        
        # check stagnate level
        stag=stag[1:]+[bests[0][0]]
        if sum([s == stag[0] for s in stag]) == len(stag):
            break
        
        # pick a random best board
        ridx = rnd.randint(0,len(bests)-1)
        #ridx = 0 # for testing
        B = np.copy(bests[ridx][1])
    '''
    # did we max out the iterations?
    if not bests[0][0] == 0:
        for b in bests:
            print "    ... Wonk Wonk Wonk ...   "
            print b[1]
            print "Score: "+str(b[0])
            print "Iterations: " +str(iter)
            print "    ... Wonk Wonk Wonk ...   "
    '''
    return score_start, int(bests[0][0]), iters,  bests[0][1]

'''
if __name__ == "__main__":
    N=10
    ITER=100
    runs = 1000
    for N in [6,8,10]:
        for s in range(1,5):
            wins = 0
            starts = []
            scores = []
            niters = []
            for n in range(runs):
                B=list2board(randomList(N))
                start, score, niter, B = findSolution(B,ITER, s)
                starts.append(int(start))
                scores.append(int(score))
                niters.append(int(niter))
                if score == 0:
                    wins=wins+1
                #print n, start, score, niter
            print s, wins, runs, (1.*wins)/runs, np.mean(starts), np.mean(scores), np.mean(niters)
'''
 
if __name__ == "__main__":
    ITER=100
    runs = 100
    for N in [8]:
        RL=[62, 31, 53, 47, 20, 26, 52, 33] # 10 1 12 ## 0 in 100
        # RL=[49, 41, 21, 33, 48, 40, 13, 23] # 14 0 6 ## 11 in 100
        # RL=[49,43,20,59,25,48,27,4] # 12 1 13 # 0 in 100
        # RL=[10,36,12,62,39,25,55,46] # 11 1 11 ## 12 in 100
        # RL=[42,26,62,52,16,4,10,39] # 6 2 9 ## 32 in 100
        # RL=[28,63,16,14,4,11,51,24] # 8 2 11 ## 5 in 100
        # RL=[38,12,46,58,59,2,11,3] # 13 2 12 ## 0 in 100
        # RL=[62,50,38,51,9,23,35,40] # 6 0 3 ## 100 in 100
        # RL=[21,54,23,12,31,24,8,29] # 9 1 11 ## 29 in 100
        # RL=[0,50,62,46,42,26,60,1] # 11 2 11 ## 0 in 100
        # RL=[2,0,28,41,19,62,47,43] # 7 2 10 ## 48 in 100
        # RL=[14,51,56,57,29,36,25,28] # 12 1 12 ## 19 in 100
        # RL=[1,41,60,63,33,49,57,19] # 16 2 11 ## 22 in 100
        B=list2board(RL)
        print(B)
        for s in [N]:
            wins = 0
            starts = []
            scores = []
            niters = []
            for n in range(runs):
                start, score, niter, B2 = findSolution(B,ITER, s)
                starts.append(int(start))
                scores.append(int(score))
                niters.append(int(niter))
                if score == 0:
                    wins=wins+1
                #print n, start, score, niter
            print s, wins, runs, (1.*wins)/runs, np.mean(starts), np.mean(scores), np.mean(niters)
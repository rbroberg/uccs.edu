function d1tod2(i,N)
    x=i/N
    y=i-x*N
    return [x,y]
end

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
    # anti trace of B included above
    return sum([binomial(s,2) for s in sums])

def scoreCols(B):
    return sum([binomial(s,2) for s in np.sum(B,axis=0)])

def scoreRows(B):
    return sum([binomial(s,2) for s in np.sum(B,axis=1)])

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
    # special exit for no sideways
    # best_last = score_start
    # call the solution "done" when score is unchanged for N rounds
    stag=[score_start]*int(N*nscale)
    stag=[score_start]*2 # this is for only one stagnant step
    for iter in range(ITER):
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
        #'''
        stag=stag[1:]+[bests[0][0]]
        if sum([s == stag[0] for s in stag]) == len(stag):
            break
        #'''
        
        # special exit for no sideways
        # if the next score is same as this score, break
		'''
        if bests[0][0] == best_last:
            break
        else:
            best_last = bests[0][0]
        '''
        # pick a random best board
        ridx = rnd.randint(0,len(bests)-1)
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
    return score_start, int(bests[0][0]), iter,  bests[0][1]


if __name__ == "__main__":
    N=10
    ITER=100
    runs = 2000
    for N in [6,7,8,9,10]:
        for s in range(1):
            wins = 0
            starts = []
            scores = []
            niters = []
            for n in range(runs):
                B=list2board(randomList(N))
                start, score, niter, B = findSolution(B,ITER, s/2.)
                starts.append(int(start))
                scores.append(int(score))
                niters.append(int(niter))
                if score == 0:
                    wins=wins+1
                #print n, start, score, niter
            print s, wins, runs, (1.*wins)/runs, np.mean(starts), np.mean(scores), np.mean(niters)
 
'''
runs=1000
stag buffer=0
 280 1000 0.28 6.603 0.845 2.651
 217 1000 0.217 8.027 1.046 3.218
 165 1000 0.165 9.753 1.21 3.85
 136 1000 0.136 11.465 1.371 4.471
 94 1000 0.094 13.003 1.512 5.042
stag buffer=1
 274 1000 0.274 6.661 0.876 2.631
 231 1000 0.231 8.072 1.045 3.21
 180 1000 0.18 9.813 1.196 3.882
 139 1000 0.139 11.275 1.339 4.413
 107 1000 0.107 12.798 1.453 5.028
runs=2000
stag buffer=0
 544 2000 0.272 6.567 0.8525 2.634
 484 2000 0.242 8.11 1.0255 3.2225
 349 2000 0.1745 9.636 1.194 3.836
 287 2000 0.1435 11.3675 1.309 4.4605
 191 2000 0.0955 12.8505 1.489 5.046
stag buffer=1
'''
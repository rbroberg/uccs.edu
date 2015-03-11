function d1tod2(i,N)
    x=int(floor(i/N));
    y=i-x*N;
    return (x+1,y+1);
end

function d2tod1(p,N)
    return (p[1]-1)*N+(p[2]-1);
end

function board2list(B)
    l=Array{Int64}[];
    rows,cols=size(B);
    for r in 1:rows
        for c in 1:cols
            if B[r,c]>0
                push!(l,[r,c]);
            end
        end
    end
    return l;
end

function list2board(ll)
    n=length(ll); # assumes board dim = number of queens
    B=Array(Int64,n,n)*0;
    for l in ll
        p=d1tod2(l,n);
        B[p[1],p[2]]=1;
    end
    return B
end

# ll=[(1,1),(6,2),(3,1),(7,3),(8,3),(4,1),(5,1),(8,8)]
function points2board(ll)
    n=length(ll); # assumes board dim = number of queens
    B=Array(Int64,n,n);
    for p in ll
        B[p[0],p[1]]=1;
    end
    return B
end

# use to initialize random board
function randomList(N)
    l=[i for i in 0:(N*N-1)]
    r=Int64[]
    for i in 1:N
        j=rand(1:length(l))
        push!(r,l[j])
        splice!(l,j)
    end
    return r
end

# assumes square diagonal
# double counts main diagonal
# Z=[1 2 3 2;6 5 4 5;8 7 9 2;3 5 7 1]
function scoreDiagonals(B)
    n=size(B)[1];
    sums=Int64[];
    for i in 1:n
        push!(sums,trace(B[1:(n-i),(i+1):n])); # upper left
        push!(sums,trace(B[(i+1):n,1:(n-i)])); # lower left
        push!(sums,antiTrace(B[1:i,1:i]));
        push!(sums,antiTrace(B[(i+1):n,(i+1):n]));
    end
    # add the main trace to complete the accounting
    push!(sums,trace(B));
    
    return sum([binomial(s,2) for s in sums])
end

function scoreCols(B)
    return sum([binomial(s,2) for s in colSum(B)])
end

function scoreRows(B)
    return sum([binomial(s,2) for s in rowSum(B)])
end

function rowSum(B)
    n=size(B)[1]
    sums=Int64[]
    for i in 1:n
        push!(sums,sum(B[i,:]))
    end
    return sums
end

function colSum(B)
    n=size(B)[1]
    sums=Int64[]
    for i in 1:n
        push!(sums,sum(B[:,i]))
    end
    return sums
end

function antiTrace(B)
    n=size(B)[1];
    sum=0;
    for i in 1:n
        sum = sum + B[i,n-i+1];
    end
    return sum
end

function scoreBoard(B)
    return scoreCols(B)+scoreRows(B)+scoreDiagonals(B)
end

# the queen can move in eight directions, clockwise
# N, NE, E, SE, S, SW, W, NW
# find how far the can the queen move in each direction

# this is sample in Marsland
# pieces=[[3,3],[4,4],[5,5],[6,6],[4,0],[5,1],[6,2],[5,7]]

#p= some point on board
#vv = eight vectors from pB
#pieces = board2list(B)

function findSolution(B,ITER,staglen)
    #ITER=100
    N=size(B)[1]
    bests=[(scoreBoard(B),B)];
    bhist=copy(bests); # format history list
    [pop!(bhist) for b in 1:length(bhist)]; # clear the history list
    pieces=board2list(B);
    score_start=scoreBoard(B);
	last = score_start+1 # for a 'no buffer' test
    # call the solution "done" when score is unchanged for N rounds
    stag=[score_start for i in 1:staglen];
    iters=0;
    for iter in 1:ITER
        iters+=1;
        boards=copy(bests); #initialize (score, board) array
        [pop!(boards) for b in 1:length(boards)]; # clear list
        push!(bhist,(scoreBoard(B),B)) # include this board as beginning of history
        for pn in 1:N
            p=pieces[pn]
            Bnew=copy(B)
            Bnew[p[1],p[2]]=0
            # North
            for i in 1:N
                if p[1]-i < 1
                    break
                end
                if B[p[1]-i,p[2]]>0
                    break
                end
                Bthis = copy(Bnew)
                Bthis[p[1]-i,p[2]]=1
                push!(boards,(scoreBoard(Bthis),Bthis))
            end
            
            # North-East
            for i in 1:N
                if p[1]-i < 1
                    break
                end
                if p[2]+i > N
                    break
                end
                if B[p[1]-i,p[2]+i]>0
                    break
                end
                Bthis = copy(Bnew)
                Bthis[p[1]-i,p[2]+i]=1
                push!(boards,(scoreBoard(Bthis),Bthis))
            end
            
            # East
            for i in 1:N
                if p[2]+i > N
                    break
                end
                if B[p[1],p[2]+i]>0
                    break
                end
                Bthis = copy(Bnew)
                Bthis[p[1],p[2]+i]=1
                push!(boards,(scoreBoard(Bthis),Bthis))
            end
            
            # South-East
            for i in 1:N
                if p[1]+i > N
                    break
                end
                if p[2]+i > N
                    break
                end
                if B[p[1]+i,p[2]+i]>0
                    break
                end
                Bthis = copy(Bnew)
                Bthis[p[1]+i,p[2]+i]=1
                push!(boards,(scoreBoard(Bthis),Bthis))
            end
            
            # South
            for i in 1:N
                if p[1]+i > N
                    break
                end
                if B[p[1]+i,p[2]]>0
                    break
                end
                Bthis = copy(Bnew)
                Bthis[p[1]+i,p[2]]=1
                push!(boards,(scoreBoard(Bthis),Bthis))
            end
            
            # South-West
            for i in 1:N
                if p[1]+i > N
                    break
                end
                if p[2]-i < 1
                    break
                end
                if B[p[1]+i,p[2]-i]>0
                    break
                end
                Bthis = copy(Bnew)
                Bthis[p[1]+i,p[2]-i]=1
                push!(boards,(scoreBoard(Bthis),Bthis))
            end
            
            # West
            for i in 1:N
                if p[2]-i < 1
                    break
                end
                if B[p[1],p[2]-i]>0
                    break
                end
                Bthis = copy(Bnew)
                Bthis[p[1],p[2]-i]=1
                push!(boards,(scoreBoard(Bthis),Bthis))
            end
            
            # North-West
            for i in 1:N
                if p[1]-i < 1
                    break
                end
                if p[2]-i < 1
                    break
                end
                if B[p[1]-i,p[2]-i]>0
                    break
                end
                Bthis = copy(Bnew)
                Bthis[p[1]-i,p[2]-i]=1
                push!(boards,(scoreBoard(Bthis),Bthis))
            end
        
        end
        
        # find a set of best boards (minimal threat pairs)
        for b in boards
            if b[1]<bests[1][1]
                bests = [b]
            elseif b[1]==bests[1][1]
                push!(bests,b)
            end
        end
               
        # did we "win" - no threats on board?
        if bests[1][1] == 0
            break
        end
        
        # check stagnate level
        splice!(stag,1);
        push!(stag, bests[1][1]);
        #println(stag)
        #println(length(stag))
        #println("..")
        #flush(STDOUT)
        if sum([s == stag[1] for s in stag]) == length(stag)
            break
        end
		
		# for no buffer
		#if last == bests[1][1]
		#	break
		#else
		#	last = bests[1][1]
        #end
		
        # pick a random best board
        ridx = rand(1:length(bests));
        # ridx=1 # for testing only
        B = copy(bests[ridx][2]);
    end
    return score_start, int(bests[1][1]), iters,  bests[1][2]
end

function main()
    ITER=100
    runs = 100
    flush(STDOUT)
    for N in [6,7,8,9,10]
        for s in [2,int(N/2),N,int(3*N/2),2*N]
            wins = 0
            starts = Int64[]
            scores = Int64[]
            niters = Int64[]
            for n in 1:runs
                B=list2board(randomList(N))
                start, score, niter, B = findSolution(B,ITER, s)
                push!(starts,int(start))
                push!(scores,int(score))
                push!(niters,int(niter))
                if score == 0
                    wins=wins+1
                end
                #println(join([start, score, niter],'\t'))
                flush(STDOUT)
            end
            println(join([N, s, wins, runs, (1.*wins)/runs, mean(starts), mean(scores), mean(niters)],'\t'))
            flush(STDOUT)
        end
    end
end

function same()
    ITER=100
    runs = 1000
	innerruns = 10
    flush(STDOUT)
    for N in [8]
		Twins = 0
		Tstarts = Float64[]
		Tscores = Float64[]
		Tniters = Float64[]
		for m in 1:runs
			# RL=[62, 31, 53, 47, 20, 26, 52, 33] # 10 1 12 ## 0 in 1000
			# RL=[49, 41, 21, 33, 48, 40, 13, 23]  # 14 0 6 ## 207 in 1000
			# RL=[49,43,20,59,25,48,27,4] # 12 1 13 ## 7 in 1000
			# RL=[10,36,12,62,39,25,55,46] # 11 1 11 ## 85 in 1000
			# RL=[42,26,62,52,16,4,10,39] # 6 2 9 ## 366 in 1000
			# RL=[28,63,16,14,4,11,51,24] # 8 2 11 ## 42 in 1000
			# RL=[38,12,46,58,59,2,11,3] # 13 2 12 ## 0 in 1000
			# RL=[62,50,38,51,9,23,35,40] # 6 0 3 ## 1000 in 1000
			# RL=[21,54,23,12,31,24,8,29] # 9 1 11 ## 262 in 1000
			# RL=[0,50,62,46,42,26,60,1] # 11 2 11 ## 0 in 1000
			# RL=[2,0,28,41,19,62,47,43] # 7 2 10 ## 452 in 1000
			# RL=[14,51,56,57,29,36,25,28] # 12 1 12 ## 175 in 1000
			# RL=[1,41,60,63,33,49,57,19] # 16 2 11 ## 187 in 1000
			# B=list2board(RL)
			B=list2board(randomList(N))
			#println(B)
			#for s in [2,int(N/2),N,int(3*N/2),2*N]
			for s in [100]
				wins = 0
				starts = Int64[]
				scores = Int64[]
				niters = Int64[]
				innie=0
				for n in 1:innerruns
					innie+=1
					start, score, niter, B2 = findSolution(B,ITER, s)
					push!(starts,int(start))
					push!(scores,int(score))
					push!(niters,int(niter))
					if score == 0
						wins=wins+1
						Twins = Twins + 1
						break
					end
					#println(join([start, score, niter],'\t'))
					#flush(STDOUT)
				end
				#println(join([innie, wins, mean(starts), mean(scores), mean(niters)],'\t'))
				#flush(STDOUT)
				push!(Tstarts,mean(starts))
				push!(Tscores,mean(scores))
				push!(Tniters,mean(niters))	
			end
		end
		println(join([runs, innerruns, Twins, (1.*Twins)/runs, mean(Tstarts), mean(Tscores), mean(Tniters)],'\t'))
    end

	#print(B)
	#print(B2)
end

#main()
same()
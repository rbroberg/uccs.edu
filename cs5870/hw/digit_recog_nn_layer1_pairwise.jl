# Pkg.add("Requests")
using Requests;

# -----------------------------------------------------------------------------
# read data
# -----------------------------------------------------------------------------
daturl="http://rhinohide.org/data/www.kaggle.com/c/digit-recognizer/download/train.csv";
res = get(daturl);
csvdat = readcsv(IOBuffer(res.data));

# -----------------------------------------------------------------------------
# prepare training data
# -----------------------------------------------------------------------------
# first line are column headers
# first column are digit labels
# size is 42001 x 785
# data is 0-255 for gray valued images
X_train=convert(Array{Float64,2},csvdat[2:end,2:end]);
Y_train=convert(Array{Int16,1},csvdat[2:end,1]);

# normalize to bipolar -1,1
X_train=(X_train/255)*2-1;

# -----------------------------------------------------------------------------
# create weights for y_in for each of 10 digits
# -----------------------------------------------------------------------------
# 45 pairwise models (0,1), (0,2) ... (1,2), (1,3) ... (7,9), (8,9)
nclass = 10;
theta = .1;
bias=ones(9,10); # only using upper right triangle
weights = randn((9,10,784)); # only using upper right triangle
weights = weights / (2*maximum(abs(weights)));

npairs=int(nclass*(nclass-1)/2)
pairs=[(0,0) for i in 1:npairs];
n=0
for i in 0:(nclass-2)
	for j in (i+1):(nclass-1)
		n=n+1
		pairs[n]=(i,j)
	end
end

# -----------------------------------------------------------------------------
# make training loop available to all threads
# -----------------------------------------------------------------------------
require("/projects/uccs.edu/cs5870/hw/iterateModel.jl");

# -----------------------------------------------------------------------------
# model training parallelized on class
# -----------------------------------------------------------------------------
function pfit(f, X, Y, weights, bias, theta, nclass)
    np = nprocs()  # determine the number of processes available
    i = 1
    # function to produce the next work item from the queue.
    # in this case it's just an index.
    nextidx() = (idx=i; i+=1; idx)
    @sync begin
        for p=1:np
            if p != myid() || np == 1
                @async begin
                    while true
                        idx = nextidx()
                        if idx > npairs # defined outside of function
                            break						
                        end
						i,j = pairs[idx] # defined outside of function
						println(idx-1,",",i,",",j)
						flush(STDOUT)
						Y_pair={};
						X_pair={};
						for k in 1:size(X_train)[1]
							if ((Y_train[k]==i) || (Y_train[k]==j))
								push!(Y_pair,Y_train[k]);
								push!(X_pair,X_train[k,:]);
							end
						end
						# array of arrays to 2d
						Xp = zeros(size(X_pair)[1],784);
						dummy=[Xp[a,:]=X_pair[a] for a in 1:size(X_pair)[1]];
						weights[i+1,j+1,:], bias[i+1,j+1] = remotecall_fetch(p, f, Xp, Y_pair, reshape(weights[i+1,j+1,:],1,784), bias[i+1,j+1], theta, i, nclass);
					end		
                    #end
                end
            end
        end
    end
    weights, bias
end

for epoch in 1:2
	println("epoch: ", epoch)
	flush(STDOUT)
	weights, bias = pfit(fitModelPairwise, X_train, Y_train, weights, bias, theta, nclass);
end

# -----------------------------------------------------------------------------
# model prediction parallelized on class
# -----------------------------------------------------------------------------
function ppred(f, preds, X, weights, bias, pairs)
    np = nprocs()  # determine the number of processes available
    i = 1
    # function to produce the next work item from the queue.
    # in this case it's just an index.
    nextidx() = (idx=i; i+=1; idx)
	nimages=size(X)[1]
    @sync begin
        for p=1:np
            if p != myid() || np == 1
                @async begin
                    while true
                        idx = nextidx()
                        if idx > nimages
                            break
                        end
						preds[idx]= remotecall_fetch(p, f, X_test[idx,:], weights, bias, pairs)
                    end
                end
            end
        end
    end
    preds
end

# -----------------------------------------------------------------------------
# run sanity check on training data
# -----------------------------------------------------------------------------
#Y_hat=zeros(size(Y_train)[1]);
#for k in 1:size(Y_train)[1]
#	x=[dot(vec(X_train[k,:]),vec(weights[pairs[a][1]+1,pairs[a][2]+1,:]))+bias[pairs[a][1]+1,pairs[a][2]+1] for a in 1:npairs]
#	x=convert(Array{Float64,1},x)
#	Y_hat[k]=indmax([dot(x,d[:,b]) for b in 1:10])-1
#end
#println(sum(Y_hat.==Y_train) / (size(Y_train)[1]))
#flush(STDOUT)

# -----------------------------------------------------------------------------
# run test
# -----------------------------------------------------------------------------
daturl="http://rhinohide.org/data/www.kaggle.com/c/digit-recognizer/download/test.csv";
res = get(daturl);
csvdat = readcsv(IOBuffer(res.data));
X_test=convert(Array{Float64,2},csvdat[2:end,1:end]);

# normalize to bipolar -1,1
X_test=(X_test/255)*2-1;

Y_hat=zeros(size(X_test)[1]);
#Y_hat=ppred(predictModelPairwise, Y_hat, X_test, weights, bias, pairs)
for k in 1:size(Y_test)[1]
	x=[dot(vec(X_test[k,:]),vec(weights[pairs[a][1]+1,pairs[a][2]+1,:]))+bias[pairs[a][1]+1,pairs[a][2]+1] for a in 1:npairs]
	x=convert(Array{Float64,1},x)
	Y_hat[k]=indmax([dot(x,d[:,b]) for b in 1:10])-1
end

# -----------------------------------------------------------------------------
# kaggle submission
# -----------------------------------------------------------------------------
rows=[i for i in 1:size(Y_hat)[1]];
y=convert(Array{Int16,2},hcat(rows,Y_hat))
writedlm("nn.csv", y, ",") 
println("manual enter 'ImageID,Label' as first row")
flush(STDOUT)

quit()

# split 0.8, epoch 10, theta 0.1, .856, .853
# split 1.0, epoch 2, theta 0.1, .852, .---


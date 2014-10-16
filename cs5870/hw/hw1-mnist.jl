# Pkg.add("Requests")
using Requests;

# -----------------------------------------------------------------------------
# read data
# -----------------------------------------------------------------------------
daturl="http://rhinohide.org/data/www.kaggle.com/c/digit-recognizer/download/train.csv";
res = get(daturl);
csvdat = readcsv(IOBuffer(res.data));

# -----------------------------------------------------------------------------
# prepare data
# -----------------------------------------------------------------------------
# first line are column headers
# first column are digit labels
# size is 42001 x 785
# data is 0-255 for gray valued images
# use 70/30 for training/testing split

idx=int(42000 * .80);
X_train=convert(Array{Float64,2},csvdat[2:idx,2:end]);
Y_train=convert(Array{Int16,1},csvdat[2:idx,1]);
X_test=convert(Array{Float64,2},csvdat[(idx+1):end,2:end]);
Y_test=convert(Array{Int16,1},csvdat[(idx+1):end,1]);

# normalize to bipolar -1,1
X_train=(X_train/255)*2-1;
X_test=(X_test/255)*2-1;

# -----------------------------------------------------------------------------
# create weights for y_in for each of 10 digits
# -----------------------------------------------------------------------------
nclass = 10
weights = randn((nclass,784)); #
weights = weights / (2*maximum(abs(weights)));

# -----------------------------------------------------------------------------
# bias and theta threshold params
# -----------------------------------------------------------------------------
bias = ones(nclass);
theta = .05;

# -----------------------------------------------------------------------------
# make training loop available to all threads
# -----------------------------------------------------------------------------
require("/projects/uccs.edu/cs5870/hw/iterateModel.jl");

# -----------------------------------------------------------------------------
# model training parallelized on class
# -----------------------------------------------------------------------------
function pfit(f, X, Y, weights, bias, theta, idx, nclass)
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
                        if idx > nclass
                            break
                        end
                        weights[idx,:],bias[idx] = remotecall_fetch(p, f, X, Y, weights, bias, theta, idx, nclass)
                    end
                end
            end
        end
    end
    weights, bias
end

for epoch in 1:40
	println("epoch: ", epoch)
	flush(STDOUT)
	weights, bias = pfit(fitModel, X_train, Y_train, weights, bias, theta, idx, nclass)
end

# -----------------------------------------------------------------------------
# model prediction parallelized on class
# -----------------------------------------------------------------------------
function ppred(f, preds, X, weights, bias, nclass)
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
                        if idx > nclass
                            break
                        end
                        preds[:,idx] = remotecall_fetch(p, f, X, weights[idx,:],bias[idx])
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
Y_preds=zeros(size(X_train)[1],nclass);
Y_preds = ppred(predictModel, Y_preds, X_train, weights, bias, nclass);
Y_hat = [int(mod(indmax(Y_preds[i,:]),nclass)) for i in 1:size(Y_preds)[1]];
println(sum(Y_hat.==Y_train) / (size(Y_train)[1]))
flush(STDOUT)

# -----------------------------------------------------------------------------
# run test
# -----------------------------------------------------------------------------
Y_preds=zeros(size(X_test)[1],nclass);
Y_preds = ppred(predictModel, Y_preds, X_test, weights, bias, nclass);
Y_hat = [int(mod(indmax(Y_preds[i,:]),nclass)) for i in 1:size(Y_preds)[1]];
println(sum(Y_hat.==Y_test) / (size(Y_test)[1]))
flush(STDOUT)

quit()

# idx=.7, epoch=4, theta=0.1, .858, .844
# idx=.7, epoch=10, theta=0.1, .875, .858
# idx=.7, epoch=40, theta=0.1, .892, .870
# idx=.7, epoch=10, theta=0.5, .852, .838
# idx=.7, epoch=10, theta=0.05, .895, .876
# idx=.7, epoch=10, theta=0.01, .879, .865
# idx=.7, epoch=40, theta=0.05, .905, .883
# idx=.8, epoch=40, theta=0.05, .867, .848


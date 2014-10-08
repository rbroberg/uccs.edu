# run kaggle training/test set

# Pkg.add("Requests")
using Requests;
# using StatsBase;

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
idx
X_train=convert(Array{Float64,2},csvdat[2:end,2:end]);
Y_train=convert(Array{Int16,1},csvdat[2:end,1]);

# normalize to bipolar -1,1
X_train=(X_train/255)*2-1;

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
theta = .1;

# -----------------------------------------------------------------------------
# make training loop available to all threads
# -----------------------------------------------------------------------------
require("/projects/kaggle.com/digit.recog/scripts/iterateModel.jl");

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
	weights, bias = pfit(fitModel, X_train, Y_train, weights, bias, theta, nclass)
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

daturl="http://rhinohide.org/data/www.kaggle.com/c/digit-recognizer/download/test.csv";
res = get(daturl);
csvdat = readcsv(IOBuffer(res.data));
X_test=convert(Array{Float64,2},csvdat[2:end,1:end]);

# normalize to bipolar -1,1
X_test=(X_test/255)*2-1;

Y_preds=zeros(size(X_test)[1],nclass);
Y_preds = ppred(predictModel, Y_preds, X_test, weights, bias, nclass);
Y_hat = [int(mod(indmax(Y_preds[i,:]),nclass)) for i in 1:size(Y_preds)[1]];

# -----------------------------------------------------------------------------
# write submission
# -----------------------------------------------------------------------------

rows=[i for i in 1:size(Y_hat)[1]];
y=convert(Array{Int16,2},hcat(rows,Y_hat))
writedlm("nn.csv", y, ",") 
# manual enter "ImageID,Label" as first row

quit()

# epoch=40, theta=0.1, .892, .870

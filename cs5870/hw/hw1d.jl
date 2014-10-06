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

idx=29400;
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
theta = 0.5;

# -----------------------------------------------------------------------------
# make training loop available to all threads
# -----------------------------------------------------------------------------
require("/projects/uccs.edu/cs5870/hw/iterateModel.jl");

# -----------------------------------------------------------------------------
# train the model
# -----------------------------------------------------------------------------
# parallel digits
# -----------------------------------------------------------------------------
for epochs in 1:784
	#@parallel for idx in 1:nclass
	for idx in 1:nclass
		weights[idx,:],bias[idx] = iterateModel(X_train, Y_train, weights, bias, theta, idx, nclass);
	end
end

# -----------------------------------------------------------------------------
# run sanity check on training data
# -----------------------------------------------------------------------------
Y_preds=zeros(size(X_train)[1],nclass);
Y_hat=zeros(size(X_train)[1]);
#@parallel for c in 1:nclass
for c in 1:nclass
		Y_preds[:,c]=predictModel(X_train,weights[c,:],bias[c]);
end
Y_hat = [int(mod(indmax(Y_preds[i,:]),nclass)) for i in 1:size(Y_preds)[1]];
print(sum(Y_hat.==Y_train) / (size(Y_train)[1]))


# -----------------------------------------------------------------------------
# run test
# -----------------------------------------------------------------------------
Y_preds=zeros(size(X_test)[1],nclass);
Y_hat=zeros(size(X_test)[1]);
#@parallel for c in 1:nclass
for c in 1:nclass
		Y_preds[:,c]=predictModel(X_test,weights[c,:],bias[c]);
end
Y_hat = [int(mod(indmax(Y_preds[i,:]),nclass)) for i in 1:size(Y_preds)[1]];
print(sum(Y_hat.==Y_test) / (size(Y_test)[1]))


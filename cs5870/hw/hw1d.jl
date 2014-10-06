# Pkg.add("Requests")
using Requests;

daturl="http://rhinohide.org/data/www.kaggle.com/c/digit-recognizer/download/train.csv";
res = get(daturl);
csvdat = readcsv(IOBuffer(res.data));

# first line are column headers
# first column are digit labels
# size is 42001 x 785
# data is 0-255 for gray valued images
# use 70/30 for training/testing split
#idx=29400;
idx=294;
X_train=convert(Array{Float64,2},csvdat[2:idx,2:end]);
Y_train=convert(Array{Int16,1},csvdat[2:idx,1]);
#X_test=convert(Array{Float64,2},csvdat[(idx+1):end,2:end]);
#Y_test=convert(Array{Int16,1},csvdat[(idx+1):end,1]);
# normalize to bipolar -1,1
X_train=(X_train/255)*2-1;
#X_test=(X_test/255)*2-1;

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
# for each character c / for each epoch (4 runs) / for each training instance i
# -----------------------------------------------------------------------------
np=nprocs()
nextidx() = (idx=i; i+=1; idx) # monotonically increasing iterator (i noes math)
for epochs in 1:3 # run through training set 4 times for each char
	@sync begin
	for p=1:np
		if p != myid() || np == 1
		@async begin
			while true
				idx = nextidx()
				if idx > nclass
					break
				end
				weights[idx,:],bias[idx]=@spawn iterateModel(X_train, Y_train, weights, bias, theta, idx, nclass);
			end
		end
	end
end

# quick training test check
for i in 1:21
	for c in 1:7
		pred = dot(vec(X_training[i]),vec(weights[c,:]))+bias[c]
		#if pred
			print(i,":",mod(i,7),":",mod(c,7),":",pred>theta,":",pred,"\n")
		#end
	end
end

# -----------------------------------------------------------------------------
# read test file
# -----------------------------------------------------------------------------
f=open(datadir*testfile)
b=ASCIIString[]
for j in 1:21
	a=""
	for i in 1:9
		a=a*readline(f)[1:7]
	end
	push!(b,a)
end
close(f)

# -----------------------------------------------------------------------------
# translate test into bipolar
# -----------------------------------------------------------------------------
X_testing = [[int(b[j][i]=='.' || b[j][i]=='@') for i in 1:63]*2-1 for j in 1:21]
# testing set has same pattern of characters as training set
Y_testing = [[int(mod(i,7)==mod(j,7)) for i in 1:21]*2-1 for j in 1:7]

# -----------------------------------------------------------------------------
# run test
# -----------------------------------------------------------------------------
for i in 1:21
	preds=Float64[]
	for c in 1:7
		push!(preds,dot(vec(X_testing[i]),vec(weights[c,:]))+bias[c])
		#if pred > 10
		#	#print(i,":",mod(i,7),":",mod(c,7),"\n")
		#	print(i,":",mod(i,7),":",mod(c,7),":",pred>theta,":",pred,"\n")
		#end
	end
	print(i,":",mod(i,7),":",mod(indmax(preds),7),"\n")
end



#'''
#The training file is organized as ABCDEJKABCDEJKABCDEJK
#Each letter is in 7*9 character array
#'''

datadir="/projects/uccs.edu/cs5870/data/";
trainfile="LetterTrainingHW1.dat";
testfile="LetterTestingHW1.dat";

# -----------------------------------------------------------------------------
# read train file
# -----------------------------------------------------------------------------
f=open(datadir*trainfile)
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
# translate training into bipolar
# -----------------------------------------------------------------------------
X_training = [[int(b[j][i]=='.') for i in 1:63]*2-1 for j in 1:21]
Y_training = [[int(mod(i,7)==mod(j,7)) for i in 1:21]*2-1 for j in 1:7]

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
# create weights for y_in
# -----------------------------------------------------------------------------
# -0.5 < weights < 0.5
weights = randn((7,63));
weights = weights / (2*maximum(abs(weights)));
bias = ones(7)
theta = 0.5
alpha = 0.05

# -----------------------------------------------------------------------------
# train the model
# -----------------------------------------------------------------------------
# for each character c / for each epoch  / for each training instance i
# -----------------------------------------------------------------------------
for c in 1:7 # for each of 7 characters
	for epochs in 1:1 # run through training set 4 times for each char
		for i in 1:21 # present each training instance
			y_in=dot(vec(X_training[i]),vec(weights[c,:]))+bias[c] # inputs * weights + bias
			y_out = int(sign(y_in)*int(abs(y_in)>theta)) # -1 < -theta else 1 > theta else 0
			if y_out != Y_training[c][i]
				bias[c] = bias[c] +  Y_training[c][i]
				weights[c,:] = weights[c,:] + alpha * transpose((Y_training[c][i] - y_in)*X_training[i])
			end
		end
	end
end

# quick training test check
# almost all predictions below 0
for i in 1:21
	for c in 1:7
		pred = dot(vec(X_training[i]),vec(weights[c,:]))+bias[c]
		print(i,":",mod(i,7),":",mod(c,7),":",pred>theta,":",pred,"\n")
	end
end

for i in 1:21
	preds=Float64[]
	for c in 1:7
		push!(preds,dot(vec(X_training[i]),vec(weights[c,:]))+bias[c])
	end
	print(i,":",mod(i,7),":",mod(indmax(preds),7),"\n")
end

# -----------------------------------------------------------------------------
# run test
# -----------------------------------------------------------------------------
# however, max value is good predictor
for i in 1:21
	for c in 1:7
		pred = dot(vec(X_testing[i]),vec(weights[c,:]))+bias[c]
		print(i,":",mod(i,7),":",mod(c,7),":",pred>theta,":",pred,"\n")
	end
end

for i in 1:21
	preds=Float64[]
	for c in 1:7
		push!(preds,dot(vec(X_testing[i]),vec(weights[c,:]))+bias[c])
	end
	print(i,":",mod(i,7),":",mod(indmax(preds),7),"\n")
end



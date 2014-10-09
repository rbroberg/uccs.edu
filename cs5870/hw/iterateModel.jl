# -----------------------------------------------------------------------------
# independent inner loop for training model on one digit/letter
# -----------------------------------------------------------------------------
function fitModel(X, Y, wghts, bs, theta, c, nclass)
	for i in 1:size(X)[1]
		y_in=dot(vec(X[i,:]),vec(wghts[c,:]))+bs[c]; # inputs * weights + biasweigt
		y_out = int(sign(y_in)*int(abs(y_in)>theta)); # -1 < -theta else 1 > theta else 0
		if (y_out == 1 && !(mod(c,nclass) == Y[i])) || (y_out == -1 && (mod(c,nclass) == Y[i]))
			y_bipolar = int(mod(c,nclass) == Y[i])*2-1;
			bs[c] = bs[c] +  y_bipolar; # -1 for no match, +1 for match
			wghts[c,:] = wghts[c,:] + y_bipolar*X[i,:];
		end
	end
	return wghts[c,:], bs[c]
end

function fitModelPairwise(X, Y, wghts, bs, theta, c, nclass)
	for i in 1:size(X)[1]
		y_in=dot(vec(X[i,:]),vec(wghts))+bs; # inputs * weights + biasweigt
		y_out = int(sign(y_in)*int(abs(y_in)>theta)); # -1 < -theta else 1 > theta else 0
		if (y_out == 1 && !(mod(c,nclass) == Y[i])) || (y_out == -1 && (mod(c,nclass) == Y[i]))
			y_bipolar = int(mod(c,nclass) == Y[i])*2-1;
			bs = bs +  y_bipolar; # -1 for no match, +1 for match
			wghts = wghts + y_bipolar*X[i,:];
		end
	end
	return wghts, bs
end
# -----------------------------------------------------------------------------
# given weights and bias, predict X
# -----------------------------------------------------------------------------
function predictModel(X, wghts, bs)
	preds = zeros(size(X)[1])
	for i in 1:size(X)[1]
		preds[i]=dot(vec(X[i,:]),vec(wghts))+bs
	end
	preds
end


# -----------------------------------------------------------------------------
# this provides a vector to get select all results relating to a single digit class 0-9 
# -----------------------------------------------------------------------------
#  [ ', ', ', ', v, ', ', ', ', X, ', ', ', ', v, ', ', ', ', X, ', ', ', ', v, ', ', ', ', X, ', ', ', ', v, ', ', ', ', X, ', ', ', ', v ]
d=[[ 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
   [-1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
   [ 0,-1, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
   [ 0, 0,-1, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
   [ 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
   [ 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0,-1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 ],
   [ 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0,-1, 0, 0, 0,-1, 0, 0, 0, 1, 1, 1, 0, 0, 0 ],
   [ 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0,-1, 0, 0, 0,-1, 0, 0,-1, 0, 0, 1, 1, 0 ],
   [ 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0,-1, 0, 0, 0,-1, 0, 0,-1, 0,-1, 0, 1 ],
   [ 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0,-1, 0, 0, 0,-1, 0, 0,-1, 0,-1,-1 ]];
d=reshape(d,45,10);

# -----------------------------------------------------------------------------
# assumes 10 classes
# -----------------------------------------------------------------------------
function predictModelPairwise(X, weights, bias, pairs)
	npairs=size(pairs)[1]
	x=[dot(vec(X),vec(weights[pairs[a][1]+1,pairs[a][2]+1,:]))+bias[pairs[a][1]+1,pairs[a][2]+1] for a in 1:npairs]
	x=convert(Array{Float64,1},x)
	return(indmax([dot(x,d[:,b]) for b in 1:10])-1)
end

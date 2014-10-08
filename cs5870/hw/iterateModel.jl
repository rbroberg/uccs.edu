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

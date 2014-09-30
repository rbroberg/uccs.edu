# -----------------------------------------------------------------------------
# Example 2:20 Training Madaline for the XOR function
# -----------------------------------------------------------------------------
# pg 92 Fausett

# NOTE: bias weights and nodes are on the tail end of their respective vectors
#       which deviates a bit from Ferguson which general shows bias on the head

# X training data input with labels
X=transpose(reshape([1,1,0,1,0,1,0,1,1,0,0,0],3,4));
# rescale to bipolar
X=X*2-1;
#4x3 Array{Int64,2}:
#  1   1  -1
#  1  -1   1
# -1   1   1
# -1  -1  -1

# -----------------------------------------------------------------------------
# STEP 0: Initialize Weights
# -----------------------------------------------------------------------------

# initial weights on first layer, bias weight on the end
w=transpose(reshape([0.05,0.2,.3,0.1,0.2,.15],3,2));
#2x3 Array{Float64,2}:
# 0.05  0.2  0.3      # weights into Z1
# 0.1   0.2  0.15     # weights into Z2

# weights into Y
v=transpose(reshape([.5,.5,.5],3,1))

# set learning rate
T=0.0; # threshold value
a=0.5; # learning rate

# -----------------------------------------------------------------------------
# STEP 1: while stopping condition is false, do STEPS 2-6
# -----------------------------------------------------------------------------
# 4 runs and stop (Ferguson calls this four epochs)
for s in 1:4
	# -------------------------------------------------------------------------
	# STEP 2: for each bipolar training pair x:y, do STEPS 3-5
	# -------------------------------------------------------------------------
	# need to fix bias handling
	for i in 1:size(X)[1]
		# -------------------------------------------------------------------------
		# STEP 3: set the activation input units x[j] = X[i,j]
		# -------------------------------------------------------------------------
		y=X[i,end]; # keep expected output
		x=X[i,:];   # get inputs
		x[end] = 1; # reuse last column for bias input

		# -------------------------------------------------------------------------
		# STEP 4: compute net input to each hidden unit
		# -------------------------------------------------------------------------
		z_in=x*transpose(w); # calculate z_in

		# -------------------------------------------------------------------------
		# STEP 5: compute net output from each hidden unit
		# -------------------------------------------------------------------------
		z_out = (z_in.>T)*2-1; # calculate z_out bipolar, threshold = 0

		# -------------------------------------------------------------------------
		# STEP 6: compute output of net
		# -------------------------------------------------------------------------
		# adding bias to zout, mult by weights of second layer and sum
		y_in = hcat(z_out,[1])*transpose(v); 
		y_out = (y_in.>T)*2-1; # T is threshold
		
		# -------------------------------------------------------------------------
		# STEP 7: determine error and update weights
		# -------------------------------------------------------------------------
		if y_out[1,1]==y
			# if y=y_out, no weight updates are performed
			dummy=1;
		elseif y_out[1,1]==-1
			# if y=1, update weights on the z unit whose net input is closest to 0
			k = indmin(z_in.^2)
			w[k,1:(end-1)]=w[k,1:(end-1)]+a*(1-z_in[k])*x[k] # z_in wts
			w[k,end]=w[k,end]+a*(1-z_in[k]) # bias wt
		elseif y_out[1,1]==1
			# if y=-1, update weights all units in Z that have postitive net input
			for k in 1:(size(w)[1])
				if z_in[k]>0 # reduce those weights
					w[k,1:(end-1)]=w[k,1:(end-1)]+a*(-1-z_in[k])*x[k] # z_in wts
					w[k,end]=w[k,end]+a*(-1-z_in[k]) # bias wt
				end
			end
		end
		print(i, w)
	end
end

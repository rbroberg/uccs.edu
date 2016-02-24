# MONTE CARLO APPROXIMATION OF INT(xexp(x))dx
# FOR TWO DIFFERENT SAMPLE SIZES
import numpy as np

np.random.seed(271828)
runs=1000

# THE FIRST APPROXIMATION USING N1 = 100 SAMPLES
N1 = 100;
x = np.random.uniform(size=N1);
I_hat_1 = sum(x*np.exp(x))/N1
I_hat_1 # 0.95254390652501486

# A SECOND APPROXIMATION USING N2 = 5000 SAMPLES
N2 = 5000;
x = np.random.uniform(size=N2);
I_hat_2 = sum(x*np.exp(x))/N2
I_hat_2 # 1.0209454002831784

# Estimate variance for N1=100
est=[]
for r in range(runs):
	x = np.random.uniform(size=N1);
	est.append(sum(x*np.exp(x))/N1)

np.var(est) # 0.0063657238770784439

# Estimate variance for N1=5000
est=[]
for r in range(runs):
	x = np.random.uniform(size=N2);
	est.append(sum(x*np.exp(x))/N2)

np.var(est) # 0.00012223069067703404

# variance decreases linearly with number of runs
0.0063657238770784439/0.00012223069067703404 
# 52.079586900955817

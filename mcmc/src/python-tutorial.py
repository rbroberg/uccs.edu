import numpy,math
import matplotlib.pyplot as plt
import random as random

# http://python4mpia.github.io/fitting_data/MC-sampling-from-Salpeter.html
# http://python4mpia.github.io/fitting_data/Metropolis-Hastings.html

random.seed(1)  # set random seed.

# Draw random samples from Salpeter IMF.
# N     ... number of samples.
# alpha ... power-law index.
# M_min ... lower bound of mass interval.
# M_max ... upper bound of mass interval.
def sampleFromSalpeter(N, alpha, M_min, M_max):
    # Convert limits from M to logM.
    log_M_Min = math.log(M_min)
    log_M_Max = math.log(M_max)
    # Since Salpeter SMF decays, maximum likelihood occurs at M_min
    maxlik = math.pow(M_min, 1.0 - alpha)

    # Prepare array for output masses.
    Masses = []
    # Fill in array.
    while (len(Masses) < N):
        # Draw candidate from logM interval.
        logM = random.uniform(log_M_Min,log_M_Max)
        M    = math.exp(logM)
        # Compute likelihood of candidate from Salpeter SMF.
        likelihood = math.pow(M, 1.0 - alpha)
        # Accept randomly.
        u = random.uniform(0.0,maxlik)
        if (u < likelihood):
            Masses.append(M)
    return Masses

# Draw samples.
Masses = sampleFromSalpeter(1000000, 2.35, 1.0, 100.0)
# Convert to logM.
LogMasses = numpy.log(numpy.array(Masses))

# Plot distribution.
plt.figure(1)
plt.hist(LogMasses, 30, histtype='step', lw=3, log=True,
         range=(0.0,math.log(100.0)))
# Overplot with Salpeter SMF.
X = []
Y = []
for n in range(101):
    logM = math.log(100.0)*float(n)/100.0
    x    = math.exp(logM)
    y    = 2.0e5*math.pow(x, 1.0-2.35)  # normalisation
    X.append(logM)
    Y.append(y)
plt.plot(X, Y, '-', lw=3, color='black')
plt.xlim(0.0,math.log(100.0))
plt.xlabel(r'$\log M$', fontsize=24)
plt.ylabel('PDF', fontsize=24)
plt.savefig('example-Monte-Carlo-sampling-from-Salpeter.png')
plt.show()

# Define logarithmic likelihood function.
# params ... array of fit params, here just alpha
# D      ... sum over log(M_n)
# N      ... number of data points.
# M_min  ... lower limit of mass interval
# M_max  ... upper limit of mass interval
def evaluateLogLikelihood(params, D, N, M_min, M_max):
    alpha = params[0]  # extract alpha
    # Compute normalisation constant.
    c = (1.0 - alpha)/(math.pow(M_max, 1.0-alpha)
                        - math.pow(M_min, 1.0-alpha))
    # return log likelihood.
    return N*math.log(c) - alpha*D

# Generate toy data.
N      = 1000000  # Draw 1 Million stellar masses.
alpha  = 2.35
M_min  = 1.0
M_max  = 100.0
Masses = sampleFromSalpeter(N, alpha, M_min, M_max)
LogM   = numpy.log(numpy.array(Masses))
D      = numpy.mean(LogM)*N

# initial guess for alpha as array.
guess = [3.0]
# Prepare storing MCMC chain as array of arrays.
A = [guess]
# define stepsize of MCMC.
stepsizes = [0.005]  # array of stepsizes
accepted  = 0.0

# Metropolis-Hastings with 10,000 iterations.
for n in range(10000):
    old_alpha  = A[len(A)-1]  # old parameter value as array
    old_loglik = evaluateLogLikelihood(old_alpha, D, N, M_min,
                    M_max)
    # Suggest new candidate from Gaussian proposal distribution.
    new_alpha = numpy.zeros([len(old_alpha)])
    for i in range(len(old_alpha)):
        # Use stepsize provided for every dimension.
        new_alpha[i] = random.gauss(old_alpha[i], stepsizes[i])
    new_loglik = evaluateLogLikelihood(new_alpha, D, N, M_min,
                    M_max)
    # Accept new candidate in Monte-Carlo fashing.
    if (new_loglik > old_loglik):
        A.append(new_alpha)
        accepted = accepted + 1.0  # monitor acceptance
    else:
        u = random.uniform(0.0,1.0)
        if (u < math.exp(new_loglik - old_loglik)):
            A.append(new_alpha)
            accepted = accepted + 1.0  # monitor acceptance
        else:
            A.append(old_alpha)

print "Acceptance rate = "+str(accepted/10000.0)

# Discard first half of MCMC chain and thin out the rest.
Clean = []
for n in range(5000,10000):
    if (n % 10 == 0):
        Clean.append(A[n][0])

# Print Monte-Carlo estimate of alpha.
print "Mean:  "+str(numpy.mean(Clean))
print "Sigma: "+str(numpy.std(Clean))

plt.figure(1)
plt.hist(Clean, 20, histtype='step', lw=3)
plt.xticks([2.346,2.348,2.35,2.352,2.354],
           [2.346,2.348,2.35,2.352,2.354])
plt.xlim(2.345,2.355)
plt.xlabel(r'$\alpha$', fontsize=24)
plt.ylabel(r'$\cal L($Data$;\alpha)$', fontsize=24)
plt.savefig('example-MCMC-results.png')
plt.show()

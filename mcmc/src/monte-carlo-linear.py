import numpy as np

# https://thetarzan.wordpress.com/2012/10/27/calculate-ols-regression-manually-in-python-using-numpy/

# known model y = a + b * x + err
# err is norm with mu = 0, sigma = 1 
# let a = 0
# let b = 1

# draw N samples from 0 to M, calculate a, b, sigma
def do_mc_ols_stderr(niter,rnge,a,b,mu,sigma):
    N=niter
    M=rnge
    X=np.ones((N,2))
    Y=np.ones((N,1))
    for i in range(N):
        X[i,1]= np.random.uniform() * M
        Y[i,0] = a + b * X[i,1] + np.random.normal(mu,sigma)
    
    ## Use the equation above (X'X)^(-1)X'Y to calculate OLS coefficient estimates:
    bh = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,Y))
    #print bh
    
    ## check your work with Numpy's built in OLS function:
    #z,resid,rank,sigma = np.linalg.lstsq(X,Y)
    #print(z)
    
    ## Calculate vector of residuals
    Yhat=(bh[0]+X[:,1]*bh[1])
    Yhat.shape=(N,1)
    res = Y-Yhat
    res.shape=(N,)
    ## Define n and k parameters
    n = N
    k = X.shape[1]
    
    ## Calculate Variance-Covariance Matrix
    VCV = np.true_divide(1,n-k)*np.dot(np.dot(res.T,res),np.linalg.inv(np.dot(X.T,X)))
     
    ## Standard errors of the estimated coefficients
    return np.sqrt(np.diagonal(VCV))


do_mc_ols_stderr(100,100,0,1,0,1)

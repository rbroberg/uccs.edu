# Math 540 HW 5
# Ron Broberg
# March 31, 2015

import numpy as np
from scipy import optimize
import numdifftools as nd # includes Hessian and gradient functions

# Interface to minimization algorithms for scalar univariate functions.
# See the ‘Golden’ method in particular.
import scipy.optimize as sopt # Return the minimum of a function

A=np.array([[2,1],[1,3]])
Z=np.array([[1],[2]])
X0=np.array([[0],[0]])
x0=[0,0]

def f(x): 
    return 2*x[0]**2 - 8*x[0] + 2*x[0]*x[1] - 14*x[1] + 3*x[1]**2 + 18

# Gradient
def df(x):
    return np.array(((4*x[0]+2*x[1]-8),(2*x[0]+6*x[1]-14)))

# Hessian
def d2f(x):
    return np.array([[4,2],[2,6]])

def vlen(x):
    return sum([(1.0*i)**2 for i in x])**0.5

# =======================
# problem 2
# =======================
epsilon=.0001
n=0
N=20
alpha=1
x0=[0,0]

while (abs(f(x0))>epsilon and n<N):
    n=n+1
    x1=list(x0-alpha*np.dot(np.linalg.inv(d2F(x0)),df(x0)))
    x0=x1
    print(x0, f(x0))

#  ----------- x------------,  f(x)
# ([0.99999999999999978, 2.0], 0.0)

# =======================
# problem 3
# =======================
def f1d(alpha):
    return f(x0 + alpha*s)

epsilon=.0001
n=0
N=20
x0=[0,0]
while (abs(f(x0))>epsilon and n<N):
    n=n+1
    s = -df(x0)
    t_crit = sopt.golden(f1d) # Return the minimum of a function
    x1 = x0 + t_crit * s
    x0 = x1
    print(x0, f(x0))

# --------------- x ----------------, f(x)
# (array([ 1.10638298,  1.93617021]), 0.021276595744676996)
# (array([ 0.99881793,  1.99763597]), 2.5149427976600691e-05)

# =======================
# problem 4
# =======================
def f1d0(alpha):
    return f([x0[0]+alpha,x0[1]])

def f1d1(alpha):
    return f([x0[0],x0[1]+alpha])

epsilon=.0001
n=0
N=10
x0=[0,0]
e=[[1,0],[0,1]]
while (abs(f(x0))>epsilon and n<=N):
    n=n+1
    for i in range(len(x0)):
        if i == 0:
            t_crit = sopt.golden(f1d0)
            x1[0]=x0[0]+t_crit
        else:
            t_crit = sopt.golden(f1d1)
            x1[1]=x0[1]+t_crit
    x1 = x0
    x0 = x1
    print(x0, f(x0))

#([0, 0]                                  ,18)
#([2.0, 1.6666666889095336]               , 1.6666666666666643)
#([1.1666666959331391, 1.9444444329355959], 0.046296312555448083)
#([1.0277778211936621, 1.9907407657783021], 0.0012860122504463334)
#([1.0046296356650151, 1.9984567996669522], 3.5722543984206823e-05)

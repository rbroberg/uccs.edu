from sys import argv
from math import *
from whrandom import random

# http://www.mas.ncl.ac.uk/~ndjw1/teaching/sim/metrop/metrop.html

def unif(a,b):
  return a+(b-a)*random()

def sdnorm(z):
  return exp(-z*z/2)

n=eval(argv[1])
alpha=eval(argv[2])
x=0
for i in range(n):
  innov=unif(-alpha,alpha)
  can=x+innov
  aprob=min(1,sdnorm(can)/sdnorm(x))
  u=random()
  if (u<aprob):
    x=can
  print x







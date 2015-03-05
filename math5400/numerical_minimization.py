
# some function in two variables
def g0(x,y):
	return x**2 + 2*x*y + 2*y**2

# partial derivative of g0 in x
def g1_x(x,y):
	return 2*x + 2

# partial derivative of g0 in y
def g1_y(x,y):
	return 4*y + 2

def partial_x(func, x, y, h):
    return (func(x+h, y) - func(x-h,y))/(2.*h)

def partial_y(func, x, y, h):
    return (func(x, y+h) - func(x,y-h))/(2.*h)

def derivative(func, x, h):
    return (func(x+h) - func(x-h))/(2.*h)

# functools
f = lambda x: x**2  # or more generally ``def f(x):``
df = functools.partial(derivative, f, 0.1)


#  scipy.misc.derivative(func, x0, dx=1.0, n=1, args=(), order=3)[source]

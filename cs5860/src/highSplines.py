#!/usr/bin/python
# http://cairnarvon.rotahall.org/2009/07/05/quadratic-spline-interpolation/
# http://cairnarvon.rotahall.org/misc/hop.py.html

import sys, re

def solve(m):
    rowswitch = lambda m, i, j:                                    \
        m[:i] + [m[j]] + m[i+1:j] + [m[i]] + m[j+1:] if i < j else \
        m[:j] + [m[i]] + m[j+1:i] + [m[j]] + m[i+1:] if i > j else m
    rowmultiply = lambda m, r, i: \
        m[:r] + [map(lambda n: n*i, m[r])] + m[r+1:]
    rowadd = lambda m, r1, r2, i: \
        m[:r2] + [map(sum, zip(map(lambda n: n*i, m[r1]), m[r2]))] + m[r2+1:]

    for i in xrange(len(m)):
        k = i
        try:
            while m[k][i] == 0: k += 1
        except:
            raise "Insoluble"

        m = rowswitch(m, i, k)
        m = rowmultiply(m, i, 1 / m[i][i])

        for j in xrange(len(m)):
            if j != i and m[j][i] != 0:
                m = rowadd(m, i, j, -m[j][i])

    return m

xs, ys = [], []
regex = re.compile(r"^\s*(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*$")

for line in sys.stdin.readlines():
    m = regex.match(line)
    if m is None:
        sys.stderr.write("Invalid input: %s\n" % line)
    xs.append(float(m.group(1)))
    ys.append(float(m.group(2)))

print "plot", " + ".join(filter(None, map(lambda (a, b): "%f * x**%d" % (b, a) if b != 0 else None, enumerate(zip(*solve(map(lambda (x, y): map(lambda (a, b): b**a, enumerate([x]*len(xs))) + [y], zip(xs, ys))))[-1]))))




#!/usr/bin/python
# http://cairnarvon.rotahall.org/2009/07/05/quadratic-spline-interpolation/
# http://cairnarvon.rotahall.org/misc/qsi.py.html

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

x, y = [], []
regex = re.compile(r"^\s*(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*$")

for line in sys.stdin.readlines():
    m = regex.match(line)
    if m is None:
        sys.stderr.write("Invalid input: %s\n" % line)
    x.append(float(m.group(1)))
    y.append(float(m.group(2)))

assert len(x) == len(y)

matrix = []

row = [0. for i in xrange(3 * len(x) - 2)]
row[0] = 1.
matrix.append(row)

# Points on the curve

for i in xrange(len(x) - 1):
    row = [0. for j in xrange(3 * len(x) - 2)]
    row[i] = x[i] * x[i]
    row[i + len(x) - 1] = x[i]
    row[i + 2 * len(x) - 2] = 1.
    row[-1] = y[i]
    matrix.append(row)

    row = [0. for j in xrange(3 * len(x) - 2)]
    row[i] = x[i + 1] * x[i + 1]
    row[i + len(x) - 1] = x[i + 1]
    row[i + 2 * len(x) - 2] = 1.
    row[-1] = y[i + 1]
    matrix.append(row)

# Slopes at inner points

for i in xrange(1, len(x) - 1):
    row = [0. for j in xrange(3 * len(x) - 2)]
    row[i - 1] = 2 * x[i]
    row[i] = -2 * x[i]
    row[i + len(x) - 2] = 1.
    row[i + len(x) - 1] = -1.
    matrix.append(row)

matrix = zip(*solve(matrix))[-1]
a = matrix[: len(matrix) / 3 ]
b = matrix[len(matrix) / 3 : 2 * len(matrix) / 3]
c = matrix[2 * len(matrix) / 3 :]

text = ""
for i in xrange(len(x) - 1):
    text += "     %f <= x && x <= %f ? %f * x * x + %f * x + %f :\\\n" % \
            (x[i], x[i + 1], a[i], b[i], c[i])

print "plot", text[5:-2], "0/0 notitle"


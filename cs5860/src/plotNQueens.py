from pylab import *
from scipy import *
import math
import matplotlib.pyplot as plt

# reading the data from a csv file
durl = 'results.csv'
rdata = genfromtxt(durl,dtype='i,i,i,f',delimiter=',')

#rdata[0] = ones(4) # cutting the label's titles
#rdata[1] = zeros(4) # cutting the global statistics

x = []
y = []
color = []
area = []

fig = plt.figure()
for data in rdata[1:]:
 x.append(data[2]) # number of retries
 y.append(data[1]) # number of neighbors
 color.append(data[3]) # larceny_theft 
 area.append(data[3]**1.5) # population
 # plotting the first eigth letters of the state's name
 plt.text(data[2], data[1]*0.5, data[3],size=11,horizontalalignment='center',verticalalignment='bottom')

plt.text(10, 500, "percent of 1000 boards solved",size=11,horizontalalignment='center',verticalalignment='bottom')

# making the scatter plot
#sct = scatter(x, y, c=color, s=area, linewidths=2, edgecolor='w')

ax = plt.gca()
ax.scatter(x ,y , c='b', alpha=0.4, s=area, linewidths=2, edgecolor='b')

#sct = scatter(x, y, s=area, linewidths=2, edgecolor='w')
#sct.set_alpha(0.75)

ax.set_yscale('log')
ax.set_xscale('log')

#axis([0,11,200,1280])
plt.xlabel('Number of neighbor searches allowed')
plt.ylabel('Number of retries on same board')
plt.title("Solution Space for 8-Queens Problem")
#fig.show()
plt.savefig("queens_space.png")
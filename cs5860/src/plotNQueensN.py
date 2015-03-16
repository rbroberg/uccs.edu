from pylab import *
from scipy import *
import math
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import NullFormatter,MultipleLocator, FormatStrFormatter

# reading the data from a csv file
durl = 'results-all.txt'
#n,nruns,retries,neighbors,nwins,pwins,start,ave,end,steps
rdata = genfromtxt(durl,dtype='i,i,i,i,i,f,f,f,f,f',delimiter=',')
pdf=pd.DataFrame(rdata[1:])
pdf.columns=['n','nruns','retries','neighbors','nwins','pwins','start','ave','end','steps']
#rdata[0] = ones(4) # cutting the label's titles
#rdata[1] = zeros(4) # cutting the global statistics

fig = plt.figure()
islog=True
majorLocator   = MultipleLocator()

# -------------------------------
N=6
# -------------------------------
ax=subplot(221)
x = []
y = []
color = []
area = []
data = np.array(pdf[pdf.n==N])
for j in range(data.shape[0]):
 x.append(data[j,3]) # number of neighbors
 y.append(data[j,2]) # number of retries
 color.append((100*data[j,5])) # larceny_theft 
 area.append((100*data[j,5])**1.5) # population
 # plotting the first eigth letters of the state's name
 pstr="%.1f" % (100*data[j,5])
 plt.text(data[j,3]*1.2, data[j,2]*0.5, pstr,size=11,horizontalalignment='left',verticalalignment='bottom')

plt.text(300,120,"N="+str(N))
ax = plt.gca()
#ax.scatter(1 ,1 , c='b', alpha=0.2, s=0.1, linewidths=2, edgecolor='w')
ax.scatter(x ,y , c='b', alpha=0.2, s=area, linewidths=2, edgecolor='b')
ax.set_yscale('log')
ax.set_xscale('log')
ax.xaxis.set_major_formatter( NullFormatter() )

# -------------------------------
N=7
# -------------------------------
ax=subplot(222)
x = []
y = []
color = []
area = []
data = np.array(pdf[pdf.n==N])
for j in range(data.shape[0]):
 x.append(data[j,3]) # number of neighbors
 y.append(data[j,2]) # number of retries
 color.append((100*data[j,5])) # larceny_theft 
 area.append((100*data[j,5])**1.5) # population
 # plotting the first eigth letters of the state's name
 pstr="%.1f" % (100*data[j,5])
 plt.text(data[j,3]*1.2, data[j,2]*0.5, pstr,size=11,horizontalalignment='left',verticalalignment='bottom')

plt.text(300,120,"N="+str(N))

ax = plt.gca()
#ax.scatter(1 ,1 , c='b', alpha=0.2, s=0.1, linewidths=2, edgecolor='w')
ax.scatter(x ,y , c='b', alpha=0.2, s=area, linewidths=2, edgecolor='b')
ax.set_yscale('log')
ax.set_xscale('log')
ax.xaxis.set_major_formatter( NullFormatter() )
ax.yaxis.set_major_formatter( NullFormatter() )
# -------------------------------
N=9
# -------------------------------
ax=subplot(223)
x = []
y = []
color = []
area = []
data = np.array(pdf[pdf.n==N])
for j in range(data.shape[0]):
 x.append(data[j,3]) # number of neighbors
 y.append(data[j,2]) # number of retries
 color.append((100*data[j,5])) # larceny_theft 
 area.append((100*data[j,5])**1.5) # population
 # plotting the first eigth letters of the state's name
 pstr="%.1f" % (100*data[j,5])
 plt.text(data[j,3]*1.2, data[j,2]*0.5, pstr,size=11,horizontalalignment='left',verticalalignment='bottom')

plt.text(300,120,"N="+str(N))

ax = plt.gca()
#ax.scatter(1 ,1 , c='b', alpha=0.2, s=0.1, linewidths=2, edgecolor='w')
ax.scatter(x ,y , c='b', alpha=0.2, s=area, linewidths=2, edgecolor='b')
ax.set_yscale('log')
ax.set_xscale('log')
#ax.xaxis.set_major_locator( majorLocator )

# -------------------------------
N=10
# -------------------------------
ax=subplot(224)
x = []
y = []
color = []
area = []
data = np.array(pdf[pdf.n==N])
for j in range(data.shape[0]):
 x.append(data[j,3]) # number of neighbors
 y.append(data[j,2]) # number of retries
 color.append((100*data[j,5])) # larceny_theft 
 area.append((100*data[j,5])**1.5) # population
 # plotting the first eigth letters of the state's name
 pstr="%.1f" % (100*data[j,5])
 plt.text(data[j,3]*1.2, data[j,2]*0.5, pstr,size=11,horizontalalignment='left',verticalalignment='bottom')

plt.text(300,120,"N="+str(N))

ax = plt.gca()
#ax.scatter(1 ,1 , c='b', alpha=0.2, s=0.1, linewidths=2, edgecolor='w')
ax.scatter(x ,y , c='b', alpha=0.2, s=area, linewidths=2, edgecolor='b')
ax.set_yscale('log')
ax.set_xscale('log')
ax.yaxis.set_major_formatter( NullFormatter() )
#ax.xaxis.set_major_locator( majorLocator )

#axis([0,11,200,1280])
#plt.xlabel('Number of neighbor searches allowed')
#plt.ylabel('Number of retries on same board')
#plt.title("Solution Space for "+str(N)+"-Queens Problem")

figtext(0.3,0.02,"Number of neighbor searches allowed",fontdict={'fontsize':14})
figtext(0.04,.7,"Number of retries on same board",fontdict={'fontsize':14},rotation=90)
subplots_adjust(wspace=0,hspace=0)
#fig.show()
plt.savefig("queens_space-multi.png")

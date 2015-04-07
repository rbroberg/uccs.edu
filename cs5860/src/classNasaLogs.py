import pandas as pd
df = pd.read_pickle("../data/nasa_access_log.pkl")

#df.iloc[1:2,]
#                src         date      time     tz getpost page  protocol  svrresp size
# 1  uplherc.upl.com  01/Aug/1995  00:00:07  -0400     get    /  HTTP/1.0      304    0

# ------------------------------------------------
# src
# ------------------------------------------------
lints=set(df.iloc[:,0])
len(lints)
#137791

# ------------------------------------------------
# date
# ------------------------------------------------
lints=set(df.iloc[:,1])
len(lints)
#58

# ------------------------------------------------
# tz
# ------------------------------------------------
lints=set(df.iloc[:,3])
len(lints)
# 1
# remove column since no information exists
df = df.drop('tz',1)

# ------------------------------------------------
# getpost
# ------------------------------------------------
# convert string to int
df.iloc[:,3][df.iloc[:,3]=='GET']=1
df.iloc[:,3][df.iloc[:,3]=='HEAD']=2
df.iloc[:,3][df.iloc[:,3]=='POST']=3
# check results
set(df.iloc[:,3])
# clean up
df=df[df.iloc[:,3]!='\xfd\xd1\xed.\x8a\x0b2\xed.\x8b>\xee']
set(df.iloc[:,3])
df['getpost'] = df['getpost'].astype(int)

# ------------------------------------------------
# page
# ------------------------------------------------
lints=set(df.iloc[:,4])
len(lints)
# 30877
# indexing is an option
# but want to preserve information
# maybe build directory tree

# ------------------------------------------------
# protocol
# ------------------------------------------------
# faster way to do this?
# many of the size have a 'dash', convert to -1
lints=set(df.iloc[:,6])
# 
# ''
# HTTP/1.0
# HTTP/V1.0
# HTTP/*
# HTML/1.0
# and other stuff indicating bad parsing

len(df[df.iloc[:,5]=='HTTP/1.0'])
# clean up
df=df[df.iloc[:,5]=='HTTP/1.0']
# remove column since no information exists
df = df.drop('protocol',1)

# ------------------------------------------------
# svrresp
# ------------------------------------------------
# faster way to do this?
# many of the size have a 'dash', convert to -1
lints=set(df.iloc[:,5])
# set(['200', '302', '304', '404', '403', '500', '501'])
df['svrresp'] = df['svrresp'].astype(int)

# ------------------------------------------------
# size
# ------------------------------------------------
# faster way to do this?
# many of the size have a 'dash', convert to -1
df.iloc[:,6][df.iloc[:,6]=='-']=-1
df.iloc[:,6]=int(df.iloc[:,6])

lints=set(df.iloc[:,6])
# len(lints)
# 16503
lints=list(lints)
for i in range(len(lints)):
    try:
        df.iloc[:,6][df.iloc[:,6]==lints[i]]=int(lints[i])
        print i
    except:
        print i, lints[i]

df['size'] = df['size'].astype(int)

# df.to_csv("../data/clean_access_log.csv")

# There are 31K page requests
# create an index for the number of requests for each page
lints=set(df.page)
lints=list(lints)
pagereq={}
for i in range(len(lints)):
        pagereq[lints[i]]=sum(df.page==lints[i])

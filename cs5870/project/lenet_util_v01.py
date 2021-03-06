from numpy import genfromtxt,vstack, hstack, zeros, ones
from theano import shared
import theano.tensor as T
import numpy.random

# since the data has a native class ratio of about 20:1
# increase the number of preictal cases
def load_data(case,npre,ninter,ntest,features, split):
    datadir="/data/www.kaggle.com/c/seizure-prediction/download/"
    f=datadir+case+"/"+features[0]+".csv"
    dat=genfromtxt(f, delimiter=',')
    # if this feature is 'cc' then reshape
    if features[0]=="cc":
        d=dat.shape[1]
        n=(int(round((d*2.*4.)**0.5))+1)/2
        dat2=zeros((dat.shape[0],n**2))
        for i in range(dat.shape[0]):
            dat2[i,:]=reshape_crosscorr(dat[i,:]).reshape(1,n**2)
        dat=dat2
    
    dat_train=dat[0:(npre+ninter),:]
    dat_test=dat[(npre+ninter):,:]
    dat_labels=hstack((ones(npre),zeros(ninter)))
    if len(features)>1:
        for n in range(1,len(features)):
            f=datadir+case+"/"+features[n]+".csv"
            dat=genfromtxt(f, delimiter=',')
            dat_train=hstack((dat_train,dat[0:(npre+ninter),:]))
            dat_test=hstack((dat_test,dat[(npre+ninter):,:]))
    
    # pad the preictal datasets to balance number of ictal
    # add gaussian white noise
    rep=ninter/npre # int
    dat_pre=zeros((rep*npre,dat_train.shape[1]))
    for n in range(rep):
        i=npre*n;j=npre*(n+1)
        rv=numpy.random.normal(0,.1,dat_train.shape[1])
        dat_pre[i:j,:]=dat_train[0:npre,:]+rv
    
    dat_train=vstack((dat_pre,dat_train[npre:,:]))
    #randomize draw
    ipre=[i for i in range(dat_train.shape[0])]
    numpy.random.shuffle(ipre)
    idx=int(dat_train.shape[0]*split)
    x_train=dat_train[ipre[0:idx],:]
    x_valid=dat_train[ipre[idx:],:]
    dat_y=hstack((ones(npre*rep),zeros(ninter)))
    y_train=dat_y[ipre[0:idx]]
    y_valid=dat_y[ipre[idx:]]
    x_test=dat_test
    y_test=zeros(x_test.shape[0])
    x_test=dat_test
    y_test=zeros(x_test.shape[0])
    return [(shared(x_train), T.cast(y_train,'int32')), (shared(x_valid), T.cast(y_valid,'int32')), (shared(x_test),T.cast(y_test,'int32'))]

def reshape_crosscorr(diag):
    d=len(diag)
    n=(int(round((d*2.*4.)**0.5))+1)/2 # form of soln for quad eqn
    m=ones((n,n))
    k=0
    for i in range(n):
        for j in range((i+1),n):
            m[i,j]=diag[k]
            m[j,i]=diag[k]
            k=k+1
    return m

from numpy import genfromtxt,vstack, hstack, zeros, ones
from theano import shared
import theano.tensor as T
import numpy.random
import os

# since the data has a native class ratio of about 20:1
# increase the number of preictal cases
# designed to take time series of cc

# sn=numer of preictal test cases series
# sm=number of preictal test cases series to include in training

def load_data_cct(case,npre,ninter,ntest,features, sn, sm):
    datadir="/projects/uccs.edu/cs5870/data/"
    f=datadir+case+"/"+features[0]+".csv"
    if not os.path.isfile(f):
        datadir="/data/www.kaggle.com/c/seizure-prediction/download/"
        f=datadir+case+"/"+features[0]+".csv"
    dat=genfromtxt(f, delimiter=',')
    dat=abs(dat)
    
    dat_pre=dat[0:npre,:]
    dat_inter=dat[npre:(npre+ninter),:]
    dat_test=dat[(npre+ninter):,:]
    dat_labels=hstack((ones(npre),zeros(ninter)))
    
    # split the preictal training and validation
    ridx=[i for i in range(sn)]
    numpy.random.shuffle(ridx)
    tidxpre=zeros(sm*6).astype(int) # index of preictals for training
    k=0
    for n in range(sm):
        for m in range(6):
            tidxpre[k]=ridx[n]*6+m
            k=k+1
    
    vidxpre=zeros((sn-sm)*6).astype(int) # index of preictals for validation
    k=0
    for n in range(sn-sm):
        for m in range(6):
            vidxpre[k]=ridx[sm+n]*6+m
            k=k+1
    
    # split the interictal training and validation
    # as there are many more interictal, just split before shuffle is sufficient
    ridx=[i+npre for i in range(ninter)]
    idxinter=int(ninter*1.*sm/sn)
    tidxinter=ridx[0:idxinter]
    vidxinter=ridx[idxinter:]
    
    # create padded preicatal data to balance size of training
    rep=ninter/npre
    dat_pret=dat[tidxpre,:]
    npret=dat_pret.shape[0]
    dat_pre=zeros((rep*npret,dat_pret.shape[1]))
    for n in range(rep):
        i=npret*n;j=npret*(n+1)
        rv=numpy.random.normal(0,.1,dat_pret.shape[1])
        dat_pre[i:j,:]=dat_pret[0:npret,:]+rv

    
    # merge the padded pre and inter validation sets
    dat_x=vstack((dat_pre,dat[tidxinter,:]))
    dat_y=hstack((ones(rep*npret),zeros(idxinter)))
    ridx=[i for i in range(dat_x.shape[0])]
    numpy.random.shuffle(ridx)
    x_train=dat[ridx,:]
    y_train=dat_y[ridx]

    # merge the pre and inter validation sets
    idxvalid=hstack((vidxpre,vidxinter))
    numpy.random.shuffle(idxvalid)
    x_valid=dat[idxvalid,:]
    y_valid=dat_y[idxvalid]
    
    # pull out the test data
    x_test=dat_test
    y_test=zeros(x_test.shape[0])
    return [(shared(x_train), T.cast(y_train,'int32')), (shared(x_valid), T.cast(y_valid,'int32')), (shared(x_test),T.cast(y_test,'int32'))]
    
# designed to take vector of unique cc values and expand into square matrix
def load_data_cc(case,npre,ninter,ntest,features, split):
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
    dat_y=hstack((ones(npre*rep),-1*ones(ninter)))
    y_train=dat_y[ipre[0:idx]]
    y_valid=dat_y[ipre[idx:]]
    x_test=dat_test
    y_test=-1*ones(x_test.shape[0])
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


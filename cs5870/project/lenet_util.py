from numpy import genfromtxt

def load_data(case,npre,ninter,ntest,features):
	datadir="/data/www.kaggle.com/c/seizure-prediction/download/"
	f=datadir+case+"/"+features[0]+".csv"
	dat=genfromtxt(f, delimiter=',')
	dat_train=dat[0:(npre+ninter),:]
	dat_test=dat[(npre+ninter):,:]
	dat_labels=np.hstack((np.ones(npre),np.zeros(ninter)))
	if len(features)>1:
		for n in range(1,len(features)):
			f=datadir+case+"/"+features[n]+".csv"
			dat=genfromtxt(f, delimiter=',')
			dat_train=np.hstack((dat_train,dat[0:(npre+ninter),:]))
			dat_test=np.hstack((dat_test,dat[(npre+ninter):,:]))
	
	return dat_train, dat_labels, dat_test

def reshape_crosscorr(diag):
	d=len(diag)
	n=(int(round((240.*4.)**0.5))+1)/2 # form of soln for quad eqn
	m=np.ones((n,n))
	k=0
	for i in range(n):
		for j in range((i+1),n):
			m[i,j]=diag[k]
			m[j,i]=diag[k]
			k=k+1
	return m

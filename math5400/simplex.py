g=[-400,-200,-250,0]

c1=[6,2,3,2000]
c2=[8,2,3,3000]
c3=[1,1,1,625]

# make array of constraints
c=[c1,c2,c3,g]
C=np.array(c)
# num variables ??? +1
n=C.shape[1]-1
# entries for slack variables
I=np.eye(C.shape[0])
C=np.hstack((C[:,0:n],I,np.array([C[:,n]]).T))

# vector of solutions
soln=[0]*(C.shape[1]-2)

# entry variable
# most negative value

# if the object variables are all positive, this is done
if sum(C[-1,:]<0):
    print("continue")

# pick entry variable
entry_var = np.argmin(C[-1,0:n])

# find pivot variable
exit_var = np.argmax(C[:,-1] / C[:,entry_var])

# normalize row for pivot variable = 1
C[exit_var,:] =  C[exit_var,:] / C[exit_var,entry_var]

# zero out the other rows
for i in range(C.shape[0]):
    if i <> exit_var:
        C[i,:]=C[i,:]-C[i,entry_var]*C[exit_var,:]

C.astype(int)
# np.sum(C[1:,0:-1]<>0,0)==1

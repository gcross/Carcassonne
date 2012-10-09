import sys
N = int(sys.argv[1])
J = float(sys.argv[2])

from numpy import array, tensordot
X = array([[0,1],[1,0]])
Z = array([[1,0],[0,-1]])

def absorbMatrixAt(M,i,V):
    return tensordot(M,V,(1,i)).transpose(tuple(range(1,i+1)) + (0,) + tuple(range(i+1,N)))

def matvec(in_v):
    in_v = in_v.reshape((2,) * N)
    out_v = 0*in_v
    for i in range(N):
        out_v += absorbMatrixAt(-Z,i,in_v)
        out_v += absorbMatrixAt(X,i,absorbMatrixAt(-J*X,(i+1)%N,in_v))
    return out_v.ravel()

from scipy.sparse.linalg import LinearOperator, eigsh
operator = LinearOperator(shape=(1 << N,)*2,matvec=matvec,dtype=float)
evals, evecs = eigsh(k=1,A=operator)
print float(evals[0])

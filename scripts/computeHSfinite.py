from __future__ import division, print_function
import sys
N = int(sys.argv[1])

from math import pi, sin
from numpy import array, complex128, dot, tensordot
X = array([[0,1],[1,0]])
Y = array([[0,-1j],[1j,0]])
Z = array([[1,0],[0,-1]])

def absorbMatrixAt(M,i,V):
    return tensordot(M,V,(1,i)).transpose(tuple(range(1,i+1)) + (0,) + tuple(range(i+1,N)))

def matvec(in_v):
    in_v = in_v.reshape((2,) * N)
    out_v = 0*in_v
    for m in range(N):
        for n in range(1,N):
            c = 1/sin(n*pi/N)**2/2
            for matrix in (X,Y,Z):
                out_v += absorbMatrixAt(matrix,m,absorbMatrixAt(c*matrix,(m+n)%N,in_v))
    return out_v.ravel()

from scipy.sparse.linalg import LinearOperator, eigsh
operator = LinearOperator(shape=(1 << N,)*2,matvec=matvec,dtype=complex128)
evals, evecs = eigsh(k=1,A=operator)
v = evecs.transpose().reshape((2,) * N)
vrc = v.ravel().conj()

print("E/N = {:.15f}".format(evals[0].real/N**3))

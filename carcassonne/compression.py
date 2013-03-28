# Imports {{{
from numpy import complex128, dot
from numpy.linalg import norm
from scipy.sparse.linalg import LinearOperator, gmres

from .data import NDArrayData
from .utils import Join, crand, prependDataContractor, unitize
# }}}

# Functions {{{
# def computeCompressor(L,R,new_dimension) {{{
@prependDataContractor(
    # 0 = L, 1 = MLD, 2 = MRU, 3 = MRD, 4 = R
    [
        Join(0,2,1,0), # L -- MLD
        Join(2,1,4,0), # MRU -- R
        Join(3,1,4,1), # MRU -- R
        Join(1,1,3,0), # MLD -- MRD
    ],
    [
        [(0,0),(4,2)],
        [(0,1),(2,0)],
    ]
)
def computeProductCompressor(formMatrix,L,R,new_dimension):
    assert L.shape[1] == L.shape[2]
    assert R.shape[0] == R.shape[1]
    assert L.shape[1] == R.shape[1]
    old_dimension = L.shape[1]
    b = L.contractWith(R,(1,2),(0,1)).toArray().ravel()
    compressor = NDArrayData.newRandom(old_dimension,new_dimension).unitize()
    for _ in range(4):
        A = formMatrix(L,compressor.conj(),compressor.transpose().conj(),compressor.transpose(),R).toArray()
        At = A.transpose().conj()
        x, info = gmres(
            LinearOperator((old_dimension*new_dimension,)*2,lambda v: dot(At,dot(A,v)),dtype=complex128),
            dot(At,b)
        )
        assert info == 0
        compressor = NDArrayData(x.reshape(old_dimension,new_dimension)).unitize()
    return compressor.transpose()
# }}}
# }}}

# Exports {{{
__all__ = [
    "computeProductCompressor",
]
# }}}

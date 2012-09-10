# Imports {{{
from copy import copy
from functools import reduce
from numpy import allclose, array, complex128, diag, identity, multiply, ndarray, ones, prod, sqrt, tensordot, zeros
from scipy.linalg import norm, svd, qr
from scipy.sparse.linalg import LinearOperator, eigs

from .utils import crand, randomComplexSample
# }}}

# Base classes {{{
class Data: # {{{
  # Instance methods {{{
    def toNDArrayData(self):
        return NDArrayData(self.toArray())
  # }}}
# }}}
# }}}

# Classes {{{
class NDArrayData(Data): # {{{
  # Class construction methods {{{
    def __init__(self,_arr): # {{{
        self._arr = _arr
    # }}}
    @classmethod # newDiagonal {{{
    def newDiagonal(cls,data):
        return cls(diag(data))
    # }}}
    @classmethod # newEnlargener {{{
    def newEnlargener(cls,old_dimension,new_dimension,dtype=None):
        if new_dimension == old_dimension:
            return (cls.newIdentity(new_dimension),)*2
        else:
            matrix = cls.newRandom(new_dimension,old_dimension).qr(mode='economic')[0]
            return matrix, matrix.conj()
    # }}}
    @classmethod # newFilled {{{
    def newFilled(cls,shape,value,dtype=None):
        if not dtype:
            dtype = type(value)
        _arr = ndarray(shape,dtype)
        _arr[...] = value
        return NDArrayData(_arr)
    # }}}
    @classmethod # newIdentity {{{
    def newIdentity(cls,N,dtype=None):
        return NDArrayData(identity(N,dtype=dtype))
    # }}}
    @classmethod # newOuterProduct {{{
    def newOuterProduct(cls,*factors):
        return cls(reduce(multiply.outer,factors))
    # }}}
    @classmethod # newRandom {{{
    def newRandom(cls,*shape):
        return cls(randomComplexSample(shape))
    # }}}
    @classmethod # newTrivial {{{
    def newTrivial(cls,shape,dtype=int):
        return cls(ones(shape,dtype=dtype))
    # }}}
    @classmethod # newZeros {{{
    def newZeros(cls,shape,dtype=int):
        return cls(zeros(shape,dtype=dtype))
    # }}}
  # }}}
  # Instance methods {{{
    def __iadd__(self,other): # {{{
        self._arr += other._arr
        return self
    # }}}
    def __imul__(self,other): # {{{
        self._arr *= other._arr
        return self
    # }}}
    def __getitem__(self,index): # {{{
        return NDArrayData(self._arr[index])
    # }}}
    def __mul__(self,other): # {{{
        return NDArrayData(self._arr * other._arr)
    # }}}
    def __repr__(self): # {{{
        return "NDArrayData(" + repr(self._arr) + ")"
    # }}}
    def __setitem__(self,index,value): # {{{
        self._arr[index] = value._arr
    # }}}
    def __str__(self): # {{{
        return "NDArrayData({})".format(self._arr)
    # }}}
    def __truediv__(self,other): # {{{
        return NDArrayData(self._arr / other._arr)
    # }}}
    def toArray(self):  #{{{
        return self._arr
    # }}}
    def adjoint(self): # {{{
        if self.ndim != 2:
            raise ValueError("Adjoint may only be computed for rank 2 tensors.")
        return self.conj().join(1,0)
    # }}}
    def absorbMatrixAt(self,axis,matrix): # {{{
        return matrix.contractWith(self,(1,),axis).join(*tuple(range(1,axis+1)) + (0,) + tuple(range(axis+1,self.ndim)))
    # }}}
    def allcloseTo(self,other,rtol=1e-05,atol=1e-08): # {{{
        return allclose(self._arr,other._arr,rtol=rtol,atol=atol)
    # }}}
    def conj(self): # {{{
        return self.__class__(self._arr.conj())
    # }}}
    def contractWith(self,other,self_axes,other_axes): # {{{
        return NDArrayData(tensordot(self._arr,other._arr,(self_axes,other_axes)))
    # }}}
    def directSumWith(self,other,*non_summed_axes): # {{{
        if not self.ndim == other.ndim:
            raise ValueError("In a direct sum the number of axes must match ({} != {})".format(self.ndim,other.ndim))
        if len(non_summed_axes) >= self.ndim:
            raise ValueError("At least one axis must be included in the direct sum;  the axes excluded from the sum were {}, counting {} in total, but the total number of axes in the tensors were {}.".format(non_summed_axes,len(non_summed_axes),self.ndim))
        new_shape = []
        left_indices = []
        right_indices = []
        for axis, m, n in zip(range(self.ndim),self.shape,other.shape):
            if axis in non_summed_axes:
                if m != n:
                    raise ValueError("All non-summed axes must agree in shape, but for axis {} I have dimension {} and oter has dimension {}.".format(axis,m,n))
                else:
                    new_shape.append(m)
                    left_indices.append(slice(0,m))
                    right_indices.append(slice(0,m))
            else:
                new_shape.append(m+n)
                left_indices.append(slice(0,m))
                right_indices.append(slice(m,m+n))

        new_data = zeros(new_shape,dtype=self.dtype)
        new_data[left_indices] = self._arr
        new_data[right_indices] = other._arr
        return NDArrayData(new_data)
    # }}}
    def extractScalar(self): # {{{
        if self.ndim != 0:
            raise ValueError("tensor is not a scalar")
        else:
            return self._arr
    # }}}
    def increaseDimensionsAndFillWithRandom(self,*axes_and_new_dimensions): # {{{
        old_shape = self.shape
        new_shape = list(old_shape)
        for axis, new_dimension in axes_and_new_dimensions:
            old_dimension = old_shape[axis]
            if new_dimension < old_dimension:
                raise ValueError("new dimension for axis {} is less than the old one ({} < {})".format(axis,new_dimension,old_dimension))
            new_shape[axis] = new_dimension
        new_arr = crand(*new_shape)
        old_indices = tuple(slice(0,d) for d in old_shape)
        new_arr[old_indices] = self._arr
        return NDArrayData(new_arr)
    # }}}
    def increaseDimensionsAndFillWithZeros(self,*axes_and_new_dimensions): # {{{
        old_shape = self.shape
        new_shape = list(old_shape)
        for axis, new_dimension in axes_and_new_dimensions:
            old_dimension = old_shape[axis]
            if new_dimension < old_dimension:
                raise ValueError("new dimension for axis {} is less than the old one ({} < {})".format(axis,new_dimension,old_dimension))
            new_shape[axis] = new_dimension
        new_arr = zeros(new_shape,dtype=complex128)
        old_indices = tuple(slice(0,d) for d in old_shape)
        new_arr[old_indices] = self._arr
        return NDArrayData(new_arr)
    # }}}
    def join(self,*groups): # {{{
        groups = [[group] if isinstance(group,int) else group for group in groups]
        _arr = self._arr.transpose([index for group in groups for index in group])
        shape = []
        index = 0
        for group in groups:
            shape.append(prod(_arr.shape[index:index+len(group)]))
            index += len(group)
        return NDArrayData(_arr.reshape(shape))
    # }}}
    def minimizeOver(self,multiplyExpectation,multiplyNormalization): # {{{
        initial = self.toArray().ravel()
        N = len(initial)
        evals, evecs = eigs(
            A=LinearOperator((N,N),dtype=self.dtype,matvec=lambda v: multiplyExpectation(NDArrayData(v.reshape(self.shape))).toArray()),
            M=LinearOperator((N,N),dtype=self.dtype,matvec=lambda v: multiplyNormalization(NDArrayData(v.reshape(self.shape))).toArray()),
            k=1,
            which='SR',
        )
        return NDArrayData(evecs.reshape(self.shape))
    # }}}
    def normalizeAxis(self,axis,sqrt_svals=False): # {{{
        if self.shape[axis] == 1:
            n = (norm(self._arr))
            if sqrt_svals:
                n = sqrt(n)
                return NDArrayData(array([[1/n]])), NDArrayData(array([[n]]))
            else:
                return NDArrayData(self._arr/n), NDArrayData(array([[1/n]])), NDArrayData(array([[n]]))
        svd_axes_to_merge = list(range(self.ndim))
        del svd_axes_to_merge[axis]
        U, S, V = self.join(svd_axes_to_merge,axis).svd(full_matrices=False)
        U_split = list(self.shape)
        del U_split[axis]
        U_split.append(U.shape[1])
        U_join = list(range(self.ndim-1))
        U_join.insert(axis,self.ndim-1)
        S = S.split(S.shape[0],1)
        if sqrt_svals:
            S = S.sqrt()
            return (V/S).conj(), V*S
        else:
            return U.split(*U_split).join(*U_join), (V/S).conj(), V*S
    # }}}
    def qr(self,mode='full'): # {{{
        q, r = qr(self._arr,mode=mode)
        return NDArrayData(q), NDArrayData(r)
    # }}}
    def split(self,*splits): # {{{
        return NDArrayData(self._arr.reshape(splits))
    # }}}
    def splitAt(self,index,*split): # {{{
        splits = [size for size in self._arr.shape]
        assert prod(split) == self._arr.shape[index]
        splits = splits[:index] + list(split) + splits[index+1:]
        return self.split(*splits)
    # }}}
    def splitAtByRoot(self,index,root): # {{{
        size = round(self._arr.shape[index]**(1.0/root))
        assert size**root == self._arr.shape[index]
        return self.splitAt(index,(size,)*root)
    # }}}
    def sqrt(self): # {{{
        return NDArrayData(sqrt(self._arr))
    # }}}
    def svd(self,full_matrices=True): # {{{
        return tuple(NDArrayData(x) for x in svd(self._arr,full_matrices=full_matrices))
    # }}}
    def transpose(self,*args): # {{{
        return NDArrayData(self._arr.transpose(*args))
    # }}}
  # }}}
  # Properties {{{
    dtype = property(fget = lambda self: self._arr.dtype)
    ndim = property(fget = lambda self: self._arr.ndim)
    shape = property(fget = lambda self: self._arr.shape)
  # }}}
# }}}
class ScalarData(Data): # {{{
  # Class construction methods {{{
    def __init__(self,value): # {{{
        self.value = value
    # }}}
    @classmethod # newRandom {{{
    def newRandom(cls):
        return crand()
    # }}}
  # }}}
  # Instance methods {{{
    def __iadd__(self,other): # {{{
        assert isinstance(other,ScalarData)
        self.value += other.value
        return self
    # }}}
    def __repr__(self): # {{{
        return "ScalarData({})".format(self.value)
    # }}}
    def contractWith(self,other,self_sum_axes,other_sum_axes): # {{{
        if isinstance(other,ScalarData):
            assert self_sum_axes == []
            assert other_sum_axes == []
            return ScalarData(self.value*other.value)
        elif isinstance(other,NDArrayData):
            return NDArrayData(self.value*other.toArray())
        else:
            raise TypeError("contraction of ScalarData with {} not supported".format(other.__class__.__name__))
    # }}}
    def extractScalar(self): # {{{
        return self.value
    # }}}
    def join(self,*grouping): # {{{
        assert grouping == []
        return copy(self)
    # }}}
    def toArray(self): # {{{
        return array(self.value)
    # }}}
    def transpose(self,transposition): # {{{
        assert transposition == [0]
        return copy(self)
    # }}}
  # }}}
  # Properties {{{
    ndim = property(lambda _: 0)
    shape = property(lambda _: ())
  # }}}
# }}}
# }}}

# Exports {{{
__all__ = [
    # Classes {{{
    "NDArrayData",
    "ScalarData",
    # }}}
]
# }}}

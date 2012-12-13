# Imports {{{
from copy import copy
from functools import partial, reduce
from math import ceil
from numpy import allclose, any, argsort, array, complex128, diag, dot, identity, isnan, multiply, ndarray, ones, prod, save, sqrt, tensordot, zeros
from scipy.linalg import norm

from ..utils import crand, dropAt, randomComplexSample, relaxOver
# }}}

# Exception classes {{{
class ARPACKError(Exception):
    pass
# }}}

# Base classes {{{
class Data: # {{{
  # Instance methods {{{
    def size(self): # {{{
        return prod(self.shape,dtype=int)
    # }}}
    def toNDArrayData(self): # {{{
        return NDArrayData(self.toArray())
    # }}}
  # }}}
# }}}
# }}}

# Classes {{{
class NDArrayData(Data): # {{{
  # Class construction methods {{{
    def __init__(self,_arr): # {{{
        self._arr = _arr
    # }}}
    @classmethod # newCollected {{{
    def newCollected(cls,datas):
        return cls(array([data.toArray() for data in datas]))
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
    @classmethod # newRandomHermitian {{{
    def newRandomHermitian(cls,*shape):
        data = cls.newRandom(*shape)
        data += data.transpose().conj()
        return data
    # }}}
    @classmethod # newNormalizedRandom {{{
    def newNormalizedRandom(cls,*shape):
        sample = randomComplexSample(shape)
        sample /= norm(sample)
        return cls(sample)
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
    def __add__(self,other): # {{{
        return NDArrayData(self._arr + other._arr)
    # }}}
    def __copy__(self): # {{{
        return NDArrayData(copy(self._arr))
    # }}}
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
    def __neg__(self): # {{{
        return NDArrayData(-self._arr)
    # }}}
    def __mul__(self,other): # {{{
        if isinstance(other,NDArrayData):
            return NDArrayData(self._arr * other._arr)
        else:
            return NDArrayData(self._arr * other)
    # }}}
    def __repr__(self): # {{{
        return "NDArrayData(" + repr(self._arr) + ")"
    # }}}
    def __rmul__(self,other): # {{{
        if isinstance(other,NDArrayData):
            return NDArrayData(other._arr * self._arr)
        else:
            return NDArrayData(other      * self._arr)
    # }}}
    def __setitem__(self,index,value): # {{{
        self._arr[index] = value._arr
    # }}}
    def __str__(self): # {{{
        return "NDArrayData({})".format(self._arr)
    # }}}
    def __sub__(self,other): # {{{
        return NDArrayData(self._arr-other._arr)
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
    def contractWithAlongAll(self,other): # {{{
        assert self.ndim == other.ndim
        return self.contractWith(other,range(self.ndim),range(self.ndim))
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
    def dropUnitAxis(self,axis): # {{{
        if self.shape[axis] != 1:
            raise ValueError("Axis {} has non-unit dimension {}.".format(axis,self.shape[axis]))
        return NDArrayData(self._arr.reshape(dropAt(self.shape,axis)))
    # }}}
    def extractScalar(self): # {{{
        if self.ndim != 0:
            raise ValueError("tensor is not a scalar")
        else:
            return self._arr
    # }}}
    def fold(self,axis): # {{{
        others = list(range(self.ndim))
        del others[axis]
        return self.join(axis,others)
    # }}}
    def hasNaN(self): # {{{
        return any(isnan(self._arr))
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
    def isCloseTo(self,other,rtol=1e-7,atol=1e-7): # {{{
        ndiff = (self-other).norm()
        if not (ndiff <= atol or ndiff/(self.norm()+other.norm())/2 <= rtol):
            print(self._arr,other._arr)
            print(self.hasNaN(),other.hasNaN())
            print(max(self._arr.ravel()),max(other._arr.ravel()))
            print(self.norm(),other.norm(),ndiff,ndiff/(self.norm()+other.norm()))
        return ndiff <= atol or ndiff/(self.norm()+other.norm())/2 <= rtol
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
    def matvecWith(self,v): # {{{
        return v.absorbMatrixAt(0,self)
    # }}}
    def norm(self): # {{{
        return norm(self._arr)
    # }}}
    def normalizeAxis(self,axis,sqrt_svals=False,dont_recip_under=1e-14): # {{{
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

        SI = copy(S.toArray())
        if dont_recip_under:
            snz = abs(SI) > dont_recip_under
            SI[snz] = 1.0/SI[snz]
        else:
            SI = 1.0/SI
        SI = NDArrayData(SI)

        if sqrt_svals:
            S = S.sqrt()
            SI = SI.sqrt()
            return (V*SI).conj(), V*S
        else:
            return U.split(*U_split).join(*U_join), (V*SI).conj(), V*S
    # }}}
    def normalized(self): # {{{
        return type(self)(self._arr/self.norm())
    # }}}
    def qr(self,mode='full'): # {{{
        q, r = qr(self._arr,mode=mode)
        return NDArrayData(q), NDArrayData(r)
    # }}}
    def ravel(self): # {{{
        return NDArrayData(self._arr.ravel())
    # }}}
    def reverseLastAxis(self): # {{{
        return NDArrayData(self._arr[...,::-1])
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
  # Constants {{{
NDArrayData.I = NDArrayData(array([[1,0],[0,1]],dtype=complex128))
NDArrayData.X = NDArrayData(array([[0,1],[1,0]],dtype=complex128))
NDArrayData.Y = NDArrayData(array([[0,1j],[1j,0]],dtype=complex128))
NDArrayData.Z = NDArrayData(array([[1,0],[0,-1]],dtype=complex128))
  # }}}
# }}}
# }}}

# Exports {{{
__all__ = [
    # Exceptions {{{
    "ARPACKError",
    # }}}
    # Classes {{{
    "NDArrayData",
    # }}}
]
# }}}

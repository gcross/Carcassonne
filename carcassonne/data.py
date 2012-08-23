# Imports {{{
from copy import copy
from functools import reduce
from numpy import allclose, array, multiply, ones, prod, tensordot

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
    @classmethod # newFilled {{{
    def newFilled(cls,shape,value,dtype=None):
        if not dtype:
            dtype = type(value)
        _arr = ndarray(shape,dtype)
        _arr[...] = value
        return NDArrayData(_arr)
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
  # }}}
  # Instance methods {{{
    def __iadd__(self,other): # {{{
        self._arr += other._arr
        return self
    # }}}
    def __getitem__(self,index): # {{{
        return NDArrayData(self._arr[index])
    # }}}
    def __setitem__(self,index,value): # {{{
        self._arr[index] = value._arr
    # }}}
    def toArray(self):  #{{{
        return self._arr
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
    def extractScalar(self): # {{{
        if self.ndim != 0:
            raise ValueError("tensor is not a scalar")
        else:
            return self._arr
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
    def transpose(self,*args): # {{{
        return NDArrayData(self._arr.transpose(*args))
    # }}}
  # }}}
  # Properties {{{

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

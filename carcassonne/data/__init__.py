# Imports {{{
from copy import copy
from functools import partial, reduce
from math import ceil
from numpy import allclose, any, array, complex128, diag, dot, identity, isnan, multiply, ndarray, ones, prod, save, sqrt, tensordot, zeros
from scipy.linalg import norm, qr, svd

from ..utils import crand, dropAt, randomComplexSample, unitize
# }}}

# Exception classes {{{
class ARPACKError(Exception):
    """Exception thrown when ARPACK signals an error."""
# }}}

# Base classes {{{
class Data: # {{{
    """Base class with common functionality for numeric data types."""
  # Instance methods {{{
    def size(self): # {{{
        """Returns the size (number of elements) of this data."""
        return prod(self.shape,dtype=int)
    # }}}
    def toNDArrayData(self): # {{{
        """Creates a copy of this numeric data of type :class:`NDArrayData`."""
        return NDArrayData(self.toArray())
    # }}}
  # }}}
# }}}
# }}}

# Classes {{{
class NDArrayData(Data): # {{{
    """Representation of numeric data using numpy's :class:`ndarray` type."""
  # Class construction methods {{{
    def __init__(self,_arr): # {{{
        self._arr = _arr
    # }}}
    @classmethod # newCollected {{{
    def newCollected(cls,datas):
        """\
Returns the result of concatenating all of the :class:`NDArrayData`s in *datas*
--- that is, all of the dimensions of each tensor in *datas* must match and the
result is a tensor whose first dimension is equal to ``len(datas)`` and whose
remaining dimensions match the dimensions of each value in *datas*.\
"""
        return cls(array([data.toArray() for data in datas]))
    # }}}
    @classmethod # newDiagonal {{{
    def newDiagonal(cls,data):
        """Returns a diagonal matrix with its diagonal entries given by *data*."""
        return cls(diag(data))
    # }}}
    @classmethod # newEnlargener {{{
    def newEnlargener(cls,old_dimension,new_dimension,dtype=None):
        """\
Let ``A, B = X.newEnlargener(..)``;  then both A and B are matrices with
dimensions (*new_dimension*,*old_dimension*) with the property that contracting
them along their first axes results in the identity.  These matrices are called
enlargeners because they enlarge both sides of a bond connecting two tensors
without changing the value of contracting them.\
"""
        if new_dimension == old_dimension:
            return (cls.newIdentity(new_dimension),)*2
        else:
            matrix = cls.newRandom(new_dimension,old_dimension).qr(mode='economic')[0]
            return matrix, matrix.conj()
    # }}}
    @classmethod # newFilled {{{
    def newFilled(cls,shape,value,dtype=None):
        """Returns a new tensor with the given *shape* filled with the given *value*, which is first casted to *dtype* if it is not `None`."""
        if not dtype:
            dtype = type(value)
        _arr = ndarray(shape,dtype)
        _arr[...] = value
        return NDArrayData(_arr)
    # }}}
    @classmethod # newIdentity {{{
    def newIdentity(cls,N,dtype=None):
        """Returns an identity matrix with dimension *N* and type *dtype*."""
        return NDArrayData(identity(N,dtype=dtype))
    # }}}
    @classmethod # newOuterProduct {{{
    def newOuterProduct(cls,*factors):
        """Returns the outer product of *factors*."""
        return cls(reduce(multiply.outer,factors))
    # }}}
    @classmethod # newRandom {{{
    def newRandom(cls,*shape):
        """Returns a new tensor with (complex) random data."""
        return cls(randomComplexSample(shape))
    # }}}
    @classmethod # newRandomHermitian {{{
    def newRandomHermitian(cls,*shape):
        """Returns a new Hermitian tensor with (complex) random data."""
        data = cls.newRandom(*shape)
        data += data.transpose().conj()
        return data
    # }}}
    @classmethod # newNormalizedRandom {{{
    def newNormalizedRandom(cls,*shape):
        """Returns a new normalized tensor with (complex) random data."""
        sample = randomComplexSample(shape)
        sample /= norm(sample)
        return cls(sample)
    # }}}
    @classmethod # newTrivial {{{
    def newTrivial(cls,shape,dtype=int):
        """Returns a new tensor with the given *shape* filled with ones of type *dtype*."""
        return cls(ones(shape,dtype=dtype))
    # }}}
    @classmethod # newZeros {{{
    def newZeros(cls,shape,dtype=int):
        """Returns a new tensor with the given *shape* filled with zeros of type *dtype*."""
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
    def toArray(self):  # {{{
        """Returns the :class:`ndarray` value wrapped by this data."""
        return self._arr
    # }}}
    def adjoint(self): # {{{
        """\
Returns the adjoint (conjugate transpose) of this data; an error is raised if
the rank is not 2.\
"""
        if self.ndim != 2:
            raise ValueError("Adjoint may only be computed for rank 2 tensors.")
        return self.conj().join(1,0)
    # }}}
    def absorbMatrixAt(self,axis,matrix): # {{{
        """\
Absorbs *matrix* into this data into *axis*.  Specifically, this method returns
the result of contracting this data at axis *axis* with *matrix* at axis 1,
rearranging the indices so that axis 0 of *matrix* is at axis *axis* of this
data.
"""
        return matrix.contractWith(self,(1,),axis).join(*tuple(range(1,axis+1)) + (0,) + tuple(range(axis+1,self.ndim)))
    # }}}
    def allcloseTo(self,other,rtol=1e-05,atol=1e-08): # {{{
        """\
Returns whether all elements of this data are close to all elements of
*other* within relative tolerance *rtol* and absolute tolerance *atol*.\
"""
        return allclose(self._arr,other._arr,rtol=rtol,atol=atol)
    # }}}
    def conj(self): # {{{
        """Returns the conjugate of this data."""
        return self.__class__(self._arr.conj())
    # }}}
    def contractWith(self,other,self_axes,other_axes): # {{{
        """Contracts *self* along *self_axes* with *other* along *other_axes*."""
        return NDArrayData(tensordot(self._arr,other._arr,(self_axes,other_axes)))
    # }}}
    def contractWithAlongAll(self,other): # {{{
        """Contracts *self* with *other* along all axes."""
        assert self.ndim == other.ndim
        return self.contractWith(other,range(self.ndim),range(self.ndim))
    # }}}
    def copy(self): # {{{
        """Makes a (deep) copy of *self* (i.e., such that the returned value has its own copy of the data)."""
        return self.__copy__()
    # }}}
    def directSumWith(self,other,*non_summed_axes): # {{{
        """\
Takes the direct sum of *self* and *other* along all axes but those listed as
*non_summed_axes*.  So for example, if ``A`` has dimension 3 and ``B`` has
diension 4, then ``A.directSumWith(B)`` has dimension 7 and is effectively the
result of appending the data in ``B`` to ``A``.  If ``A`` has dimensions (1,2)
and ``B`` has dimensions (3,4), then ``A.directSumWith(B)`` is a block-diagonal
matrix with ``A`` in the upper-left block and ``B`` in the lower-right block.

If *non_summed_axes* is non-empty, then the given axes are shared rather than
being summed over, which implies in particular that the dimensions of the two
tensor along each axis in *non_summed_axes* must match with each other.  For
example if ``A`` and ``B`` have dimension 3 then ``A.directSumWith(B,0)`` also
has dimension 3 and is the result of taking the element-wise sum of ``A`` and
``B``; if ``A`` has dimensions (1,5) and ``B`` has dimensions (2,5) then
``A.directSumWith(B,1)`` has dimensions ``(3,5)``, it's first row is equal to
``A``, and its last two rows are equal to ``B``; finally, if both ``A`` and
``B`` have dimensions (1,2) then ``A.directSumWith(B,0,1)`` (which is
equivalent to ``A.directSumWith(B,1,0)``) has dimensions (1,2) and is equal to
``A`` and ``B`` added together element-wise.\
"""
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
        """Drops a given axis with dimension 1;  throws a :exc:`ValueError` if the dimension is not 1."""
        if self.shape[axis] != 1:
            raise ValueError("Axis {} has non-unit dimension {}.".format(axis,self.shape[axis]))
        return NDArrayData(self._arr.reshape(dropAt(self.shape,axis)))
    # }}}
    def extractScalar(self): # {{{
        """\
Given a tensor with rank zero, returns the scalar value it represents; if
the tensor has non-zero rank then an :exc:`ValueError` is thrown.\
"""
        if self.ndim != 0:
            raise ValueError("tensor is not a scalar")
        else:
            return self._arr
    # }}}
    def fold(self,axis): # {{{
        """Moves the given *axis* to first and flattens all other axes, resulting in a matrix."""
        others = list(range(self.ndim))
        del others[axis]
        return self.join(axis,others)
    # }}}
    def hasNaN(self): # {{{
        """Returns whether any component in the array is NaN."""
        return any(isnan(self._arr))
    # }}}
    def increaseDimensionsAndFillWithRandom(self,*axes_and_new_dimensions): # {{{
        """\
Increases the dimension of each axis by the given increment, where
*axes_and_new_dimensions* is a list of axis/increment pairs;  the newly
created space is filled with random complex numbers.\
"""
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
        """\
Increases the dimension of each axis by the given increment, where
*axes_and_new_dimensions* is a list of axis/increment pairs;  the newly
created space is filled with zeros.\
"""
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
        """\
Returns whether *self* is close to *other* (using the norm) to within either
*atol* absolute tolerance of *rtol* relative tolerance.\
"""
        ndiff = (self-other).norm()
        if not (ndiff <= atol or ndiff/(self.norm()+other.norm())/2 <= rtol):
            print(self._arr,other._arr)
            print(self.hasNaN(),other.hasNaN())
            print(max(self._arr.ravel()),max(other._arr.ravel()))
            print(self.norm(),other.norm(),ndiff,ndiff/(self.norm()+other.norm()))
        return ndiff <= atol or ndiff/(self.norm()+other.norm())/2 <= rtol
    # }}}
    def join(self,*groups): # {{{
        """\
Given *groups*, which must be a collection of collections of axes such that
each axis in *self* appears in exactly one *group* and every group contains
only axes of *self*, returns a tensor of rank ``len(groups)`` where each axis
is formed by flattening all of the axes together in the corresponding group in
*groups*.\
"""
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
        """Returns the result of multiplying this matrix with *v*."""
        return v.absorbMatrixAt(0,self)
    # }}}
    def norm(self): # {{{
        """Returns the norm of the data."""
        return norm(self._arr)
    # }}}
    def normalizeAxis(self,axis,dont_recip_under=1e-14): # {{{
        """\
Let ``A, B, C = X.normalizeAxis(...)``.  Then `A` is the *normalized* matrix,
which means it has the property that if you contract it with its conjugate
along all axes except *axis* then the result is the identity matrix, `B` is the
*normalizer* matrix, which means that it has the property that absorbing it
into `X` at *axis* obtains `A`, and `C` is the *denormalizer* matrix, which
means that the result of contracting it with `B` along axis 1 (for both
matrices) is the identity matrix.

Since, computing `B` requires taking the inverse of the (diagonal) matrix of
singular values, the *dont_recip_under* parameter indicates the threshold below
which a singular value is interpreted to be zero.\
"""
        if self.shape[axis] == 1:
            n = (norm(self._arr))
            return NDArrayData(self._arr/n), NDArrayData(array([[1/n]])), NDArrayData(array([[n]]))
        svd_axes_to_merge = list(range(self.ndim))
        del svd_axes_to_merge[axis]
        M = self.join(svd_axes_to_merge,axis)
        if M.shape[0] < M.shape[1]:
            raise ValueError("the total number of degrees of freedom in all other axes ({}) are not enough to normalize axis ({}) with dimension ({})".format(M.shape[0],axis,M.shape[1]))
        U, S, V = M.svd(full_matrices=False)

        U = U.contractWith(V,(1,),(0,))

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

        return U.split(*U_split).join(*U_join), V.transpose().contractWith((V*SI).conj(),(1,),(0,)), V.transpose().conj().contractWith(V*S,(1,),(0,))
    # }}}
    def normalizeAxisAndDenormalize(self,axis_to_norm,axis_to_denorm,data_to_denormalize=None,dont_recip_under=1e-14): # {{{
        """\
Returns the result of normalizing *self* along the axis *axis_to_norm* and
*data_to_denormalize* along the axis *axis_to_denorm*; that is, this method
calls *normalizeAxis* on *self* with *axis_to_norm* and *dont_recip_under* and
returns the normalized data and the result of absorbing the denormalizer into
*data_to_denormalize* at axis *axis_to_denorm*.  This function is method
because one often wants to normalize a given tensor and then push the "junk"
needed to keep the tensor network contraction equal into a different tensor
that doesn't need to be normalized.  If *data_to_denormalize* is `None`, then
it defaults to be equal to *self*.\
"""
        if data_to_denormalize is None:
            data_to_denormalize = self
        if self.shape[axis_to_norm] != data_to_denormalize.shape[axis_to_denorm]:
            raise ValueError("Normalized axis and denormalized axis have different sizes ({} != {}).".format(self.shape[axis_to_norm],data_to_denormalize.shape[axis_to_denorm]))
        normalized_data, _, denormalizer = self.normalizeAxis(axis_to_norm,dont_recip_under)
        denormalized_data = data_to_denormalize.absorbMatrixAt(axis_to_denorm,denormalizer)
        return normalized_data, denormalized_data
    # }}}
    def normalized(self): # {{{
        """Normalizes this tensor by dividing every element by the norm of this tensor."""
        return type(self)(self._arr/self.norm())
    # }}}
    def qr(self,mode='full'): # {{{
        """\
Computes the QR composition, i.e. ``Q, R = x.qr()``; the full matrices are
returned if *mode* equals 'full' (the default).\
"""
        q, r = qr(self._arr,mode=mode)
        return NDArrayData(q), NDArrayData(r)
    # }}}
    def ravel(self): # {{{
        """Returns the result of raveling this tensor into a vector."""
        return NDArrayData(self._arr.ravel())
    # }}}
    def reverseLastAxis(self): # {{{
        """Reverses the order of the data in the last axis."""
        return NDArrayData(self._arr[...,::-1])
    # }}}
    def size(self): # {{{
        """Returns the total number of elements."""
        return prod(self.shape)
    # }}}
    def split(self,*splits): # {{{
        """Reshapes this tensor to the dimensions specified in *splits*."""
        return NDArrayData(self._arr.reshape(splits))
    # }}}
    def splitAt(self,index,*split): # {{{
        """\
Splits the axis at the given index to have the shape given by *split*; put
another way, it takes the tensor's current shape and replaces the dimension
of axis with the dimensions given by *split*; an error is raised if the old and
new shapes are not compatible.
"""
        splits = [size for size in self._arr.shape]
        assert prod(split) == self._arr.shape[index]
        splits = splits[:index] + list(split) + splits[index+1:]
        return self.split(*splits)
    # }}}
    def splitAtByRoot(self,index,root): # {{{
        """\
Splits index into a *root* number of equal dimensions, so for example if
root is 2 and *index* has dimension 9 then it splits *index* into 3 and 3; if
the dimension is not equal to some value raised to *root* then an error is
raised.\
"""
        size = round(self._arr.shape[index]**(1.0/root))
        assert size**root == self._arr.shape[index]
        return self.splitAt(index,(size,)*root)
    # }}}
    def sqrt(self): # {{{
        """Returns the element-wise square root of this tensor."""
        return NDArrayData(sqrt(self._arr))
    # }}}
    def svd(self,full_matrices=True): # {{{
        """Computes the singular value composition, i.e. `U, S, V = x.svd()``; the full matrices are returned."""
        return tuple(NDArrayData(x) for x in svd(self._arr,full_matrices=full_matrices))
    # }}}
    def transpose(self,*args): # {{{
        """Reverses the order of the axes."""
        return NDArrayData(self._arr.transpose(*args))
    # }}}
    def unitize(self): # {{{
        """Returns the best approximation of this matrix to a unitary, using the `unitize` function in :mod:`carcassonne.utils`."""
        return NDArrayData(unitize(self.toArray()))
    # }}}
  # }}}
  # Properties {{{
    dtype = property(fget = lambda self: self._arr.dtype)
    """The type of the contained data."""

    ndim = property(fget = lambda self: self._arr.ndim)
    """The rank (number of dimensions) of the tensor."""

    shape = property(fget = lambda self: self._arr.shape)
    """The shape of the tensor."""
  # }}}
  # Constants {{{
NDArrayData.I = NDArrayData(array([[1,0],[0,1]],dtype=complex128))
NDArrayData.X = NDArrayData(array([[0,1],[1,0]],dtype=complex128))
NDArrayData.Y = NDArrayData(array([[0,-1j],[1j,0]],dtype=complex128))
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

# Imports {{{
from copy import copy
import numpy
from numpy import all, allclose, array, dot, identity, product, tensordot
from numpy.linalg import norm
from numpy.random import rand
from paycheck import *
from random import randint, shuffle
import unittest

from ..utils import *
# }}}

# Decorators {{{

class prependContractor: # {{{
    def __init__(self,*args,**keywords):
        self.contractor = formContractor(*args,**keywords)
    def __call__(self,f):
        def new_f(*args,**keywords):
            return f(self.contractor,*args,**keywords)
        return new_f
# }}}

# }}}

# Functions {{{

def ensurePhysicalDimensionSufficientlyLarge(tensor,index,dimension): # {{{
    if dimension > product(withoutIndex(tensor.dimensions(),index)):
        new_shape = list(tensor.dimensions())
        new_shape[0] = dimension
        return type(tensor)(crand(*new_shape))
    else:
        return tensor
# }}}

def ensurePhysicalDimensionSufficientlyLargeToNormalize(tensor,index): # {{{
    return ensurePhysicalDimensionSufficientlyLarge(tensor,index,tensor.dimension(index))
# }}}

def randomIndex(ndim): # {{{
    return randint(0,ndim-1)
# }}}

def randomNormalizableTensorAndIndex(ndim): # {{{
    index = randomIndex(ndim)
    shape = randomShape(ndim)
    shape[index] = randint(1,product(withoutIndex(shape,index)))
    return crand(*shape), index
# }}}

def randomPermutation(size): # {{{
    permutation = list(range(size))
    shuffle(permutation)
    return permutation
# }}}

def randomShape(ndim,maximum=5): # {{{
    return tuple(randint(1,maximum) for _ in range(ndim))
# }}}

def randomShapeAgreeingWith(ndim,index,other_dimension): # {{{
    shape = randomShape(ndim)
    shape[index] = other_dimension
    return shape
# }}}

def randomShuffledPartition(elements): # {{{
    elements = list(elements)
    shuffle(elements)
    return randomPartition(elements)
# }}}

def randomTensor(ndim): # {{{
    return crand(*randomShape(ndim))
# }}}

def randomTensorAgreeingWith(ndim,index,other_dimension): # {{{
    return crand(*randomShapeAgreeingWith(ndim,index,other_dimension))
# }}}

def randomTensorAndIndex(ndim): # {{{
    return randomTensor(ndim), randomIndex(ndim)
# }}}

def randomTensorAndIndexAgreeingWith(ndim,other_dimension): # {{{
    index = randomIndex(ndim)
    return randomTensorAgreeingWith(ndim,index,other_dimension), index

# }}}

# }}}

# Classes {{{

class Dummy: # {{{
    def __init__(self,**fields):
        for name, value in fields.items():
            setattr(self,name,value)
# }}}

class TestCase(unittest.TestCase): # {{{
    def assertAllClose(self,v1,v2): # {{{
        v1 = array(v1)
        v2 = array(v2)
        self.assertEqual(v1.shape,v2.shape)
        self.assertTrue(allclose(v1,v2))
    # }}}

    def assertAllEqual(self,v1,v2): # {{{
        v1 = array(v1)
        v2 = array(v2)
        self.assertEqual(v1.shape,v2.shape)
        self.assertTrue(all(v1 == v2))
    # }}}

    def assertDataAlmostEqual(self,v1,v2,rtol=1e-05,atol=1e-08): # {{{
        self.assertEqual(v1.shape,v2.shape)
        self.assertTrue(v1.allcloseTo(v2,rtol=rtol,atol=atol))
    # }}}

    def assertNormalized(self,tensor,index): # {{{
        self.assertAllClose(
            tensordot(tensor.conj(),tensor,(withoutIndex(range(tensor.ndim),index),)*2),
            identity(tensor.shape[index])
        )
    # }}}

    def assertVanishing(self,v): # {{{
        self.assertAlmostEqual(norm(v),0)
    # }}}
# }}}

# }}}

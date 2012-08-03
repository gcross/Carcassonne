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

class prependContractor(object): # {{{
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

def randomPartition(elements): # {{{
    number_of_elements = len(elements)
    number_of_partitions = randint(0,number_of_elements)
    partition_indices = [randint(0,number_of_elements) for _ in xrange(number_of_partitions)]
    partition_indices.sort()
    partition_indices = [0] + partition_indices + [number_of_elements]
    return [elements[partition_indices[i]:partition_indices[i+1]] for i in xrange(number_of_partitions+1)]
# }}}

def randomShape(ndim): # {{{
    return [randint(1,5) for _ in range(ndim)]
# }}}

def randomShapeAgreeingWith(ndim,index,other_dimension): # {{{
    shape = randomShape(ndim)
    shape[index] = other_dimension
    return shape
# }}}

def randomShuffledPartition(elements):
    elements = list(elements)
    shuffle(elements)
    return randomPartition(elements)

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

class TestCase(unittest.TestCase):
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

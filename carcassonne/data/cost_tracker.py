# Imports {{{
from numpy import prod

from ..data import Data
# }}}

# Classes {{{
class CostTracker(Data): # {{{
    """\
An instance of :class:`Data` that keeps track of the cost of performing a tensor
network contraction; *shape* is the shape of the tensor and *cost* is the cost
of constructing this tensor (if it is the result of a tensor contraction).\
"""
    def __init__(self,shape,cost=0): # {{{
        self.shape = shape
        self.ndim = len(shape)
        self.cost = cost
    # }}}
    def __repr__(self): # {{{
        return "CostTracker({},{})".format(self.shape,self.cost)
    # }}}
    def contractWith(self,other,self_axes,other_axes): # {{{
        """\
Contracts *self* along *self_axes* with *other* along *other_axes*; the
resulting tensor has its *cost* field set to the cost of performing this
contraction plus the *cost* of *self* and *other*.
"""
        shape = tuple(x for i,x in enumerate(self.shape)  if i not in self_axes) + \
                tuple(x for i,x in enumerate(other.shape) if i not in other_axes)
        cost = self.cost + other.cost + prod(shape,dtype=int)*prod([x for i,x in enumerate(self.shape) if i in self_axes],dtype=int)
        return CostTracker(shape,cost)
    # }}}
    def extractScalar(self): # {{{
        """Asserts that this tensor has rank 0, and then returns itself."""
        assert len(self.shape) == 0
        return self
    # }}}
    def join(self,*groups): # {{{
        """\
Given *groups*, which must be a collection of collections of axes such that
each axis in *self* appears in exactly one *group* and every group contains
only axes of *self*, returns a tensor of rank ``len(groups)`` where each axis
is formed by flattening all of the axes together in the corresponding group in
*groups*.\
"""
        return \
            CostTracker(
                tuple(prod([self.shape[i] for i in group],dtype=int) for group in groups),
                self.cost
            )
    # }}}
# }}}
# }}}

# Functions {{{
def computeCostOfContracting(contractor,*tensors): # {{{
    """\
Returns the cost of contracting the given tensor network, where *contractor* is
the function that takes a list of tensors as input and returns the result of
contracting them as output, and *tensors* is a list of values each of which is
either a tuple that gives the shape of a tensor or which is some other value
which has a *shape* field.\
"""
    return contractor(*(CostTracker(tensor.shape) if not isinstance(tensor,tuple) else CostTracker(tensor) for tensor in tensors)).cost
# }}}
# }}}

# Exports {{{
__all__ = [
    "CostTracker",

    "computeCostOfContracting",
] # }}}

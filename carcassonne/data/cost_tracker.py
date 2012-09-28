# Imports {{{
from numpy import prod

from ..data import Data
# }}}

# Classes {{{
class CostTracker(Data): # {{{
    def __init__(self,shape,cost=0): # {{{
        self.shape = shape
        self.ndim = len(shape)
        self.cost = cost
    # }}}
    def __repr__(self): # {{{
        return "CostTracker({},{})".format(self.shape,self.cost)
    # }}}
    def contractWith(self,other,self_axes,other_axes): # {{{
        shape = tuple(x for i,x in enumerate(self.shape)  if i not in self_axes) + \
                tuple(x for i,x in enumerate(other.shape) if i not in other_axes)
        cost = self.cost + other.cost + prod(shape,dtype=int)*prod([x for i,x in enumerate(self.shape) if i in self_axes],dtype=int)
        return CostTracker(shape,cost)
    # }}}
    def extractScalar(self): # {{{
        assert len(self.shape) == 0
        return self
    # }}}
    def join(self,*groups): # {{{
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
    return contractor(*(CostTracker(tensor.shape) for tensor in tensors)).cost
# }}}
# }}}

# Exports {{{
__all__ = [
    "CostTracker",

    "computeCostOfContracting",
] # }}}

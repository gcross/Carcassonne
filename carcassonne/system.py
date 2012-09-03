# Imports {{{
from numpy import prod, zeros
from numpy.linalg import eigh

from .sparse import Operator
from .tensors.sparse import absorbSparseSideIntoCornerFromLeft, absorbSparseSideIntoCornerFromRight, absorbSparseCenterSOSIntoSide, formExpectationAndNormalizationMultipliers
from .utils import L, R
# }}}

# Classes {{{
class System: # {{{
    def __init__(self,corners,sides,state_center_data,operator_center_tensor,state_center_data_conj=None): # {{{
        self.corners = list(corners)
        self.sides = list(sides)
        self.operator_center_tensor = operator_center_tensor
        self.state_center_data = state_center_data
        if state_center_data_conj is None:
            self.state_center_data_conj = self.state_center_data.conj()
    # }}}
    def absorbCenter(self,direction): # {{{
        self.corners[direction] = absorbSparseSideIntoCornerFromLeft(self.corners[direction],self.sides[L(direction)])
        self.sides[direction] = absorbSparseCenterSOSIntoSide(direction,self.sides[direction],self.state_center_data,self.operator_center_tensor,self.state_center_data_conj)
        self.corners[R(direction)] = absorbSparseSideIntoCornerFromRight(self.corners[R(direction)],self.sides[R(direction)])
    # }}}
    def computeExpectation(self): # {{{
        return self.state_center_data_conj.contractWith(self.formExpectationMultiplier()(self.state_center_data),range(5),range(5)).extractScalar()
    # }}}
    def computeNormalization(self): # {{{
        return self.state_center_data_conj.contractWith(self.formNormalizationMultiplier()(self.state_center_data),range(5),range(5)).extractScalar()
    # }}}
    def formExpectationAndNormalizationMultipliers(self): # {{{
        return formExpectationAndNormalizationMultipliers(self.corners,self.sides,self.operator_center_tensor)
    # }}}
    def formExpectationMultiplier(self): # {{{
        return self.formExpectationAndNormalizationMultipliers()[0]
    # }}}
    def formNormalizationMultiplier(self): # {{{
        return self.formExpectationAndNormalizationMultipliers()[1]
    # }}}
    def minimizeExpectation(self): # {{{
        state_center_data = self.state_center_data
        if prod(state_center_data.shape[:4]) == 1:
            N = prod(state_center_data.shape)
            operator = state_center_data.newZeros(shape=(N,N),dtype=state_center_data.dtype)
            for tag, data in self.operator_center_tensor.items():
                if isinstance(tag,Operator):
                    operator += data
            evals, evecs = eigh(operator.toArray())
            self.state_center_data = type(state_center_data)(evecs[:,0].reshape(state_center_data.shape))
        else:
            self.state_center_data = state_center_data.minimizeOver(*self.formExpectationAndNormalizationMultipliers())
    # }}}
    def setStateCenter(self,state_center_data,state_center_data_conj=None): # {{{
        self.state_center_data = state_center_data
        if state_center_data_conj is None:
            self.state_center_data_conj = state_center_data.conj()
        else:
            self.state_center_data_conj = state_center_data_conj
    # }}}
# }}}
# }}}

# Exports {{{
__all__ = [
    "System",
]
# }}}

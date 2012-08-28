# Imports {{{
from .tensors.sparse import absorbSparseSideIntoCornerFromLeft, absorbSparseSideIntoCornerFromRight, absorbSparseCenterSOSIntoSide, formExpectationMultiplier
from .utils import L, R
# }}}

# Classes {{{
class Expectation: # {{{
    def __init__(self,corners,sides,operator_center_tensor): # {{{
        self.corners = list(corners)
        self.sides = list(sides)
        self.operator_center_tensor = operator_center_tensor
    # }}}
    def absorbCenter(self,direction,state_center_data,state_center_data_conj=None): # {{{
        self.corners[direction] = absorbSparseSideIntoCornerFromLeft(direction,self.corners[direction],self.sides[L(direction)])
        self.sides[direction] = absorbSparseCenterSOSIntoSide(direction,self.sides[direction],state_center_data,self.operator_center_tensor,state_center_data_conj)
        self.corners[R(direction)] = absorbSparseSideIntoCornerFromRight(R(direction),self.corners[R(direction)],self.sides[R(direction)])
    # }}}
    def computeExpectation(self,state_center_data,state_center_data_conj=None): # {{{
        if state_center_data_conj is None:
            state_center_data_conj = state_center_data.conj()
        return state_center_data_conj.contractWith(self.formMultiplier()(state_center_data),range(5),range(5)).extractScalar()
    # }}}
    def formMultiplier(self): # {{{
        return formExpectationMultiplier(self.corners,self.sides,self.operator_center_tensor)
    # }}}
# }}}
# }}}

# Exports {{{
__all__ = [
    "Expectation",
]
# }}}

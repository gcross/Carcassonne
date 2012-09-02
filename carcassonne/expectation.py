# Imports {{{
from .tensors.sparse import absorbSparseSideIntoCornerFromLeft, absorbSparseSideIntoCornerFromRight, absorbSparseCenterSOSIntoSide, formExpectationAndNormalizationMultipliers
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
        self.corners[direction] = absorbSparseSideIntoCornerFromLeft(self.corners[direction],self.sides[L(direction)])
        self.sides[direction] = absorbSparseCenterSOSIntoSide(direction,self.sides[direction],state_center_data,self.operator_center_tensor,state_center_data_conj)
        self.corners[R(direction)] = absorbSparseSideIntoCornerFromRight(self.corners[R(direction)],self.sides[R(direction)])
    # }}}
    def computeExpectation(self,state_center_data,state_center_data_conj=None): # {{{
        if state_center_data_conj is None:
            state_center_data_conj = state_center_data.conj()
        return state_center_data_conj.contractWith(self.formExpectationMultiplier()(state_center_data),range(5),range(5)).extractScalar()
    # }}}
    def computeNormalization(self,state_center_data,state_center_data_conj=None): # {{{
        if state_center_data_conj is None:
            state_center_data_conj = state_center_data.conj()
        return state_center_data_conj.contractWith(self.formNormalizationMultiplier()(state_center_data),range(5),range(5)).extractScalar()
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
# }}}
# }}}

# Exports {{{
__all__ = [
    "Expectation",
]
# }}}

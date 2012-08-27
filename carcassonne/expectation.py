# Imports {{{
# }}}

# Classes {{{
class Expectation: # {{{
    def __init__(self,corners,sides,operator_center_tensor): # {{{
        self.corners = list(corners)
        self.sides = list(sides)
        self.operator_center_tensor = operator_center_tensor
    # }}}
    def absorbCenter(self,direction,state_center_data,state_center_data_conj=None): # {{{
        self.corners[direction] = self.corners[direction].absorbFromLeft(direction,self.sides[(direction+1)%4])
        self.sides[direction] = self.sides[direction].absorbCenterSOS(direction,state_center_data,self.operator_center_tensor,state_center_data_conj)
        self.corners[(direction-1)%4] = self.corners[(direction-1)%4].absorbFromRight(direction-1,self.sides[(direction-1)%4])
    # }}}
    def computeNormalization(self,state_center_data,state_center_data_conj=None): # {{{
        if state_center_data_conj is None:
            state_center_data_conj = state_center_data.conj()
        return state_center_data_conj.contractWith(self.formMultiplier()(state_center_data),range(5),range(5)).extractScalar()
    # }}}
    def formMultiplier(self): # {{{
        return self.sides[0].formMultiplier(self.corners,self.sides,self.operator_center_tensor)
    # }}}
# }}}
# }}}

# Exports {{{
__all__ = [
    "Expectation",
]
# }}}

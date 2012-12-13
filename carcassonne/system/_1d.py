# Imports {{{
from ..tensors._1d import *
# }}}

# Classes {{{
class System(BaseSystem): # {{{
  # Instance methods {{{
    def __init__(self,left_environment,right_environment,center_state,center_operator): # {{{
        self.left_environment = left_environment
        self.right_environment = right_environment
        self.center_state = center_state
        self.center_operator = center_operator
    # }}}
    def computeExpectation(self): # {{{
        return self.computeScalarUsingMultiplier(self.formExpectationMultiplier())
    # }}}
    def computeScalarUsingMultiplier(self,multiply): # {{{
        return self.center_state.contractWith(multiply(self.center_state),range(3),range(3)).extractScalar()
    # }}}
    def contract(self,direction): # {{{
        if direction == 0:
            self.contractRight()
        elif direction == 1:
            self.contractLeft()
        else:
            raise ValueError("Direction must be 0 for right or 1 for left, not {}.".format(direction))
    # }}}
    def contractLeft(self): # {{{
        normalized_center_state, denormalized_center_state =\
            normalizeAndDenormalize(self.center_state,1,self.center_state,0)
        self.left_environment = \
            absorbCenterOSSIntoLeftEnvironment(
                self.left_environment,
                self.center_operator,
                normalized_center_state,
                normalized_center_state.conj(),
            )
        self.center_state = denormalized_center_state
    # }}}
    def contractRight(self): # {{{
        normalized_center_state, denormalized_center_state =\
            normalizeAndDenormalize(self.center_state,0,self.center_state,1)
        self.right_environment = \
            absorbCenterOSSIntoRightEnvironment(
                self.right_environment,
                self.center_operator,
                normalized_center_state,
                normalized_center_state.conj(),
            )
        self.center_state = denormalized_center_state
    # }}}
    def formExpectationMultiplier(self): # {{{
        return \
            formExpectationMultiplier(
                self.left_environment,
                self.right_environment,
                self.center_operator
            )
    # }}}
    def setCenterState(self,center_state,center_state_conj=None): # {{{
        self.center_state = center_state
        if center_state_conj is None:
            center_state_conj = conter_state.conj()
        self.center_state_conj = center_state_conj
    # }}}
  # }}}
# }}}
# }}}

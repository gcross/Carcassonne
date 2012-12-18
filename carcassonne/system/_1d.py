# Imports {{{
from numpy import array, complex128, conj, dot, identity, multiply, sqrt, tensordot, zeros
from scipy.linalg import svd

from .base import *
from ..tensors._1d import *
from ..utils import buildProductTensor, computeAndCheckNewDimension, computeLimitingLinearCoefficient, crand, relaxOver, normalize, normalizeAndDenormalize
# }}}

# Classes {{{
class System(BaseSystem): # {{{
  # Instance methods {{{
    def __init__(self,left_operator_boundary,right_operator_boundary,center_operator,center_state=None,left_state_boundary=None,right_state_boundary=None): # {{{
        self.left_operator_boundary = array(left_operator_boundary)
        self.right_operator_boundary = array(right_operator_boundary)
        if left_state_boundary is None:
            self.left_environment = buildProductTensor(left_operator_boundary,[1],[1])
        else:
            self.left_environment = buildProductTensor(left_operator_boundary,left_state_boundary,list(map(conj,left_state_boundary)))
        if right_state_boundary is None:
            self.right_environment = buildProductTensor(right_operator_boundary,[1],[1])
        else:
            self.right_environment = buildProductTensor(right_operator_boundary,right_state_boundary,list(map(conj,right_state_boundary)))
        self.center_operator = center_operator
        if center_state is None:
            self.setCenterState(ones((1,1,center_operator.shape[2])))
        else:
            self.setCenterState(center_state)
        assert self.left_operator_boundary.ndim == 1
        assert self.right_operator_boundary.ndim == 1
        assert self.left_environment.ndim == 3
        assert self.right_environment.ndim == 3
        assert self.center_operator.ndim == 4
        assert self.center_state.ndim == 3
    # }}}
    def computeExpectation(self): # {{{
        return self.computeScalarUsingMultiplier(self.formExpectationMultiplier())
    # }}}
    def computeOneSiteExpectation(self): # {{{
        assert self.left_environment.size >= 2
        assert self.left_operator_boundary.size >= 2
        right_O_environment_shape = self.right_environment.shape
        right_N_environment_shape = (self.center_state.shape[1],)*2
        right_N_environment_size = prod(right_N_environment_shape)
        normalized_center_state = normalize(self.center_state,1)
        normalized_center_state_conj = normalized_center_state.conj()
        return \
            computeLimitingLinearCoefficient(
                prod(right_O_environment_shape),
                lambda v: \
                    absorbCenterOSSIntoRightEnvironment(
                        v.reshape(right_O_environment_shape),
                        self.center_operator,
                        normalized_center_state,
                        normalized_center_state_conj
                    ).ravel(),
                lambda v: \
                    absorbCenterSSIntoRightEnvironment(
                        v.reshape(right_N_environment_shape),
                        normalized_center_state,
                        normalized_center_state_conj
                    ).ravel(),
                lambda vs: \
                    tensordot(
                        self.left_operator_boundary,
                        vs.reshape((vs.shape[0],) + right_O_environment_shape),
                        (0,1)
                    ).reshape(vs.shape[0],right_N_environment_size),
                lambda vs: \
                    tensordot(
                        self.right_operator_boundary,
                        vs.reshape((vs.shape[0],) + right_O_environment_shape),
                        (0,1)
                    ).reshape(vs.shape[0],right_N_environment_size)
            )
    # }}}
    def computeScalarUsingMultiplier(self,multiply): # {{{
        return self.center_state.contractWith(multiply(self.center_state),range(3),range(3)).extractScalar()
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
    def contractTowards(self,direction): # {{{
        if direction == 0:
            self.contractRight()
        elif direction == 1:
            self.contractLeft()
        else:
            raise ValueError("Direction must be 0 for right or 1 for left, not {}.".format(direction))
    # }}}
    def formExpectationMultiplier(self): # {{{
        return \
            formExpectationMultiplier(
                self.left_environment,
                self.right_environment,
                self.center_operator
            )
    # }}}
    def increaseBandwidth(self,direction=0,by=None,to=None,do_as_much_as_possible=False): # {{{
        if direction != 0:
            raise ValueError("Direction for bandwidth increase must be 0, not {}.".format(direction))
        old_dimension, new_dimension, increment = \
            computeAndCheckNewDimension(
                self.center_state,
                direction,
                by=by,
                to=to,
                do_as_much_as_possible=do_as_much_as_possible
            )
        X = svd(crand(new_dimension,old_dimension),full_matrices=False)[0]
        XC = X.conj()
        XT = X.transpose()
        XH = XT.conj()
        self.setCenterState(tensordot(tensordot(X,self.center_state,(1,0)),XH,(1,0)).transpose(0,2,1))
        self.left_environment = tensordot(tensordot(self.left_environment,XH,(1,0)).transpose(0,2,1),XT,(2,0))
        self.right_environment = tensordot(XC,tensordot(X,self.right_environment,(1,1)).transpose(1,0,2),(1,2)).transpose(1,2,0)
    # }}}
    def minimizeExpectation(self): # {{{
        self.setCenterState(relaxOver(
            initial=NDArrayData(self.center_state),
            expectation_multiplier=self.formExpectationMultiplier(),
            maximum_number_of_multiplications=100
        ).toArray())
    # }}}
    def setCenterState(self,center_state,center_state_conj=None): # {{{
        self.center_state = center_state
        if center_state_conj is None:
            center_state_conj = center_state.conj()
        self.center_state_conj = center_state_conj
    # }}}
  # }}}
# }}}
# }}}

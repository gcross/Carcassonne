# Imports {{{
from numpy import array, complex128, conj, dot, identity, multiply, sqrt, tensordot, zeros
from scipy.linalg import svd

from .base import *
from ..tensors._1d import *
from ..utils import buildProductTensor, computeAndCheckNewDimension, computeLimitingLinearCoefficient, crand, relaxOver, normalize, normalizeAndDenormalize
# }}}

# Classes {{{
class System(BaseSystem): # {{{
  # Class methods {{{
    @classmethod # newRandom {{{
    def newRandom(cls,operator_dimension,state_dimension,physical_dimension):
        return cls(
            crand(operator_dimension),
            crand(operator_dimension),
            (lambda x: x+x.transpose(0,1,3,2).conj())(crand(operator_dimension,operator_dimension,physical_dimension,physical_dimension)),
            crand(state_dimension,state_dimension,physical_dimension),
            crand(state_dimension),
            crand(state_dimension),
        )
    # }}}
  # }}}
  # Instance methods {{{
    def __init__(self,left_operator_boundary,right_operator_boundary,center_operator,center_state=None,left_state_boundary=None,right_state_boundary=None): # {{{
        self.left_operator_boundary = NDArrayData(array(left_operator_boundary))
        self.right_operator_boundary = NDArrayData(array(right_operator_boundary))
        if left_state_boundary is None:
            self.left_environment = NDArrayData(buildProductTensor(left_operator_boundary,[1],[1]))
        else:
            self.left_environment = NDArrayData(buildProductTensor(left_operator_boundary,left_state_boundary,list(map(conj,left_state_boundary))))
        if right_state_boundary is None:
            self.right_environment = NDArrayData(buildProductTensor(right_operator_boundary,[1],[1]))
        else:
            self.right_environment = NDArrayData(buildProductTensor(right_operator_boundary,right_state_boundary,list(map(conj,right_state_boundary))))
        self.center_operator = NDArrayData(center_operator)
        if center_state is None:
            self.setCenterState(NDArrayData.newTrivial(1,1,center_operator.shape[2]))
        else:
            self.setCenterState(NDArrayData(center_state))
        assert self.left_operator_boundary.ndim == 1
        assert self.right_operator_boundary.ndim == 1
        assert self.left_environment.ndim == 3
        assert self.right_environment.ndim == 3
        assert self.center_operator.ndim == 4
        assert self.center_state.ndim == 3
    # }}}
    def computeExpectation(self): # {{{
        return self.computeScalarUsingMultiplier(self.formExpectationMultiplier()).real
    # }}}
    def computeOneSiteExpectation(self): # {{{
        assert self.left_environment.size() >= 2
        assert self.left_operator_boundary.size() >= 2
        right_O_environment_shape = self.right_environment.shape
        right_N_environment_shape = (self.center_state.shape[1],)*2
        right_N_environment_size = prod(right_N_environment_shape)
        normalized_center_state = self.center_state.normalizeAxis(1)[0]
        normalized_center_state_conj = normalized_center_state.conj()
        return \
            computeLimitingLinearCoefficient(
                prod(right_O_environment_shape),
                lambda v: \
                    absorbCenterOSSIntoRightEnvironment(
                        NDArrayData(v.reshape(right_O_environment_shape)),
                        self.center_operator,
                        normalized_center_state,
                        normalized_center_state_conj
                    ).ravel().toArray(),
                lambda v: \
                    absorbCenterSSIntoRightEnvironment(
                        NDArrayData(v.reshape(right_N_environment_shape)),
                        normalized_center_state,
                        normalized_center_state_conj
                    ).ravel().toArray(),
                lambda vs: \
                    (self.left_operator_boundary
                    ).contractWith(
                        NDArrayData(vs.reshape((vs.shape[0],) + right_O_environment_shape)), (0,), (1,)
                    ).join(0,(1,2)).toArray(),
                lambda vs: \
                    (self.right_operator_boundary
                    ).contractWith(
                        NDArrayData(vs.reshape((vs.shape[0],) + right_O_environment_shape)), (0,), (1,)
                    ).join(0,(1,2)).toArray(),
            )
    # }}}
    def computeScalarUsingMultiplier(self,multiply): # {{{
        return self.center_state_conj.ravel().contractWith(multiply(self.center_state).ravel(),(0,),(0,)).extractScalar()
    # }}}
    def contractLeft(self): # {{{
        normalized_center_state, denormalized_center_state = \
            self.center_state.normalizeAxisAndDenormalize(1,0)
        self.left_environment = \
            absorbCenterOSSIntoLeftEnvironment(
                self.left_environment,
                self.center_operator,
                normalized_center_state,
                normalized_center_state.conj(),
            )
        self.setCenterState(denormalized_center_state)
    # }}}
    def contractRight(self): # {{{
        normalized_center_state, denormalized_center_state = \
            self.center_state.normalizeAxisAndDenormalize(0,1)
        self.right_environment = \
            absorbCenterOSSIntoRightEnvironment(
                self.right_environment,
                self.center_operator,
                normalized_center_state,
                normalized_center_state.conj(),
            )
        self.setCenterState(denormalized_center_state)
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
    def getCenterStateAsArray(self): # {{{
        return self.center_state.toArray()
    # }}}
    def increaseBandwidth(self,direction=0,by=None,to=None,do_as_much_as_possible=False): # {{{
        if direction != 0:
            raise ValueError("Direction for bandwidth increase must be 0, not {}.".format(direction))
        old_dimension, new_dimension, increment = \
            computeAndCheckNewDimension(
                self.center_state.shape,
                direction,
                by=by,
                to=to,
                do_as_much_as_possible=do_as_much_as_possible
            )
        X = NDArrayData(svd(crand(new_dimension,old_dimension),full_matrices=False)[0])
        XC = X.conj()
        self.setCenterState(self.center_state.absorbMatrixAt(0,X).absorbMatrixAt(1,XC))
        self.left_environment = self.left_environment.absorbMatrixAt(1,XC).absorbMatrixAt(2,X)
        self.right_environment = self.right_environment.absorbMatrixAt(1,X).absorbMatrixAt(2,XC)
    # }}}
    def minimizeExpectation(self): # {{{
        self.setCenterState(relaxOver(
            initial=self.center_state,
            expectation_multiplier=self.formExpectationMultiplier(),
            maximum_number_of_multiplications=100
        ))
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

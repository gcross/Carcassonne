# Imports {{{
from numpy import array, complex128, conj, dot, identity, multiply, sqrt, tensordot, zeros
from scipy.linalg import svd

from .base import *
from ..tensors._1d import *
from ..utils import buildProductTensor, computeCompressorForMatrixTimesItsDagger, computeLimitingLinearCoefficient, computeNewDimension, crand, dropAt, relaxOver, normalize, normalizeAndDenormalize
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
    def __init__(self,right_operator_boundary,left_operator_boundary,operator_center_data,state_center_data=None,right_state_boundary=None,left_state_boundary=None): # {{{
        BaseSystem.__init__(self)
        self.right_operator_boundary = NDArrayData(array(right_operator_boundary))
        self.left_operator_boundary = NDArrayData(array(left_operator_boundary))
        if right_state_boundary is None:
            self.right_environment = NDArrayData(buildProductTensor(right_operator_boundary,[1],[1]))
        else:
            self.right_environment = NDArrayData(buildProductTensor(right_operator_boundary,right_state_boundary,list(map(conj,right_state_boundary))))
        if left_state_boundary is None:
            self.left_environment = NDArrayData(buildProductTensor(left_operator_boundary,[1],[1]))
        else:
            self.left_environment = NDArrayData(buildProductTensor(left_operator_boundary,left_state_boundary,list(map(conj,left_state_boundary))))
        self.operator_center_data = NDArrayData(operator_center_data)
        if state_center_data is None:
            self.setStateCenter(NDArrayData.newTrivial(1,1,operator_center_data.shape[2]))
        else:
            self.setStateCenter(NDArrayData(state_center_data))
        assert self.left_operator_boundary.ndim == 1
        assert self.right_operator_boundary.ndim == 1
        assert self.left_environment.ndim == 3
        assert self.right_environment.ndim == 3
        assert self.operator_center_data.ndim == 4
        assert self.state_center_data.ndim == 3
    # }}}
    def computeExpectation(self): # {{{
        return self.computeScalarUsingMultiplier(self.formExpectationMultiplier()).real
    # }}}
    def computeOneSiteExpectation(self): # {{{
        assert self.left_environment.size() >= 2
        assert self.left_operator_boundary.size() >= 2
        right_O_environment_shape = self.right_environment.shape
        right_N_environment_shape = (self.state_center_data.shape[0],)*2
        right_N_environment_size = prod(right_N_environment_shape)
        normalized_state_center_data = self.state_center_data.normalizeAxis(1)[0]
        normalized_state_center_data_conj = normalized_state_center_data.conj()
        return \
            computeLimitingLinearCoefficient(
                prod(right_O_environment_shape),
                lambda v: \
                    absorbCenterOSSIntoRightEnvironment(
                        NDArrayData(v.reshape(right_O_environment_shape)),
                        self.operator_center_data,
                        normalized_state_center_data,
                        normalized_state_center_data_conj
                    ).ravel().toArray(),
                lambda v: \
                    absorbCenterSSIntoRightEnvironment(
                        NDArrayData(v.reshape(right_N_environment_shape)),
                        normalized_state_center_data,
                        normalized_state_center_data_conj
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
        return self.state_center_data_conj.ravel().contractWith(multiply(self.state_center_data).ravel(),(0,),(0,)).extractScalar()
    # }}}
    def contractLeftNormalized(self,state_center_data): # {{{
        if state_center_data.shape[1] != self.left_environment.shape[1]:
            raise ValueError("state dimension of the left environment ({}) does not match the left dimension of the center state ({})".format(self.left_environment.shape[1],state_center_data.shape[1]))
        self.contractLeftUnnormalized(state_center_data.normalizeAxis(0)[0])
    # }}}
    def contractLeftUnnormalized(self,state_center_data): # {{{
        self.left_environment = \
            absorbCenterOSSIntoLeftEnvironment(
                self.left_environment,
                self.operator_center_data,
                state_center_data,
                state_center_data.conj(),
            )
    # }}}
    def contractRightNormalized(self,state_center_data): # {{{
        if state_center_data.shape[0] != self.right_environment.shape[1]:
            raise ValueError("state dimension of the right environment ({}) does not match the right dimension of the center state ({})".format(self.right_environment.shape[1],state_center_data.shape[0]))
        self.contractRightUnnormalized(state_center_data.normalizeAxis(1)[0])
    # }}}
    def contractRightUnnormalized(self,state_center_data=None): # {{{
        if state_center_data is None:
            state_center_data = self.state_center_data
        self.right_environment = \
            absorbCenterOSSIntoRightEnvironment(
                self.right_environment,
                self.operator_center_data,
                state_center_data,
                state_center_data.conj(),
            )
    # }}}
    def contractTowards(self,direction): # {{{
        tensor_to_contract, _, matrix_to_absorb = self.state_center_data.normalizeAxis(1-direction)
        self.contractUnnormalizedTowards(direction,tensor_to_contract)
        self.setStateCenter(self.state_center_data.normalizeAxis(direction)[0].absorbMatrixAt(direction,matrix_to_absorb.transpose()))
    # }}}
    def contractNormalizedTowards(self,direction,state_center_data): # {{{
        if direction == 0:
            self.contractRightNormalized(state_center_data)
        elif direction == 1:
            self.contractLeftNormalized(state_center_data)
        else:
            raise ValueError("Direction must be 0 for right or 1 for left, not {}.".format(direction))
    # }}}
    def contractUnnormalizedTowards(self,direction,state_center_data): # {{{
        if direction == 0:
            self.contractRightUnnormalized(state_center_data)
        elif direction == 1:
            self.contractLeftUnnormalized(state_center_data)
        else:
            raise ValueError("Direction must be 0 for right or 1 for left, not {}.".format(direction))
    # }}}
    def formExpectationMatrix(self): # {{{
        return self.formExpectationMultiplier().formMatrix()
    # }}}
    def formExpectationMultiplier(self): # {{{
        return \
            formExpectationMultiplier(
                self.right_environment,
                self.left_environment,
                self.operator_center_data
            )
    # }}}
    def increaseBandwidth(self,direction=0,by=None,to=None,do_as_much_as_possible=False): # {{{
        if direction != 0:
            raise ValueError("Direction for bandwidth increase must be 0, not {}.".format(direction))
        return self._increaseBandwidth((0,1),by,to,do_as_much_as_possible)
    # }}}
    def minimizeExpectation(self): # {{{
        self.setStateCenter(relaxOver(
            initial=self.state_center_data,
            expectation_multiplier=self.formExpectationMultiplier(),
            maximum_number_of_multiplications=100
        ))
    # }}}
    def setStateCenter(self,state_center_data,state_center_data_conj=None): # {{{
        self.state_center_data = state_center_data
        if state_center_data_conj is None:
            state_center_data_conj = state_center_data.conj()
        self.state_center_data_conj = state_center_data_conj
    # }}}
  # }}}
# }}}
# }}}

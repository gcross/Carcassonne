# Imports {{{
from .base import *
from ..tensors._1d import *
from ..utils import relaxOver
# }}}

# Classes {{{
class System(BaseSystem): # {{{
  # Instance methods {{{
    def __init__(self,left_operator_boundary,right_operator_boundary,center_operator): # {{{
        self.left_operator_boundary = array(left_operator_boundary)
        self.left_environment = buildProductTensor(left_operator_boundary,[1],[1])
        self.right_environment = buildProductTensor(right_operator_boundary,[1],[1])
        self.center_state = ones((1,1,center_operator.shape[2]))
        self.center_operator = center_operator
    # }}}
    def computeExpectation(self): # {{{
        return self.computeScalarUsingMultiplier(self.formExpectationMultiplier())
    # }}}
    def computeOneSiteExpectation(self): # {{{
        def multiplyRightBoundary(v):
            return \
                absorbCenterOSSIntoRightEnvironment(
                    v.reshape(self.right_environment.shape),
                    self.center_operator,
                    self.center_state,
                    self.center_state_conj,
                ).ravel()
        ovecs = eigs(LinearOperator(N=prod(self.right_environment.shape),matmul=multiplyRightBoundary),k=2,which='LM')[1].transpose()

        Omatrix = zeros((2,2),dtype=complex128)
        for i in range(2):
            for j in range(2):
                Omatrix[i,j] = dot(ovecs[i].conj(),multiplyRightBoundary(ovecs[j]))
        numerator = sqrt(trace(dot(Omatrix.transpose().conj(),Omatrix))-2)

        lnvecs = tensordot(self.left_operator_boundary,ovecs.reshape((2,) + self.left_operator_boundary.shape,(0,1))
        rnvecs = tensordot(self.right_operator_boundary,ovecs.reshape((2,) + self.right_operator_boundary.shape,(0,1))
        Nmatrix = zeros((2,2),dtype=complex128)
        for i in range(2):
            for j in range(2):
                Nmatrix[i,j] = dot(lnvecs[i].conj().ravel(),absorbCenterSSIntoRightEnvironment(rnvecs[j],self.center_state,self.center_state_conj).ravel())
        denominator = sqrt(trace(dot(Nmatrix.transpose().conj(),Nmatrix)))
        return numerator/denominator
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

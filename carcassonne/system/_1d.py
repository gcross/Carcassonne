# Imports {{{
from numpy import argmax, array, complex128, conj, dot, identity, multiply, sqrt, tensordot, trace, zeros
from scipy.linalg import eigvals, lstsq
from scipy.sparse.linalg import LinearOperator, eigs

from .base import *
from ..tensors._1d import *
from ..utils import buildProductTensor,relaxOver
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
        assert self.left_environment.size >= 2
        assert self.left_operator_boundary.size >= 2

        n = self.left_environment.size
        if n <= 3:
            matrix = []
            for i in range(n):
                matrix.append(multiplyRightBoundary(array([0]*i+[1]+[0]*(n-1-i))))
            matrix = array(matrix)
            evals = eigvals(matrix)
            lam = evals[argmax(abs(evals))]
            matrix -= lam*identity(n)
            soln1, _, _, _ = lstsq(matrix,zeros((n,)))
            matrix += multiply.outer(soln1,soln1.conj().transpose())
            soln2, _, _, _ = lstsq(matrix,zeros((n,)))
            ovecs = array([soln1,soln2])
            print("ovecs=",ovecs)
        else:
            ovecs = eigs(LinearOperator((self.right_environment.size,)*2,matvec=multiplyRightBoundary),k=2,which='LM',ncv=9)[1].transpose()

        Omatrix = zeros((2,2),dtype=complex128)
        for i in range(2):
            for j in range(2):
                Omatrix[i,j] = dot(ovecs[i].conj(),multiplyRightBoundary(ovecs[j]))
        numerator = sqrt(trace(dot(Omatrix.transpose().conj(),Omatrix))-2)

        lnvecs = tensordot(self.left_operator_boundary,ovecs.reshape((2,) + self.left_environment.shape),(0,1))
        rnvecs = tensordot(self.right_operator_boundary,ovecs.reshape((2,) + self.right_environment.shape),(0,1))
        Nmatrix = zeros((2,2),dtype=complex128)
        for i in range(2):
            for j in range(2):
                Nmatrix[i,j] = dot(lnvecs[i].conj().ravel(),absorbCenterSSIntoRightEnvironment(rnvecs[j],self.center_state,self.center_state_conj).ravel())
        denominator = sqrt(trace(dot(Nmatrix.transpose().conj(),Nmatrix)))
        print(repr(Omatrix))
        print(repr(Nmatrix))
        print(numerator,denominator)
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

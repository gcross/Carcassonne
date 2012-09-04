# Imports {{{
from numpy import prod, zeros
from numpy.linalg import eigh

from .sparse import Identity, Operator
from .tensors.dense import formNormalizationMultiplier, formNormalizationSubmatrix
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
        multiplyExpectation, multiplyNormalization = self.formExpectationAndNormalizationMultipliers()
        unnormalized_expectation = self.computeScalarUsingMultiplier(multiplyExpectation)
        normalization = self.computeScalarUsingMultiplier(multiplyNormalization)
        return unnormalized_expectation/normalization
    # }}}
    def computeNormalization(self): # {{{
        return self.computeScalarUsingMultiplier(self.formNormalizationMultiplier())
    # }}}
    def computeScalarUsingMultiplier(self,multiply): # {{{
        return self.state_center_data_conj.contractWith(multiply(self.state_center_data),range(5),range(5)).extractScalar()
    # }}}
    def formExpectationAndNormalizationMultipliers(self): # {{{
        return formExpectationAndNormalizationMultipliers(self.corners,self.sides,self.operator_center_tensor)
    # }}}
    def formExpectationMultiplier(self): # {{{
        return self.formExpectationAndNormalizationMultipliers()[0]
    # }}}
    def formNormalizationMultiplier(self): # {{{
        return formNormalizationMultiplier(tuple(corner[Identity()] for corner in self.corners),tuple(side[Identity()] for side in self.sides))
    # }}}
    def formNormalizationSubmatrix(self): # {{{
        return formNormalizationSubmatrix(tuple(corner[Identity()] for corner in self.corners),tuple(side[Identity()] for side in self.sides))
    # }}}
    def increaseBandwidth(self,direction,by=None,to=None): # {{{
        state_center_data = self.state_center_data
        direction %= 2
        old_dimension = state_center_data.shape[direction]
        if by is None and to is None:
            raise ValueError("Either 'by' or 'to' must not be None.")
        elif by is not None and to is not None:
            raise ValueError("Both 'by' ({}) and 'to' ({}) cannot be None.".format(by,to))
        elif by is not None:
            new_dimension = old_dimension + by
        elif to is not None:
            new_dimension = to
        assert new_dimension >= old_dimension
        matrix, matrix_conj = state_center_data.newEnlargener(old_dimension,new_dimension)
        self.state_center_data = state_center_data.absorbMatrixAt(direction,matrix).absorbMatrixAt(direction+2,matrix)
        self.state_center_data_conj = self.state_center_data.conj()
        for i in (direction,direction+2):
            self.sides[i] = {tag: data.absorbMatrixAt(2,matrix_conj).absorbMatrixAt(3,matrix) for tag, data in self.sides[i].items()}
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

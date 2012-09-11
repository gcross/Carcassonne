# Imports {{{
from numpy import complex128, prod, zeros
from numpy.linalg import eigh
from random import randint

from .data import NDArrayData
from .sparse import Identity, Operator, directSumListsOfSparse, directSumSparse, mapOverSparseData
from .tensors.dense import formNormalizationMultiplier, formNormalizationSubmatrix
from .tensors.sparse import absorbSparseSideIntoCornerFromLeft, absorbSparseSideIntoCornerFromRight, absorbSparseCenterSOSIntoSide, formExpectationAndNormalizationMultipliers
from .utils import computeNewDimension, L, R
# }}}

# Classes {{{
class System: # {{{
  # Class methods {{{
    @classmethod # newEnlargener {{{
    def newEnlargener(cls,O,bandwidth_dimensions):
        system = cls(
            tuple({Identity():NDArrayData.newTrivial((1,)*4)} for _ in range(4)),
            tuple({Identity():NDArrayData.newTrivial((1,)*4)+(d,)*2} for d in bandwidth_dimensions),
            NDArrayData.newRandom(*tuple(bandwidth_dimensions)+tuple(O.shape[:1])),
            {Identity():None,Operator():O}
        )
        system.assertDimensionsAreConsistent()
        system.assertNormalizationIsHermitian()
        for direction in [0,2,1,3]:
            system.absorbCenter(direction)
        return system
    # }}}
    @classmethod # newRandom {{{
    def newRandom(cls,makeOperator=None,DataClass=NDArrayData,maximum_dimension=2,O=None):
        assert not (makeOperator is not None and O is not None)
        randomDimension = lambda: randint(1,maximum_dimension)
        randomDimensions = lambda n: tuple(randomDimension() for _ in range(n))
        spoke_sizes = randomDimensions(2)*2
        sides_data = tuple(DataClass.newRandom(*randomDimensions(1)*4+(spoke_sizes[i],)*2) for i in range(4))
        for side_data in sides_data:
            side_data += side_data.join(1,0,3,2,5,4).conj()
        corners_data = tuple(DataClass.newRandom(*(sides_data[(i+1)%4].shape[2],)*2+(sides_data[i].shape[0],)*2) for i in range(4))
        for corner_data in corners_data:
            corner_data += corner_data.join(1,0,3,2).conj()
        if O:
            physical_dimension = O.shape[0]
        else:
            physical_dimension = randomDimension()
        state_center_data = DataClass.newRandom(*spoke_sizes + (physical_dimension,))
        if O is None:
            if makeOperator is None:
                O = DataClass.newRandom(physical_dimension,physical_dimension)
                O += O.join(1,0).conj()
            else:
                O = makeOperator(physical_dimension)
        operator_center_tensor = {Identity():None,Operator():O}
        system = cls(
            tuple({Identity():corner_data} for corner_data in corners_data),
            tuple({Identity():side_data} for side_data in sides_data),
            state_center_data,
            operator_center_tensor,
        )
        system.assertDimensionsAreConsistent()
        system.assertNormalizationIsHermitian()
        return system
    # }}}
    @classmethod # newTrivial {{{
    def newTrivial(cls,O,DataClass=NDArrayData):
        return cls(
            tuple({Identity():DataClass.newTrivial((1,)*4,dtype=complex128)} for _ in range(4)),
            tuple({Identity():DataClass.newTrivial((1,)*6,dtype=complex128)} for _ in range(4)),
            DataClass.newTrivial((1,1,1,1,O.shape[0]),dtype=complex128),
            {Identity():None,Operator():O}
        )
    # }}}
  # }}}
  # Instance methods {{{
    def __init__(self,corners,sides,state_center_data,operator_center_tensor,state_center_data_conj=None): # {{{
        self.corners = list(corners)
        self.sides = list(sides)
        self.operator_center_tensor = operator_center_tensor
        self.state_center_data = state_center_data
        if state_center_data_conj is None:
            self.state_center_data_conj = self.state_center_data.conj()
    # }}}
    def __add__(self,other): # {{{
        return System(
            directSumListsOfSparse(self.corners,other.corners),
            directSumListsOfSparse(self.sides,other.sides),
            self.state_center_data.directSumWith(other.state_center_data,4),
            self.operator_center_tensor
        )
    # }}}
    def absorbCenter(self,direction): # {{{
        self.corners[direction] = absorbSparseSideIntoCornerFromLeft(self.corners[direction],self.sides[L(direction)])
        self.sides[direction] = absorbSparseCenterSOSIntoSide(direction,self.sides[direction],self.state_center_data,self.operator_center_tensor,self.state_center_data_conj)
        self.corners[R(direction)] = absorbSparseSideIntoCornerFromRight(self.corners[R(direction)],self.sides[R(direction)])
    # }}}
    def assertDimensionsAreConsistent(self): # {{{
        assert self.state_center_data.shape == self.state_center_data_conj.shape
        if self.state_center_data.shape[0] != self.state_center_data.shape[2]:
            raise AssertionError("state center's left and right dimensions do not agree ({} != {})".format(self.state_center_data.shape[2] != self.state_center_data.shape[0]))
        if self.state_center_data.shape[1] != self.state_center_data.shape[3]:
            raise AssertionError("state center's up and down dimensions do not agree ({} != {})".format(self.state_center_data.shape[1] != self.state_center_data.shape[3]))
        for i, side in enumerate(self.sides):
            normalization_data = side[Identity()]
            for tag, data in side.items():
                if data.shape != normalization_data.shape:
                    raise AssertionError("for side {} the data tagged with {} does not match the shape of the data tagged with Identity() ({} != {})".format(i,tag,data.shape,normalization_data.shape))
            side_shape = normalization_data.shape
            for d in (0,2,4):
                if side_shape[d] != side_shape[d+1]:
                    raise AssertionError("side {}'s dimension {} does not match its dimension {} ({} != {})".format(i,d,d+1,side_shape[d],side_shape[d+1]))
            if side_shape[0] != side_shape[2]:
                raise AssertionError("side {}'s left and right dimensions do not agree ({} != {})".format(side_shape[0],side_shape[2]))
            if side_shape[4] != self.state_center_data.shape[i]:
                raise AssertionError("side {}'s center-facing dimensions do not match the corresponding state dimension ({} != {})".format(i,side_shape[4],self.state_center_data.shape[i]))
        for i, corner in enumerate(self.corners):
            normalization_data = corner[Identity()]
            for tag, data in corner.items():
                if data.shape != normalization_data.shape:
                    raise AssertionError("for corner {} the data tagged with {} does not match the shape of the data tagged with Identity() ({} != {})".format(i,tag,data.shape,normalization_data.shape))
            corner_shape = normalization_data.shape
            for d in (0,2):
                if corner_shape[d] != corner_shape[d+1]:
                    raise AssertionError("corner {}'s dimension {} does not match its dimension {} ({} != {})".format(i,d,d+1,corner_shape[d],corner_shape[d+1]))
            if corner_shape[2] != self.sides[i][Identity()].shape[0]:
                raise AssertionError("corner {}'s right dimensions do not match side {}'s left dimensions ({} != {})".format(i,i,corner_shape[2],self.sides[i][Identity()].shape[0]))
            if corner_shape[0] != self.sides[L(i)][Identity()].shape[2]:
                raise AssertionError("corner {}'s left dimensions do not match side {}'s right dimensions ({} != {})".format(i,L(i),corner_shape[0],self.sides[L(i)][Identity()].shape[2]))
    # }}}
    def assertNormalizationIsHermitian(self): # {{{
        for i in range(4):
            if not self.sides[i][Identity()].allcloseTo(self.sides[i][Identity()].join(1,0,3,2,5,4).conj()):
                raise AssertionError("side {} is not hermitian".format(i))
            if not self.corners[i][Identity()].allcloseTo(self.corners[i][Identity()].join(1,0,3,2).conj()):
                raise AssertionError("corner {} is not hermitian".format(i))
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
    def computeUnnormalizedExpectation(self): # {{{
        return self.computeScalarUsingMultiplier(self.formExpectationMultipliers())
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
    def increaseBandwidth(self,increments): # {{{
        state_center_data = self.state_center_data
        new_sides = []
        new_bandwidths = []
        for direction, increment in enumerate(increments):
            if increment < 1:
                raise ValueError("All increments must be at least 1;  observed increment of {} for direction {}.".format(increment,direction))
            new_bandwidths.append(state_center_data.shape[direction]+increment)
            indices_to_merge = list(range(5))
            del indices_to_merge[direction]
            U, S, V = state_center_data.join(indices_to_merge,direction).svd(full_matrices=False)
            shrinker = V[:increment,:]
            shrinker_conj = shrinker.conj()
            new_sides.append(mapOverSparseData(lambda data: data.absorbMatrixAt(4,shrinker).absorbMatrixAt(5,shrinker_conj),self.sides[direction]))
        self.corners = directSumListsOfSparse(self.corners,self.corners)
        self.sides = directSumListsOfSparse(self.sides,new_sides)
        self.setStateCenter(self.state_center_data.increaseDimensionsAndFillWithZeros(*enumerate(new_bandwidths)))
    # }}}
    def increaseBandwidthAndThenNormalize(self,direction,by=None,to=None): # {{{
        self.increaseBandwidth(direction,by,to)
        self.normalize()
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
    def normalize(self): # {{{
        for corner_id in range(4):
            for direction in range(2):
                self.normalizeCornerAndDenormalizeSide(corner_id,direction)
        for side_id in range(4):
            self.normalizeSideAndDenormalizeCenter(side_id)
    # }}}
    def normalizeCenterAndDenormalizeSide(self,direction): # {{{
        normalizer_for_center, denormalizer_for_side_axis1 = self.state_center_data.normalizeAxis(direction,True)
        self.state_center_data = self.state_center_data.absorbMatrixAt(direction,normalizer_for_center)
        self.state_center_data_conj = self.state_center_data.conj()
        denormalizer_for_side_axis2 = denormalizer_for_side_axis1.conj()
        self.sides[direction] = {
            tag: side_data.absorbMatrixAt(4,denormalizer_for_side_axis1).absorbMatrixAt(5,denormalizer_for_side_axis2)
            for tag, side_data in self.sides[direction].items()
        }
    # }}}
    def normalizeCornerAndDenormalizeSide(self,corner_id,direction): # {{{
        side_id = (corner_id+1-direction)%4
        corner_axis1 = direction*2+0
        corner_axis2 = direction*2+1
        side_axis1 = (1-direction)*2+0
        side_axis2 = (1-direction)*2+1
        corner_data = self.corners[corner_id][Identity()]
        normalizer_for_corner_axis1, denormalizer_for_side_axis1 = corner_data.normalizeAxis(corner_axis1,True)
        normalizer_for_corner_axis2 = normalizer_for_corner_axis1.conj()
        denormalizer_for_side_axis2 = denormalizer_for_side_axis1.conj()
        self.corners[corner_id] = {
            tag: corner_data.absorbMatrixAt(corner_axis1,normalizer_for_corner_axis1).absorbMatrixAt(corner_axis2,normalizer_for_corner_axis2)
            for tag, corner_data in self.corners[corner_id].items()
        }
        self.sides[side_id] = {
            tag: side_data.absorbMatrixAt(side_axis1,denormalizer_for_side_axis1).absorbMatrixAt(side_axis2,denormalizer_for_side_axis2)
            for tag, side_data in self.sides[side_id].items()
        }
    # }}}
    def normalizeSideAndDenormalizeCenter(self,side_id): # {{{
        side_data = self.sides[side_id][Identity()]
        normalizer_for_side_axis1, denormalizer_for_center = side_data.normalizeAxis(4,True)
        normalizer_for_side_axis2 = normalizer_for_side_axis1.conj()
        self.sides[side_id] = {
            tag: side_data.absorbMatrixAt(4,normalizer_for_side_axis1).absorbMatrixAt(5,normalizer_for_side_axis2)
            for tag, side_data in self.sides[side_id].items()
        }
        self.state_center_data = self.state_center_data.absorbMatrixAt(side_id,denormalizer_for_center)
        self.state_center_data_conj = self.state_center_data.conj()
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
# }}}

# Exports {{{
__all__ = [
    "System",
]
# }}}

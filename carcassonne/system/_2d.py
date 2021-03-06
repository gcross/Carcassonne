# Imports {{{
from copy import copy
from numpy import array, complex128, dot, prod, sqrt, zeros
from numpy.linalg import cond
from scipy.linalg import eigh
from random import randint
from scipy.sparse.linalg import LinearOperator, eigs, eigsh

from .base import BaseSystem
from ..compression import computeProductCompressor
from ..data import NDArrayData
from ..sparse import Identity, OneSiteOperator, TwoSiteOperator, TwoSiteOperatorCompressed, directSumListsOfSparse, directSumSparse, makeSimpleSparseOperator, makeSparseOperator, mapOverSparseData, stripAllButIdentityFrom
from ..tensors._2d.dense import formNormalizationMultiplier, formNormalizationSubmatrix
from ..tensors._2d.sparse import absorbSparseSideIntoCornerFromLeft, absorbSparseSideIntoCornerFromRight, absorbSparseCenterSOSIntoSide, formExpectationAndNormalizationMultipliers
from ..utils import InvariantViolatedError, Multiplier, computeCompressor, computeCompressorForMatrixTimesItsDagger, computeNewDimension, dropAt, relaxOver, L, O, R
# }}}

# Classes {{{
class System(BaseSystem): # {{{
  # Class methods {{{
    @classmethod # newEnlargener {{{
    def newEnlargener(cls,O,bandwidth_dimensions):
        system = cls(
            tuple({Identity():NDArrayData.newTrivial((1,)*6)} for _ in range(4)),
            tuple({Identity():NDArrayData.newTrivial((1,)*6)+(d,)*2} for d in bandwidth_dimensions),
            NDArrayData.newRandom(*tuple(bandwidth_dimensions)+tuple(O.shape[:1])),
            {Identity():O.newIdentity(O.shape[0]),OneSiteOperator(None):O}
        )
        system.assertDimensionsAreConsistent()
        system.assertNormalizationIsHermitian()
        for direction in [0,2,1,3]:
            system.contractUnnormalizedTowards(direction)
        return system
    # }}}
    @classmethod # newRandom {{{
    def newRandom(cls,makeOperator=None,DataClass=NDArrayData,maximum_dimension=2,O=None):
        assert not (makeOperator is not None and O is not None)
        randomDimension = lambda: randint(1,maximum_dimension)
        randomDimensions = lambda n: tuple(randomDimension() for _ in range(n))
        spoke_sizes = randomDimensions(2)*2
        sides_dimensions = [randomDimension() for _ in range(4)]
        sides_data = tuple(DataClass.newRandom(*((sides_dimensions[i],)*2+(1,))*2+(spoke_sizes[i],)*2) for i in range(4))
        for side_data in sides_data:
            side_data += side_data.join(1,0,2,4,3,5,7,6).conj()
        corners_data = tuple(DataClass.newRandom(*(sides_data[L(i)].shape[3],)*2+(1,)+(sides_data[i].shape[0],)*2+(1,)) for i in range(4))
        for corner_data in corners_data:
            corner_data += corner_data.join(1,0,2,4,3,5).conj()
        if O is not None:
            physical_dimension = O.shape[0]
        else:
            physical_dimension = max(2,randomDimension())
        state_center_data = DataClass.newRandom(*spoke_sizes + (physical_dimension,))
        if O is None:
            if makeOperator is None:
                O = DataClass.newRandom(physical_dimension,physical_dimension)
                O += O.join(1,0).conj()
            else:
                O = makeOperator(physical_dimension)
        operator_center_tensor = {Identity():DataClass.newIdentity(physical_dimension),OneSiteOperator(None):O}
        system = cls(
            tuple({Identity():corner_data} for corner_data in corners_data),
            tuple({Identity():side_data} for side_data in sides_data),
            state_center_data,
            operator_center_tensor,
        )
        system.assertDimensionsAreConsistent()
        system.assertNormalizationIsHermitian()
        system.assertHasNoNaNs()
        return system
    # }}}
    @classmethod # newTrivial {{{
    def newTrivial(cls,operator_center_tensor,DataClass=NDArrayData):
        physical_dimension = None
        for data in operator_center_tensor.values():
            if data is not None:
                physical_dimension = data.shape[0]
                break
        if physical_dimension is None:
            raise ValueError("Operator tensor must have at least one non-identity component.")
        return cls(
            tuple({Identity():DataClass.newTrivial((1,)*6,dtype=complex128)} for _ in range(4)),
            tuple({Identity():DataClass.newTrivial((1,)*8,dtype=complex128)} for _ in range(4)),
            DataClass.newFilled((1,1,1,1,physical_dimension),1.0/sqrt(physical_dimension),dtype=complex128),
            operator_center_tensor,
        )
    # }}}
    @classmethod # newTrivialWithSimpleSparseOperator {{{
    def newTrivialWithSimpleSparseOperator(cls,O=None,OO_UD=None,OO_LR=None):
        return cls.newTrivial(makeSimpleSparseOperator(O=O,OO_UD=OO_UD,OO_LR=OO_LR))
    # }}}
    @classmethod # newTrivialWithSparseOperator {{{
    def newTrivialWithSparseOperator(cls,Os=[],OO_UDs=[],OO_LRs=[]):
        return cls.newTrivial(makeSparseOperator(Os=Os,OO_UDs=OO_UDs,OO_LRs=OO_LRs))
    # }}}
  # }}}
  # Instance methods {{{
    def __init__(self,corners,sides,state_center_data,operator_center_tensor,state_center_data_conj=None): # {{{
        BaseSystem.__init__(self)
        self.corners = list(corners)
        self.sides = list(sides)
        self.operator_center_tensor = operator_center_tensor
        self.state_center_data = state_center_data
        if state_center_data_conj is None:
            self.state_center_data_conj = self.state_center_data.conj()
        else:
            self.state_center_data_conj = state_center_data_conj
        self.just_increased_bandwidth = False
    # }}}
    def __add__(self,other): # {{{
        return System(
            directSumListsOfSparse(self.corners,other.corners),
            directSumListsOfSparse(self.sides,other.sides),
            self.state_center_data.directSumWith(other.state_center_data,4),
            self.operator_center_tensor
        )
    # }}}
    def __iadd__(self,other): # {{{
        self.corners = directSumListsOfSparse(self.corners,other.corners)
        self.sides = directSumListsOfSparse(self.sides,other.sides)
        self.state_center_data.directSumWith(other.state_center_data,4)
    # }}}
    def __copy__(self): # {{{
        return \
            System(
                copy(self.corners),
                copy(self.sides),
                self.state_center_data,
                copy(self.operator_center_tensor),
                self.state_center_data_conj,
            )
    # }}}
    def assertDimensionsAreConsistent(self): # {{{
        assert self.state_center_data.shape == self.state_center_data_conj.shape
        if self.state_center_data.shape[0] != self.state_center_data.shape[2]:
            raise AssertionError("state center's left and right dimensions do not agree ({} != {})".format(self.state_center_data.shape[2] != self.state_center_data.shape[0]))
        if self.state_center_data.shape[1] != self.state_center_data.shape[3]:
            raise AssertionError("state center's up and down dimensions do not agree ({} != {})".format(self.state_center_data.shape[1] != self.state_center_data.shape[3]))
        for i, side in enumerate(self.sides):
            normalization_data = side[Identity()]
            if normalization_data.ndim != 8:
                raise AssertionError("for side {} the normalization data has rank {} instead of rank 8".format(i,normalization_data.ndim))
            for tag, data in side.items():
                if data.shape != normalization_data.shape:
                    raise AssertionError("for side {} the data tagged with {} does not match the shape of the data tagged with Identity() ({} != {})".format(i,tag,data.shape,normalization_data.shape))
            side_shape = normalization_data.shape
            for d in (0,3,6):
                if side_shape[d] != side_shape[d+1]:
                    raise AssertionError("side {}'s dimension {} does not match its dimension {} ({} != {})".format(i,d,d+1,side_shape[d],side_shape[d+1]))
            if side_shape[0] != side_shape[3]:
                raise AssertionError("side {}'s left and right dimensions do not agree ({} != {})".format(side_shape[0],side_shape[3]))
            if side_shape[6] != self.state_center_data.shape[i]:
                raise AssertionError("side {}'s center-facing dimensions do not match the corresponding state dimension ({} != {})".format(i,side_shape[6],self.state_center_data.shape[i]))
        for i, corner in enumerate(self.corners):
            normalization_data = corner[Identity()]
            if normalization_data.ndim != 6:
                raise AssertionError("for corner {} the normalization data has rank {} instead of rank 8".format(i,normalization_data.ndim))
            for tag, data in corner.items():
                if data.shape != normalization_data.shape:
                    raise AssertionError("for corner {} the data tagged with {} does not match the shape of the data tagged with Identity() ({} != {})".format(i,tag,data.shape,normalization_data.shape))
            corner_shape = normalization_data.shape
            for d in (0,3):
                if corner_shape[d] != corner_shape[d+1]:
                    raise AssertionError("corner {}'s dimension {} does not match its dimension {} ({} != {})".format(i,d,d+1,corner_shape[d],corner_shape[d+1]))
            if corner_shape[3] != self.sides[i][Identity()].shape[0]:
                raise AssertionError("corner {}'s right dimensions do not match side {}'s left dimensions ({} != {})".format(i,i,corner_shape[3],self.sides[i][Identity()].shape[0]))
            if corner_shape[0] != self.sides[L(i)][Identity()].shape[3]:
                raise AssertionError("corner {}'s left dimensions do not match side {}'s right dimensions ({} != {})".format(i,L(i),corner_shape[0],self.sides[L(i)][Identity()].shape[3]))
    # }}}
    def assertHasNoNaNs(self): # {{{
        for i, corner in enumerate(self.corners):
            for tag, data in corner.items():
                if data.hasNaN():
                    raise AssertionError("corner {} has a NaN in component {}".format(i,tag))
        for i, side in enumerate(self.sides):
            for tag, data in side.items():
                if data.hasNaN():
                    raise AssertionError("side {} has a NaN in component {}".format(i,tag))
        if self.state_center_data.hasNaN():
            raise AssertionError("state center has a NaN")
        for tag, data in self.operator_center_tensor.items():
            if data is not None and data.hasNaN():
                raise AssertionError("operator center has a NaN in component {}".format(tag))
    # }}}
    def assertNormalizationIsHermitian(self): # {{{
        for i in range(4):
            if not self.sides[i][Identity()].allcloseTo(self.sides[i][Identity()].join(1,0,2,4,3,5,7,6).conj()):
                raise AssertionError("side {} is not hermitian".format(i))
            if not self.corners[i][Identity()].allcloseTo(self.corners[i][Identity()].join(1,0,2,4,3,5).conj()):
                raise AssertionError("corner {} is not hermitian".format(i))
    # }}}
    def compressCornerStateTowards(self,corner_id,direction,new_dimension): # {{{
        if direction == 0:
            return self.compressCornerStateTowardsLeft(corner_id,new_dimension)
        elif direction == 1:
            return self.compressCornerStateTowardsRight(corner_id,new_dimension)
        else:
            raise ValueError("compression direction must be 0 or 1, not " + str(direction))
    # }}}
    def compressCornerStateTowardsLeft(self,corner_id,new_dimension): # {{{
        side_data_joined = self.sides[L(corner_id)][Identity()].join((0,1,2,6,7),3,4,5)
        corner_data_joined = self.corners[corner_id][Identity()].join(0,1,2,(3,4,5))
        compressor = computeProductCompressor(side_data_joined,corner_data_joined,new_dimension)
        self.sides[L(corner_id)] = mapOverSparseData(lambda data: (data
            ).absorbMatrixAt(3,compressor
            ).absorbMatrixAt(4,compressor.conj())
        ,self.sides[L(corner_id)]
        )
        self.corners[corner_id] = mapOverSparseData(lambda data: (data
            ).absorbMatrixAt(0,compressor.conj()
            ).absorbMatrixAt(1,compressor)
        ,self.corners[corner_id]
        )
    # }}}
    def compressCornerStateTowardsRight(self,corner_id,new_dimension): # {{{
        corner_data_joined = self.corners[corner_id][Identity()].join((0,1,2),3,4,5)
        side_data_joined = self.sides[corner_id][Identity()].join(0,1,2,(3,4,5,6,7))
        compressor = computeProductCompressor(corner_data_joined,side_data_joined,new_dimension)
        self.corners[corner_id] = mapOverSparseData(lambda data: (data
            ).absorbMatrixAt(3,compressor
            ).absorbMatrixAt(4,compressor.conj())
        ,self.corners[corner_id]
        )
        self.sides[corner_id] = mapOverSparseData(lambda data: (data
            ).absorbMatrixAt(0,compressor.conj()
            ).absorbMatrixAt(1,compressor)
        ,self.sides[corner_id]
        )
    # }}}
    def compressCornerTwoSiteOperatorTowards(self,corner_id,direction,new_dimension,normalize=False): # {{{
        axis = 3*direction+2
        # Gather the terms {{{
        new_corner = {}
        sparse_data = []
        sparse_position_map = {}
        old_compressed_data = None
        next_index = 0
        for tag, data in self.corners[corner_id].items():
            if isinstance(tag,TwoSiteOperator) and tag.direction == direction:
                assert tag.position not in sparse_position_map
                sparse_position_map[tag.position] = next_index
                next_index += 1
                sparse_data.append(data)
            elif isinstance(tag,TwoSiteOperatorCompressed) and tag.direction == direction:
                assert old_compressed_data is None
                old_compressed_data = data
            else:
                new_corner[tag] = data
        old_compressed_data_exists = old_compressed_data is not None
        # }}}
        # Compute the old dimension {{{
        old_dimension = len(sparse_data)
        if old_compressed_data_exists:
            old_dimension += old_compressed_data.shape[axis]
        if old_dimension == 0:
            return
        # }}}
        # Compute the first submatrix {{{
        slice1 = slice(0,len(sparse_data))
        collected_data = array([data.toArray().ravel() for data in sparse_data])
        matrix1 = dot(collected_data.conj(),collected_data.transpose())
        # }}}
        # Compute the second submatrix {{{
        slice2 = slice(len(sparse_data),old_dimension)
        if old_compressed_data_exists:
            data2 = old_compressed_data.fold(axis)
            data1 = data2.conj()
            matrix2 = data1.contractWith(data2,(1,),(1,)).toArray()
            del data1
            del data2
        else:
            matrix2 = zeros((0,0),dtype=complex128)
        # }}}
        # Compute the compressors  {{{
        def multiply(in_v):
            out_v = 0*in_v
            out_v[slice1] = dot(matrix1,in_v[slice1])
            out_v[slice2] = dot(matrix2,in_v[slice2])
            return out_v
        def formMatrix():
            matrix = zeros((old_dimension,old_dimension),dtype=complex128)
            matrix[slice1,slice1] = matrix1
            matrix[slice2,slice2] = matrix2
            return matrix
        corner_multiplier, side_multiplier_conj = \
            computeCompressor(
                old_dimension,
                new_dimension,
                Multiplier(
                    (old_dimension,)*2,
                    multiply,
                    matrix1.shape[0]*matrix1.shape[1]+matrix2.shape[0]*matrix2.shape[1],
                    formMatrix,
                    0,
                ),
                complex128,
                normalize
            )
        corner_multiplier = NDArrayData(corner_multiplier)
        side_multiplier = NDArrayData(side_multiplier_conj.conj())
        del matrix1
        del matrix2
        del multiply
        del formMatrix
        # }}}
        # Compute the new corner {{{
        collected_data_shape = list(self.corners[corner_id][Identity()].shape)
        del collected_data_shape[axis]
        collected_data_shape.insert(0,old_dimension)
        collected_data_transposition = list(range(1,6))
        collected_data_transposition.insert(axis,0)
        compressed_collected_data = \
            NDArrayData(collected_data.reshape(collected_data_shape)).absorbMatrixAt(0,corner_multiplier[:,slice1]).transpose(collected_data_transposition)

        if old_compressed_data_exists:
            new_compressed_data = old_compressed_data.absorbMatrixAt(axis,corner_multiplier[:,slice2])
            new_compressed_data += compressed_collected_data
        else:
            new_compressed_data = compressed_collected_data

        new_corner[TwoSiteOperatorCompressed(direction)] = new_compressed_data
        self.corners[corner_id] = new_corner
        del collected_data
        # }}}
        # Compute the new side {{{
        side_id = sideFromCorner(corner_id,direction)
        direction = 1-direction
        axis = 3*direction+2
        collected_data = [None]*len(sparse_data)
        old_compressed_data = None
        new_side = {}
        for tag, data in self.sides[side_id].items():
            if isinstance(tag,TwoSiteOperator) and tag.direction == direction:
                collected_data[sparse_position_map[tag.position]] = data
            elif isinstance(tag,TwoSiteOperatorCompressed) and tag.direction == direction:
                assert old_compressed_data_exists
                assert old_compressed_data is None
                old_compressed_data = data
            else:
                new_side[tag] = data
        assert None not in collected_data
        if old_compressed_data_exists:
            assert old_compressed_data is not None
        if collected_data:
            collected_data = collected_data[0].newCollected(collected_data).dropUnitAxis(axis+1)
        else:
            assert old_compressed_data_exists
            collected_data_shape = list(self.sides[side_id][Identity()].shape)
            del collected_data_shape[axis]
            collected_data_shape.insert(0,0)
            collected_data = old_compressed_data.newZeros(collected_data_shape,dtype=complex128)
        collected_data_transposition = list(range(1,8))
        collected_data_transposition.insert(axis,0)
        compressed_collected_data = \
            collected_data.absorbMatrixAt(0,side_multiplier[:,slice1]).transpose(collected_data_transposition)
        if old_compressed_data_exists:
            new_compressed_data = old_compressed_data.absorbMatrixAt(axis,side_multiplier[:,slice2])
            new_compressed_data += compressed_collected_data
        else:
            new_compressed_data = compressed_collected_data
        new_side[TwoSiteOperatorCompressed(direction)] = new_compressed_data
        self.sides[side_id] = new_side
        # }}}
    # }}}
    def computeCenterSiteExpectation(self): # {{{
        return self.computeExpectation()-self.computeExpectationWithoutCenter()
    # }}}
    def computeExpectation(self,operator_center_tensor=None): # {{{
        return self.computeExpectationAndNormalization(operator_center_tensor)[0]
    # }}}
    def computeExpectationWithoutCenter(self): # {{{
        return self.computeExpectationAndNormalizationWithoutCenter()[0]
    # }}}
    def computeExpectationAndNormalization(self,operator_center_tensor=None): # {{{
        multiplyExpectation, multiplyNormalization = self.formExpectationAndNormalizationMultipliers(operator_center_tensor)
        unnormalized_expectation = self.computeScalarUsingMultiplier(multiplyExpectation)
        normalization = self.computeScalarUsingMultiplier(multiplyNormalization)
        return unnormalized_expectation/normalization, normalization
    # }}}
    def computeExpectationAndNormalizationWithoutCenter(self): # {{{
        return self.computeExpectationAndNormalization({
            tag: value
                for (tag,value) in self.operator_center_tensor.items()
                if tag is Identity() or (
                    isinstance(tag,TwoSiteOperator) and
                    tag.position == 0 and
                    tag.direction in (2,3)
                )
        })
    # }}}
    def computeNormalization(self): # {{{
        return self.computeScalarUsingMultiplier(self.formNormalizationMultiplier())
    # }}}
    def computeNormalizationMatrixConditionNumber(self): # {{{
        return cond(self.formNormalizationMatrix().toArray())
    # }}}
    def computeOneSiteExpectation(self): # {{{
        expectation = 0
        stripped_self = self.stripExpectationEnvironment()

        for tag, value in self.operator_center_tensor.items():
            if isinstance(tag,OneSiteOperator):
                system = copy(stripped_self)
                system.operator_center_tensor[tag] = value
                expectation += system.computeExpectation()
                del system

            if isinstance(tag,TwoSiteOperator):
                if tag.position != 0:
                    continue
                if tag.direction == 0:
                    other_tag = tag.withNewDirectionAndPosition(2,0)
                    system = copy(stripped_self)
                    system.operator_center_tensor[tag] = value
                    system.operator_center_tensor[other_tag] = self.operator_center_tensor[other_tag]
                    system.contractTowards(0)
                    expectation += system.computeExpectation()
                    del system
                if tag.direction == 1:
                    other_tag = tag.withNewDirectionAndPosition(3,0)
                    system = copy(stripped_self)
                    system.operator_center_tensor[tag] = value
                    system.operator_center_tensor[other_tag] = self.operator_center_tensor[other_tag]
                    system.contractTowards(1)
                    expectation += system.computeExpectation()
                    del system

        return expectation
    # }}}
    def computeScalarUsingMultiplier(self,multiply): # {{{
        return self.state_center_data_conj.contractWith(multiply(self.state_center_data),range(5),range(5)).extractScalar()
    # }}}
    def computeUnnormalizedExpectation(self): # {{{
        return self.computeScalarUsingMultiplier(self.formExpectationMultipliers())
    # }}}
    def contractTowards(self,direction): # {{{
        tensor_to_contract, _, matrix_to_absorb = self.state_center_data.normalizeAxis(O(direction))
        self.contractUnnormalizedTowards(direction,tensor_to_contract)
        self.setStateCenter(self.state_center_data.normalizeAxis(direction)[0].absorbMatrixAt(direction,matrix_to_absorb))
    # }}}
    def contractNormalizedTowards(self,direction,state_center_data): # {{{
        self.contractUnnormalizedTowards(direction,state_center_data.normalizeAxis(O(direction))[0])
    # }}}
    def contractUnnormalizedTowards(self,direction,state_center_data=None,state_center_data_conj=None): # {{{
        if state_center_data is None:
            state_center_data = self.state_center_data
            state_center_data_conj = self.state_center_data_conj
        if state_center_data_conj is None:
            state_center_data_conj = state_center_data.conj()
        self.corners[direction] = absorbSparseSideIntoCornerFromLeft(self.corners[direction],self.sides[L(direction)])
        self.sides[direction] = absorbSparseCenterSOSIntoSide(direction,self.sides[direction],state_center_data,self.operator_center_tensor,state_center_data_conj)
        self.corners[R(direction)] = absorbSparseSideIntoCornerFromRight(self.corners[R(direction)],self.sides[R(direction)])
        if self.just_increased_bandwidth:
            raise InvariantViolatedError("Contracting the current center would blow up the condition number of the normalization matrix;  optimize it or replace it first.")
    # }}}
    def formExpectationAndNormalizationMultipliers(self,operator_center_tensor=None): # {{{
        if operator_center_tensor is None:
            operator_center_tensor = self.operator_center_tensor
        return formExpectationAndNormalizationMultipliers(self.corners,self.sides,operator_center_tensor)
    # }}}
    def formExpectationMatrix(self): # {{{
        return self.formExpectationMultiplier().formMatrix()
    # }}}
    def formExpectationMultiplier(self): # {{{
        return self.formExpectationAndNormalizationMultipliers()[0]
    # }}}
    def formNormalizationMatrix(self): # {{{
        return self.formNormalizationMultiplier().formMatrix()
    # }}}
    def formNormalizationMultiplier(self): # {{{
        return \
            formNormalizationMultiplier(
                tuple(corner[Identity()] for corner in self.corners),
                tuple(side[Identity()] for side in self.sides),
                self.operator_center_tensor[Identity()]
            )
    # }}}
    def formNormalizationSubmatrix(self): # {{{
        return formNormalizationSubmatrix(tuple(corner[Identity()] for corner in self.corners),tuple(side[Identity()] for side in self.sides))
    # }}}
    def increaseBandwidth(self,direction,by=None,to=None,do_as_much_as_possible=False,enlargeners=None): # {{{
        if direction not in (0,1):
            raise ValueError("Direction for bandwidth increase must be either 0 (for horizontal axes) or 1 (for vertical axes), not {}.".format(direction))
        return self._increaseBandwidth(direction,by,to,do_as_much_as_possible,enlargeners)
    # }}}
    def increaseBandwidthAndThenNormalize(self,direction,by=None,to=None): # {{{
        self.increaseBandwidth(direction,by,to)
        self.normalize()
    # }}}
    def minimizeExpectation(self): # {{{
        self.setStateCenter(
            relaxOver(
                self.state_center_data,
                *self.formExpectationAndNormalizationMultipliers(),
                maximum_number_of_multiplications=100
            )
        )
    # }}}
    def minimizeExpectationUsingFullEigensolver(self): # {{{
        matrix = self.formExpectationMultiplier().formMatrix().toArray()
        evals, evecs = eigh(matrix,self.formNormalizationMultiplier().formMatrix().toArray())
        self.setStateCenter(NDArrayData(evecs[:,0].reshape(self.state_center_data.shape)))
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
            tag: side_data.absorbMatrixAt(6,denormalizer_for_side_axis1).absorbMatrixAt(7,denormalizer_for_side_axis2)
            for tag, side_data in self.sides[direction].items()
        }
    # }}}
    def normalizeCornerAndDenormalizeSide(self,corner_id,direction): # {{{
        side_id = sideFromCorner(corner_id,direction)
        corner_axis1 = direction*3+0
        corner_axis2 = direction*3+1
        side_axis1 = (1-direction)*3+0
        side_axis2 = (1-direction)*3+1
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
        normalizer_for_side_axis1, denormalizer_for_center = side_data.normalizeAxis(6,True)
        normalizer_for_side_axis2 = normalizer_for_side_axis1.conj()
        self.sides[side_id] = {
            tag: side_data.absorbMatrixAt(6,normalizer_for_side_axis1).absorbMatrixAt(7,normalizer_for_side_axis2)
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
        self.just_increased_bandwidth = False
    # }}}
    def stripExpectationEnvironment(self): # {{{
        return \
            type(self)(
                map(stripAllButIdentityFrom,self.corners),
                map(stripAllButIdentityFrom,self.sides),
                self.state_center_data,
                stripAllButIdentityFrom(self.operator_center_tensor),
                self.state_center_data_conj,
            )
    # }}}
  # }}}
# }}}
# }}}

# Functions {{{
def sideFromCorner(corner_id,direction): # {{{
    return (corner_id+1-direction)%4
# }}}
# }}}

# Exports {{{
__all__ = [
    "System",

    "sideFromCorner",
]
# }}}

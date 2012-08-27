# Imports {{{
from copy import copy
from numpy import zeros
from random import randint

from . import *
from ..data import *
from ..expectation import *
from ..sparse import *
from ..tensors.dense import *
from ..tensors.sparse import *
# }}}

class TestExpectation(TestCase): # {{{
    @staticmethod # generateData # {{{
    def generateData():
        corners_dense_shapes = tuple((randint(1,2),randint(1,2)) for i in range(4))
        corners_tensor = tuple(randomSparseTensor((randint(5,6),randint(5,6)),corners_dense_shapes[i],4) for i in range(4))
        state_center_data = NDArrayData.newRandom(*(randint(1,2) for _ in range(5)))
        operator_center_tensor = randomSparseTensor(tuple(randint(5,6) for _ in range(4)),(state_center_data.shape[-1],)*2,4)
        sides_tensor = tuple(
            randomSparseTensor(
                (
                    corners_tensor[i].dimensions[1],
                    corners_tensor[(i-1)%4].dimensions[0],
                    operator_center_tensor.dimensions[i]
                ),(
                    corners_dense_shapes[i][1],
                    corners_dense_shapes[(i-1)%4][0],
                    state_center_data.shape[i],
                    state_center_data.shape[i]
                ),
                4
            )
            for i in range(4)
        )
        corners = tuple(mapSparseChunkValues(DenseCorner,corner_tensor) for corner_tensor in corners_tensor)
        sides = tuple(SparseSide(mapSparseChunkValues(DenseSide,side_tensor)) for side_tensor in sides_tensor)
        return corners_tensor, sides_tensor, state_center_data, operator_center_tensor, corners, sides
    # }}}
    @with_checker
    def test_absorbCenter(self,i=irange(0,3)): # {{{
        corners_tensor, sides_tensor, state_center_data, operator_center_tensor, corners, sides = self.generateData()
        expectation = Expectation(corners,sides,operator_center_tensor)
        expectation.absorbCenter(i,state_center_data)
        corners = list(corners)
        sides = list(sides)
        corners[i] = absorbSparseSideIntoCornerFromLeft(i,corners[i],sides[(i+1)%4])
        sides[i] = sides[i].absorbCenterSOS(i,state_center_data,operator_center_tensor)
        corners[(i-1)%4] = absorbSparseSideIntoCornerFromRight(i,corners[(i-1)%4],sides[(i-1)%4])
        self.assertSparseTensorWithWrappedDataAlmostEqual(expectation.corners[0],corners[0])
        self.assertSparseTensorWithWrappedDataAlmostEqual(expectation.corners[1],corners[1])
        self.assertSparseTensorWithWrappedDataAlmostEqual(expectation.corners[2],corners[2])
        self.assertSparseTensorWithWrappedDataAlmostEqual(expectation.corners[3],corners[3])
        self.assertSparseTensorWithWrappedDataAlmostEqual(expectation.sides[0].tensor,sides[0].tensor)
        self.assertSparseTensorWithWrappedDataAlmostEqual(expectation.sides[1].tensor,sides[1].tensor)
        self.assertSparseTensorWithWrappedDataAlmostEqual(expectation.sides[2].tensor,sides[2].tensor)
        self.assertSparseTensorWithWrappedDataAlmostEqual(expectation.sides[3].tensor,sides[3].tensor)
    # }}}
    @with_checker
    def test_formMultiplier(self): # {{{
        corners_tensor, sides_tensor, state_center_data, operator_center_tensor, corners, sides = self.generateData()
        corners_data = [NDArrayData(formDenseTensor(corner_tensor,NDArrayData.toArray)) for corner_tensor in corners_tensor]
        sides_data = [NDArrayData(formDenseTensor(side_tensor,NDArrayData.toArray)) for side_tensor in sides_tensor]
        operator_center_data = NDArrayData(formDenseTensor(operator_center_tensor,NDArrayData.toArray))

        observed = Expectation(corners,sides,operator_center_tensor).formMultiplier()(state_center_data)
        exact = formDataContractor(
            [Join(i,(1,3),i+4,(0,3)) for i in range(4)]+
            [Join(i,(0,2),(i+1)%4+4,(1,4)) for i in range(4)]+
            [Join(i+4,2,8,i) for i in range(4)]+
            [Join(i+4,5,9,i) for i in range(4)]+
            [Join(8,5,9,4)],
            [[(i+4,6)] for i in range(4)]+[[(8,4)]]
        )(*corners_data + sides_data + [operator_center_data,state_center_data,])
        self.assertDataAlmostEqual(observed,exact,rtol=1e-5)
    # }}}
# }}}

# Imports {{{
from copy import copy
from numpy import any, isnan, zeros
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
        corners = corners_tensor
        sides = sides_tensor
        return corners_tensor, sides_tensor, state_center_data, operator_center_tensor, corners, sides
    # }}}
    @with_checker(number_of_calls=40)
    def test_absorbCenter(self,i=irange(0,3)): # {{{
        corners_tensor, sides_tensor, state_center_data, operator_center_tensor, corners, sides = self.generateData()
        expectation = Expectation(corners,sides,operator_center_tensor)
        expectation.absorbCenter(i,state_center_data)
        corners = list(corners)
        sides = list(sides)
        corners[i] = absorbSparseSideIntoCornerFromLeft(i,corners[i],sides[(i+1)%4])
        sides[i] = absorbSparseCenterSOSIntoSide(i,sides[i],state_center_data,operator_center_tensor)
        corners[(i-1)%4] = absorbSparseSideIntoCornerFromRight(i,corners[(i-1)%4],sides[(i-1)%4])
        self.assertSparseTensorsAlmostEqual(expectation.corners[0],corners[0])
        self.assertSparseTensorsAlmostEqual(expectation.corners[1],corners[1])
        self.assertSparseTensorsAlmostEqual(expectation.corners[2],corners[2])
        self.assertSparseTensorsAlmostEqual(expectation.corners[3],corners[3])
        self.assertSparseTensorsAlmostEqual(expectation.sides[0],sides[0])
        self.assertSparseTensorsAlmostEqual(expectation.sides[1],sides[1])
        self.assertSparseTensorsAlmostEqual(expectation.sides[2],sides[2])
        self.assertSparseTensorsAlmostEqual(expectation.sides[3],sides[3])
    # }}}
    @with_checker(number_of_calls=10)
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
        try:
            self.assertDataAlmostEqual(observed,exact,atol=1e-5)
        except:
            if not any(isnan(exact.toArray())):
                raise
    # }}}
# }}}

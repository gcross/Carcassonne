# Imports {{{
from copy import copy
from numpy import any, isnan, zeros
from random import randint

from . import *
from ..data import *
from ..normalization import *
from ..tensors.dense import *
from ..utils import L, R
# }}}

class TestNormalization(TestCase): # {{{
    @staticmethod # generateData # {{{
    def generateData():
        n = 5
        corners_data = tuple(NDArrayData.newRandom(*(randint(1,n) for _ in range(2))) for _ in range(4))
        center_data = NDArrayData.newRandom(*(randint(1,n) for _ in range(5)))
        sides_data = tuple(NDArrayData.newRandom(corners_data[i].shape[1],corners_data[(i-1)%4].shape[0],center_data.shape[i],center_data.shape[i]) for i in range(4))
        corners = tuple(corners_data)
        sides = tuple(DenseSide(sides_data[i]) for i in range(4))
        return corners_data, sides_data, center_data, corners, sides
    # }}}
    @with_checker
    def test_absorbCenter(self,i=irange(0,3)): # {{{
        corners_data, sides_data, center_data, corners, sides = self.generateData()
        normalization = Normalization(corners,sides)
        normalization.absorbCenter(i,center_data)
        corners = list(corners)
        sides = list(sides)
        corners[i] = absorbDenseSideIntoCornerFromLeft(corners[i],sides[L(i)])
        sides[i] = sides[i].absorbCenterSS(i,center_data)
        corners[R(i)] = absorbDenseSideIntoCornerFromRight(corners[R(i)],sides[R(i)])
        self.assertDataAlmostEqual(normalization.corners[0],corners[0])
        self.assertDataAlmostEqual(normalization.corners[1],corners[1])
        self.assertDataAlmostEqual(normalization.corners[2],corners[2])
        self.assertDataAlmostEqual(normalization.corners[3],corners[3])
        self.assertDataAlmostEqual(normalization.sides[0].data,sides[0].data)
        self.assertDataAlmostEqual(normalization.sides[1].data,sides[1].data)
        self.assertDataAlmostEqual(normalization.sides[2].data,sides[2].data)
        self.assertDataAlmostEqual(normalization.sides[3].data,sides[3].data)
    # }}}
    @with_checker
    def test_formMultiplier(self): # {{{
        corners_data, sides_data, center_data, corners, sides = self.generateData()

        observed = Normalization(corners,sides).formMultiplier()(center_data)
        exact = formDataContractor(
            [Join(i,1,i+4,0) for i in range(4)]+
            [Join(i,0,(i+1)%4+4,1) for i in range(4)]+
            [Join(i+4,2,8,i) for i in range(4)],
            [[(i+4,3)] for i in range(4)]+[[(8,4)]]
        )(*corners_data + sides_data + (center_data,))
        try:
            self.assertDataAlmostEqual(observed,exact,atol=1e-5)
        except:
            if not any(isnan(exact.toArray())):
                raise
    # }}}
    @with_checker
    def test_W_state(self, # {{{
        moves=(irange(0,1),)*4
    ):
        center = zeros((4,)*4+(2,))
        center[0,0,0,0,0] = 1
        center[1,0,1,0,0] = 1
        center[0,1,0,1,0] = 1
        center[1,1,1,1,0] = 1
        center[1,2,0,2,0] = 1
        center[2,0,2,1,0] = 1
        center[1,3,0,3,0] = 1
        center[3,0,3,1,0] = 1
        center[3,2,2,3,1] = 1
        center = NDArrayData(center)

        corners = [NDArrayData.newTrivial((1,1))]*4
        sides = [
            DenseSide(NDArrayData.newOuterProduct([1],[1],[0,1,0,1],[0,1,0,1])),
            DenseSide(NDArrayData.newOuterProduct([1],[1],[1,0,1,0],[1,0,1,0])),
            DenseSide(NDArrayData.newOuterProduct([1],[1],[1,0,1,0],[1,0,1,0])),
            DenseSide(NDArrayData.newOuterProduct([1],[1],[0,1,0,1],[0,1,0,1])),
        ]
        normalization = Normalization(corners,sides)

        directions = sum(([i]*moves[i] for i in range(4)),[])
        shuffle(directions)
        for direction in directions:
            normalization.absorbCenter(direction,center)
        self.assertEqual(
            normalization.computeNormalization(center),
            (1+moves[0]+moves[2])*(1+moves[1]+moves[3])
        )
    # }}}
# }}}

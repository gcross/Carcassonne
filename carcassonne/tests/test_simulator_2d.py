# Imports {{{
from functools import partial
from math import pi, sqrt
from numpy import ones
from paycheck import *

from . import *
from ..policies import *
from ..system._2d import System
# }}}

class TestSimulator2D(TestCase): # {{{
    def test_on_transverse_Ising(self): # {{{
        OO = [NDArrayData.X,-0.01*NDArrayData.X]
        system = System.newTrivialWithSimpleSparseOperator(O=-NDArrayData.Z,OO_LR=OO,OO_UD=OO)
        system.setPolicy("state compression",ConstantStateCompressionPolicy(2))
        system.setPolicy("sweep convergence",RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7))
        system.setPolicy("run convergence",RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7))
        system.setPolicy("bandwidth increase",AllDirectionsIncrementBandwidthIncreasePolicy())
        system.setPolicy("contraction",RepeatPatternContractionPolicy(range(4)))
        system.runUntilConverged()
        self.assertAlmostEqual(system.computeOneSiteExpectation(),1.000050,places=6)
    # }}}
# }}}

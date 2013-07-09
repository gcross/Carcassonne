# Imports {{{
from functools import partial
from math import pi, sqrt
from numpy import ones
from paycheck import *

from . import *
from ..policies import *
from ..system._1d import System
# }}}

class TestSimulator1D(TestCase): # {{{
    @ with_checker(number_of_calls=10) # test_on_magnetic_field {{{
    def test_on_magnetic_field(self):
        system = \
            System(
                buildProductTensor([1,0]),
                buildProductTensor([0,1]),
                buildTensor((2,2,2,2),{
                   (0,0): Pauli.I,
                   (1,1): Pauli.I,
                   (0,1): -Pauli.Z,
                }),
                ones((1,1,2)),
            )
        system.setPolicy("sweep convergence",RelativeStateDifferenceThresholdConvergencePolicy(1e-5))
        system.setPolicy("run convergence",RelativeEstimatedOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-2))
        system.setPolicy("bandwidth increase",OneDirectionIncrementBandwidthIncreasePolicy(0))
        system.setPolicy("contraction",RepeatPatternContractionPolicy([0,1]))
        system.runUntilConverged()
        self.assertAlmostEqual(abs(system.computeEstimatedOneSiteExpectation(0)),1,places=2)
    # }}}
    @ with_checker(number_of_calls=10) # test_on_ferromagnetic_coupling {{{
    def dont_test_on_ferromagnetic_coupling(self):
        system = \
            System(
                [1,0,0],
                [0,0,1],
                buildTensor((3,3,2,2),{
                    (0,0): Pauli.I,
                    (0,1): -Pauli.Z,
                    (1,2): Pauli.Z,
                    (2,2): Pauli.I,
                }),
                ones((1,1,2)),
            )
        system.setPolicy("sweep convergence",RelativeStateDifferenceThresholdConvergencePolicy(1e-5))
        system.setPolicy("run convergence",RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-2))
        system.setPolicy("bandwidth increase",OneDirectionIncrementBandwidthIncreasePolicy(0))
        system.setPolicy("contraction",RepeatPatternContractionPolicy([0,1]))
        system.runUntilConverged()
        self.assertAlmostEqual(abs(system.computeOneSiteExpectation()),1,places=2)
    # }}}
    def test_on_transverse_Ising(self): # {{{
        system = \
            System(
                [1,0,0],
                [0,0,1],
                buildTensor((3,3,2,2),{
                    (0,0): Pauli.I,
                    (0,2): Pauli.Z,
                    (0,1): -0.01*Pauli.X,
                    (1,2): Pauli.X,
                    (2,2): Pauli.I,
                }),
                ones((1,1,2)),
            )
        system.setPolicy("sweep convergence",RelativeStateDifferenceThresholdConvergencePolicy(1e-5))
        system.setPolicy("run convergence",RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7))
        system.setPolicy("bandwidth increase",OneDirectionIncrementBandwidthIncreasePolicy(0,2))
        system.setPolicy("contraction",RepeatPatternContractionPolicy([0,1]))
        system.runUntilConverged()
        self.assertAlmostEqual(system.computeOneSiteExpectation(),1.0000250001562545,places=6)
    # }}}
    def test_on_Haldane_Shastry(self): # {{{
        # n = 6, cutoff = 1000
        # a = array([ 4.82625404e-01, 3.63171856e-01, 2.56789132e-02, 1.92857348e-04, 1.25118032e-01, 3.21293315e-03])
        # b = array([ 0.05930634, 0.3228463, 0.84625552, 0.99028958, 0.63260458, 0.9513444 ])

        # n = 9, cutoff = 1000
        # a = array([ -1.39798300e+01,   1.28034382e+01,   5.61535861e+00, -1.26781677e+01,   1.94601671e-04,  -5.61212883e+00, 4.82347734e-01,   1.40055820e+01,   3.63205362e-01])
        # b = array([ 0.84596776,  0.63229175,  0.95118618,  0.6322922 ,  0.99024125, 0.95118618,  0.0592535 ,  0.84596776,  0.32259284])

        # n = 9, cutoff = 100
        #a = array([ -1.39798300e+01,   1.28034382e+01,   5.61535861e+00, -1.26781677e+01,   1.94601671e-04,  -5.61212883e+00, 4.82347734e-01,   1.40055820e+01,   3.63205362e-01])
        #b = array([ 0.84596776,  0.63229175,  0.95118618,  0.6322922 ,  0.99024125, 0.95118618,  0.0592535 ,  0.84596776,  0.32259284])

        # n = 9, cutoff = 10000
        a = array([6.18505736e-04,   3.56927507e-01,   7.04055807e-05, 1.77859581e-02,   3.78493975e-03,   6.54917336e-02, 1.83235170e-01,   4.09930918e-06,   3.72081681e-01])
        b = array([0.97613415,  0.22719877,  0.99279374,  0.85346561,  0.93626109, 0.70473391,  0.48229086,  0.99858369,  0.0402559])

        assert len(a) == len(b)
        matrix = {}
        n = len(a)
        l = 3*n
        r = l+1
        matrix[l,l] = Pauli.I
        matrix[r,r] = Pauli.I
        for i in range(n):
            matrix[l,0*n+i] = a[i]*Pauli.X
            matrix[l,1*n+i] = a[i]*Pauli.Y
            matrix[l,2*n+i] = a[i]*Pauli.Z
            matrix[0*n+i,0*n+i] = b[i]*Pauli.I
            matrix[1*n+i,1*n+i] = b[i]*Pauli.I
            matrix[2*n+i,2*n+i] = b[i]*Pauli.I
            matrix[0*n+i,r] = Pauli.X
            matrix[1*n+i,r] = Pauli.Y
            matrix[2*n+i,r] = Pauli.Z

        system = \
            System(
                [0]*(3*n)+[1,0],
                [0]*(3*n)+[0,1],
                buildTensor((3*n+2,3*n+2,2,2),matrix),
                crand(1,1,2),
            )
        system.setPolicy("sweep convergence",RelativeStateDifferenceThresholdConvergencePolicy(1e-5))
        system.setPolicy("run convergence",RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-2))
        system.setPolicy("bandwidth increase",OneDirectionIncrementBandwidthIncreasePolicy(0,1))
        system.setPolicy("contraction",RepeatPatternContractionPolicy([0,1]))
        system.runUntilConverged()
        self.assertAlmostEqual(system.computeOneSiteExpectation(),pi*pi/6,places=3)
    # }}}
    def test_on_XY(self): # {{{
        system = \
            System(
                [1,0,0,0],
                [0,0,0,1],
                buildTensor((4,4,2,2),{
                    (0,0): Pauli.I,
                    (0,1): Pauli.X,
                    (0,2): Pauli.Z,
                    (1,3): -Pauli.X,
                    (2,3): -Pauli.Z,
                    (3,3): Pauli.I,
                }),
                ones((1,1,2)),
            )
        system.setPolicy("sweep convergence",RelativeStateDifferenceThresholdConvergencePolicy(1e-5))
        system.setPolicy("run convergence",RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-2))
        system.setPolicy("bandwidth increase",OneDirectionIncrementBandwidthIncreasePolicy(0,2))
        system.setPolicy("contraction",RepeatPatternContractionPolicy([0,1]))
        system.runUntilConverged()
        self.assertAlmostEqual(system.computeOneSiteExpectation(),1.27,places=2)
    # }}}
    def test_on_Heisenberg_ferromagnet(self): # {{{
        system = \
            System(
                [1,0,0,0,0],
                [0,0,0,0,1],
                buildTensor((5,5,2,2),{
                    (0,0): Pauli.I,
                    (0,1): Pauli.X,
                    (0,2): Pauli.Y,
                    (0,3): Pauli.Z,
                    (1,4): -Pauli.X,
                    (2,4): -Pauli.Y,
                    (3,4): -Pauli.Z,
                    (4,4): Pauli.I,
                }),
                ones((1,1,2)),
            )
        system.setPolicy("sweep convergence",RelativeStateDifferenceThresholdConvergencePolicy(1e-5))
        system.setPolicy("run convergence",RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-2))
        system.setPolicy("bandwidth increase",OneDirectionIncrementBandwidthIncreasePolicy(0,2))
        system.setPolicy("contraction",RepeatPatternContractionPolicy([0,1]))
        system.runUntilConverged()
        self.assertAlmostEqual(system.computeOneSiteExpectation(),1,places=2)
    # }}}
# }}}

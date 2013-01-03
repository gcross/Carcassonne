# Imports {{{
from numpy import array, complex128, cos, sin, sqrt, zeros
from paycheck import *

from . import *
from ..policies import *
from ..utils import *
from ..system._1d import System
# }}}

# Tests {{{

class TestSystem1D(TestCase): # {{{
    @with_checker # test_computeOneSiteExpectation_Z_field_all_up {{{
    def test_computeOneSiteExpectation_Z_field_all_up(self,phase=float,field_strength=float):
        system = \
            System(
                [1,0],
                [0,1],
                buildTensor((2,2,2,2),{
                    (0,0): Pauli.I,
                    (0,1): field_strength*Pauli.Z,
                    (1,1): Pauli.I,
                }),
                array([[[cos(phase)+1j*sin(phase),0]]]),
            )
        self.assertAlmostEqual(system.computeOneSiteExpectation(),abs(field_strength))
    # }}}
    @with_checker # test_computeOneSiteExpectation_ZZ_field_all_up {{{
    def test_computeOneSiteExpectation_ZZ_field_all_up(self,phase=float,field_strength=float):
        system = \
            System(
                [1,0,0],
                [0,0,1],
                buildTensor((3,3,2,2),{
                    (0,0): Pauli.I,
                    (0,1): field_strength*Pauli.Z,
                    (1,2): Pauli.Z,
                    (2,2): Pauli.I,
                }),
                array([[[cos(phase)+1j*sin(phase),0]]]),
            )
        self.assertAlmostEqual(system.computeOneSiteExpectation(),abs(field_strength))
    # }}}
    @with_checker # test_computeOneSiteExpectation_ZZZ_field_all_up {{{
    def test_computeOneSiteExpectation_ZZZ_field_all_up(self,phase=float,field_strength=float):
        system = \
            System(
                [1,0,0,0],
                [0,0,0,1],
                buildTensor((4,4,2,2),{
                    (0,0): Pauli.I,
                    (0,1): field_strength*Pauli.Z,
                    (1,2): Pauli.Z,
                    (2,3): Pauli.Z,
                    (3,3): Pauli.I,
                }),
                array([[[cos(phase)+1j*sin(phase),0]]]),
            )
        self.assertAlmostEqual(system.computeOneSiteExpectation(),abs(field_strength))
    # }}}
    @with_checker # test_increaseBandwidth
    def test_increaseBandwidth(self,o=irange(1,5),s=irange(1,5),p=irange(1,5)):
        system1 = System.newRandom(o,s,p)
        system1.minimizeExpectation()

        old_shape = system1.center_state.shape
        new_bandwidth = randint(old_shape[0]+1,2*old_shape[0])
        system2 = copy(system1)

        system1.increaseBandwidth(0,to=new_bandwidth)
        system2.contractRightUnnormalized()
        system2.contractLeftUnnormalized()
        self.assertAlmostEqual(system1.computeExpectation(),system2.computeExpectation())
# }}}

# }}}

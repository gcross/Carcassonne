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
    @with_checker # test_computeOneSiteExpectation_Z_field_all_random {{{
    def test_computeOneSiteExpectation_Z_field_random(self,field_strength=float):
        system = \
            System(
                [1,0],
                [0,1],
                buildTensor((2,2,2,2),{
                    (0,0): Pauli.I,
                    (0,1): field_strength*Pauli.Z,
                    (1,1): Pauli.I,
                }),
                normalize(crand(2,2,2),1),
                [1,1],
                [1,1],
            )
        for i in range(100):
            system.contractLeft()
            system.contractRight()
        self.assertAlmostEqual(system.computeOneSiteExpectation(),system.computeExpectation()/201)
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
    @with_checker # test_computeOneSiteExpectation_Z_field_all_up {{{
    def test_computeOneSiteExpectation_Z_field_all_up(self,directions=[oneof(0,1)]):
        system = \
            System(
                [1,0],
                [0,1],
                buildTensor((2,2,2,2),{
                    (0,0): Pauli.I,
                    (0,1): Pauli.Z,
                    (1,1): Pauli.I,
                }),
                array([[[1,0]]]),
            )
        for direction in directions:
            system.contractTowards(direction)
        self.assertAlmostEqual(len(directions)+1,system.computeExpectation())
    # }}}
# }}}

# }}}

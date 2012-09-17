# Imports {{{
from . import *
from ..system import System
# }}}

class TestTwoSiteOperator(TestCase):
    @with_checker # def test_no_steps {{{
    def test_no_steps(self,physical_dimension=irange(1,5)):
        Os = [NDArrayData.newRandomHermitian(physical_dimension,physical_dimension) for _ in range(4)]
        system = System.newTrivialWithSparseOperator(OO_UD=(Os[0],Os[1]),OO_LR=(Os[2],Os[3]))
        self.assertAlmostEqual(system.computeExpectation(),0)
    # }}}
    @with_checker # def test_UD_one_step_right {{{
    def test_UD_one_step_right(self,physical_dimension=irange(1,5)):
        OO_UD = [NDArrayData.newRandomHermitian(physical_dimension,physical_dimension) for _ in range(2)]
        system = System.newTrivialWithSparseOperator(OO_UD=OO_UD)
        system.absorbCenter(0)
        self.assertAlmostEqual(system.computeExpectation(),0)
    # }}}
    @with_checker # def test_UD_one_step_left {{{
    def test_UD_one_step_left(self,physical_dimension=irange(1,5)):
        OO_UD = [NDArrayData.newRandomHermitian(physical_dimension,physical_dimension) for _ in range(2)]
        system = System.newTrivialWithSparseOperator(OO_UD=OO_UD)
        system.absorbCenter(2)
        self.assertAlmostEqual(system.computeExpectation(),0)
    # }}}
    @with_checker(number_of_calls=10) # def test_UD_horizontal_steps {{{
    def test_UD_horizontal_steps(self,physical_dimension=irange(1,5),directions=[choiceof((0,2))]):
        OO_UD = [NDArrayData.newRandomHermitian(physical_dimension,physical_dimension) for _ in range(2)]
        system = System.newTrivialWithSparseOperator(OO_UD=OO_UD)
        for direction in directions:
            system.absorbCenter(direction)
        self.assertAlmostEqual(system.computeExpectation(),0)
    # }}}
    @with_checker # def test_LR_one_step_up {{{
    def test_LR_one_step_up(self,physical_dimension=irange(1,5)):
        OO_LR = [NDArrayData.newRandomHermitian(physical_dimension,physical_dimension) for _ in range(2)]
        system = System.newTrivialWithSparseOperator(OO_LR=OO_LR)
        system.absorbCenter(1)
        self.assertAlmostEqual(system.computeExpectation(),0)
    # }}}
    @with_checker # def test_LR_one_step_down {{{
    def test_LR_one_step_down(self,physical_dimension=irange(1,5)):
        OO_LR = [NDArrayData.newRandomHermitian(physical_dimension,physical_dimension) for _ in range(2)]
        system = System.newTrivialWithSparseOperator(OO_LR=OO_LR)
        system.absorbCenter(3)
        self.assertAlmostEqual(system.computeExpectation(),0)
    # }}}
    @with_checker(number_of_calls=10) # def test_LR_horizontal_steps {{{
    def test_LR_horizontal_steps(self,physical_dimension=irange(1,5),directions=[choiceof((1,3))]):
        OO_LR = [NDArrayData.newRandomHermitian(physical_dimension,physical_dimension) for _ in range(2)]
        system = System.newTrivialWithSparseOperator(OO_LR=OO_LR)
        for direction in directions:
            system.absorbCenter(direction)
        self.assertAlmostEqual(system.computeExpectation(),0)
    # }}}
    @with_checker # def test_LR_one_step_right {{{
    def test_LR_one_step_right(self,physical_dimension=irange(1,5)):
        OO_LR = [NDArrayData.newRandomHermitian(physical_dimension,physical_dimension) for _ in range(2)]
        OO_L, OO_R = OO_LR
        system = System.newTrivialWithSparseOperator(OO_LR=OO_LR)
        state_center_data_L = NDArrayData.newNormalizedRandom(1,1,1,1,physical_dimension)
        state_center_data_R = NDArrayData.newNormalizedRandom(1,1,1,1,physical_dimension)
        system.setStateCenter(state_center_data_R)
        self.assertAlmostEqual(system.computeNormalization(),1)
        system.absorbCenter(0)
        system.setStateCenter(state_center_data_L)
        self.assertAlmostEqual(system.computeNormalization(),1)

        correct_expectation_L = OO_L.contractWith(state_center_data_L.ravel(),(1,),(0,)).contractWith(state_center_data_L.ravel().conj(),(0,),(0,)).extractScalar()
        correct_expectation_R = OO_R.contractWith(state_center_data_R.ravel(),(1,),(0,)).contractWith(state_center_data_R.ravel().conj(),(0,),(0,)).extractScalar()
        correct_expectation = correct_expectation_L*correct_expectation_R/system.computeNormalization()

        self.assertAlmostEqual(system.computeExpectation(),correct_expectation)
    # }}}

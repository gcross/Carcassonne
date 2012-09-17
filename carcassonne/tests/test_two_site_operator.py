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
    @with_checker # def test_LR_one_step_left {{{
    def test_LR_one_step_left(self,physical_dimension=irange(1,5)):
        OO_LR = [NDArrayData.newRandomHermitian(physical_dimension,physical_dimension) for _ in range(2)]
        OO_L, OO_R = OO_LR
        system = System.newTrivialWithSparseOperator(OO_LR=OO_LR)
        state_center_data_L = NDArrayData.newNormalizedRandom(1,1,1,1,physical_dimension)
        state_center_data_R = NDArrayData.newNormalizedRandom(1,1,1,1,physical_dimension)
        system.setStateCenter(state_center_data_L)
        self.assertAlmostEqual(system.computeNormalization(),1)
        system.absorbCenter(2)
        system.setStateCenter(state_center_data_R)
        self.assertAlmostEqual(system.computeNormalization(),1)

        correct_expectation_L = OO_L.contractWith(state_center_data_L.ravel(),(1,),(0,)).contractWith(state_center_data_L.ravel().conj(),(0,),(0,)).extractScalar()
        correct_expectation_R = OO_R.contractWith(state_center_data_R.ravel(),(1,),(0,)).contractWith(state_center_data_R.ravel().conj(),(0,),(0,)).extractScalar()
        correct_expectation = correct_expectation_L*correct_expectation_R/system.computeNormalization()

        self.assertAlmostEqual(system.computeExpectation(),correct_expectation)
    # }}}
    @with_checker # def test_UD_one_step_up {{{
    def test_UD_one_step_up(self,physical_dimension=irange(1,5)):
        OO_UD = [NDArrayData.newRandomHermitian(physical_dimension,physical_dimension) for _ in range(2)]
        OO_U, OO_D = OO_UD
        system = System.newTrivialWithSparseOperator(OO_UD=OO_UD)
        state_center_data_U = NDArrayData.newNormalizedRandom(1,1,1,1,physical_dimension)
        state_center_data_D = NDArrayData.newNormalizedRandom(1,1,1,1,physical_dimension)
        system.setStateCenter(state_center_data_U)
        self.assertAlmostEqual(system.computeNormalization(),1)
        system.absorbCenter(1)
        system.setStateCenter(state_center_data_D)
        self.assertAlmostEqual(system.computeNormalization(),1)

        correct_expectation_U = OO_U.contractWith(state_center_data_U.ravel(),(1,),(0,)).contractWith(state_center_data_U.ravel().conj(),(0,),(0,)).extractScalar()
        correct_expectation_D = OO_D.contractWith(state_center_data_D.ravel(),(1,),(0,)).contractWith(state_center_data_D.ravel().conj(),(0,),(0,)).extractScalar()
        correct_expectation = correct_expectation_U*correct_expectation_D/system.computeNormalization()

        self.assertAlmostEqual(system.computeExpectation(),correct_expectation)
    # }}}
    @with_checker # def test_UD_one_step_down {{{
    def test_UD_one_step_down(self,physical_dimension=irange(1,5)):
        OO_UD = [NDArrayData.newRandomHermitian(physical_dimension,physical_dimension) for _ in range(2)]
        OO_U, OO_D = OO_UD
        system = System.newTrivialWithSparseOperator(OO_UD=OO_UD)
        state_center_data_U = NDArrayData.newNormalizedRandom(1,1,1,1,physical_dimension)
        state_center_data_D = NDArrayData.newNormalizedRandom(1,1,1,1,physical_dimension)
        system.setStateCenter(state_center_data_D)
        self.assertAlmostEqual(system.computeNormalization(),1)
        system.absorbCenter(3)
        system.setStateCenter(state_center_data_U)
        self.assertAlmostEqual(system.computeNormalization(),1)

        correct_expectation_U = OO_U.contractWith(state_center_data_U.ravel(),(1,),(0,)).contractWith(state_center_data_U.ravel().conj(),(0,),(0,)).extractScalar()
        correct_expectation_D = OO_D.contractWith(state_center_data_D.ravel(),(1,),(0,)).contractWith(state_center_data_D.ravel().conj(),(0,),(0,)).extractScalar()
        correct_expectation = correct_expectation_U*correct_expectation_D/system.computeNormalization()

        self.assertAlmostEqual(system.computeExpectation(),correct_expectation)
    # }}}
    @with_checker # def test_LR_two_steps_right {{{
    def test_LR_two_steps_right(self,physical_dimension=irange(1,5)):
        OO_LR = [NDArrayData.newRandomHermitian(physical_dimension,physical_dimension) for _ in range(2)]
        OO_L, OO_R = OO_LR
        system = System.newTrivialWithSparseOperator(OO_LR=OO_LR)
        Ss = [NDArrayData.newNormalizedRandom(1,1,1,1,physical_dimension) for _ in range(3)]
        SL, SM, SR = Ss
        system.setStateCenter(SR)
        self.assertAlmostEqual(system.computeNormalization(),1)
        system.absorbCenter(0)
        system.setStateCenter(SM)
        self.assertAlmostEqual(system.computeNormalization(),1)
        system.absorbCenter(0)
        system.setStateCenter(SL)
        self.assertAlmostEqual(system.computeNormalization(),1)

        E1L = OO_L.contractWith(SL.ravel(),(1,),(0,)).contractWith(SL.ravel().conj(),(0,),(0,)).extractScalar()
        E1M = OO_R.contractWith(SM.ravel(),(1,),(0,)).contractWith(SM.ravel().conj(),(0,),(0,)).extractScalar()
        E1 = E1L*E1M

        E2M = OO_L.contractWith(SM.ravel(),(1,),(0,)).contractWith(SM.ravel().conj(),(0,),(0,)).extractScalar()
        E2R = OO_R.contractWith(SR.ravel(),(1,),(0,)).contractWith(SR.ravel().conj(),(0,),(0,)).extractScalar()
        E2 = E2M*E2R

        self.assertAlmostEqual(system.computeExpectation(),E1+E2)
    # }}}
    @with_checker # def test_LR_two_steps_left {{{
    def test_LR_two_steps_left(self,physical_dimension=irange(1,5)):
        OO_LR = [NDArrayData.newRandomHermitian(physical_dimension,physical_dimension) for _ in range(2)]
        OO_L, OO_R = OO_LR
        system = System.newTrivialWithSparseOperator(OO_LR=OO_LR)
        Ss = [NDArrayData.newNormalizedRandom(1,1,1,1,physical_dimension) for _ in range(3)]
        SL, SM, SR = Ss
        system.setStateCenter(SL)
        self.assertAlmostEqual(system.computeNormalization(),1)
        system.absorbCenter(2)
        system.setStateCenter(SM)
        self.assertAlmostEqual(system.computeNormalization(),1)
        system.absorbCenter(2)
        system.setStateCenter(SR)
        self.assertAlmostEqual(system.computeNormalization(),1)

        E1L = OO_L.contractWith(SL.ravel(),(1,),(0,)).contractWith(SL.ravel().conj(),(0,),(0,)).extractScalar()
        E1M = OO_R.contractWith(SM.ravel(),(1,),(0,)).contractWith(SM.ravel().conj(),(0,),(0,)).extractScalar()
        E1 = E1L*E1M

        E2M = OO_L.contractWith(SM.ravel(),(1,),(0,)).contractWith(SM.ravel().conj(),(0,),(0,)).extractScalar()
        E2R = OO_R.contractWith(SR.ravel(),(1,),(0,)).contractWith(SR.ravel().conj(),(0,),(0,)).extractScalar()
        E2 = E2M*E2R

        self.assertAlmostEqual(system.computeExpectation(),E1+E2)
    # }}}
    @with_checker # def test_UD_two_steps_up {{{
    def test_UD_two_steps_up(self,physical_dimension=irange(1,5)):
        OO_UD = [NDArrayData.newRandomHermitian(physical_dimension,physical_dimension) for _ in range(2)]
        OO_U, OO_D = OO_UD
        system = System.newTrivialWithSparseOperator(OO_UD=OO_UD)
        Ss = [NDArrayData.newNormalizedRandom(1,1,1,1,physical_dimension) for _ in range(3)]
        SU, SM, SD = Ss
        system.setStateCenter(SU)
        self.assertAlmostEqual(system.computeNormalization(),1)
        system.absorbCenter(1)
        system.setStateCenter(SM)
        self.assertAlmostEqual(system.computeNormalization(),1)
        system.absorbCenter(1)
        system.setStateCenter(SD)
        self.assertAlmostEqual(system.computeNormalization(),1)

        E1U = OO_U.contractWith(SU.ravel(),(1,),(0,)).contractWith(SU.ravel().conj(),(0,),(0,)).extractScalar()
        E1M = OO_D.contractWith(SM.ravel(),(1,),(0,)).contractWith(SM.ravel().conj(),(0,),(0,)).extractScalar()
        E1 = E1U*E1M

        E2M = OO_U.contractWith(SM.ravel(),(1,),(0,)).contractWith(SM.ravel().conj(),(0,),(0,)).extractScalar()
        E2D = OO_D.contractWith(SD.ravel(),(1,),(0,)).contractWith(SD.ravel().conj(),(0,),(0,)).extractScalar()
        E2 = E2M*E2D

        self.assertAlmostEqual(system.computeExpectation(),E1+E2)
    # }}}
    @with_checker # def test_UD_two_steps_down {{{
    def test_UD_two_steps_down(self,physical_dimension=irange(1,5)):
        OO_UD = [NDArrayData.newRandomHermitian(physical_dimension,physical_dimension) for _ in range(2)]
        OO_U, OO_D = OO_UD
        system = System.newTrivialWithSparseOperator(OO_UD=OO_UD)
        Ss = [NDArrayData.newNormalizedRandom(1,1,1,1,physical_dimension) for _ in range(3)]
        SU, SM, SD = Ss
        system.setStateCenter(SD)
        self.assertAlmostEqual(system.computeNormalization(),1)
        system.absorbCenter(3)
        system.setStateCenter(SM)
        self.assertAlmostEqual(system.computeNormalization(),1)
        system.absorbCenter(3)
        system.setStateCenter(SU)
        self.assertAlmostEqual(system.computeNormalization(),1)

        E1U = OO_U.contractWith(SU.ravel(),(1,),(0,)).contractWith(SU.ravel().conj(),(0,),(0,)).extractScalar()
        E1M = OO_D.contractWith(SM.ravel(),(1,),(0,)).contractWith(SM.ravel().conj(),(0,),(0,)).extractScalar()
        E1 = E1U*E1M

        E2M = OO_U.contractWith(SM.ravel(),(1,),(0,)).contractWith(SM.ravel().conj(),(0,),(0,)).extractScalar()
        E2D = OO_D.contractWith(SD.ravel(),(1,),(0,)).contractWith(SD.ravel().conj(),(0,),(0,)).extractScalar()
        E2 = E2M*E2D

        self.assertAlmostEqual(system.computeExpectation(),E1+E2)
    # }}}

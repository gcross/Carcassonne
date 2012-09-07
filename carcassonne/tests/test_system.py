# Imports {{{
from numpy import array, complex128, isnan
from numpy.linalg import norm

from . import *
from ..system import System
from ..sparse import Identity, Operator
# }}}

class TestSystem(TestCase): # {{{
    @with_checker(number_of_calls=10) # test_increaseBandwidth_one_step # {{{
    def test_increaseBandwidth_one_step(self,direction=irange(0,3),increment=irange(0,4)):
        system = System.newRandom()
        expectation1 = system.computeExpectation()
        normalization1 = system.computeNormalization()
        system.increaseBandwidth(direction,by=increment)
        expectation2 = system.computeExpectation()
        normalization2 = system.computeNormalization()
        self.assertAlmostEqual(expectation1,expectation2)
        self.assertAlmostEqual(normalization1,normalization2)
    # }}}
    @with_checker # test_expectation_of_identity_after_no_steps # {{{
    def test_expectation_of_sum_of_identities_after_no_steps(self):
        self.assertAlmostEqual(System.newRandom(makeOperator=lambda N: NDArrayData.newIdentity(N)).computeExpectation(),1)
    # }}}
    @with_checker # test_expectation_of_identity_after_some_steps # {{{
    def test_expectation_of_sum_of_identities_after_some_steps(self,moves=(irange(0,1),)*4):
        system = System.newRandom(makeOperator=lambda N: NDArrayData.newIdentity(N))
        directions = sum(([i]*moves[i] for i in range(4)),[])
        width = 1
        height = 1
        for direction in directions:
            system.assertDimensionsAreConsistent()
            system.assertNormalizationIsHermitian()
            system.absorbCenter(direction)
            system.assertDimensionsAreConsistent()
            system.assertNormalizationIsHermitian()
            system.increaseBandwidth(direction=direction+1,by=1)
            system.assertDimensionsAreConsistent()
            system.assertNormalizationIsHermitian()
            if direction == 0 or direction == 2:
                width += 1
            else:
                height += 1
        if isnan(system.computeNormalization()):
            return
        self.assertAlmostEqual(system.computeExpectation(),width*height)
    # }}}
    @with_checker # test_formNormalizationMultiplier_same_both_ways {{{
    def test_formNormalizationMultiplier_same_both_ways(self):
        system = System.newRandom()
        random_data = NDArrayData.newRandom(*system.state_center_data.shape)
        m1 = system.formNormalizationMultiplier()(random_data)
        m2 = system.formExpectationAndNormalizationMultipliers()[1](random_data)
        self.assertDataAlmostEqual(m1,m2)
    # }}}
    @with_checker # test_formNormalizationMultiplier_same_asformNormalizationSubmatrix {{{
    def test_formNormalizationMultiplier_same_asformNormalizationSubmatrix(self):
        system = System.newRandom()
        random_data = NDArrayData.newRandom(*system.state_center_data.shape)
        m1 = system.formNormalizationMultiplier()(random_data)
        m2 = system.formNormalizationSubmatrix().contractWith(random_data.join(range(4),4),(1,),(0,)).split(*random_data.shape)
        self.assertDataAlmostEqual(m1,m2)
    # }}}
    @with_checker # test_minimizer_works_after_some_steps {{{
    def dont_test_minimizer_works_after_some_steps(self,moves=(irange(0,1),)*4):
        system = System.newRandom(makeOperator=lambda N: NDArrayData.newDiagonal([1]*(N-1)+[-1]))
        N = system.state_center_data.shape[-1]
        system.minimizeExpectation()
        self.assertDataAlmostEqual(system.state_center_data,NDArrayData.newOuterProduct([1],[1],[1],[1],[0]*(N-1)+[1]))
        directions = sum(([i]*moves[i] for i in range(4)),[])
        for direction in directions:
            system.absorbCenter(direction)
            system.increaseBandwidth(direction=direction+1,by=1)
    # }}}
    @with_checker # test_normalizeCornerAndDenormalizeSide {{{
    def test_normalizeCornerAndDenormalizeSide(self,corner_id=irange(0,3),direction=irange(0,1)):
        system = System.newRandom()
        normalization1 = system.computeNormalization()
        expectation1 = system.computeExpectation()
        system.normalizeCornerAndDenormalizeSide(corner_id,direction)
        system.assertNormalizationIsHermitian()
        normalization2 = system.computeNormalization()
        expectation2 = system.computeExpectation()
        self.assertAlmostEqual(normalization1,normalization2)
        self.assertAlmostEqual(expectation1,expectation2)
    # }}}
    @with_checker # test_normalizeSideAndDenormalizeCenter {{{
    def test_normalizeSideAndDenormalizeCenter(self,side_id=irange(0,3)):
        system = System.newRandom()
        normalization1 = system.computeNormalization()
        expectation1 = system.computeExpectation()
        system.normalizeSideAndDenormalizeCenter(side_id)
        system.assertNormalizationIsHermitian()
        normalization2 = system.computeNormalization()
        expectation2 = system.computeExpectation()
        self.assertAlmostEqual(normalization1,normalization2)
        self.assertAlmostEqual(expectation1,expectation2)
    # }}}
# }}}

class TestSystemSillyFieldWalk(TestCase): # {{{
    @staticmethod
    def makeSillySystem(): # {{{
        corners = [{Identity():NDArrayData.newTrivial((1,1,1,1))}]*4
        sides = [{Identity():NDArrayData.newTrivial((1,1,1,1,1,1))}]*4
        I = O = NDArrayData.newTrivial((1,1))
        operator_center_tensor = {Identity():None,Operator():O}
        state_center_data = NDArrayData.newTrivial((1,)*5)
        return System(corners, sides, state_center_data, operator_center_tensor)
    # }}}
    def test_silly_field_no_steps(self): # {{{
        system = self.makeSillySystem()
        self.assertEqual(system.computeExpectation(),1)
    # }}}
    @with_checker(number_of_calls=10)
    def test_silly_field_single_step(self,direction=irange(0,3)): # {{{
        system = self.makeSillySystem()
        system.absorbCenter(direction)
        self.assertEqual(system.computeExpectation(),2)
        self.assertEqual(system.computeNormalization(),1)
    # }}}
    @with_checker(number_of_calls=40)
    def test_silly_field_double_step(self,direction1=irange(0,3),direction2=irange(0,3)): # {{{
        system = self.makeSillySystem()
        width = 1
        height = 1
        system.absorbCenter(direction1)
        system.absorbCenter(direction2)
        if direction1 == 0 or direction1 == 2:
            width += 1
        else:
            height += 1
        if direction2 == 0 or direction2 == 2:
            width += 1
        else:
            height += 1
        self.assertEqual(system.computeExpectation(),width*height)
        self.assertEqual(system.computeNormalization(),1)
    # }}}
    @with_checker(number_of_calls=10)
    def test_silly_field_random_walk(self,directions=[irange(0,3)]): # {{{
        system = self.makeSillySystem()
        width = 1
        height = 1
        for direction in directions:
            system.absorbCenter(direction)
            if direction == 0 or direction == 2:
                width += 1
            else:
                height += 1
        self.assertEqual(system.computeExpectation(),width*height)
        self.assertEqual(system.computeNormalization(),1)
    # }}}
# }}}

class TestSystemMagneticFieldWalk(TestCase): # {{{
    @staticmethod
    def makeMagneticField(): # {{{
        corners = [{Identity():NDArrayData.newTrivial((1,1,1,1),dtype=complex128)}]*4
        sides = [{Identity():NDArrayData.newTrivial((1,1,1,1,1,1),dtype=complex128)}]*4
        Z = NDArrayData(array([[1,0],[0,-1]],dtype=complex128))
        operator_center_tensor = {Identity():None,Operator():Z}
        state_up = NDArrayData(array([[[[[1,0]]]]],dtype=complex128))
        state_down = NDArrayData(array([[[[[0,1]]]]],dtype=complex128))
        return System(corners, sides, state_up, operator_center_tensor), [state_up, state_down], [1,-1]
    # }}}
    @with_checker(number_of_calls=10)
    def test_magnetic_field_no_steps(self,s1=irange(0,1),s2=irange(0,1)): # {{{
        system, states, spins = self.makeMagneticField()
        system.setStateCenter(states[s1])
        self.assertEqual(system.computeExpectation(),spins[s1])
        self.assertEqual(system.computeNormalization(),1)
        system.setStateCenter(states[s2])
        system.minimizeExpectation()
        self.assertDataAlmostEqual(system.state_center_data,states[1])
    # }}}
    @with_checker(number_of_calls=10)
    def test_magnetic_field_single_step(self,direction=irange(0,3),s1=irange(0,1),s2=irange(0,1)): # {{{
        system, states, spins = self.makeMagneticField()
        system.setStateCenter(states[s1])
        system.absorbCenter(direction)
        system.setStateCenter(states[s2])
        self.assertEqual(system.computeExpectation(),spins[s1]+spins[s2])
        self.assertEqual(system.computeNormalization(),1)
    # }}}
    @with_checker(number_of_calls=10)
    def test_magnetic_field_double_step_same_direction(self,direction=irange(0,3),s1=irange(0,1),s2=irange(0,1),s3=irange(0,1)): # {{{
        system, states, spins = self.makeMagneticField()
        system.setStateCenter(states[s1])
        system.absorbCenter(direction)
        system.setStateCenter(states[s2])
        system.absorbCenter(direction)
        system.setStateCenter(states[s3])
        self.assertEqual(system.computeExpectation(),spins[s1]+spins[s2]+spins[s3])
        self.assertEqual(system.computeNormalization(),1)
    # }}}
    @with_checker(number_of_calls=10)
    def test_magnetic_field_random_walk(self, # {{{
        directions_and_spins=[(irange(0,3),irange(0,1))],
        final_spin=irange(0,1)
    ):
        system, states, spins = self.makeMagneticField()
        UL = 0
        U  = 0
        UR = 0
        L  = 0
        R  = 0
        DL = 0
        D  = 0
        DR = 0
        for direction, spin in directions_and_spins:
            system.setStateCenter(states[spin])
            system.absorbCenter(direction)
            if direction == 0:
                R += spins[spin]
                UR += U
                DR += D
            elif direction == 1:
                U += spins[spin]
                UL += L
                UR += R
            elif direction == 2:
                L += spins[spin]
                UL += U
                DR += D
            elif direction == 3:
                D += spins[spin]
                DL += L
                DR += R
        C = spins[final_spin]
        N = UL+U+UR+L+R+C+DL+D+DR
        system.setStateCenter(states[final_spin])
        self.assertEqual(system.computeExpectation(),N)
        self.assertEqual(system.computeNormalization(),1)
        self.assertDataAlmostEqual(system.formExpectationMultiplier()(states[final_spin]),NDArrayData(N*states[final_spin].toArray()))
        self.assertDataAlmostEqual(system.formNormalizationMultiplier()(states[final_spin]),NDArrayData(states[final_spin].toArray()))
    # }}}
# }}}

# Imports {{{
from numpy import array, complex128

from . import *
from ..expectation import Expectation
from ..sparse import Identity, Operator
# }}}

class TestExpectation(TestCase): # {{{
    @staticmethod
    def makeSillyFieldForASillyState(d=1): # {{{
        corners = [{Identity():NDArrayData.newTrivial((1,1))}]*4
        sides = [{Identity():NDArrayData.newTrivial((1,1,1,1))}]*4
        I = O = NDArrayData.newTrivial((1,1))
        operator_center_tensor = {Identity():None,Operator():O}
        state_center_data = NDArrayData.newTrivial((1,)*5)
        return Expectation(corners, sides, operator_center_tensor), state_center_data
    # }}}
    def test_silly_field_no_steps(self): # {{{
        expectation, state_center_data = self.makeSillyFieldForASillyState()
        self.assertEqual(expectation.computeExpectation(state_center_data),1)
    # }}}
    @with_checker(number_of_calls=10)
    def test_silly_field_single_step(self,direction=irange(0,3)): # {{{
        expectation, state_center_data = self.makeSillyFieldForASillyState()
        expectation.absorbCenter(direction,state_center_data)
        self.assertEqual(expectation.computeExpectation(state_center_data),2)
        self.assertEqual(expectation.computeNormalization(state_center_data),2)
    # }}}
    @with_checker(number_of_calls=40)
    def test_silly_field_double_step(self,direction1=irange(0,3),direction2=irange(0,3)): # {{{
        expectation, state_center_data = self.makeSillyFieldForASillyState()
        width = 1
        height = 1
        expectation.absorbCenter(direction1,state_center_data)
        expectation.absorbCenter(direction2,state_center_data)
        if direction1 == 0 or direction1 == 2:
            width += 1
        else:
            height += 1
        if direction2 == 0 or direction2 == 2:
            width += 1
        else:
            height += 1
        self.assertEqual(expectation.computeExpectation(state_center_data),width*height)
        self.assertEqual(expectation.computeNormalization(state_center_data),1)
    # }}}
    @with_checker(number_of_calls=10)
    def test_silly_field_random_walk(self,directions=[irange(0,3)]): # {{{
        expectation, state_center_data = self.makeSillyFieldForASillyState()
        width = 1
        height = 1
        for direction in directions:
            expectation.absorbCenter(direction,state_center_data)
            if direction == 0 or direction == 2:
                width += 1
            else:
                height += 1
        self.assertEqual(expectation.computeExpectation(state_center_data),width*height)
        self.assertEqual(expectation.computeNormalization(state_center_data),1)
    # }}}
    @staticmethod
    def makeMagneticField(): # {{{
        corners = [{Identity():NDArrayData.newTrivial((1,1),dtype=complex128)}]*4
        sides = [{Identity():NDArrayData.newTrivial((1,1,1,1),dtype=complex128)}]*4
        Z = NDArrayData(array([[1,0],[0,-1]],dtype=complex128))
        operator_center_tensor = {Identity():None,Operator():Z}
        state_up = NDArrayData(array([[[[[1,0]]]]],dtype=complex128))
        state_down = NDArrayData(array([[[[[0,1]]]]],dtype=complex128))
        return Expectation(corners, sides, operator_center_tensor), [state_up, state_down], [1,-1]
    # }}}
    @with_checker(number_of_calls=10)
    def test_silly_field_no_steps(self,s1=irange(0,1),s2=irange(0,1)): # {{{
        expectation, states, spins = self.makeMagneticField()
        self.assertEqual(expectation.computeExpectation(states[s1]),spins[s1])
        self.assertEqual(expectation.computeNormalization(states[s1]),1)
        self.assertDataAlmostEqual(expectation.minimizeStartingWith(states[s2]),states[1])
    # }}}
    @with_checker(number_of_calls=10)
    def test_silly_field_single_step(self,direction=irange(0,3),s1=irange(0,1),s2=irange(0,1)): # {{{
        expectation, states, spins = self.makeMagneticField()
        expectation.absorbCenter(direction,states[s1])
        self.assertEqual(expectation.computeExpectation(states[s2]),spins[s1]+spins[s2])
        self.assertEqual(expectation.computeNormalization(states[s2]),1)
    # }}}
    @with_checker(number_of_calls=10)
    def test_silly_field_double_step_same_direction(self,direction=irange(0,3),s1=irange(0,1),s2=irange(0,1),s3=irange(0,1)): # {{{
        expectation, states, spins = self.makeMagneticField()
        expectation.absorbCenter(direction,states[s1])
        expectation.absorbCenter(direction,states[s2])
        self.assertEqual(expectation.computeExpectation(states[s3]),spins[s1]+spins[s2]+spins[s3])
        self.assertEqual(expectation.computeNormalization(states[s3]),1)
    # }}}
    @with_checker(number_of_calls=10)
    def test_magnetic_field_random_walk(self, # {{{
        directions_and_spins=[(irange(0,3),irange(0,1))],
        final_spin=irange(0,1)
    ):
        expectation, states, spins = self.makeMagneticField()
        UL = 0
        U  = 0
        UR = 0
        L  = 0
        R  = 0
        DL = 0
        D  = 0
        DR = 0
        for direction, spin in directions_and_spins:
            expectation.absorbCenter(direction,states[spin])
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
        self.assertEqual(expectation.computeExpectation(states[final_spin]),N)
        self.assertEqual(expectation.computeNormalization(states[final_spin]),1)
        self.assertDataAlmostEqual(expectation.formExpectationMultiplier()(states[final_spin]),NDArrayData(N*states[final_spin].toArray()))
        self.assertDataAlmostEqual(expectation.formNormalizationMultiplier()(states[final_spin]),NDArrayData(states[final_spin].toArray()))
    # }}}
# }}}

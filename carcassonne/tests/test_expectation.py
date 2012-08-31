# Imports {{{
from numpy import array

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
    def dont_test_silly_field_no_steps(self): # {{{
        expectation, state_center_data = self.makeSillyFieldForASillyState()
        self.assertEqual(expectation.computeExpectation(state_center_data),1)
    # }}}
    def dont_test_silly_field_step_0(self): # {{{
        expectation, state_center_data = self.makeSillyFieldForASillyState()
        expectation.absorbCenter(0,state_center_data)
        self.assertEqual(expectation.computeExpectation(state_center_data),2)
    # }}}
    @with_checker(number_of_calls=10)
    def test_silly_field_single_step(self,direction=irange(0,3)): # {{{
        expectation, state_center_data = self.makeSillyFieldForASillyState()
        expectation.absorbCenter(direction,state_center_data)
        self.assertEqual(expectation.computeExpectation(state_center_data),2)
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
    # }}}
    @staticmethod
    def makeMagneticField(): # {{{
        corners = [{Identity():NDArrayData.newTrivial((1,1))}]*4
        sides = [{Identity():NDArrayData.newTrivial((1,1,1,1))}]*4
        Z = NDArrayData(array([[1,0],[0,-1]]))
        operator_center_tensor = {Identity():None,Operator():Z}
        state_up = NDArrayData(array([[[[[1,0]]]]]))
        state_down = NDArrayData(array([[[[[0,1]]]]]))
        return Expectation(corners, sides, operator_center_tensor), [state_up, state_down], [1,-1]
    # }}}
    @with_checker(number_of_calls=10)
    def test_silly_field_single_step(self,direction=irange(0,3),s1=irange(0,1),s2=irange(0,1)): # {{{
        expectation, states, spins = self.makeMagneticField()
        expectation.absorbCenter(direction,states[s1])
        self.assertEqual(expectation.computeExpectation(states[s2]),spins[s1]+spins[s2])
    # }}}
    @with_checker(number_of_calls=10)
    def test_silly_field_double_step_same_direction(self,direction=irange(0,3),s1=irange(0,1),s2=irange(0,1),s3=irange(0,1)): # {{{
        expectation, states, spins = self.makeMagneticField()
        expectation.absorbCenter(direction,states[s1])
        expectation.absorbCenter(direction,states[s2])
        self.assertEqual(expectation.computeExpectation(states[s3]),spins[s1]+spins[s2]+spins[s3])
    # }}}
# }}}

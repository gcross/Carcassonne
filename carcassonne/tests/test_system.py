# Imports {{{
from numpy import array, complex128

from . import *
from ..system import System
from ..sparse import Identity, Operator
# }}}

class TestSystem(TestCase): # {{{
    @staticmethod # randomSystem # {{{
    def randomInitialSystem(n=2,makeOperator=None):
        spoke_sizes = (randint(1,n),randint(1,n))*2
        sides_data = tuple(NDArrayData.newRandom(*(randint(1,n),)*2+(spoke_sizes[i],)*2) for i in range(4))
        corners_data = tuple(NDArrayData.newRandom(sides_data[(i+1)%4].shape[1],sides_data[i].shape[0]) for i in range(4))
        state_center_data = NDArrayData.newRandom(*spoke_sizes + (randint(1,n),))
        for i in range(4):
            assert sides_data[i].shape[0] == corners_data[i].shape[1]
            assert sides_data[i].shape[1] == corners_data[(i-1)%4].shape[0]
            assert sides_data[i].shape[2] == state_center_data.shape[i]
            assert sides_data[i].shape[3] == state_center_data.shape[i]
        if makeOperator is None:
            O = NDArrayData.newRandom(state_center_data.shape[-1],state_center_data.shape[-1])
        else:
            O = makeOperator(state_center_data.shape[-1])
        operator_center_tensor = {Identity():None,Operator():O}
        return \
            System(
                tuple({Identity():corner_data} for corner_data in corners_data),
                tuple({Identity():side_data} for side_data in sides_data),
                state_center_data,
                operator_center_tensor,
            )
    # }}}
    @with_checker(number_of_calls=10) # test_increaseBandwidth_one_step # {{{
    def test_increaseBandwidth_one_step(self,direction=irange(0,3),increment=irange(0,4)):
        system = self.randomInitialSystem()
        expectation1 = system.computeExpectation()
        normalization1 = system.computeNormalization()
        system.increaseBandwidth(direction,by=increment)
        expectation2 = system.computeExpectation()
        normalization2 = system.computeNormalization()
        self.assertAlmostEqual(expectation1,expectation2)
        self.assertAlmostEqual(normalization1,normalization2)
    # }}}
    @with_checker # test_expectation_of_identity_is_1_after_no_steps # {{{
    def test_expectation_of_sum_of_identities_after_no_steps(self):
        self.assertAlmostEqual(self.randomInitialSystem(makeOperator=lambda N: NDArrayData.newIdentity(N)).computeExpectation(),1)
    # }}}
    @with_checker # test_expectation_of_identity_is_1_after_some_steps # {{{
    def test_expectation_of_sum_of_identities_after_some_steps(self,moves=(irange(0,1),)*4):
        system = self.randomInitialSystem(makeOperator=lambda N: NDArrayData.newIdentity(N))
        directions = sum(([i]*moves[i] for i in range(4)),[])
        width = 1
        height = 1
        for direction in directions:
            system.absorbCenter(direction)
            system.increaseBandwidth(direction=direction+1,by=1)
            if direction == 0 or direction == 2:
                width += 1
            else:
                height += 1
        self.assertAlmostEqual(system.computeExpectation(),width*height)
    # }}}
    @with_checker # test_minimizer_works_after_some_steps {{{
    def dont_test_minimizer_works_after_some_steps(self,moves=(irange(0,1),)*4):
        system = self.randomInitialSystem(makeOperator=lambda N: NDArrayData.newDiagonal([1]*(N-1)+[-1]))
        N = system.state_center_data.shape[-1]
        system.minimizeExpectation()
        self.assertDataAlmostEqual(system.state_center_data,NDArrayData.newOuterProduct([1],[1],[1],[1],[0]*(N-1)+[1]))
        directions = sum(([i]*moves[i] for i in range(4)),[])
        for direction in directions:
            system.absorbCenter(direction)
            system.increaseBandwidth(direction=direction+1,by=1)
    # }}}
# }}}

class TestSystemSillyFieldWalk(TestCase): # {{{
    @staticmethod
    def makeSillySystem(): # {{{
        corners = [{Identity():NDArrayData.newTrivial((1,1))}]*4
        sides = [{Identity():NDArrayData.newTrivial((1,1,1,1))}]*4
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
        corners = [{Identity():NDArrayData.newTrivial((1,1),dtype=complex128)}]*4
        sides = [{Identity():NDArrayData.newTrivial((1,1,1,1),dtype=complex128)}]*4
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

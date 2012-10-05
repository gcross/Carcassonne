# Imports {{{
from numpy import array, complex128, prod, isnan
from numpy.linalg import norm

from . import *
from ..system import *
from ..sparse import *
from ..utils import *
# }}}

class TestSystem(TestCase): # {{{
    def test___add___trivial(self): # test___add___trivial {{{
        O = NDArrayData.newIdentity(1)
        system1 = System.newTrivialWithSparseOperator(O=O)
        system2 = system1 + system1
        self.assertAlmostEqual(system2.computeNormalization(),2)
        self.assertAlmostEqual(system2.computeExpectation(),1)
    # }}}
    @with_checker(number_of_calls=10) # test___add__ {{{
    def test___add___self(self,moves=(irange(0,1),)*4):
        system1 = System.newRandom()
        for direction in sum(([i]*moves[i] for i in range(4)),[]):
            system1.contractTowards(direction)
        system2 = system1 + system1
        try:
            self.assertAlmostEqual(
                system2.computeNormalization(),
                system1.computeNormalization()*2
            )
        except:
            self.assertAlmostEqual(system2.computeNormalization()/system1.computeNormalization(),2)
        try:
            self.assertAlmostEqual(
                system2.computeExpectation(),
                system1.computeExpectation()
            )
        except:
            self.assertAlmostEqual(system2.computeExpectation()/system1.computeExpectation(),1)
    # }}}
    @with_checker(number_of_calls=10) # test_compressCornerStateTowards_down_by_half {{{
    def test_compressCornerStateTowards_down_by_half(self,corner_id=irange(0,3),direction=irange(0,1),normalize=bool):
        system = System.newRandom()
        for i in range(4):
            system.contractTowards(i)
        expectation1, normalization1 = system.computeExpectationAndNormalization()

        corner_data = system.corners[corner_id][Identity()]
        axis = 3*direction
        old_dimension = corner_data.shape[axis]
        new_dimension = 2*(old_dimension+1)
        enlargener, enlargener_conj = NDArrayData.newEnlargener(old_dimension,new_dimension)
        system.corners[corner_id] = mapOverSparseData(lambda data: data.absorbMatrixAt(axis,enlargener).absorbMatrixAt(axis+1,enlargener_conj),system.corners[corner_id])
        side_id = sideFromCorner(corner_id,direction)

        axis = 3-axis
        side_enlargener = enlargener_conj
        side_enlargener_conj = enlargener
        system.sides[side_id] = mapOverSparseData(lambda data: data.absorbMatrixAt(axis,side_enlargener).absorbMatrixAt(axis+1,side_enlargener_conj),system.sides[side_id])

        expectation2, normalization2 = system.computeExpectationAndNormalization()
        self.assertAlmostEqual(normalization2/normalization1,1)
        self.assertAlmostEqual(expectation2/expectation1,1)

        system.compressCornerStateTowards(corner_id,direction,old_dimension,normalize)
        normalization3 = system.computeNormalization()
        expectation3 = system.computeExpectation()
        self.assertAlmostEqual(normalization3/normalization1,1)
        self.assertAlmostEqual(expectation3/expectation1,1)
    # }}}
    @with_checker(number_of_calls=10) # test_compressCornerStateTowards_new_same_as_old {{{
    def test_compressCornerStateTowards_new_same_as_old(self,corner_id=irange(0,3),direction=irange(0,1),normalize=bool):
        system = System.newRandom(maximum_dimension=4)
        expectation1, normalization1 = system.computeExpectationAndNormalization()
        dimension = system.corners[corner_id][Identity()].shape[3*direction]
        system.compressCornerStateTowards(corner_id,direction,dimension,normalize)
        expectation2, normalization2 = system.computeExpectationAndNormalization()
        if isnan(normalization1) or isnan(normalization2) or isnan(expectation1) or isnan(expectation2):
            return
        self.assertAlmostEqual(normalization2/normalization1,1)
        self.assertAlmostEqual(expectation2/expectation1,1)
    # }}}
    @with_checker(number_of_calls=10) # test_compressCornerTwoSiteOperatorTowards_down_by_half {{{
    def test_compressCornerTwoSiteOperatorTowards_down_by_half(self,
        corner_id=irange(0,3),
        direction=irange(0,1),
        physical_dimension=irange(1,4),
        new_dimension=irange(1,10),
    ):
        side_id = sideFromCorner(corner_id,direction)
        side_direction = 1-direction

        operator_data = [NDArrayData.newRandomHermitian(physical_dimension,physical_dimension) for _ in range(2)]
        if side_id in (1,3):
            operator_direction = "OO_LR"
        else:
            operator_direction = "OO_UD"
        system = System.newTrivialWithSparseOperator(**{operator_direction:operator_data})

        if direction == 0:
            system.contractTowards(R(side_id))
        else:
            system.contractTowards(L(side_id))
        for _ in range(new_dimension):
            system.contractTowards(side_id)
        state_center_data = system.state_center_data
        for _ in range(new_dimension*2+1):
            system.contractTowards(side_id)
        system.setStateCenter(state_center_data)

        expectation1, normalization1 = system.computeExpectationAndNormalization()
        system.compressCornerTwoSiteOperatorTowards(corner_id,direction,new_dimension)
        expectation2, normalization2 = system.computeExpectationAndNormalization()

        self.assertAlmostEqual(normalization2,normalization1)
        self.assertAlmostEqual(expectation2,expectation1)
    # }}}
    @with_checker(number_of_calls=10) # test_compressCornerTwoSiteOperatorTowards_new_same_as_old {{{
    def test_compressCornerTwoSiteOperatorTowards_new_same_as_old(self,
        corner_id=irange(0,3),
        direction=irange(0,1),
        physical_dimension=irange(1,4),
        number_of_terms=irange(0,10),
    ):
        side_id = sideFromCorner(corner_id,direction)
        side_direction = 1-direction

        operator_data = [NDArrayData.newRandomHermitian(physical_dimension,physical_dimension) for _ in range(2)]
        if side_id in (1,3):
            operator_direction = "OO_LR"
        else:
            operator_direction = "OO_UD"
        system = System.newTrivialWithSparseOperator(**{operator_direction:operator_data})

        if direction == 0:
            system.contractTowards(R(side_id))
        else:
            system.contractTowards(L(side_id))
        for _ in range(number_of_terms):
            system.contractTowards(side_id)

        number_of_corner_two_site_terms = 0
        for tag, data in system.corners[corner_id].items():
            if isinstance(tag,TwoSiteOperator):
                self.assertEqual(tag.direction,direction)
                number_of_corner_two_site_terms += 1
        self.assertEqual(number_of_corner_two_site_terms,number_of_terms)
        del number_of_corner_two_site_terms

        number_of_side_two_site_terms = 0
        for tag, data in system.sides[side_id].items():
            if isinstance(tag,TwoSiteOperator) and tag.direction != side_direction:
                number_of_side_two_site_terms += 1
        self.assertEqual(number_of_side_two_site_terms,number_of_terms)
        del number_of_side_two_site_terms

        expectation1, normalization1 = system.computeExpectationAndNormalization()
        system.compressCornerTwoSiteOperatorTowards(corner_id,direction,number_of_terms)
        if number_of_terms > 0:
            self.assertTrue(TwoSiteOperatorCompressed(direction) in system.corners[corner_id])
            self.assertTrue(TwoSiteOperatorCompressed(1-direction) in system.sides[side_id])
        expectation2, normalization2 = system.computeExpectationAndNormalization()

        self.assertAlmostEqual(normalization2,normalization1)
        self.assertAlmostEqual(expectation2,expectation1)
    # }}}
    @with_checker # test_compressCornerTwoSiteOperatorTowards_trivial {{{
    def test_compressCornerTwoSiteOperatorTowards_trivial(self,
        corner_id=irange(0,3),
        direction=irange(0,1),
        new_dimension=irange(1,10),
    ):
        system = System.newTrivialWithSparseOperator(O=NDArrayData.newIdentity(1))
        expectation1, normalization1 = system.computeExpectationAndNormalization()
        system.compressCornerTwoSiteOperatorTowards(corner_id,direction,new_dimension)
        expectation2, normalization2 = system.computeExpectationAndNormalization()
        self.assertAlmostEqual(normalization2,normalization1)
        self.assertAlmostEqual(expectation2,expectation1)
    # }}}
    @with_checker # test_computeCenterSiteExpectation {{{
    def test_computeCenterSiteExpectation(self,physical_dimension=irange(1,4),moves=(irange(0,1),)*4):
        O = NDArrayData.newRandomHermitian(physical_dimension,physical_dimension)
        system = System.newRandom(O=O)

        directions = sum([[i]*n for i,n in enumerate(moves)],[])
        shuffle(directions)
        for direction in directions:
            system.contractTowards(direction)
            system.setStateCenter(NDArrayData.newRandom(*system.state_center_data.shape))

        state_center_data = system.formNormalizationMultiplier()(system.state_center_data)
        state_center_data_conj = system.state_center_data_conj

        self.assertAlmostEqual(
            system.computeCenterSiteExpectation(),
            state_center_data_conj.ravel().contractWith(state_center_data.absorbMatrixAt(4,O).ravel(),(0,),(0,)).extractScalar()/state_center_data_conj.ravel().contractWith(state_center_data.ravel(),(0,),(0,)).extractScalar()
        )
    # }}}
    @with_checker(number_of_calls=10) # test_contractTowardsAndNormalize {{{
    def test_contractTowardsAndNormalize(self,direction=irange(0,3),moves=(irange(0,2),)*4):
        system1 = System.newRandom()
        directions = sum(([i]*moves[i] for i in range(4)),[])
        shuffle(directions)
        for direction in directions:
            system1.contractTowards(direction)
        system2 = copy(system1)

        system1.contractTowards(direction)
        system2.contractNormalizedTowards(direction)

        expectation1, normalization1 = system1.computeExpectationAndNormalization()
        expectation2, normalization2 = system2.computeExpectationAndNormalization()

        self.assertAlmostEqual(normalization2/normalization1,1.0)
        self.assertAlmostEqual(expectation2/expectation1,1.0)
    # }}}
    @with_checker # test_expectation_of_identity_after_no_steps # {{{
    def test_expectation_of_sum_of_identities_after_no_steps(self):
        self.assertAlmostEqual(System.newRandom(makeOperator=lambda N: NDArrayData.newIdentity(N)).computeExpectation(),1)
    # }}}
    @with_checker(number_of_calls=10) # test_expectation_of_identity_after_some_steps # {{{
    def test_expectation_of_sum_of_identities_after_some_steps(self,moves=(irange(0,1),)*4):
        system = System.newRandom(makeOperator=lambda N: NDArrayData.newIdentity(N))
        directions = sum(([i]*moves[i] for i in range(4)),[])
        width = 1
        height = 1
        for direction in directions:
            system.assertDimensionsAreConsistent()
            system.assertNormalizationIsHermitian()
            system.contractTowards(direction)
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
    @with_checker(number_of_calls=10) # test_formExpectationMatrix {{{
    def test_formExpectationMatrix(self,moves=(irange(0,2),)*4):
        system = System.newRandom()
        directions = sum(([i]*moves[i] for i in range(4)),[])
        shuffle(directions)
        for direction in directions:
            system.contractTowards(direction)

        multiply = system.formExpectationMultiplier()
        matrix = system.formExpectationMatrix()

        state_center_data = system.state_center_data.newRandom(*system.state_center_data.shape)
        result1 = multiply(state_center_data).ravel()
        result2 = state_center_data.ravel().absorbMatrixAt(0,matrix)

        self.assertDataAlmostEqual(result1,result2)
    # }}}
    @with_checker(number_of_calls=10) # test_formExpectationMultiplier {{{
    def test_formExpectationAndNormalizationMultipliers_have_correct_shape(self,moves=(irange(0,2),)*4):
        system = System.newRandom()

        expectation_mutiplier, normalization_multiplier = system.formExpectationAndNormalizationMultipliers()

        self.assertEqual(expectation_mutiplier.shape,(system.state_center_data.size(),)*2)
        self.assertEqual(normalization_multiplier.shape,(system.state_center_data.size(),)*2)
    # }}}
    @with_checker # test_formNormalizationMatrix {{{
    def test_formNormalizationMatrix(self):
        system = System.newRandom()
        multiply = system.formNormalizationMultiplier()
        matrix = system.formNormalizationMatrix()

        state_center_data = system.state_center_data.newRandom(*system.state_center_data.shape)
        result1 = multiply(state_center_data).ravel()
        result2 = state_center_data.ravel().absorbMatrixAt(0,matrix)

        self.assertDataAlmostEqual(result1,result2)
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
    @with_checker(number_of_calls=10) # test_increaseBandwidth# {{{
    def test_increaseBandwidth(self,direction=irange(0,1),physical_dimension=irange(3,4)):
        system1 = System.newRandom()
        system1.minimizeExpectation()

        old_shape = system1.state_center_data.shape
        increment = randint(0,maximumBandwidthIncrement(direction,old_shape))
        system2 = copy(system1)

        system1.increaseBandwidth(direction,by=increment)
        system2.contractTowards(O(direction))
        system2.contractTowards(direction)
        self.assertAlmostEqual(system1.computeNormalization(),system2.computeNormalization())
        self.assertAlmostEqual(system1.computeExpectation(),system2.computeExpectation())
    # }}}
    @with_checker(number_of_calls=10) # test_increaseBandwidth_trivial # {{{
    def test_increaseBandwidth_trivial(self,direction=irange(0,1),physical_dimension=irange(2,8)):
        system1 = System.newTrivialWithSparseOperator(NDArrayData.newRandomHermitian(physical_dimension,physical_dimension))
        system1.minimizeExpectation()
        system2 = copy(system1)

        system1.increaseBandwidth(direction,by=1)
        system2.contractTowards(O(direction))
        system2.contractTowards(direction)
        self.assertAlmostEqual(system1.computeNormalization(),system2.computeNormalization())
        self.assertAlmostEqual(system1.computeExpectation(),system2.computeExpectation())

        self.assertAlmostEqual(system1.computeNormalizationMatrixConditionNumber(),1)
    # }}}
    @with_checker # test_minimizer_works_after_some_steps {{{
    def dont_test_minimizer_works_after_some_steps(self,moves=(irange(0,1),)*4):
        system = System.newRandom(makeOperator=lambda N: NDArrayData.newDiagonal([1]*(N-1)+[-1]))
        N = system.state_center_data.shape[-1]
        system.minimizeExpectation()
        self.assertDataAlmostEqual(system.state_center_data,NDArrayData.newOuterProduct([1],[1],[1],[1],[0]*(N-1)+[1]))
        directions = sum(([i]*moves[i] for i in range(4)),[])
        for direction in directions:
            system.contractTowards(direction)
            system.increaseBandwidth(direction=direction+1,by=1)
    # }}}
    @with_checker # test_normalizeCenterAndDenormalizeSide {{{
    def test_normalizeCenterAndDenormalizeSide(self,direction=irange(0,3)):
        system = System.newRandom()
        expectation1, normalization1 = system.computeExpectationAndNormalization()
        system.normalizeCenterAndDenormalizeSide(direction)
        system.assertNormalizationIsHermitian()
        expectation2, normalization2 = system.computeExpectationAndNormalization()
        self.assertAlmostEqual(normalization1,normalization2)
        self.assertAlmostEqual(expectation1,expectation2)
    # }}}
    @with_checker(number_of_calls=10) # test_normalizeCornerAndDenormalizeSide {{{
    def test_normalizeCornerAndDenormalizeSide(self,corner_id=irange(0,3),direction=irange(0,1)):
        system = System.newRandom()
        expectation1, normalization1 = system.computeExpectationAndNormalization()
        system.normalizeCornerAndDenormalizeSide(corner_id,direction)
        system.assertNormalizationIsHermitian()
        expectation2, normalization2 = system.computeExpectationAndNormalization()
        self.assertAlmostEqual(normalization1,normalization2)
        self.assertAlmostEqual(expectation1,expectation2)
    # }}}
    @with_checker # test_normalizeSideAndDenormalizeCenter {{{
    def test_normalizeSideAndDenormalizeCenter(self,side_id=irange(0,3)):
        system = System.newRandom()
        expectation1, normalization1 = system.computeExpectationAndNormalization()
        system.normalizeSideAndDenormalizeCenter(side_id)
        system.assertNormalizationIsHermitian()
        expectation2, normalization2 = system.computeExpectationAndNormalization()
        self.assertAlmostEqual(normalization1,normalization2)
        self.assertAlmostEqual(expectation1,expectation2)
    # }}}
# }}}

class TestSystemSillyFieldWalk(TestCase): # {{{
    @staticmethod # def makeSillySystem # {{{
    def makeSillySystem():
        system = System.newTrivialWithSparseOperator(O=NDArrayData.newIdentity(1))
        system.assertDimensionsAreConsistent()
        return system
    # }}}
    def test_silly_field_no_steps(self): # {{{
        system = self.makeSillySystem()
        self.assertEqual(system.computeExpectation(),1)
    # }}}
    @with_checker(number_of_calls=10) # test_silly_field_single_step {{{
    def test_silly_field_single_step(self,direction=irange(0,3)):
        system = self.makeSillySystem()
        system.contractTowards(direction)
        self.assertEqual(system.computeExpectation(),2)
        self.assertEqual(system.computeNormalization(),1)
    # }}}
    @with_checker(number_of_calls=40) # test_silly_field_double_step {{{
    def test_silly_field_double_step(self,direction1=irange(0,3),direction2=irange(0,3)):
        system = self.makeSillySystem()
        width = 1
        height = 1
        system.contractTowards(direction1)
        system.contractTowards(direction2)
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
    @with_checker(number_of_calls=10) # test_silly_field_random_walk {{{
    def test_silly_field_random_walk(self,directions=[irange(0,3)]):
        system = self.makeSillySystem()
        width = 1
        height = 1
        for direction in directions:
            system.contractTowards(direction)
            if direction == 0 or direction == 2:
                width += 1
            else:
                height += 1
        self.assertEqual(system.computeExpectation(),width*height)
        self.assertEqual(system.computeNormalization(),1)
    # }}}
# }}}

class TestSystemMagneticFieldWalk(TestCase): # {{{
    @staticmethod # def makeMagneticField # {{{
    def makeMagneticField():
        system = System.newTrivialWithSparseOperator(O=NDArrayData.Z)
        system.assertDimensionsAreConsistent()
        state_up = NDArrayData(array([[[[[1,0]]]]],dtype=complex128))
        state_down = NDArrayData(array([[[[[0,1]]]]],dtype=complex128))
        return system, (state_up, state_down), (1,-1)
    # }}}
    @with_checker(number_of_calls=10) # test_magnetic_field_no_steps {{{
    def test_magnetic_field_no_steps(self,s1=irange(0,1),s2=irange(0,1)):
        system, states, spins = self.makeMagneticField()
        system.setStateCenter(states[s1])
        self.assertEqual(system.computeExpectation(),spins[s1])
        self.assertEqual(system.computeNormalization(),1)
        system.setStateCenter(states[s2])
        system.minimizeExpectation()
        self.assertDataAlmostEqual(system.state_center_data,states[1])
    # }}}
    @with_checker(number_of_calls=10) # test_magnetic_field_single_step {{{
    def test_magnetic_field_single_step(self,direction=irange(0,3),s1=irange(0,1),s2=irange(0,1)):
        system, states, spins = self.makeMagneticField()
        system.setStateCenter(states[s1])
        system.contractTowards(direction)
        system.setStateCenter(states[s2])
        self.assertEqual(system.computeExpectation(),spins[s1]+spins[s2])
        self.assertEqual(system.computeNormalization(),1)
    # }}}
    @with_checker(number_of_calls=10) # test_magnetic_field_double_step_same_direction {{{
    def test_magnetic_field_double_step_same_direction(self,direction=irange(0,3),s1=irange(0,1),s2=irange(0,1),s3=irange(0,1)):
        system, states, spins = self.makeMagneticField()
        system.setStateCenter(states[s1])
        system.contractTowards(direction)
        system.setStateCenter(states[s2])
        system.contractTowards(direction)
        system.setStateCenter(states[s3])
        self.assertEqual(system.computeExpectation(),spins[s1]+spins[s2]+spins[s3])
        self.assertEqual(system.computeNormalization(),1)
    # }}}
    @with_checker(number_of_calls=10) # test_magnetic_field_random_walk {{{
    def test_magnetic_field_random_walk(self,
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
            system.contractTowards(direction)
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

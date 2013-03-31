# Imports {{{
from functools import partial
import itertools
from numpy import complex128, prod

from .dense import *
from ...sparse import Identity, Complete, OneSiteOperator, TwoSiteOperator, TwoSiteOperatorCompressed, addStandardCompleteAndIdentityTerms, contractSparseTensors, formSparseContractor, getInformationFromOperatorCenter
from ...utils import Multiplier, multiplyBySingleSiteOperator, L, R, O
# }}}

# Functions {{{
def absorbSparseSideIntoCornerFromLeft(corner,side): # {{{
    terms = {}
    absorb = absorbDenseSideIntoCornerFromLeft
    addStandardCompleteAndIdentityTerms(terms,absorb)
    terms.update({
        (TwoSiteOperator,Identity): lambda r,l: (r.matchesSideIdentityOnLeft(),absorb),
        (Identity,TwoSiteOperator): lambda r,l: (l.matchesCornerIdentityOnRight(),absorb),
        (TwoSiteOperator,TwoSiteOperator): lambda r,l: (l.matches(r),absorb),
        (TwoSiteOperatorCompressed,TwoSiteOperatorCompressed): lambda r,l: (l.matches(r),absorb),
    })
    return contractSparseTensors(terms,corner,side)
# }}}
def absorbSparseSideIntoCornerFromRight(corner,side): # {{{
    terms = {}
    absorb = absorbDenseSideIntoCornerFromRight
    addStandardCompleteAndIdentityTerms(terms,absorb)
    terms.update({
        (TwoSiteOperator,Identity): lambda l,r: (l.matchesSideIdentityOnRight(),absorb),
        (Identity,TwoSiteOperator): lambda l,r: (r.matchesCornerIdentityOnLeft(),absorb),
        (TwoSiteOperator,TwoSiteOperator): lambda l,r: (l.matches(r),absorb),
        (TwoSiteOperatorCompressed,TwoSiteOperatorCompressed): lambda r,l: (l.matches(r),absorb),
    })
    return contractSparseTensors(terms,corner,side)
# }}}
def absorbSparseCenterSOSIntoSide(direction,side,state_center_data,operator_center_data,state_center_data_conj=None): # {{{
    if state_center_data_conj is None:
        state_center_data_conj = state_center_data.conj()
    def contractSS(side_data,_):
        return absorbDenseCenterSSIntoSide(direction,side_data,state_center_data,state_center_data_conj)
    def contractSOS(side_data,operator_center_data_data):
        return absorbDenseCenterSOSIntoSide(direction,side_data,state_center_data,operator_center_data_data,state_center_data_conj)
    terms = {
        # Complete/Identity terms
        (Identity,Identity): lambda s,c: (Identity(),contractSS),
        (Complete,Identity): lambda s,c: (Complete(),contractSS),

        # OneSiteOperator term
        (Identity,OneSiteOperator): lambda s,c: (Complete(),contractSOS),

        # TwoSiteOperator terms
        (TwoSiteOperator,Identity): lambda s,c: (s.matchesCenterIdentity(),contractSS),
        (Identity,TwoSiteOperator): lambda s,c: (c.matchesSideIdentityOutward(direction),contractSOS),
        (TwoSiteOperator,TwoSiteOperator): lambda s,c: (s.matchesCenter(direction,c),contractSOS),
        (TwoSiteOperatorCompressed,Identity): lambda s,c: (s,contractSS)
    }
    return contractSparseTensors(terms,side,operator_center_data)
# }}}
def formExpectationAndNormalizationMultipliers(corners,sides,operator_center_data): # {{{
    return formExpectationStage3(
        formExpectationStage2(
            formExpectationStage1(corners[0],sides[0]),
            formExpectationStage1(corners[1],sides[1]),
        ),
        formExpectationStage2(
            formExpectationStage1(corners[2],sides[2]),
            formExpectationStage1(corners[3],sides[3]),
        ),
        operator_center_data
    )
# }}}
def formExpectationStage1(corner,side): # {{{
    terms = {}
    formStage = formNormalizationStage1
    addStandardCompleteAndIdentityTerms(terms,formStage)
    terms.update({
        (TwoSiteOperator,Identity): lambda l,r: (l.matchesSideIdentityOnRightForStage1(),formStage),
        (Identity,TwoSiteOperator): lambda l,r: (r.matchesCornerIdentityOnLeftForStage1(),formStage),
        (TwoSiteOperator,TwoSiteOperator): lambda l,r: (l.matches(r),formStage),
        (TwoSiteOperatorCompressed,Identity): lambda l,r: (l.matchesSideIdentityOnRightForStage1(),formStage),
        (Identity,TwoSiteOperatorCompressed): lambda l,r: (r.matchesCornerIdentityOnLeftForStage1(),formStage),
        (TwoSiteOperatorCompressed,TwoSiteOperatorCompressed): lambda l,r: (l.matches(r),formStage),
    })
    return contractSparseTensors(terms,corner,side)
# }}}
def formExpectationStage2(right,left): # {{{
    terms = {}
    formStage = formNormalizationStage2
    addStandardCompleteAndIdentityTerms(terms,formStage)
    terms.update({
        (TwoSiteOperator,Identity): lambda r,l: (r.matchesStage1IdentityOnLeft(),formStage),
        (Identity,TwoSiteOperator): lambda r,l: (l.matchesStage1IdentityOnRight(),formStage),
        (TwoSiteOperator,TwoSiteOperator): lambda r,l: (l.matches(r),formStage),
        (TwoSiteOperatorCompressed,Identity): lambda r,l: (r.matchesStage1IdentityOnLeft(),formStage),
        (Identity,TwoSiteOperatorCompressed): lambda r,l: (l.matchesStage1IdentityOnRight(),formStage),
        (TwoSiteOperatorCompressed,TwoSiteOperatorCompressed): lambda r,l: (l.matches(r),formStage),
    })
    return contractSparseTensors(terms,right,left)
# }}}
def formExpectationStage3(stage2_0,stage2_1,operator_center): # {{{
    rules = {
        # Complete and Identity rules
        (Complete,Identity,Identity): lambda x,y,z: True,
        (Identity,Complete,Identity): lambda x,y,z: True,

        # OneSiteOperator rules
        (Identity,Identity,OneSiteOperator): lambda x,y,z: True,

        # TwoSiteOperator rules
        (TwoSiteOperator,Identity,TwoSiteOperator): lambda x,y,z: x.matchesCenterForStage3(0,z),
        (Identity,TwoSiteOperator,TwoSiteOperator): lambda x,y,z: y.matchesCenterForStage3(1,z),
        (TwoSiteOperator,TwoSiteOperator,Identity): lambda x,y,z: x.matchesForStage3(y),
        (TwoSiteOperatorCompressed,TwoSiteOperatorCompressed,Identity): lambda x,y,z: x.matchesForStage3(y),
    }

    multipliers = []

    tensors = (stage2_0,stage2_1,operator_center)
    for tags in itertools.product(*tensors):
        types = tuple(map(type,tags))
        if not rules.get(types,lambda x,y,z: False)(*tags):
            continue
        if tags[-1] == Identity():
            formStage3 = formNormalizationStage3
        else:
            formStage3 = formDenseStage3
        multipliers.append(formStage3(*(tensor[tag] for tensor, tag in zip(tensors,tags))))
        
    def multiplyExpectation(center):
        result = center.newZeros(center.shape,dtype=center.dtype)
        for multiplier in multipliers:
            result += multiplier(center)
        return result

    bandwidth_dimensions = (
        stage2_0[Identity()].shape[2],
        stage2_0[Identity()].shape[3],
        stage2_1[Identity()].shape[2],
        stage2_1[Identity()].shape[3],
    )
    bandwidth_dimension = prod(bandwidth_dimensions)
    physical_dimension, dtype, DataClass = getInformationFromOperatorCenter(operator_center)
    dimension = bandwidth_dimension*physical_dimension
    def formExpectationMatrix():
        matrix = DataClass.newZeros((dimension,dimension),dtype=complex128)
        for multiplier in multipliers:
            matrix += multiplier.formMatrix()
        return matrix

    normalization_multiplier = formNormalizationStage3(stage2_0[Identity()],stage2_1[Identity()],DataClass.newIdentity(physical_dimension))

    expectation_multiplier = Multiplier(
        normalization_multiplier.shape,
        multiplyExpectation,
        sum((multiplier.cost_of_multiply for multiplier in multipliers),0),
        formExpectationMatrix,
        sum((multiplier.cost_of_formMatrix for multiplier in multipliers),0),
    )

    return expectation_multiplier, normalization_multiplier
# }}}
# }}}

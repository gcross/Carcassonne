# Imports {{{
from functools import partial
import itertools

from .dense import *
from ..sparse import Identity, Complete, OneSiteOperator, TwoSiteOperator, addStandardCompleteAndIdentityTerms, contractSparseTensors, formSparseContractor
from ..utils import multiplyBySingleSiteOperator, L, R, O
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
    })
    return contractSparseTensors(terms,corner,side)
# }}}
def absorbSparseCenterSOSIntoSide(direction,side,center_state,center_operator,center_state_conj=None): # {{{
    if center_state_conj is None:
        center_state_conj = center_state.conj()
    def contractSS(side_data,_):
        return absorbDenseCenterSSIntoSide(direction,side_data,center_state,center_state_conj)
    def contractSOS(side_data,center_operator_data):
        return absorbDenseCenterSOSIntoSide(direction,side_data,center_state,center_operator_data,center_state_conj)
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
    }
    return contractSparseTensors(terms,side,center_operator)
# }}}
def formExpectationAndNormalizationMultipliers(corners,sides,center_operator): # {{{
    return formExpectationStage3(
        formExpectationStage2(
            formExpectationStage1(corners[0],sides[0]),
            formExpectationStage1(corners[1],sides[1]),
        ),
        formExpectationStage2(
            formExpectationStage1(corners[2],sides[2]),
            formExpectationStage1(corners[3],sides[3]),
        ),
        center_operator
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
    })
    return contractSparseTensors(terms,corner,side)
# }}}
def formExpectationStage2(corner,side): # {{{
    terms = {}
    formStage = formNormalizationStage2
    addStandardCompleteAndIdentityTerms(terms,formStage)
    terms.update({
        (TwoSiteOperator,Identity): lambda r,l: (r.matchesStage1IdentityOnLeft(),formStage),
        (Identity,TwoSiteOperator): lambda r,l: (l.matchesStage1IdentityOnRight(),formStage),
        (TwoSiteOperator,TwoSiteOperator): lambda r,l: (l.matches(r),formStage),
    })
    return contractSparseTensors(terms,corner,side)
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
    }

    def makeMultiplier(stage2_0_data,stage2_1_data,operator_center_data):
        dense_multiplier = formNormalizationStage3(stage2_0_data,stage2_1_data)
        if operator_center_data is None:
            return dense_multiplier
        else:
            def multiplier(state_center_data):
                return dense_multiplier(multiplyBySingleSiteOperator(state_center_data,operator_center_data))
            return multiplier
    multipliers = []

    for tags in itertools.product(stage2_0,stage2_1,operator_center):
        types = tuple(type(tag) for tag in tags)
        if not rules.get(types,lambda x,y,z: False)(*tags):
            continue
        multipliers.append(makeMultiplier(stage2_0[tags[0]],stage2_1[tags[1]],operator_center[tags[2]]))
        
    def multiplyExpectation(center):
        result = center.newZeros(center.shape,dtype=center.dtype)
        for multiplier in multipliers:
            result += multiplier(center)
        return result

    multiplyNormalization = formNormalizationStage3(stage2_0[Identity()],stage2_1[Identity()])

    return multiplyExpectation, multiplyNormalization
# }}}
# }}}

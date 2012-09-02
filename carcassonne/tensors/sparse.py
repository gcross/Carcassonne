# Imports {{{
from functools import partial

from .dense import *
from ..sparse import Identity, Complete, Operator, contractSparseTensors, formSparseContractor
from ..utils import multiplyBySingleSiteOperator
# }}}

# Functions {{{
# def absorbSparseSideIntoCornerFromLeft {{{
absorbSparseSideIntoCornerFromLeft = \
    formSparseContractor({
            (Identity,Identity): lambda x,y: (Identity(),absorbDenseSideIntoCornerFromLeft),
            (Complete,Identity): lambda x,y: (Complete(),absorbDenseSideIntoCornerFromLeft),
            (Identity,Complete): lambda x,y: (Complete(),absorbDenseSideIntoCornerFromLeft),
    })
# }}}
# def absorbSparseSideIntoCornerFromRight {{{
absorbSparseSideIntoCornerFromRight = \
    formSparseContractor({
            (Identity,Identity): lambda x,y: (Identity(),absorbDenseSideIntoCornerFromRight),
            (Complete,Identity): lambda x,y: (Complete(),absorbDenseSideIntoCornerFromRight),
            (Identity,Complete): lambda x,y: (Complete(),absorbDenseSideIntoCornerFromRight),
    })
# }}}
# def absorbSparseCenterSOSIntoSide {{{
def absorbSparseCenterSOSIntoSide(direction,side,center_state,center_operator,center_state_conj=None):
    if center_state_conj is None:
        center_state_conj = center_state.conj()
    def contractSS(side_data,_):
        return absorbDenseCenterSSIntoSide(direction,side_data,center_state,center_state_conj)
    def contractSOS(side_data,center_operator_data):
        return absorbDenseCenterSOSIntoSide(direction,side_data,center_state,center_operator_data,center_state_conj)
    return contractSparseTensors({
            (Identity,Identity): lambda x,y: (Identity(),contractSS),
            (Complete,Identity): lambda x,y: (Complete(),contractSS),
            (Identity,Operator): lambda x,y: (Complete(),contractSOS),
    },side,center_operator)
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
# def formExpectationStage1 {{{
formExpectationStage1 = \
    formSparseContractor({
            (Identity,Identity): lambda x,y: (Identity(),formNormalizationStage1),
            (Complete,Identity): lambda x,y: (Complete(),formNormalizationStage1),
            (Identity,Complete): lambda x,y: (Complete(),formNormalizationStage1),
    })
# }}}
# def formExpectationStage2 {{{
formExpectationStage2 = \
    formSparseContractor({
            (Identity,Identity): lambda x,y: (Identity(),formNormalizationStage2),
            (Complete,Identity): lambda x,y: (Complete(),formNormalizationStage2),
            (Identity,Complete): lambda x,y: (Complete(),formNormalizationStage2),
    })
# }}}
def formExpectationStage3(stage2_0,stage2_1,center_operator): # {{{
    complete_multipliers = [
        formNormalizationStage3(stage2_0[tag_0],stage2_1[tag_1])
        for tag_0, tag_1 in ((Complete(),Identity()),(Identity(),Complete()))
        if tag_0 in stage2_0 and tag_1 in stage2_1
    ]
    identity_multiplier = formNormalizationStage3(stage2_0[Identity()],stage2_1[Identity()])
    multipliers = []
    for tag in center_operator:
        if isinstance(tag,Identity):
            multipliers += complete_multipliers
        elif isinstance(tag,Operator):
            def operator_multiplier(operator,center):
                return identity_multiplier(multiplyBySingleSiteOperator(center,operator))
            multipliers.append(partial(operator_multiplier,center_operator[tag]))
    def multiplyExpectation(center):
        result = center.newZeros(center.shape)
        for multiplier in multipliers:
            result += multiplier(center)
        return result
    def multiplyNormalization(center):
        return identity_multiplier(center)
    return multiplyExpectation, multiplyNormalization
# }}}
# }}}

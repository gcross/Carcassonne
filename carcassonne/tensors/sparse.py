# Imports {{{
from functools import partial
import itertools

from .dense import *
from ..sparse import Identity, Complete, OneSiteOperator, TwoSiteOperator, contractSparseTensors, formSparseContractor
from ..utils import multiplyBySingleSiteOperator, L, R, O
# }}}

# Functions {{{
# def absorbSparseSideIntoCornerFromLeft(corner,side) {{{
absorbSparseSideIntoCornerFromLeft = formSparseContractor({
    (Identity,Identity): lambda r,l: (Identity(),absorbDenseSideIntoCornerFromLeft),
    (Complete,Identity): lambda r,l: (Complete(),absorbDenseSideIntoCornerFromLeft),
    (Identity,Complete): lambda r,l: (Complete(),absorbDenseSideIntoCornerFromLeft),
    (TwoSiteOperator,Identity): lambda r,l:
        (r.moveOut(),absorbDenseSideIntoCornerFromLeft)
            if r.direction == 1 else None,
    (Identity,TwoSiteOperator): lambda r,l:
        ({
            0:l,
            1:None,
            2:TwoSiteOperator(1,0)
         }[l.direction],
         absorbDenseSideIntoCornerFromLeft
        ),
    (TwoSiteOperator,TwoSiteOperator): lambda r,l:
        (Complete(),absorbDenseSideIntoCornerFromLeft)
            if l.direction == 1 and r.direction == 0 and l.position == r.position else None,
})
# }}}
# def absorbSparseSideIntoCornerFromRight(corner,side) {{{
absorbSparseSideIntoCornerFromRight = formSparseContractor({
    (Identity,Identity): lambda l,r: (Identity(),absorbDenseSideIntoCornerFromRight),
    (Complete,Identity): lambda l,r: (Complete(),absorbDenseSideIntoCornerFromRight),
    (Identity,Complete): lambda l,r: (Complete(),absorbDenseSideIntoCornerFromRight),
    (TwoSiteOperator,Identity): lambda l,r:
        (l.moveOut(),absorbDenseSideIntoCornerFromRight)
            if l.direction == 0 else None,
    (Identity,TwoSiteOperator): lambda l,r:
        ({
            0:None,
            1:r,
            2:TwoSiteOperator(0,0)
         }[r.direction],
         absorbDenseSideIntoCornerFromRight
        ),
    (TwoSiteOperator,TwoSiteOperator): lambda l,r:
        (Complete(),absorbDenseSideIntoCornerFromRight)
            if l.direction == 1 and r.direction == 0 and l.position == r.position else None,
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
    identity_tags = {
        L(direction): TwoSiteOperator(0,0),
        R(direction): TwoSiteOperator(1,0),
        O(direction): TwoSiteOperator(2),
        direction: None
    }
    return contractSparseTensors({
            (Identity,Identity): lambda s,c: (Identity(),contractSS),
            (Complete,Identity): lambda s,c: (Complete(),contractSS),
            (Identity,OneSiteOperator): lambda s,c: (Complete(),contractSOS),
            (TwoSiteOperator,Identity): lambda s,c: (s.moveOut(),contractSS) if s.direction != 2 else None,
            (Identity,TwoSiteOperator): lambda s,c: (identity_tags[c.direction],contractSOS),
            (TwoSiteOperator,TwoSiteOperator): lambda s,c: (Complete(),contractSOS) if s.direction == 2 and c.direction == direction else None
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
# def formExpectationStage1(corner,side) {{{
formExpectationStage1 = formSparseContractor({
    (Identity,Identity): lambda l,r: (Identity(),formNormalizationStage1),
    (Complete,Identity): lambda l,r: (Complete(),formNormalizationStage1),
    (Identity,Complete): lambda l,r: (Complete(),formNormalizationStage1),
    (TwoSiteOperator,Identity): lambda l,r: (l,formNormalizationStage1) if l.direction == 0 else None,
    (Identity,TwoSiteOperator): lambda l,r: (r,formNormalizationStage1) if r.direction != 0 else None,
    (TwoSiteOperator,TwoSiteOperator): lambda l,r:
        (Complete(),formNormalizationStage1)
            if l.direction == 1 and r.direction == 0 and l.position == r.position else None,
})
# }}}
# def formExpectationStage2(stage1_0,stage1_1) {{{
formExpectationStage2 = formSparseContractor({
    (Identity,Identity): lambda r,l: (Identity(),formNormalizationStage2),
    (Complete,Identity): lambda r,l: (Complete(),formNormalizationStage2),
    (Identity,Complete): lambda r,l: (Complete(),formNormalizationStage2),
    (TwoSiteOperator,Identity): lambda r,l:
        ({
            0:None,
            1:r,
            2:TwoSiteOperator(2,0)
         }[r.direction],
         formNormalizationStage2
        ),
    (Identity,TwoSiteOperator): lambda r,l:
        ({
            0:l,
            1:None,
            2:TwoSiteOperator(2,1)
         }[l.direction],
         formNormalizationStage2
        ),
    (TwoSiteOperator,TwoSiteOperator): lambda r,l:
        (Complete(),formNormalizationStage2)
            if l.direction == 1 and r.direction == 0 and l.position == r.position else None,
})
# }}}
def formExpectationStage3(stage2_0,stage2_1,operator_center): # {{{
    rules = {
        (Complete,Identity,Identity): lambda x,y,z: True,
        (Identity,Complete,Identity): lambda x,y,z: True,
        (Identity,Identity,OneSiteOperator): lambda x,y,z: True,
        (TwoSiteOperator,Identity,TwoSiteOperator): lambda x,y,z: x.direction == 2 and x.position == z.direction,
        (Identity,TwoSiteOperator,TwoSiteOperator): lambda x,y,z: y.direction == 2 and y.position+2 == z.direction,
        (TwoSiteOperator,TwoSiteOperator,Identity): lambda x,y,z: (x.direction,y.direction) in ((0,1),(1,0)) and x.position == y.position,
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

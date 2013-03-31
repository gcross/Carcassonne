# Imports {{{
from functools import partial
from numpy import prod

from ...data.cost_tracker import CostTracker, computeCostOfContracting
from ...utils import Join, Multiplier, formDataContractor, prepend, prependDataContractor, L, R, O
# }}}

# Functions {{{
# def absorbDenseSideIntoCornerFromLeft(corner,side) # {{{
absorbDenseSideIntoCornerFromLeft = formDataContractor(
    [Join(0,range(3),1,range(3,6))],
    [[(1,i)] for i in range(3)] + [[(0,3+i),(1,6+i)] for i in range(2)] + [[(0,5)]]
)
# }}}
# def absorbDenseSideIntoCornerFromRight(corner,side) # {{{
absorbDenseSideIntoCornerFromRight = formDataContractor(
    [Join(0,range(3,6),1,range(3))],
    [[(0,i),(1,6+i)] for i in range(2)] + [[(0,2)]] + [[(1,i)] for i in range(3,6)]
)
# }}}
# def absorbDenseCenterSSIntoSide(direction,side,center,center_conj=None) # {{{
@prepend([formDataContractor(
    # 0 = side, 1 = center, 2 = center*
    [
        Join(0,6,1,i),
        Join(0,7,2,i),
        Join(1,4,2,4),
    ],
    [
        [(0,0),(1,L(i))],
        [(0,1),(2,L(i))],
        [(0,2)],
        [(0,3),(1,R(i))],
        [(0,4),(2,R(i))],
        [(0,5)],
        [(1,O(i))],
        [(2,O(i))],
    ]
) for i in range(4)])
def absorbDenseCenterSSIntoSide(contractors,direction,side,center,center_conj=None):
    if center_conj is None:
        center_conj = center.conj()
    return \
      contractors[direction](
        side,
        center,
        center_conj,
      )
# }}}
# def absorbDenseCenterSOSIntoSide(side,center,center_conj=None) {{{
@prepend([formDataContractor(
    # 0 = side, 1 = state, 2 = state*, 3 = operator
    [
        Join(3,0,2,4),
        Join(3,1,1,4),
        Join(0,6,1,i),
        Join(0,7,2,i),
    ],
    [
        [(0,0),(1,L(i))],
        [(0,1),(2,L(i))],
        [(0,2)],
        [(0,3),(1,R(i))],
        [(0,4),(2,R(i))],
        [(0,5)],
        [(1,O(i))],
        [(2,O(i))],
    ]
) for i in range(4)])
def absorbDenseCenterSOSIntoSide(contractors,direction,side,state_center_data,operator_center_data,state_center_data_conj=None):
    if state_center_data_conj is None:
        state_center_data_conj = state_center_data.conj()
    return \
        contractors[direction](
            side,
            state_center_data,
            state_center_data_conj,
            operator_center_data,
        )
# }}}
def formNormalizationMultiplier(corners,sides,center_identity): # {{{
    return formNormalizationStage3(
        formNormalizationStage2(
            formNormalizationStage1(corners[0],sides[0]),
            formNormalizationStage1(corners[1],sides[1]),
        ),
        formNormalizationStage2(
            formNormalizationStage1(corners[2],sides[2]),
            formNormalizationStage1(corners[3],sides[3]),
        ),
        center_identity
    )
# }}}
# def formNormalizationStage1(corner,side) {{{
formNormalizationStage1 = formDataContractor(
    [Join(0,range(3,6),1,range(3))],
    [[(0,i) for i in range(3)]] + [[(1,i) for i in range(3,6)]] + [[(1,i)] for i in range(6,8)]
)
# }}}
# def formNormalizationStage2(stage1_0,stage1_1) {{{
formNormalizationStage2 = formDataContractor(
    [Join(0,0,1,1)],
    [
        [(1,0)],
        [(0,1)],
        [(0,2)],
        [(1,2)],
        [(0,3)],
        [(1,3)],
    ]
)
# }}}
# def formNormalizationStage3(stage2_0,stage2_1) {{{
@prependDataContractor(
    [
        Join(0,[3,4],2,[0,1]),
        Join(1,[3,4],2,[2,3]),
        Join(0,0,1,0),
    ],
    [
        [(0,1)],
        [(0,2)],
        [(1,1)],
        [(1,2)],
        [(2,4)],
    ]
)
def formNormalizationStage3(contractor,stage2_0,stage2_1,center_identity):
    stage2_0_joined = stage2_0.join((0,1),4,5,2,3)
    stage2_1_joined = stage2_1.join((1,0),4,5,2,3)
    dummy_state_center_data = CostTracker(stage2_0_joined.shape[-2:] + stage2_1_joined.shape[-2:] + (center_identity.shape[0],))
    return \
        Multiplier(
            (dummy_state_center_data.size(),)*2,
            partial(
                contractor,
                stage2_0_joined,
                stage2_1_joined
            ),
            computeCostOfContracting(contractor,stage2_0_joined,stage2_1_joined,dummy_state_center_data),
            *formDenseStage3_formMatrix_and_cost(stage2_0,stage2_1,center_identity)
        )
# }}}
# def formDenseStage3_multiply_and_cost(stage2_0,stage2_1,site_operator) {{{
@prependDataContractor(
    [
        Join(2,1,3,4),
        Join(0,[3,4],3,[0,1]),
        Join(1,[3,4],3,[2,3]),
        Join(0,0,1,0),
    ],
    [
        [(0,1)],
        [(0,2)],
        [(1,1)],
        [(1,2)],
        [(2,0)],
    ]
)
def formDenseStage3_multiply_and_cost(contractor,stage2_0,stage2_1,site_operator):
    stage2_0_joined = stage2_0.join((0,1),4,5,2,3)
    stage2_1_joined = stage2_1.join((1,0),4,5,2,3)
    state_shape = stage2_0_joined.shape[-2:] + stage2_1_joined.shape[-2:] + (site_operator.shape[0],)
    return (
        partial(
            contractor,
            stage2_0_joined,
            stage2_1_joined,
            site_operator,
        ),
        computeCostOfContracting(contractor,stage2_0_joined,stage2_1_joined,site_operator,state_shape)
    )
# }}}
# def formDenseStage3_formMatrix_and_cost(stage2_0,stage2_1,site_operator) {{{
@prependDataContractor(
    [
        Join(0,[0,1],1,[1,0]),
    ],
    [
        [(i,j) for i in (0,1) for j in (4,5)] + [(2,0)],
        [(i,j) for i in (0,1) for j in (2,3)] + [(2,1)],
    ]
)
def formDenseStage3_formMatrix_and_cost(contractor,stage2_0,stage2_1,site_operator):
    return (
        partial(
            contractor,
            stage2_0,
            stage2_1,
            site_operator,
        ),
        computeCostOfContracting(contractor,stage2_0,stage2_1,site_operator)
    )
# }}}
def formDenseStage3(stage2_0,stage2_1,site_operator): # {{{
    return \
        Multiplier(*
            ((prod(stage2_0.shape[-2:]+stage2_1.shape[-2:]+(site_operator.shape[0],)),)*2,)+
            formDenseStage3_multiply_and_cost(stage2_0,stage2_1,site_operator)+ 
            formDenseStage3_formMatrix_and_cost(stage2_0,stage2_1,site_operator)
        )
# }}}
# def formNormalizationSubmatrix + friends {{{
@prependDataContractor(
    [
        Join(0,[0,1],1,[1,0]),
    ],
    [
        [(i,j) for i in (0,1) for j in (4,5)],
        [(i,j) for i in (0,1) for j in (2,3)],
    ]
)
def formNormalizationSubmatrix(contractor,corners,sides):
    return contractor(
        formNormalizationStage2(
            formNormalizationStage1(corners[0],sides[0]),
            formNormalizationStage1(corners[1],sides[1]),
        ),
        formNormalizationStage2(
            formNormalizationStage1(corners[2],sides[2]),
            formNormalizationStage1(corners[3],sides[3]),
        ),
    )
# }}}
# }}}

# Exports {{{
__all__ = [
    "absorbDenseSideIntoCornerFromLeft",
    "absorbDenseSideIntoCornerFromRight",
    "absorbDenseCenterSSIntoSide",
    "absorbDenseCenterSOSIntoSide",
    "formDenseStage3",
    "formDenseStage3_formMatrix_and_cost",
    "formDenseStage3_multiply_and_cost",
    "formNormalizationMultiplier",
    "formNormalizationStage1",
    "formNormalizationStage2",
    "formNormalizationStage3",
    "formNormalizationSubmatrix",
]
# }}}

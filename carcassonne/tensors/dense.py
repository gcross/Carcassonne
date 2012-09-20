# Imports {{{
from ..utils import Join, formDataContractor, prepend, prependDataContractor, L, R, O
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
def absorbDenseCenterSOSIntoSide(contractors,direction,side,center_state,center_operator,center_state_conj=None):
    if center_state_conj is None:
        center_state_conj = center_state.conj()
    return \
        contractors[direction](
            side,
            center_state,
            center_state_conj,
            center_operator,
        )
# }}}
def formNormalizationMultiplier(corners,sides): # {{{
    return formNormalizationStage3(
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
def formNormalizationStage3(contractor,stage2_0,stage2_1):
    data0 = stage2_0.join((0,1),4,5,2,3)
    data1 = stage2_1.join((1,0),4,5,2,3)
    def multiply(center):
        return contractor(data0,data1,center)
    return multiply
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
    "formNormalizationMultiplier",
    "formNormalizationStage1",
    "formNormalizationStage2",
    "formNormalizationStage3",
    "formNormalizationSubmatrix",
]
# }}}

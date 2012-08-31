# Imports {{{
from ..utils import Join, formDataContractor, prepend, prependDataContractor
# }}}

# Functions {{{
# def absorbDenseSideIntoCornerFromLeft + friends # {{{
@prependDataContractor(
    [Join(0,0,1,1)],
    [
        [(1,0)],
        [(0,1),(1,2)],
    ]
)
def absorbDenseSideIntoCornerFromLeft(contractor,corner,side):
    return contractor(corner,side.join(0,1,(2,3)))
# }}}
# def absorbDenseSideIntoCornerFromRight + friends # {{{
@prependDataContractor(
    [Join(0,1,1,0)],
    [
        [(0,0),(1,2)],
        [(1,1)],
    ]
)
def absorbDenseSideIntoCornerFromRight(contractor,corner,side):
    return contractor(corner,side.join(0,1,(2,3)))
# }}}
# def absorbDenseCenterSSIntoSide + friends # {{{
@prepend([formDataContractor(
        # 0 = side, 1 = center, 2 = center*
        [
            Join(0,2,1,i),
            Join(0,3,2,i),
            Join(1,4,2,4),
        ],
        [
            [(0,0),(1,(i+1)%4),(2,(i+1)%4)],
            [(0,1),(1,(i-1)%4),(2,(i-1)%4)],
            [(1,(i+2)%4)],
            [(2,(i+2)%4)],
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
# def absorbDenseCenterSOSIntoSide + friends {{{
@prepend([formDataContractor(
    # 0 = side, 1 = state, 2 = state*, 3 = operator
    [
        Join(3,0,2,4),
        Join(3,1,1,4),
        Join(0,2,1,i),
        Join(0,3,2,i),
    ],
    [
        [(0,0),(1,(i+1)%4),(2,(i+1)%4)],
        [(0,1),(1,(i-1)%4),(2,(i-1)%4)],
        [(1,(i+2)%4)],
        [(2,(i+2)%4)],
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
# def formNormalizationStage1 + friends {{{
@prependDataContractor(
    [Join(0,1,1,0)],
    [
        [(0,0)],
        [(1,1)],
        [(1,2)],
        [(1,3)],
    ]
)
def formNormalizationStage1(contractor,corner,side):
    return contractor(corner,side)
# }}}
# def formNormalizationStage2 + friends {{{
@prependDataContractor(
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
def formNormalizationStage2(contractor,stage1_0,stage1_1):
    return contractor(stage1_0,stage1_1)
# }}}
# def formNormalizationStage3 + friends {{{
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
]
# }}}

# Imports {{{
from ..utils import Join, formDataContractor, prependDataContractor
# }}}

# Classes {{{
class DenseCorner: # {{{
    def __init__(self,data): # {{{
        self.data = data
    # }}}
    def __iadd__(self,other): # {{{
        self.data += other.data
        return self
    # }}}
    # def absorbFromLeft + friends # {{{
    _absorbFromLeft = staticmethod(formDataContractor(
        [Join(0,0,1,1)],
        [
            [(1,0)],
            [(0,1),(1,2)],
        ]
    ))
    def absorbFromLeft(self,side):
        return DenseCorner(self._absorbFromLeft(self.data,side.data.join(0,1,(2,3))))
    # }}}
    # def absorbFromRight + friends # {{{
    _absorbFromRight = staticmethod(formDataContractor(
        [Join(0,1,1,0)],
        [
            [(0,0),(1,2)],
            [(1,1)],
        ]
    ))
    def absorbFromRight(self,side):
        return DenseCorner(self._absorbFromRight(self.data,side.data.join(0,1,(2,3))))
    # }}}
# }}}

class DenseSide: # {{{
    def __init__(self,data): # {{{
        self.data = data
    # }}}
    def __iadd__(self,other): # {{{
        self.data += other.data
        return self
    # }}}
    # def absorbCenterSS + friends # {{{
    _absorbCenterSS = [
        formDataContractor(
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
        ) for i in range(4)
    ]
    def absorbCenterSS(self,direction,center):
        return DenseSide(
          self._absorbCenterSS[direction](
            self.data,
            center,
            center.conj(),
          )
        )
    # }}}
    # def absorbCenterSOS + friends {{{
    _absorbCenterSOS = [formDataContractor(
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
    ) for i in range(4)]
    def absorbCenterSOS(self,direction,center_state,center_operator,center_state_conj=None):
        if center_state_conj is None:
            center_state_conj = center_state.conj()
        return DenseSide(
            self._absorbCenterSOS[direction](
                self.data,
                center_state,
                center_state_conj,
                center_operator,
            )
        )
    # }}}
    @staticmethod # def formMultiplier {{{
    def formMultiplier(corners,sides):
        return formDenseStage3(
            formDenseStage2(
                formDenseStage1(corners[0],sides[0]),
                formDenseStage1(corners[1],sides[1]),
            ),
            formDenseStage2(
                formDenseStage1(corners[2],sides[2]),
                formDenseStage1(corners[3],sides[3]),
            ),
        )
    # }}}
# }}}
# }}}

# Functions {{{
# def formDenseStage1 + friends {{{
@prependDataContractor(
    [Join(0,1,1,0)],
    [
        [(0,0)],
        [(1,1)],
        [(1,2)],
        [(1,3)],
    ]
)
def formDenseStage1(contractor,corner,side):
    return contractor(corner.data,side.data)
# }}}
# def formDenseStage2 + friends {{{
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
def formDenseStage2(contractor,stage1_0,stage1_1):
    return contractor(stage1_0,stage1_1)
# }}}
# def formDenseStage3 + friends {{{
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
def formDenseStage3(contractor,stage2_0,stage2_1):
    data0 = stage2_0.join((0,1),4,5,2,3)
    data1 = stage2_1.join((1,0),4,5,2,3)
    def multiply(center):
        return contractor(data0,data1,center)
    return multiply
# }}}
# }}}

# Exports {{{
__all__ = [
    "DenseCorner",
    "DenseSide",

    "formDenseStage1",
    "formDenseStage2",
    "formDenseStage3",
]
# }}}

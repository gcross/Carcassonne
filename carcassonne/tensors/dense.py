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
        return DenseStage3(
            DenseStage2(
                DenseStage1(corners[0],sides[0]),
                DenseStage1(corners[1],sides[1]),
            ),
            DenseStage2(
                DenseStage1(corners[2],sides[2]),
                DenseStage1(corners[3],sides[3]),
            ),
        )
    # }}}
# }}}

class DenseStage1: # {{{
    # def init + friends # {{{
    @prependDataContractor(
        [Join(0,1,1,0)],
        [
            [(0,0)],
            [(1,1)],
            [(1,2)],
            [(1,3)],
        ]
    )
    def __init__(contractor,self,corner,side):
        self.data = contractor(corner.data,side.data)
    # }}}
# }}}

class DenseStage2: # {{{
    # def absorbFromLeft + friends # {{{
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
    def __init__(contractor,self,stage1_0,stage1_1):
        self.data = contractor(stage1_0.data,stage1_1.data)
    # }}}
# }}}

class DenseStage3: # {{{
    def __init__(self,stage2_0,stage2_1): # {{{
        self.data0 = stage2_0.data.join((0,1),4,5,2,3)
        self.data1 = stage2_1.data.join((1,0),4,5,2,3)
    # }}}
    # operator() + friends # {{{
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
    def __call__(contractor,self,center):
        return contractor(self.data0,self.data1,center)
    # }}}
# }}}
# }}}

# Exports {{{
__all__ = [
    "DenseCorner",
    "DenseSide",
]
# }}}

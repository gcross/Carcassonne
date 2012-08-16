# Imports {{{
from ..utils import Join, makeDataContractor, prependDataContractor
# }}}

# Classes {{{
class NormalizationCorner(object): # {{{
    def __init__(self,data): # {{{
        self.data = data
    # }}}
    # def absorbFromLeft + friends # {{{
    @prependDataContractor(
        [Join(0,0,1,1)],
        [
            [(1,0)],
            [(0,1),(1,2)],
        ]
    )
    def absorbFromLeft(contractor,self,side):
        return NormalizationCorner(contractor(self.data,side.data.join(0,1,(2,3))))
    # }}}
    # def absorbFromRight + friends # {{{
    @prependDataContractor(
        [Join(0,1,1,0)],
        [
            [(0,0),(1,2)],
            [(1,1)],
        ]
    )
    def absorbFromRight(contractor,self,side):
        return NormalizationCorner(contractor(self.data,side.data.join(0,1,(2,3))))
    # }}}
# }}}

class NormalizationSide(object): # {{{
    def __init__(self,direction,data): # {{{
        self.direction = direction
        self.data = data
    # }}}
    # def absorbCenter + friends # {{{
    _absorbCenter = [
        makeDataContractor(
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
    def absorbCenter(self,center):
        direction = self.direction
        return NormalizationSide(direction,
          self._absorbCenter[direction](
            self.data,
            center,
            center.conj(),
          )
        )
    # }}}
    @staticmethod # def formMultiplier {{{
    def formMultiplier(corners,sides):
        return NormalizationStage3(
            NormalizationStage2(
                NormalizationStage1(corners[0],sides[0]),
                NormalizationStage1(corners[1],sides[1]),
            ),
            NormalizationStage2(
                NormalizationStage1(corners[2],sides[2]),
                NormalizationStage1(corners[3],sides[3]),
            ),
        )
    # }}}
# }}}

class NormalizationStage1(object): # {{{
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

class NormalizationStage2(object): # {{{
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

class NormalizationStage3(object): # {{{
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

# Exports {{{
__all__ = [
    "NormalizationCorner",
    "NormalizationSide",
]
# }}}

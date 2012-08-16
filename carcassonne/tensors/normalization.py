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
        return NormalizationCorner(contractor(self.data,side.data))
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
        return NormalizationCorner(contractor(self.data,side.data))
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
                [(1,(i+2)%4),(2,(i+2)%4)],
            ]
        ) for i in xrange(4)
    ]
    def absorbCenter(self,center):
        direction = self.direction
        return NormalizationSide(direction,
          self._absorbCenter[direction](
            self.data.splitAt(2,(center.shape[direction],center.shape[direction])),
            center,
            center.conj(),
          )
        )
    # }}}
# }}}

# Exports {{{
__all__ = [
    "NormalizationCorner",
    "NormalizationSide",
]
# }}}

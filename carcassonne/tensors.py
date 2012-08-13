# Imports {{{
from .utils import FromRight, FromBoth, formAbsorber, formContractor
# }}}

# Classes {{{
class NormalizationCorner(object): # {{{
    def __init__(self): # {{{
        self.tensor = ones((1,1))
    # }}}
    # def absorbFromLeft + friends # {{{
    _absorbFromLeft = staticmethod(formAbsorber((0,),(1,),(FromRight(0),FromBoth(1,2))))
    def absorbFromLeft(self,side):
        self.tensor = self._absorbFromLeft(self.tensor,side.tensor)
    # }}}
    # def absorbFromRight + friends # {{{
    _absorbFromRight = staticmethod(formAbsorber((1,),(0,),(FromBoth(0,2),FromRight(1))))
    def absorbFromRight(self,side):
        self.tensor = self._absorbFromRight(self.tensor,side.tensor)
    # }}}
# }}}

class NormalizationSide(object): # {{{
    def __init__(self): # {{{
        self.tensor = ones((1,1,1))
    # }}}
    # def absorbFromCenter + friends # {{{
    _absorbCenterFrom = [
        formContractor(
            ['side','center*','center'],
            [
                (('side',2),('center*',i)),
                (('side',3),('center',i)),
                (('center*',4),('center',4)),
            ],
            [
                [('side',0),('center*',(i+1)%4),('center',(i+1)%4)],
                [('side',1),('center*',(i-1)%4),('center',(i-1)%4)],
                [('center*',(i+2)%4),('center',(i+2)%4)]
            ]
        ) for i in xrange(4)
    ]
    def absorbCenter(self,center,direction):
        self.tensor = self._absorbCenterFrom[direction](
            self.tensor.reshape(self.tensor.shape[0],self.tensor.shape[1],center.tensor.shape[direction],center.tensor.shape[direction],),
            center.tensor.conj(),
            center.tensor
        )
    # }}}
# }}}
# }}}

# Exports {{{
__all__ = [
    "NormalizationCorner",
    "NormalizationSide",
]
# }}}

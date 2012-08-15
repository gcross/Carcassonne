# Imports {{{
from ..utils import FromRight, FromBoth, formAbsorber, formContractor
# }}}

# Classes {{{
class NormalizationCorner(object): # {{{
    def __init__(self,tensor=None): # {{{
        if tensor is None:
            tensor = ones((1,1))
        self.tensor = tensor
    # }}}
    # def absorbFromLeft + friends # {{{
    _absorbFromLeft = staticmethod(formAbsorber((0,),(1,),(FromRight(0),FromBoth(1,2))))
    def absorbFromLeft(self,side):
        return NormalizationCorner(self._absorbFromLeft(self.tensor,side.tensor))
    # }}}
    # def absorbFromRight + friends # {{{
    _absorbFromRight = staticmethod(formAbsorber((1,),(0,),(FromBoth(0,2),FromRight(1))))
    def absorbFromRight(self,side):
        return NormalizationSide(self._absorbFromRight(self.tensor,side.tensor))
    # }}}
# }}}

class NormalizationSide(object): # {{{
    def __init__(self,direction,tensor=None): # {{{
        self.direction = direction
        if tensor is None:
            tensor = ones((1,1,1))
        self.tensor = tensor
    # }}}
    # def absorbCenter + friends # {{{
    _absorbCenter = [
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
    def absorbCenter(self,center):
        direction = self.direction
        return NormamalizationSide(self._absorbCenter[direction](
            self.tensor.reshape(self.tensor.shape[0],self.tensor.shape[1],center.tensor.shape[direction],center.tensor.shape[direction],),
            center.tensor.conj(),
            center.tensor
        ))
    # }}}
# }}}
# }}}

# Exports {{{
__all__ = [
    "NormalizationCorner",
    "NormalizationSide",
]
# }}}

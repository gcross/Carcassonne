# Imports {{{
from .tensors.dense import absorbDenseSideIntoCornerFromLeft, absorbDenseSideIntoCornerFromRight
# }}}

# Classes {{{
class Normalization(): # {{{
    def __init__(self,corners,sides): # {{{
        self.corners = list(corners)
        self.sides = list(sides)
    # }}}
    def absorbCenter(self,direction,center): # {{{
        self.corners[direction] = absorbDenseSideIntoCornerFromLeft(self.corners[direction],self.sides[(direction+1)%4])
        self.sides[direction] = self.sides[direction].absorbCenterSS(direction,center)
        self.corners[(direction-1)%4] = absorbDenseSideIntoCornerFromRight(self.corners[(direction-1)%4],self.sides[(direction-1)%4])
    # }}}
    def computeNormalization(self,center): # {{{
        return center.conj().contractWith(self.formMultiplier()(center),range(5),range(5)).extractScalar()
    # }}}
    def formMultiplier(self): # {{{
        return self.sides[0].formMultiplier(self.corners,self.sides)
    # }}}
# }}}
# }}}

# Exports {{{
__all__ = [
    "Normalization",
]
# }}}

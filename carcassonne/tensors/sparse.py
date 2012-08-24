# Imports {{{
from ..sparse import SparseTensor, formSparseContractor
from ..tensors.dense import DenseCorner
from ..utils import FromRight, FromBoth
# }}}

# Values {{{
special_indices = [
    {
        "indices_to_ignore": {(0,1)},
        "indices_to_redirect": {(2,3):2},
    },
    {
        "indices_to_ignore": {(2,3)},
        "indices_to_redirect": {(0,1):0},
    },
]
corner_left_special_indices = [special_indices[i] for i in  [1,0,0,1]]
corner_right_special_indices = [special_indices[i] for i in [1,0,0,1]]
side_left_special_indices = [special_indices[i] for i in    [1,1,0,0]]
side_right_special_indices = [special_indices[i] for i in   [0,0,1,1]]
# }}}

# Classes {{{
class SparseCorner: # {{{
    def __init__(self,tensor): # {{{
        self.tensor = tensor
    # }}}
    @staticmethod # constructUsing # {{{
    def constructUsing(direction,datacls):
        return SparseCorner(direction,SparseTensor(
            (1,1),
            {(1,1): datacls.newTrivial((1,1))},
        ))
    # }}}
    # def absorbFromLeft + friends # {{{
    _absorbFromLeft = [formSparseContractor(
        (0,), # my join indices
        (1,), # side's join indices
        (FromRight(0),FromBoth(1,2,**special_indices)),
        DenseCorner.absorbFromLeft
    ) for special_indices in corner_left_special_indices]
    def absorbFromLeft(self,direction,side):
        return SparseCorner(self._absorbFromLeft[direction](self.tensor,side.tensor))
    # }}}
    # def absorbFromRight + friends # {{{
    _absorbFromRight = [formSparseContractor(
        (1,), # my join indices
        (0,), # side's join indices
        (FromBoth(0,2,**special_indices),FromRight(1)),
        DenseCorner.absorbFromRight
    ) for special_indices in corner_right_special_indices]
    def absorbFromRight(self,direction,side):
        return SparseCorner(self._absorbFromRight[direction](self.tensor,side.tensor))
    # }}}
# }}}
class SparseSide: # {{{
    def __init__(self,tensor): # {{{
        self.tensor = tensor
    # }}}
    # def absorbCenterSOS + friends {{{
    _absorbCenterSOS = [formSparseContractor( 
        (2,),(i,),(
            FromBoth(0,(i+1)%4,**side_left_special_indices[i]),
            FromBoth(1,(i-1)%4,**side_right_special_indices[i]),
            FromRight((i+2)%4)
        )
    ) for i in range(4)]
    def absorbCenterSOS(self,direction,state_center_data,operator_center_tensor,state_center_data_conj=None):
        if state_center_data_conj is None:
            state_center_data_conj = state_center_data.conj()
        def contractChunks(side,operator_center_data):
            return side.absorbCenterSOS(direction,state_center_data,operator_center_data,state_center_data_conj)
        return SparseSide(
            self._absorbCenterSOS[direction](
                contractChunks,
                self.tensor,
                operator_center_tensor
            )
        )
    # }}}
# }}}
# }}}

# Exports {{{
__all__ = [
    "SparseCorner",
    "SparseSide",
]
# }}}

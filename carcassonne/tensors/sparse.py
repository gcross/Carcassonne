# Imports {{{
from ..sparse import SparseTensor, formSparseContractor
from ..tensors.dense import DenseCorner, formDenseStage1, formDenseStage2
from ..utils import FromLeft, FromRight, FromBoth, Join, formDataContractor 
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
class SparseStage1: # {{{
    # def init + friends # {{{
    _contractor = staticmethod(formSparseContractor(
        (1,),
        (0,),
        [
            FromLeft(0),
            FromRight(1),
            FromRight(2),
        ],
        formDenseStage1
    ))
    def __init__(self,corner,side):
        self.tensor = self._contractor(corner.tensor,side.tensor)
    # }}}
# }}}
class SparseStage2: # {{{
    # def init + friends # {{{
    _contractor = staticmethod(formSparseContractor(
        (0,),
        (1,),
        [
            FromRight(0),
            FromLeft(1),
            FromLeft(2),
            FromRight(2),
        ],
        formDenseStage2
    ))
    def __init__(self,stage1_0,stage1_1):
        self.tensor = self._contractor(stage1_0.tensor,stage1_1.tensor)
    # }}}
# }}}
class SparseStage3: # {{{
    def __init__(self,stage2_0,stage2_1): # {{{
        self.tensor0 = stage2_0.tensor
        self.tensor1 = stage2_1.tensor
    # }}}
    # call + friends # {{{
    _dense_contractor_0 = staticmethod(formDataContractor( # {{{
        [
            Join(1,4,2,1),
            Join(0,(2,3),1,(0,1)),
        ],
        [
            [(0,4)],
            [(0,5)],
            [(1,2)],
            [(1,3)],
            [(2,0)],
            [(0,1)],
            [(0,0)],
        ]
    )) # }}}
    _dense_contractor_1 = staticmethod(formDataContractor( # {{{
        [
            Join(0,(0,1,2,3),1,(5,6,2,3)),
        ],
        [
            [(1,0)],
            [(1,1)],
            [(0,4)],
            [(0,5)],
            [(1,4)],
        ]
    )) # }}}
    _sparse_contractor_0 = staticmethod(formSparseContractor( # {{{
        (2,3), # stage 2 tensor 0 indices
        (0,1), # center operator tensor indices
        [
            FromLeft(1),
            FromLeft(0),
            FromRight(2),
            FromRight(3),
        ]
    )) # }}}
    _sparse_contractor_1 = staticmethod(formSparseContractor( # {{{
        (0,1,2,3), # stage 2 tensor 1 indices
        (0,1,2,3), # intermediate tensor indices
        []
    )) # }}}
    def __call__(self,state_center_data,operator_center_tensor):
        def contractChunks(dense,operator_center_data):
            return self._dense_contractor_0(dense.data,state_center_data,operator_center_data)
        intermediate_tensor = self._sparse_contractor_0(contractChunks,self.tensor0,operator_center_tensor)
        final_tensor = self._sparse_contractor_1(lambda x,y: self._dense_contractor_1(x.data,y),self.tensor1,intermediate_tensor)
        assert final_tensor.dimensions == ()
        try:
            return final_tensor.chunks[()]
        except KeyError:
            return state_center_data.newZeros(shape=state_center_data.shape,dtype=state_center_data.dtype)
    # }}}
# }}}
# }}}

# Exports {{{
__all__ = [
    "SparseCorner",
    "SparseSide",

    "SparseStage1",
    "SparseStage2",
    "SparseStage3",
]
# }}}

# Imports {{{
from ..sparse import SparseTensor, formSparseContractor, prependSparseContractor
from ..tensors.dense import DenseCorner, formDenseStage1, formDenseStage2
from ..utils import FromLeft, FromRight, FromBoth, Join, formDataContractor, prepend
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
    @staticmethod # def formMultiplier {{{
    def formMultiplier(corners,sides,operator_center_tensor):
        return formSparseStage3(
            formSparseStage2(
                formSparseStage1(corners[0],sides[0]),
                formSparseStage1(corners[1],sides[1]),
            ),
            formSparseStage2(
                formSparseStage1(corners[2],sides[2]),
                formSparseStage1(corners[3],sides[3]),
            ),
            operator_center_tensor,
        )
    # }}}
# }}}
# }}}

# def formSparseStage1 {{{
@prependSparseContractor(
    (1,),
    (0,),
    [
        FromLeft(0),
        FromRight(1),
        FromRight(2),
    ],
    formDenseStage1
)
def formSparseStage1(contractor,corner,side):
    return contractor(corner.tensor,side.tensor)
# }}}

# def formSparseStage2: {{{
@prependSparseContractor(
    (0,),
    (1,),
    [
        FromRight(0),
        FromLeft(1),
        FromLeft(2),
        FromRight(2),
    ],
    formDenseStage2
)
def formSparseStage2(contractor,stage1_0,stage1_1):
    return contractor(stage1_0,stage1_1)
# }}}

# def formSparseStage3: {{{
@prepend( # {{{
    formDataContractor( # dense contractor 0 {{{
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
    ), # }}}
    formDataContractor( # dense contractor 1 {{{
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
    ), # }}}
    formSparseContractor( # sparse contractor 0 {{{
        (2,3), # stage 2 tensor 0 indices
        (0,1), # center operator tensor indices
        [
            FromLeft(1),
            FromLeft(0),
            FromRight(2),
            FromRight(3),
        ]
    ), # }}}
    formSparseContractor( # sparse contractor 1 {{{
        (0,1,2,3), # stage 2 tensor 1 indices
        (0,1,2,3), # intermediate tensor indices
        []
    ), # }}}
) # }}}
def formSparseStage3( # {{{
    denseContractor0,
    denseContractor1,
    sparseContractor0,
    sparseContractor1,
    stage2_0,
    stage2_1,
    operator_center_tensor,
):
    def multiply(state_center_data):
        result = sparseContractor1(
            denseContractor1,
            stage2_1,
            sparseContractor0(
                lambda stage2_0_data,operator_center_data:
                    denseContractor0(stage2_0_data,state_center_data,operator_center_data),
                stage2_0,
                operator_center_tensor
            )
        )
        assert result.dimensions == ()
        try:
            return result.chunks[()]
        except KeyError:
            return state_center_data.newZeros(shape=state_center_data.shape,dtype=state_center_data.dtype)
    return multiply
# }}}
# }}}

# Exports {{{
__all__ = [
    "SparseCorner",
    "SparseSide",

    "formSparseStage1",
    "formSparseStage2",
    "formSparseStage3",
]
# }}}

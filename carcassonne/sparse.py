# Imports {{{
from collections import namedtuple
from functools import partial
# }}}

# Base classes {{{
class Singleton: # {{{
    __slots__ = []
    def __new__(cls):
        try:
            return cls.instance
        except AttributeError:
            instance = object.__new__(cls)
            cls.instance = instance
            return instance
    def __hash__(self):
        return hash(type)
# }}}
# }}}

# Tags {{{
class Identity(Singleton): # {{{
    __slots__ = []
# }}}
class Complete(Singleton): # {{{
    __slots__ = []
# }}}
class OneSiteOperator(Singleton): # {{{
    __slots__ = []
# }}}
class TwoSiteOperator: # {{{
    __slots__ = ["direction","position"]
    def __init__(self,direction,position=None):
        self.direction = direction
        self.position = position
    def moveOut(self):
        return TwoSiteOperator(self.direction,self.position+1)
# }}}
# }}}

# Functions {{{
def contractSparseTensors(dense_contractors,tensor_1,tensor_2): # {{{
    result_tensor = {}
    for tag_1, data_1 in tensor_1.items():
        tag_1_type = type(tag_1)
        for tag_2, data_2 in tensor_2.items():
            tag_2_type = type(tag_2)
            if (tag_1_type,tag_2_type) in dense_contractors:
                handler = dense_contractors[tag_1_type,tag_2_type](tag_1,tag_2)
                if handler is not None:
                    result_tag, contractDenseTensors = handler
                    if result_tag is not None:
                        result_data = contractDenseTensors(data_1,data_2)
                        if result_tag in result_tensor:
                            result_tensor[result_tag] += result_data
                        else:
                            result_tensor[result_tag]  = result_data
    return result_tensor
# }}}
def directSumListsOfSparse(list1,list2): # {{{
    return [directSumSparse(sparse1,sparse2) for (sparse1,sparse2) in zip(list1,list2)]
# }}}
def directSumSparse(sparse1,sparse2): # {{{
    sparses = (sparse1,sparse2)
    zeroses = tuple(sparse[Identity()].newZeros(sparse[Identity()].shape,dtype=sparse[Identity()].dtype) for sparse in sparses)
    result = {}
    for tag in sparse1.keys() | sparse2.keys():
        sparse_datas = tuple(
            sparse[tag] if tag in sparse else zeros
            for sparse, zeros in zip(sparses,zeroses)
        )
        result[tag] = sparse_datas[0].directSumWith(sparse_datas[1])
    return result
# }}}
def formSparseContractor(dense_contractors): # {{{
    return partial(contractSparseTensors,dense_contractors)
# }}}
def makeSparseOperator(O=None,OO_UD=None,OO_LR=None): # {{{
    operator = {Identity():None}
    if O is not None:
        operator[OneSiteOperator()] = O
    if OO_UD is not None:
        O_U, O_D = OO_UD
        operator[TwoSiteOperator(3,0)] = O_U
        operator[TwoSiteOperator(1,0)] = O_D
    if OO_LR is not None:
        O_L, O_R = OO_LR
        operator[TwoSiteOperator(0,0)] = O_L
        operator[TwoSiteOperator(2,0)] = O_R
    return operator
# }}}
def mapOverSparseData(f,sparse): # {{{
    return {tag: f(data) for tag, data in sparse.items()}
# }}}
# }}}

# Exports {{{
__all__ = [
    "Side",

    "Identity",
    "Complete",
    "OneSiteOperator",

    "contractSparseTensors",
    "directSumListsOfSparse",
    "directSumSparse",
    "formSparseContractor",
    "makeSparseOperator",
    "mapOverSparseData",
]
# }}}

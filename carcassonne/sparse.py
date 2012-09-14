# Imports {{{
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
# }}}

# Functions {{{
def contractSparseTensors(dense_contractors,tensor_1,tensor_2): # {{{
    result_tensor = {}
    for tag_1, data_1 in tensor_1.items():
        tag_1_type = type(tag_1)
        for tag_2, data_2 in tensor_2.items():
            tag_2_type = type(tag_2)
            result_tag = None
            contractDenseTensors = None
            if (tag_1_type,tag_2_type) in dense_contractors:
                result_tag, contractDenseTensors = dense_contractors[tag_1_type,tag_2_type](tag_1,tag_2)
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
    "mapOverSparseData",
]
# }}}

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
class Operator(Singleton): # {{{
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
def formSparseContractor(dense_contractors): # {{{
    return partial(contractSparseTensors,dense_contractors)
# }}}
# }}}

# Exports {{{
__all__ = [
    "Side",

    "Identity",
    "Complete",
    "Operator",

    "contractSparseTensors",
    "formSparseContractor",
]
# }}}

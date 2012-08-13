# Imports {{{
from numpy import tensordot
from numpy.random import rand
# }}}

# Classes {{{
# Result dimension sources {{{
class FromLeft(object): # {{{
    # Constants {{{
    number_of_left_dimensions = 1
    number_of_right_dimensions = 0
    # }}}
    def __init__(self,dimension): # {{{
        self.dimension = dimension
    # }}}
    def appendDimensionsToTransposition(self,transposition,left_transpose_offsets,right_transpose_offsets): # {{{
        transposition.append(left_transpose_offsets[self.dimension])
    # }}}
    def getResultDimension(self,left_dimensions,right_dimensions): # {{{
        return left_dimensions[self.dimension]
    # }}}
    def getResultIndex(self,right_dimensions,left_indices,right_indices): # {{{
        return left_indices[self.dimension]
    # }}}
# }}}
class FromRight(object): # {{{
    # Constants {{{
    number_of_left_dimensions = 0
    number_of_right_dimensions = 1
    # }}}
    def __init__(self,dimension): # {{{
        self.dimension = dimension
    # }}}
    def appendDimensionsToTransposition(self,transposition,left_transpose_offsets,right_transpose_offsets): # {{{
        transposition.append(right_transpose_offsets[self.dimension])
    # }}}
    def getResultDimension(self,left_dimensions,right_dimensions): # {{{
        return right_dimensions[self.dimension]
    # }}}
    def getResultIndex(self,right_dimensions,left_indices,right_indices): # {{{
        return right_indices[self.dimension]
    # }}}
# }}}
class FromBoth(object): # {{{
    # Constants {{{
    number_of_left_dimensions = 1
    number_of_right_dimensions = 1
    # }}}
    def __init__(self,left_dimension,right_dimension,indices_to_ignore=frozenset(),indices_to_redirect=frozenset()): # {{{
        self.left_dimension = left_dimension
        self.right_dimension = right_dimension
        self.indices_to_ignore = indices_to_ignore
        self.indices_to_redirect = indices_to_redirect
    # }}}
    def appendDimensionsToTransposition(self,transposition,left_transpose_offsets,right_transpose_offsets): # {{{
        transposition.append(left_transpose_offsets[self.left_dimension])
        transposition.append(right_transpose_offsets[self.right_dimension])
    # }}}
    def getResultDimension(self,left_dimensions,right_dimensions): # {{{
        return left_dimensions[self.left_dimension]*right_dimensions[self.right_dimension]
    # }}}
    def getResultIndex(self,right_dimensions,left_indices,right_indices): # {{{
        left_index = left_indices[self.left_dimension]
        right_index = right_indices[self.right_dimension]
        indices = (left_index,right_index)
        if indices in self.indices_to_ignore:
            return None
        if indices in self.indices_to_redirect:
            return self.indices_to_redirect[indices]
        return left_index * right_dimensions[self.right_dimension] + right_index
    # }}}
# }}}
# }}}
# }}}

# Functions {{{
def crand(*shape): # {{{
    return rand(*shape)*2-1+rand(*shape)*2j-1j
# }}}
def formAbsorber(left_join_dimensions,right_join_dimensions,result_dimension_sources): # {{{
    # Compute the numbers of dimensions {{{
    number_of_join_dimensions = len(left_join_dimensions)
    assert number_of_join_dimensions == len(right_join_dimensions)
    number_of_left_dimensions = number_of_join_dimensions + sum(x.number_of_left_dimensions for x in result_dimension_sources)
    number_of_right_dimensions = number_of_join_dimensions + sum(x.number_of_right_dimensions for x in result_dimension_sources)
    # }}}
    # Compute the transposition {{{
    next_transpose_dimension = 0

    left_transpose_offsets = {}
    for dimension in xrange(number_of_left_dimensions):
        if dimension not in left_join_dimensions:
            left_transpose_offsets[dimension] = next_transpose_dimension
            next_transpose_dimension += 1

    right_transpose_offsets = {}
    for dimension in xrange(number_of_right_dimensions):
        if dimension not in right_join_dimensions:
            right_transpose_offsets[dimension] = next_transpose_dimension
            next_transpose_dimension += 1

    transposition = []
    for result_dimension_source in result_dimension_sources:
        result_dimension_source.appendDimensionsToTransposition(transposition,left_transpose_offsets,right_transpose_offsets)
    transposition = tuple(transposition)
    # }}}
    join_dimensions = (left_join_dimensions,right_join_dimensions)
    def absorb(left,right): # {{{
        left_shape = left.shape
        right_shape = right.shape
        result_shape = tuple(x.getResultDimension(left_shape,right_shape) for x in result_dimension_sources)
        return tensordot(left,right,join_dimensions).transpose(transposition).reshape(result_shape)
    # }}}
    return absorb
# }}}
def formContractor(order,joins,result_joins): # {{{
    observed_tensor_indices = {}

    # Tabulate all of the observed tensor indices {{{
    for (tensor_id,index) in sum([list(x) for x in joins] + [list(x) for x in result_joins],[]):
        if index < 0:
            raise ValueError("index {} of tensor {} is negative".format(index,tensor_id))
        try:
            observed_indices = observed_tensor_indices[tensor_id]
        except KeyError:
            observed_indices = set()
            observed_tensor_indices[tensor_id] = observed_indices
        if index in observed_indices:
            raise ValueError("index {} of tensor {} appears more than once in the joins".format(index,tensor_id))
        observed_indices.add(index)

    for tensor_id in observed_tensor_indices:
        if tensor_id not in order:
            raise ValueError("tensor {} does not appear in the list of arguments ({})".format(tensor_id,order))
    # }}}

    tensor_join_ids = {}

    # Check the observed tensor indices and initialize the map of tensor joins {{{
    for (tensor_id,observed_indices) in observed_tensor_indices.items():
        tensor_dimension = max(observed_indices)+1
        expected_indices = set(range(tensor_dimension))
        invalid_indices = observed_indices - expected_indices
        missing_indices = expected_indices - observed_indices
        if len(invalid_indices) > 0:
            raise ValueError('the invalid indices {} have appeared in joins involving tendor {}'.format(invalid_indices,tensor_id))
        if len(missing_indices) > 0:
            raise ValueError('the expected indices {} do not appear in any of the joins for tensor {}'.format(missing_indices,tensor_id))
        if tensor_id not in order:
            raise ValueError('tensor {} does not appear in the list of arguments'.format(tensor_id))
        tensor_join_ids[tensor_id] = [None]*tensor_dimension
    # }}}

    result_join_ids = []

    # Label each join with a unique id {{{
    current_join_id = 0
    for join in joins:
        for (tensor_id,index) in join:
            tensor_join_ids[tensor_id][index] = current_join_id
        current_join_id += 1
    for join in result_joins:
        join_ids = []
        for (tensor_id,index) in join:
            join_ids.append(current_join_id)
            tensor_join_ids[tensor_id][index] = current_join_id
            current_join_id += 1
        result_join_ids.append(join_ids)
    # }}}

    argument_join_ids = [tensor_join_ids[tensor_id] for tensor_id in order]

    # Form the contractor function {{{
    def contract(*arguments):
        if len(arguments) != len(order):
            raise ValueError("wrong number of arguments;  expected {} but received {}".format(len(order),len(arguments)))
        for (i, (tensor_id, argument)) in enumerate(zip(order,arguments)):
            if argument.ndim != len(tensor_join_ids[tensor_id]):
                raise ValueError("argument {} ('{}') has rank {} when it was expected to have rank {}".format(i,order[i],argument.ndim,len(tensor_join_ids[tensor_id])))
        arguments = list(arguments)
        join_ids_index = -1
        current_tensor = arguments.pop()
        current_join_ids = argument_join_ids[join_ids_index]
        while len(arguments) > 0:
            join_ids_index -= 1
            next_tensor = arguments.pop()
            next_join_ids = argument_join_ids[join_ids_index]
            try:
                first_axes = []
                second_axes = []
                first_axis_index = 0
                common_join_ids = set()
                for join_id in current_join_ids:
                    if join_id in next_join_ids:
                        common_join_ids.add(join_id)
                        first_axes.append(first_axis_index)
                        second_axes.append(next_join_ids.index(join_id))
                    first_axis_index += 1
                current_tensor = tensordot(current_tensor,next_tensor,(first_axes,second_axes))
                current_join_ids = [i for i in current_join_ids+next_join_ids if i not in common_join_ids]
            except Exception as e:
                raise ValueError("Error when joining tensor {}: '{}' (first tensor axes are {} with dimensions {}, second ({}) axes are {} with dimensions {})".format(order[join_ids_index],str(e),first_axes,[current_tensor.shape[i] for i in first_axes],order[join_ids_index],second_axes,[next_tensor.shape[i] for i in second_axes]))
        current_tensor = current_tensor.transpose([current_join_ids.index(i) for i in sum([list(x) for x in result_join_ids],[])])
        old_shape = current_tensor.shape
        new_shape = []
        index = 0
        for join in result_join_ids:
            dimension = 1
            for _ in join:
                dimension *= old_shape[index]
                index += 1
            new_shape.append(dimension)
        return current_tensor.reshape(new_shape)
    # }}}

    return contract
# }}}
# }}}

# Exports {{{
__all__ = [
    "FromLeft",
    "FromRight",
    "FromBoth",

    "crand",
    "formAbsorber",
    "formContractor",
]
# }}}

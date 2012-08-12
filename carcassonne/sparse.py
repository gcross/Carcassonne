# Imports {{{
from collections import namedtuple
import itertools
from numpy import ndarray, zeros
# }}}

# Types {{{
SparseTensor = namedtuple("SparseTensor",["dimensions","chunks"])
# }}}

# Classes {{{
# Result dimension sources {{{
class FromLeft(object): # {{{
    number_of_left_dimensions = 1
    number_of_right_dimensions = 0
    def __init__(self,dimension):
        self.dimension = dimension
    def getResultDimension(self,left_dimensions,right_dimensions):
        return left_dimensions[self.dimension]
    def getResultIndex(self,right_dimensions,left_indices,right_indices):
        return left_indices[self.dimension]
# }}}
class FromRight(object): # {{{
    number_of_left_dimensions = 0
    number_of_right_dimensions = 1
    def __init__(self,dimension):
        self.dimension = dimension
    def getResultDimension(self,left_dimensions,right_dimensions):
        return right_dimensions[self.dimension]
    def getResultIndex(self,right_dimensions,left_indices,right_indices):
        return right_indices[self.dimension]
# }}}
class FromBoth(object): # {{{
    number_of_left_dimensions = 1
    number_of_right_dimensions = 1
    def __init__(self,left_dimension,right_dimension,indices_to_ignore=frozenset(),indices_to_sum=frozenset()):
        self.left_dimension = left_dimension
        self.right_dimension = right_dimension
    def getResultDimension(self,left_dimensions,right_dimensions):
        return left_dimensions[self.left_dimension]*right_dimensions[self.right_dimension]
    def getResultIndex(self,right_dimensions,left_indices,right_indices):
        left_index = left_indices[self.left_dimension]
        right_index = right_indices[self.right_dimension]
        indices = (left_index,right_index)
        if indices in self.indices_to_ignore:
            return None
        if indices in self.indices_to_sum:
            return self.indices_to_sum[indices]
        return left_index * right_dimensions[self.right_dimension] + right_index
# }}}
# }}}
# }}}

# Functions {{{
def formDenseTensor(sparse_tensor,dtype=None): # {{{
    dimensions = sparse_tensor.dimensions
    if not dtype and sparse_tensor.chunks:
        value = next(sparse_tensor.chunks.itervalues())
        if isinstance(value,ndarray):
            dtype = value.dtype
            dimensions +=  value.shape
        else:
            dtype = type(value)
    dense_tensor = zeros(shape=dimensions,dtype=dtype)
    for indices, value in sparse_tensor.chunks.iteritems():
        dense_tensor[indices] += value
    return dense_tensor
# }}}
def formSparseContractor(left_join_dimensions,right_join_dimensions,result_dimension_sources,contractChunks): # {{{
    # Compute the numbers of dimensions {{{
    number_of_left_dimensions = len(left_join_dimensions) + sum(dimension_source.number_of_left_dimensions for dimension_source in result_dimension_sources)
    number_of_right_dimensions = len(right_join_dimensions) + sum(dimension_source.number_of_right_dimensions for dimension_source in result_dimension_sources)
    number_of_result_dimensions = len(result_dimension_sources)
    number_of_join_dimensions = len(left_join_dimensions)
    # }}}
    # Perform sanity checks for the numbers of dimensions {{{
    assert(number_of_join_dimensions == len(right_join_dimensions))
    assert(number_of_result_dimensions == number_of_left_dimensions + number_of_right_dimensions - 2*number_of_join_dimensions)
    # }}}
    def absorb(left_sparse_tensor,right_sparse_tensor): # {{{
        # Cache the components of the tensors {{{
        left_dimensions = left_sparse_tensor.dimensions
        left_chunks = left_sparse_tensor.chunks
        right_dimensions = right_sparse_tensor.dimensions
        right_chunks = right_sparse_tensor.chunks
        # }}}
        # Compute the result dimensions {{{
        result_dimensions = tuple(dimension_source.getResultDimension(left_dimensions,right_dimensions) for dimension_source in result_dimension_sources)
        # }}}
        # Determine which join indices overlap between the left and right, and from them create a dictionary of chunk lists {{{
        chunk_lists = {index: ([],[]) for index in (
            frozenset(tuple(indices[join_dimension] for join_dimension in left_join_dimensions) for indices in left_chunks.iterkeys()) &
            frozenset(tuple(indices[join_dimension] for join_dimension in right_join_dimensions) for indices in right_chunks.iterkeys())
        )}
        # }}}
        # Add all overlapping left chunks to the chunk list {{{
        for indexed_chunk in left_chunks.iteritems():
            indices = indexed_chunk[0]
            join_indices = tuple(indices[join_dimension] for join_dimension in left_join_dimensions)
            if join_indices in chunk_lists:
                chunk_lists[join_indices][0].append(indexed_chunk)
        # }}}
        # Add all overlapping right chunks to the chunk list {{{
        for indexed_chunk in right_chunks.iteritems():
            indices = indexed_chunk[0]
            join_indices = tuple(indices[join_dimension] for join_dimension in right_join_dimensions)
            if join_indices in chunk_lists:
                chunk_lists[join_indices][1].append(indexed_chunk)
        # }}}
        # Compute the result chunks {{{
        result_chunks = {}
        for indexed_chunk_lists_pair in chunk_lists.iteritems():
            for ((left_indices,left_chunk),(right_indices,right_chunk)) in itertools.product(*indexed_chunk_lists_pair[1]):
                result_indices = tuple(dimension_source.getResultIndex(right_dimensions,left_indices,right_indices) for dimension_source in result_dimension_sources)
                if None in result_indices:
                    continue
                result_chunk = contractChunks(left_chunk,right_chunk)
                if result_indices in result_chunks:
                    result_chunks[result_indices] += result_chunk
                else:
                    result_chunks[result_indices]  = result_chunk
        # }}}
        # Return the result {{{
        return SparseTensor(result_dimensions,result_chunks)
        # }}}
    # }}}
    return absorb
# }}}
# }}}

# Exports {{{
__all__ = [
    "SparseTensor",

    "FromLeft",
    "FromRight",
    "FromBoth",

    "formDenseTensor",
    "formSparseContractor",
]# }}}

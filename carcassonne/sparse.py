# Imports {{{
from collections import namedtuple
from functools import partial
import itertools
from numpy import ndarray, zeros
# }}}

# Types {{{
SparseTensor = namedtuple("SparseTensor",["dimensions","chunks"])
# }}}

# Functions {{{
def formDenseTensor(sparse_tensor,toArray=None,shape=None,dtype=None): # {{{
    if toArray is None:
        toArray = lambda x: x
    dimensions = sparse_tensor.dimensions
    if sparse_tensor.chunks:
        value = toArray(next(iter(sparse_tensor.chunks.values())))
        if dtype is None:
            if hasattr(value,"dtype"):
                dtype = value.dtype
            else:
                dtype = type(value)
        if shape is None:
            if hasattr(value,"shape"):
                shape = value.shape
            else:
                shape = ()
    if shape is None:
        shape = ()
    dense_tensor = zeros(shape=sparse_tensor.dimensions+shape,dtype=dtype)
    for indices, value in sparse_tensor.chunks.items():
        dense_tensor[indices] += toArray(value)
    return dense_tensor
# }}}
def formSparseContractor(left_join_dimensions,right_join_dimensions,result_dimension_sources,contractChunks=None): # {{{
    # Compute the numbers of dimensions {{{
    number_of_join_dimensions = len(left_join_dimensions)
    assert(number_of_join_dimensions == len(right_join_dimensions))
    number_of_result_dimensions = len(result_dimension_sources)
    number_of_left_dimensions = number_of_join_dimensions + sum(x.number_of_left_dimensions for x in result_dimension_sources)
    number_of_right_dimensions = number_of_join_dimensions + sum(x.number_of_right_dimensions for x in result_dimension_sources)
    # }}}
    # Check that no left join dimensions are repeated {{{
    observed_left_dimensions = set()
    for dimension in left_join_dimensions:
        if dimension in observed_left_dimensions:
            raise ValueError("left join dimension {} appears more than once".format(dimension))
        observed_left_dimensions.add(dimension)
    # }}}
    # Check that no right join dimensions are repeated {{{
    observed_right_dimensions = set()
    for dimension in right_join_dimensions:
        if dimension in observed_right_dimensions:
            raise ValueError("right join dimension {} appears more than once".format(dimension))
        observed_right_dimensions.add(dimension)
    # }}}
    # Check that the result dimensions do not conflict with the join dimensions or each other {{{
    for result_dimension_source in result_dimension_sources:
        for dimension in result_dimension_source.left_dimensions:
            if dimension in observed_left_dimensions:
                if dimension in left_join_dimensions:
                    raise ValueError("left dimension {} appears both as a join dimension and as a result dimension".format(dimension))
                else:
                    raise ValueError("left dimension {} appears more than once in the result dimensions".format(dimension))
            observed_left_dimensions.add(dimension)
        for dimension in result_dimension_source.right_dimensions:
            if dimension in observed_right_dimensions:
                if dimension in right_join_dimensions:
                    raise ValueError("right dimension {} appears both as a join dimension and as a result dimension".format(dimension))
                else:
                    raise ValueError("right dimension {} appears more than once in the result dimensions".format(dimension))
            observed_right_dimensions.add(dimension)
    # }}}
    # Check that the obserbed numbers of dimensions are consistent with the expected numbers {{{
    missing_left_dimensions = frozenset(range(max(observed_left_dimensions)+1 if observed_left_dimensions else 0))-observed_left_dimensions
    if missing_left_dimensions:
        raise ValueError("The following left dimensions do not appear in the arguments: {}".format(missing_left_dimensions))
    assert number_of_left_dimensions == len(observed_left_dimensions)
    missing_right_dimensions = frozenset(range(max(observed_right_dimensions)+1 if observed_right_dimensions else 0))-observed_right_dimensions
    if missing_right_dimensions:
        raise ValueError("The following right dimensions do not appear in the arguments: {}".format(missing_right_dimensions))
    assert number_of_right_dimensions == len(observed_right_dimensions)
    # }}}
    def absorb(contractChunks,left_sparse_tensor,right_sparse_tensor): # {{{
        # Cache the components of the tensors {{{
        left_dimensions = left_sparse_tensor.dimensions
        left_chunks = left_sparse_tensor.chunks
        right_dimensions = right_sparse_tensor.dimensions
        right_chunks = right_sparse_tensor.chunks
        # }}}
        # Check that the tensors have the correct numbers of dimensions {{{
        if len(left_dimensions) != number_of_left_dimensions:
            raise ValueError("The left tensor was expected to have {} dimensions but it actually has {} dimensions.".format(number_of_left_dimensions,len(left_dimensions)))
        if len(right_dimensions) != number_of_right_dimensions:
            raise ValueError("The right tensor was expected to have {} dimensions but it actually has {} dimensions.".format(number_of_right_dimensions,len(right_dimensions)))
        # }}}
        # Compute the result dimensions {{{
        result_dimensions = tuple(dimension_source.getResultDimension(left_dimensions,right_dimensions) for dimension_source in result_dimension_sources)
        # }}}
        # Determine which join indices overlap between the left and right, and from them create a dictionary of chunk lists {{{
        chunk_lists = {index: ([],[]) for index in (
            frozenset(tuple(indices[join_dimension] for join_dimension in left_join_dimensions) for indices in left_chunks.keys()) &
            frozenset(tuple(indices[join_dimension] for join_dimension in right_join_dimensions) for indices in right_chunks.keys())
        )}
        # }}}
        # Add all overlapping left chunks to the chunk list {{{
        for indexed_chunk in left_chunks.items():
            indices = indexed_chunk[0]
            join_indices = tuple(indices[join_dimension] for join_dimension in left_join_dimensions)
            if join_indices in chunk_lists:
                chunk_lists[join_indices][0].append(indexed_chunk)
        # }}}
        # Add all overlapping right chunks to the chunk list {{{
        for indexed_chunk in right_chunks.items():
            indices = indexed_chunk[0]
            join_indices = tuple(indices[join_dimension] for join_dimension in right_join_dimensions)
            if join_indices in chunk_lists:
                chunk_lists[join_indices][1].append(indexed_chunk)
        # }}}
        # Compute the result chunks {{{
        result_chunks = {}
        for indexed_chunk_lists_pair in chunk_lists.items():
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
    if contractChunks is not None:
        return partial(absorb,contractChunks)
    else:
        return absorb
# }}}
def mapSparseChunkValues(f,tensor): # {{{
    return SparseTensor(tensor.dimensions,{x: f(y) for (x,y) in tensor.chunks.items()})
# }}}
# }}}

# Exports {{{
__all__ = [
    "SparseTensor",

    "formDenseTensor",
    "formSparseContractor",
    "mapSparseChunkValues",
]# }}}

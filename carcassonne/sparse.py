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
def formDenseTensor(sparse_tensor,dtype=None): # {{{
    dimensions = sparse_tensor.dimensions
    if not dtype and sparse_tensor.chunks:
        value = next(iter(sparse_tensor.chunks.values()))
        if isinstance(value,ndarray):
            dtype = value.dtype
            dimensions +=  value.shape
        else:
            dtype = type(value)
    dense_tensor = zeros(shape=dimensions,dtype=dtype)
    for indices, value in sparse_tensor.chunks.items():
        dense_tensor[indices] += value
    return dense_tensor
# }}}
def formSparseContractor(left_join_dimensions,right_join_dimensions,result_dimension_sources,contractChunks=None): # {{{
    # Compute the numbers of dimensions {{{
    number_of_join_dimensions = len(left_join_dimensions)
    assert(number_of_join_dimensions == len(right_join_dimensions))
    number_of_result_dimensions = len(result_dimension_sources)
    # }}}
    def absorb(contractChunks,left_sparse_tensor,right_sparse_tensor): # {{{
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

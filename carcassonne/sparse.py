# Imports {{{
from collections import namedtuple
from itertools import product
# }}}

# Types {{{
class IndexRange(object):
    __slots__ = ["start","end"]

    def __init__(self,start,end):
        assert start <= end
        self.start = start
        self.end = end

    def __eq__(self,other):
        return self.start == other.start and self.end == other.end

    def __lt__(self,other):
        return self.start < other.start and self.end <= other.start

    def __gt__(self,other):
        return self.start >= other.end and self.start > other.start
# }}}

def formSparseAbsorber(left_join_index,right_join_index,right_external_index,merged_indices):
    # Compute and check the rank {{{
    left_nrank = 1 + len(merged_indices)
    for index in xrange(left_nrank):
        if not (index in merged_indices or index == right_external_index):
            raise ValueError("Computed {} indices for the left tensor, but unable to identify a source for index {}".format(left_nrank,index)
    # }}}
    def absorb(left_sparse_chunks, left_sparse_dimensions, right_sparse_chunks, right_sparse_dimensions):
        # Compute the new sparse dimensions {{{
        new_sparse_dimensions = copy(left_sparse_dimensions)
        for index in xrange(left_nrank):
            if index in merged_indices:
                new_sparse_dimensions[index] *= right_sparse_dimensions[merged_indices[index]]
            else
                new_sparse_dimensions[index]  = right_external_index
        # }}}
        # Determine which join indices overlap between the left and right, and from them create a dictionary of chunk lists {{{
        chunk_lists = {index: ([],[]) for index in (
            frozenset(indices[left_join_index] for indices in left_sparse_chunks.iterkeys()) &
            frozenset(indices[right_join_index] for indices in right_sparse_chunks.iterkeys())
        )}
        # }}}
        # Add all overlapping left chunks to the chunk list {{{
        for indices, chunk in left_sparse_chunks.iteritems():
            index = indices[left_join_index] 
            if index in chunk_lists:
                chunk_lists[index][0].append((indices,chunk))
        # }}}
        # Add all overlapping right chunks to the chunk list {{{
        for indices, chunk in right_sparse_chunks.iteritems():
            index = indices[right_join_index] 
            if index in chunk_lists:
                chunk_lists[index][1].append((indices,chunk))
        # }}}
        # Initialize the resulting chunks {{{
        new_sparse_chunks = type(left_sparse_chunks)()
        new_index_as_list = [None]*left_nrank
        # }}}
        for (left_indexed_chunks,right_indexed_chunks) in chunk_lists.iteriterms():
            

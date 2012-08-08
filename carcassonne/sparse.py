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
    # Compute and check the ranks {{{
    left_nrank = 1 + len(merged_indices)
    for index in xrange(left_nrank):
        if not (index in merged_indices or index == right_external_index):
            raise ValueError("Computed {} indices for the left tensor, but unable to identify a source for index {}".format(left_nrank,index)
    right_nrank = left_nrank + 1 if right_external_index is not None else 0
    sparse_nrank = right_nrank - 1
    # }}}
    def absorb(left_sparse_chunks, left_sparse_dimensions, right_sparse_chunks, right_sparse_dimensions):
        # Compute the new sparse dimensions {{{
        new_sparse_dimensions = copy(left_sparse_dimensions)
        for index in xrange(left_nrank):
            if index in merged_indices:
                new_sparse_dimensions[index] *= right_sparse_dimensions[merged_indices[index]]
            else
                new_sparse_dimensions[index]  = right_sparse_dimensions[right_external_index] if right_external_index is not None else None
        if right_external_index is None:
            del new_sparse_dimensopns[right_external_index]
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
        new_sparse_dimensions = [0] * sparse_nrank
        new_sparse_chunks = type(left_sparse_chunks)()
        new_index_as_list = [None]*left_nrank
        destination_table = ({} for _ in xrange(left_nrank))
        for indexed_chunk_lists_pair in chunk_lists.iteriterms():
            for ((left_indices,left_chunk),(right_indices,right_chunk)) in izip(*indexed_chunk_lists_pair):
                new_indices = []
                for i in xrange(left_nrank):
                    if i in merged_indices:
                        new_indices.append((left_indices[i],right_indices[merged_indices[i]]))
                    elif right_external_index is not None:
                        new_indices.append(right_external_index)
                if new_indices in destination_table:
                    new_indices = destination_table[new_indices]
                else:
                    

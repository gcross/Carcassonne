#ifndef CARCASSONNE_SPARSE_HPP
#define CARCASSONNE_SPARSE_HPP

#include <boost/array.hpp>
#include <boost/foreach.hpp>
#include <boost/function.hpp>
#include <boost/optional.hpp>
#include <boost/range/algorithm/transform.hpp>
#include <boost/range/irange.hpp>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>
#include <boost/utility.hpp>
#include <boost/utility/result_of.hpp>
#include <boost/variant.hpp>

#include "either.hpp"

namespace Carcassonne {

// Simple type aliases {{{
typedef size_t Dimension;
typedef uint64_t DimensionSize;
typedef DimensionSize Index;
// }}}

struct DimensionMergeInformation { // {{{
    Dimension left_dimension, right_dimension;
    boost::unordered_set<std::pair<Index,Index> > indices_to_ignore;
    boost::unordered_map<std::pair<Index,Index>,Index> indices_to_sum;
}; // }}}

typedef boost::variant<Left<Dimension>,Right<Dimension>,DimensionMergeInformation> ResultDimensionInformation;

namespace SparseTensorImplementation { // {{{

using boost::array;
using boost::irange;
using boost::none;
using boost::optional;

template<Dimension number_of_left_dimensions, Dimension number_of_right_dimensions> struct GetResultDimension : std::unary_function<DimensionSize,ResultDimensionInformation> { // {{{
    array<DimensionSize,number_of_left_dimensions> const& left_dimensions;
    array<DimensionSize,number_of_right_dimensions> const& right_dimensions;

    GetResultDimension(
        array<DimensionSize,number_of_left_dimensions> const& left_dimensions
       ,array<DimensionSize,number_of_right_dimensions> const& right_dimensions
    ) : left_dimensions(left_dimensions)
      , right_dimensions(right_dimensions)
    {}

    DimensionSize operator()(Left<Dimension> const& left_dimension) const { return left_dimensions[left_dimension]; }
    DimensionSize operator()(Right<Dimension> const& right_dimension) const { return right_dimensions[right_dimension]; }
    DimensionSize operator()(DimensionMergeInformation const& merge_information) const {
        return left_dimensions[merge_information.left_dimension] * right_dimensions[merge_information.right_dimension];
    }
};
template<
    Dimension number_of_left_dimensions
   ,Dimension number_of_right_dimensions
> GetResultDimension<number_of_left_dimensions, number_of_right_dimensions> makeGetResultDimension(
    array<DimensionSize,number_of_left_dimensions> const& left_dimensions
   ,array<DimensionSize,number_of_right_dimensions> const& right_dimensions
) {
    return GetResultDimension<number_of_left_dimensions,number_of_right_dimensions>(left_dimensions,right_dimensions);
}
// }}}

template<Dimension number_of_left_dimensions, Dimension number_of_right_dimensions> struct GetResultIndex : std::unary_function<optional<Dimension>,ResultDimensionInformation> { // {{{
    array<Dimension,number_of_right_dimensions> const& right_dimensions;
    array<Index,number_of_left_dimensions> const& left_indices;
    array<Index,number_of_right_dimensions> const& right_indices;

    GetResultIndex(
        array<Dimension,number_of_right_dimensions> const& right_dimensions
       ,array<Index,number_of_left_dimensions> const& left_indices
       ,array<Index,number_of_right_dimensions> const& right_indices
    ) : right_dimensions(right_dimensions)
      , left_indices(left_indices)
      , right_indices(right_indices)
    {}

    optional<Dimension> operator()(Left<Dimension> const& left_dimension) const { return left_indices[left_dimension]; }
    optional<Dimension> operator()(Right<Dimension> const& right_dimension) const { return right_indices[right_dimension]; }

    optional<Dimension> operator()(DimensionMergeInformation const& merge_information) const {
        std::pair<Index,Index> index_pair(left_indices[merge_information.left_dimension],right_indices[merge_information.right_dimension]);
        if(merge_information.indices_to_ignore.find(index_pair) != merge_information.indices_to_ignore.end()) return none;
        boost::unordered_map<std::pair<Index,Index>,Index>::const_iterator destination_index_ptr = merge_information.indices_to_sum.find(index_pair);
        if(destination_index_ptr != merge_information.indices_to_sum.end()) { return destination_index_ptr->second; }
        return index_pair.first * number_of_right_dimensions + index_pair.second;
    }
};
template<
    Dimension number_of_left_dimensions
   ,Dimension number_of_right_dimensions
> GetResultIndex<number_of_left_dimensions, number_of_right_dimensions> makeGetResultIndex(
    array<Index,number_of_right_dimensions> const& right_dimensions
   ,array<Index,number_of_left_dimensions> const& left_indices
   ,array<Index,number_of_right_dimensions> const& right_indices
) {
    return GetResultIndex<number_of_left_dimensions,number_of_right_dimensions>(right_dimensions,left_indices,right_indices);
}
// }}}

// function populateChunkLists {{{
template<
    typename IndexedChunks
   ,typename IndexedChunkLists
   ,Dimension number_of_join_dimensions
> void populateChunkLists(
    boost::array<Dimension,number_of_join_dimensions> const& join_dimensions
   ,IndexedChunks const& chunks
   ,IndexedChunkLists const& chunk_lists
) {
    BOOST_FOREACH(typename IndexedChunks::value_type const& chunk, chunks) {
        boost::array<Index,number_of_join_dimensions> join_indices;
        BOOST_FOREACH(Dimension const i, irange((Dimension)0u,number_of_join_dimensions)) {
            join_indices[i] = chunk.first[join_dimensions[i]];
        }
        chunk_lists[join_indices].append(&chunk);
    }
} // }}}

} // }}}

template<typename Chunk, Dimension number_of_dimensions_> class SparseTensor : boost::noncopyable { // {{{
private:
    BOOST_MOVABLE_BUT_NOT_COPYABLE(SparseTensor)

public:
    // Constants {{{
    Dimension const static number_of_dimensions;
    // }}}

    // Type aliases {{{
    typedef boost::array<DimensionSize,number_of_dimensions_> Dimensions;
    typedef Dimensions Indices;
    typedef boost::unordered_map<Indices,Chunk> IndexedChunks;
    typedef typename IndexedChunks::value_type IndexedChunk;
    // }}}

    // Fields {{{
    Dimensions dimensions;
    IndexedChunks chunks;
    // }}}

    // Constructors {{{
    SparseTensor(Dimensions const& dimensions) : dimensions(dimensions) {}

    SparseTensor(BOOST_RV_REF(SparseTensor) other) : dimensions(other.dimensions), chunks(boost::move(chunks)) {}
    // }}}

    // Operators {{{
    SparseTensor& operator=(BOOST_RV_REF(SparseTensor) other) {
        if(this != &other) {
            dimensions = other.dimensions;
            chunks = boost::move(chunks);
        }
        return; 
    }
    // }}}

    // Methods {{{

    // method contractSparseTensors {{{
    template<
         typename RightChunk
        ,Dimension number_of_right_dimensions
        ,typename ResultChunk
        ,Dimension number_of_joined_dimensions
    > SparseTensor<ResultChunk,number_of_dimensions_ + number_of_right_dimensions - 2u*number_of_joined_dimensions> contractWith(
         SparseTensor<RightChunk,number_of_right_dimensions> const& right_tensor
        ,boost::array<Dimension,number_of_joined_dimensions> const& left_join_dimensions
        ,boost::array<Dimension,number_of_joined_dimensions> const& right_join_dimensions
        ,boost::array<ResultDimensionInformation,number_of_dimensions_ + number_of_right_dimensions - 2u*number_of_joined_dimensions> result_dimension_sources
        ,boost::function<ResultChunk(Chunk,RightChunk)> contractChunks
    ) const {
        // Usings {{{
        using boost::array;
        using boost::irange;
        using boost::unordered_map;
        using std::pair;
        using std::vector;

        using namespace SparseTensorImplementation;
        // }}}

        Dimension const number_of_result_dimensions = number_of_dimensions_ + number_of_right_dimensions - 2u*number_of_joined_dimensions;

        // Define some type aliases for convenience {{{
        typedef array<Index,number_of_joined_dimensions> JoinDimensions;

        typedef Chunk LeftChunk;
        typedef SparseTensor LeftSparseTensor;
        typedef typename LeftSparseTensor::Dimensions LeftDimensions;
        typedef typename LeftSparseTensor::Indices LeftIndices;
        typedef typename LeftSparseTensor::IndexedChunk LeftIndexedChunk;
        typedef typename LeftSparseTensor::IndexedChunks LeftIndexedChunks;
        typedef vector<LeftIndexedChunk const*> LeftIndexedChunkList;
        typedef unordered_map<JoinDimensions,LeftIndexedChunkList> LeftJoinIndexedChunkLists;
        typedef typename LeftJoinIndexedChunkLists::value_type LeftJoinIndexedChunkList;

        typedef SparseTensor<RightChunk,number_of_right_dimensions> RightSparseTensor;
        typedef typename RightSparseTensor::Dimensions RightDimensions;
        typedef typename RightSparseTensor::Indices RightIndices;
        typedef typename RightSparseTensor::IndexedChunk RightIndexedChunk;
        typedef typename RightSparseTensor::IndexedChunks RightIndexedChunks;
        typedef vector<RightIndexedChunk const*> RightIndexedChunkList;
        typedef unordered_map<JoinDimensions,RightIndexedChunkList> RightJoinIndexedChunkLists;
        typedef typename RightJoinIndexedChunkLists::value_type RightJoinIndexedChunkList;

        typedef SparseTensor<ResultChunk,number_of_dimensions_ + number_of_right_dimensions - 2u*number_of_joined_dimensions> ResultSparseTensor;
        typedef typename ResultSparseTensor::Dimensions ResultDimensions;
        typedef typename ResultSparseTensor::Indices ResultIndices;
        typedef typename ResultSparseTensor::IndexedChunk ResultIndexedChunk;
        typedef typename ResultSparseTensor::IndexedChunks ResultIndexedChunks;
        // }}}

        // Define some value aliases for convenience {{{
        LeftSparseTensor const& left_tensor = *this;
        ResultSparseTensor result_tensor;

        LeftIndexedChunks const& left_chunks  = left_tensor.chunks;
        LeftDimensions const& left_dimensions = left_tensor.dimensions;

        RightIndexedChunks const& right_chunks = right_tensor.chunks;
        RightDimensions const& right_dimensions = right_tensor.dimensions;

        ResultIndexedChunks const& result_chunks = result_tensor.chunks;
        ResultDimensions const& result_dimensions = result_tensor.dimensions;
        // }}}

        // Compute the dimensions of the resulting tensor {{{
        boost::transform(result_dimension_sources,result_dimensions.begin(),boost::apply_visitor(makeGetResultDimension(left_dimensions,right_dimensions)));
        // }}}

        // Construct the join chunk lists {{{
        LeftJoinIndexedChunkLists left_join_chunk_lists;
        populateChunkLists(left_join_dimensions,left_chunks,left_join_chunk_lists);

        RightJoinIndexedChunkLists right_join_chunk_lists;
        populateChunkLists(right_join_dimensions,right_chunks,right_join_chunk_lists);
        // }}}

        // Iterate over all the join indices and associated chunk lists in the left chunk lists {{{
        BOOST_FOREACH(LeftJoinIndexedChunkList const& left_indexed_chunk_list, left_join_chunk_lists) {
            // See if there is a right chunk with the same join indices; otherwise move on {{{
            typename RightJoinIndexedChunkLists::const_iterator const right_indexed_chunk_list_ptr = right_join_chunk_lists.find(left_indexed_chunk_list.first);
            if(right_indexed_chunk_list_ptr == right_join_chunk_lists.end()) continue;
            // }}}

            // Create aliases for the left and right chunk lists {{{
            LeftJoinIndexedChunkList const& left_chunk_list = left_indexed_chunk_list.second;
            RightJoinIndexedChunkList const& right_chunk_list = right_indexed_chunk_list_ptr->second;
            // }}}

            // Now we iterate over all the left and right chunks in the Cartesian product of the two lists {{{
            BOOST_FOREACH(LeftIndexedChunk const* left_indexed_chunk_ptr, left_chunk_list) {
                // Create aliases for the left indices and chunk {{{
                LeftIndices const& left_indices = left_indexed_chunk_ptr->first;
                LeftChunk const& left_chunk = left_indexed_chunk_ptr->second;
                // }}}
                BOOST_FOREACH(RightIndexedChunk const* right_indexed_chunk_ptr, right_chunk_list) {
                    // Create alises for the right indices and chunk {{{
                    RightIndices const& right_indices = right_indexed_chunk_ptr->first;
                    RightChunk const& right_chunk = right_indexed_chunk_ptr->second;
                    // }}}

                    // Compute the result indices, aborting this term if one of them is in the ignored set {{{
                    ResultIndices indices;
                    bool skip_chunk = false;
                    BOOST_FOREACH(size_t const i, irange((Dimension)0u,number_of_result_dimensions)) {
                        boost::optional<Index> maybe_index = boost::apply_visitor(makeGetResultIndex(right_dimensions,left_indices,right_indices),result_dimension_sources[i]);
                        if(maybe_index) {
                            indices[i] = *maybe_index;
                        } else {
                            skip_chunk = true;
                        }
                    }
                    if(skip_chunk) continue;
                    // }}}

                    // Contract the left and right chunks and store the result in the result chunks {{{
                    ResultChunk chunk = contractChunks(left_chunk,right_chunk);
                    typename ResultIndexedChunks::iterator existing_indexed_chunk_ptr = result_chunks.find(indices);
                    if(existing_indexed_chunk_ptr == result_chunks.end()) {
                        chunks.emplace(indices,boost::move(chunk));
                    } else {
                        *existing_indexed_chunk_ptr += chunk;
                    }
                    // }}}
                }
            } // }}}
        } // }}}

        return ResultSparseTensor(result_dimensions,boost::move(result_chunks));

        // Return the result {{{
        return ResultSparseTensor(result_dimensions,boost::move(result_chunks));
        // }}}
    } // }}}

    // Methods }}}

}; // }}}
template<typename Chunk, Dimension number_of_dimensions_>
    Dimension const SparseTensor<Chunk,number_of_dimensions_>::number_of_dimensions = number_of_dimensions_;

} // Namespace Carcassonne
#endif

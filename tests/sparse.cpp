// Includes {{{
#include "sparse.hpp"

#include "test_utils.hpp"
// }}}

// Usings {{{
using boost::array;
using boost::none;
using boost::none_t;

using namespace Carcassonne;
// }}}

TEST_SUITE(Sparse) { // {{{
    TEST_SUITE(none_t) { // {{{
        TEST_CASE(scalar_is_constructable) {
            array<DimensionSize,0u> dimensions;
            SparseTensor<none_t,0u> tensor(dimensions);
        }
        TEST_CASE(scalar_is_contractable) {
            array<DimensionSize,0u> dimensions;
            SparseTensor<none_t,0u> left(dimensions), right(dimensions);

            array<Dimension,0u> join_dimensions;
            array<ResultDimensionInformation,0u> result_dimension_sources;
            left.contractWith<none_t,0u,ConstantAbsorber<none_t,none_t,none_t>,0u>(
                right
               ,join_dimensions
               ,join_dimensions
               ,result_dimension_sources
               ,makeConstantAbsorber(none)
            );
        }
    } // }}}
} // }}}

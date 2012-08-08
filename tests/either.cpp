// Includes {{{
#include <boost/move/move.hpp>

#include "either.hpp"

#include "test_utils.hpp"
// }}}

TEST_SUITE(Either) { // {{{
    TEST_SUITE(Left) { // {{{
        TEST_CASE(constructors_work_correctly) { // {{{
            CopyMoveRecorder recorder1, recorder2;
            Left<CopyMoveRecorder> l1(recorder1), l2(boost::move(recorder2));
            ASSERT_TRUE(recorder1.copied);
            ASSERT_FALSE(recorder1.moved);
            ASSERT_FALSE(recorder2.copied);
            ASSERT_TRUE(recorder2.moved);
        } // }}}
        TEST_CASE(assignment_works_correctly) { // {{{
            Left<CopyMoveRecorder> ol1, ol2, nl1, nl2;
            nl1 = ol1;
            nl2 = boost::move(ol2);
            ASSERT_TRUE(ol1.value.copied);
            ASSERT_FALSE(ol1.value.moved);
            ASSERT_FALSE(ol2.value.copied);
            ASSERT_TRUE(ol2.value.moved);
        } // }}}
    } // }}}
    TEST_SUITE(Right) { // {{{
        TEST_CASE(constructors_work_correctly) { // {{{
            CopyMoveRecorder recorder1, recorder2;
            Right<CopyMoveRecorder> l1(recorder1), l2(boost::move(recorder2));
            ASSERT_TRUE(recorder1.copied);
            ASSERT_FALSE(recorder1.moved);
            ASSERT_FALSE(recorder2.copied);
            ASSERT_TRUE(recorder2.moved);
        } // }}}
        TEST_CASE(assignment_works_correctly) { // {{{
            Right<CopyMoveRecorder> ol1, ol2, nl1, nl2;
            nl1 = ol1;
            nl2 = boost::move(ol2);
            ASSERT_TRUE(ol1.value.copied);
            ASSERT_FALSE(ol1.value.moved);
            ASSERT_FALSE(ol2.value.copied);
            ASSERT_TRUE(ol2.value.moved);
        } // }}}
    } // }}}
} // }}}

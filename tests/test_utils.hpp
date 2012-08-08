#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

// Includes {{{
#include <illuminate.hpp>

#include <boost/move/move.hpp>
// }}}

// Usings {{{
using namespace Carcassonne;
// }}}

// Helper Classes {{{
struct CopyMoveRecorder {
private:
    BOOST_COPYABLE_AND_MOVABLE(CopyMoveRecorder);
public:
    mutable bool copied, moved;

    CopyMoveRecorder() : copied(false), moved(false) {}
    CopyMoveRecorder(CopyMoveRecorder const& other) : copied(false), moved(false) { other.copied = true; }
    CopyMoveRecorder(BOOST_RV_REF(CopyMoveRecorder) other) : copied(false), moved(false) { other.moved = true; }

    CopyMoveRecorder& operator=(CopyMoveRecorder const& other) {
        other.copied = true;
        return *this;
    }
    CopyMoveRecorder& operator=(BOOST_RV_REF(CopyMoveRecorder) other) {
        other.moved = true;
        return *this;
    }
};
// }}}

#endif

#ifndef CARCASSONNE_EITHER_HPP
#define CARCASSONNE_EITHER_HPP

#include <boost/move/move.hpp>

namespace Carcassonne {

#define GENERATE_LEFT_OR_RIGHT(Which) \
    template<typename T> class Which { \
    private: \
        BOOST_COPYABLE_AND_MOVABLE(Which) \
    public: \
        T value; \
        Which() {} \
        Which(T const& value) : value(value) {} \
        Which(BOOST_RV_REF(T) value) : value(value) {} \
        Which(Which const& other) : value(other.value) {} \
        Which& operator=(BOOST_COPY_ASSIGN_REF(Which) other) { \
            if(this != &other) { value = other.value; } \
            return *this; \
        } \
        Which(BOOST_RV_REF(Which) other) : value(boost::move(other.value)) {} \
        Which& operator=(BOOST_RV_REF(Which) other) { \
            if(this != &other) { value = boost::move(other.value); } \
            return *this; \
        } \
    };

GENERATE_LEFT_OR_RIGHT(Left)
GENERATE_LEFT_OR_RIGHT(Right)

#undef GENERATE_LEFT_OR_RIGHT

}

#endif

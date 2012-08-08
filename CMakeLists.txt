cmake_minimum_required (VERSION 2.8.8)
project (Carcassonne CXX)

set(VERSION 1)

# Find packages {{{
# Boost {{{
find_package( Boost 1.48 COMPONENTS filesystem program_options signals system thread REQUIRED )
link_directories ( ${Boost_LIBRARY_DIRS} )
include_directories ( ${Boost_INCLUDE_DIRS} )
# }}}
# }}}

include_directories ("includes")

# add_subdirectory ("sources")

# Configure tests {{{
find_package( Illuminate )

if(${Illuminate_FOUND})
    add_subdirectory ("tests")
else(${Illuminate_FOUND})
    add_custom_target(test
        COMMAND ${CMAKE_COMMAND} -E echo "The Illuminate library is required to build the tests."
    )
endif(${Illuminate_FOUND})
# }}}

project(twinning)
cmake_minimum_required(VERSION 2.8)
FIND_PACKAGE( Boost 1.40 COMPONENTS program_options REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )
add_executable(alg2_1 "alg2_1.cpp" )
add_executable(alg3_1 "alg3_1.cpp" )
TARGET_LINK_LIBRARIES(alg2_1 ${Boost_LIBRARIES})
TARGET_LINK_LIBRARIES(alg3_1 ${Boost_LIBRARIES})
set_target_properties(alg2_1 PROPERTIES CXX_STANDARD 14)
set_target_properties(alg3_1 PROPERTIES CXX_STANDARD 14)
INCLUDE_DIRECTORIES ( "~/src/wmath" )
INCLUDE_DIRECTORIES ( "~/src/asu3" )
INCLUDE_DIRECTORIES ( "/usr/include/eigen3/" )
INCLUDE_DIRECTORIES ( "~/src/LBFGSpp/include" )
aux_source_directory(. SRC_LIST)
add_library(wmath SHARED IMPORTED)
SET_TARGET_PROPERTIES(wmath PROPERTIES LINKER_LANGUAGE CXX CXX_STANDARD 14)
SET(GCC_COVERAGE_COMPILE_FLAGS "${GCC_COVERAGE_COMPILE_FLAGS} -std=c++14 -lboost_system -lboost_thread -pthread  -fext-numeric-literals -march=native")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS} -std=c++14 -lboost_system -lboost_thread -pthread -fext-numeric-literals -O3 -Wfatal-errors" )

cmake_minimum_required (VERSION 3.1)

option(CM_ALL_VTK_OUTPUT "VTK output routine" OFF)
option(CM_ALL_VORONOI "Voronoi-based loadbalancing scheme (requires Voro++)" OFF)
option(CM_ALL_FORTRAN "VTK output routine" OFF)

enable_language(CXX)

if (CM_ALL_FORTRAN)
    enable_language(Fortran)
endif (CM_ALL_FORTRAN)

set(CMAKE_CXX_STANDARD 11)

project(ALL)

find_package(MPI REQUIRED)

if(CM_ALL_VTK_OUTPUT)
    message("Using VTK output")
    FIND_PACKAGE(VTK REQUIRED)
    if(NOT VTK_FOUND)
        message(FATAL_ERROR "VTK not found, help CMake to find it by setting VTK_LIBRARY and VTK_INCLUDE_DIR")
    endif()
    add_compile_options("-DALL_VTK_OUTPUT")
endif(CM_ALL_VTK_OUTPUT)

if(CM_ALL_VORONOI)
    message(STATUS "compiling voro++ version in contrib/voro++")
    add_subdirectory(contrib/voro++)
    set(CM_ALL_VORO_INC ../contrib/voro++/include)
    set(CM_ALL_VORO_LIB "-L../contrib/voro++/src -lvoro")
    add_compile_options("-DALL_VORONOI")
endif()

add_subdirectory(src)
add_subdirectory(example)


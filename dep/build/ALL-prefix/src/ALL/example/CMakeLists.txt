add_executable(ALL_test ALL_test.cpp)
if(CM_ALL_VTK_OUTPUT)
    target_include_directories(ALL_test PUBLIC ${VTK_INCLUDE_DIRS})
    target_link_libraries(ALL_test PUBLIC ${VTK_LIBRARY_DIRS})
endif(CM_ALL_VTK_OUTPUT)
#if(CM_ALL_VORONOI)
#    message("${CM_ALL_VORO_INC} - ${CM_ALL_VORO_LIB}")
#    target_include_directories(ALL_test PUBLIC ${CM_ALL_VORO_INC})
#    target_link_libraries(ALL_test PUBLIC ${CM_ALL_VORO_LIB})
#endif(CM_ALL_VORONOI)

target_link_libraries (ALL_test LINK_PUBLIC ALL)

install(TARGETS ALL_test 
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        INCLUDES DESTINATION include)

# helper program to convert ascii input to mpi binary input
add_executable(ASCII2MPIBIN ascii2mpibin.cpp)

target_link_libraries(ASCII2MPIBIN LINK_PUBLIC ALL)

install(TARGETS ASCII2MPIBIN
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        INCLUDES DESTINATION include)

if(CM_ALL_FORTRAN)
    add_executable(ALL_test_f ALL_test_f.f90)
    if(CM_ALL_VTK_OUTPUT)
        target_include_directories(ALL_test_f PUBLIC ${VTK_INCLUDE_DIRS})
        target_link_libraries(ALL_test_f PUBLIC ${VTK_LIBRARY_DIRS})
    endif(CM_ALL_VTK_OUTPUT)
    #if(CM_ALL_VORONOI)
    #    target_include_directories(ALL_test_f PUBLIC ${CM_ALL_VORO_INC})
    #    target_link_libraries(ALL_test_f PUBLIC ${CM_ALL_VORO_LIB})
    #endif(CM_ALL_VORONOI)

    target_link_libraries(ALL_test_f LINK_PUBLIC ALL)
    target_link_libraries(ALL_test_f LINK_PUBLIC ALL_fortran)
    target_include_directories(ALL_test_f PUBLIC ${CMAKE_BINARY_DIR}/modules)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR}/modules)
    set_property(TARGET ALL_test_f PROPERTY LINKER_LANGUAGE Fortran)

    install(TARGETS ALL_test_f
            RUNTIME DESTINATION bin
            LIBRARY DESTINATION lib
            INCLUDES DESTINATION include)
endif(CM_ALL_FORTRAN)    

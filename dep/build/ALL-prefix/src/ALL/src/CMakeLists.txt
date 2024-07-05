set(ALL_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(ALL_HEADER_FILES ${ALL_INCLUDE_DIR}/ALL.hpp 
                     ${ALL_INCLUDE_DIR}/ALL_Staggered.hpp 
                     ${ALL_INCLUDE_DIR}/ALL_Tensor.hpp
                     ${ALL_INCLUDE_DIR}/ALL_Unstructured.hpp
                     ${ALL_INCLUDE_DIR}/ALL_Voronoi.hpp
                     ${ALL_INCLUDE_DIR}/ALL_Point.hpp
                     ${ALL_INCLUDE_DIR}/ALL_CustomExceptions.hpp)


add_library (ALL ALL.cpp ALL_fortran.cpp ${ALL_HEADER_FILES})

if(CM_ALL_VTK_OUTPUT)
    target_include_directories(ALL PUBLIC ${VTK_INCLUDE_DIRS})
    target_link_libraries(ALL PUBLIC ${VTK_LIBRARIES})
endif(CM_ALL_VTK_OUTPUT)
if(CM_ALL_VORONOI)
    target_include_directories(ALL PUBLIC ${CM_ALL_VORO_INC})
    target_link_libraries(ALL PUBLIC ${CM_ALL_VORO_LIB})
endif(CM_ALL_VORONOI)

target_include_directories(ALL PUBLIC ${MPI_CXX_INCLUDE_PATH})
target_include_directories (ALL PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${ALL_INCLUDE_DIR})
target_link_libraries(ALL PUBLIC ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES})

install(TARGETS ALL 
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        INCLUDES DESTINATION include)

install(FILES ${ALL_HEADER_FILES}
        DESTINATION include)

if (CM_ALL_FORTRAN)
    set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/modules)
    add_library (ALL_fortran ALL_module.f90)
    set_property(TARGET ALL_fortran PROPERTY LINKER_LANGUAGE Fortran)

    if(CM_ALL_VTK_OUTPUT)
        target_include_directories(ALL_fortran PUBLIC ${VTK_INCLUDE_DIRS})
        target_link_libraries(ALL_fortran PUBLIC ${VTK_LIBRARIES})
    endif(CM_ALL_VTK_OUTPUT)
    #if(CM_ALL_VORONOI)
    #    target_include_directories(ALL_fortran PUBLIC ${CM_ALL_VORO_INC})
    #    target_link_libraries(ALL_fortran PUBLIC ${CM_ALL_VORO_LIB})
    #endif(CM_ALL_VORONOI)

    target_include_directories(ALL_fortran PUBLIC ${MPI_CXX_INCLUDE_PATH})
    target_include_directories(ALL_fortran PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${ALL_INCLUDE_DIR})
    target_link_libraries(ALL_fortran PUBLIC ${MPI_Fortran_LINK_FLAGS} ${MPI_Fortran_LIBRARIES} ALL)

    install(TARGETS ALL_fortran 
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        INCLUDES DESTINATION include)

    install(DIRECTORY ${CMAKE_BINARY_DIR}/modules DESTINATION include)
endif(CM_ALL_FORTRAN)


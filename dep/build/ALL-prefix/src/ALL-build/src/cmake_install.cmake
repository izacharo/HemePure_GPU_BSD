# Install script for directory: /home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_GPU_BSD/HemePure_GPU_BSD/dep/build/ALL-prefix/src/ALL/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_GPU_BSD/HemePure_GPU_BSD/dep/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_GPU_BSD/HemePure_GPU_BSD/dep/build/ALL-prefix/src/ALL-build/src/libALL.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_GPU_BSD/HemePure_GPU_BSD/dep/build/ALL-prefix/src/ALL/include/ALL.hpp"
    "/home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_GPU_BSD/HemePure_GPU_BSD/dep/build/ALL-prefix/src/ALL/include/ALL_Staggered.hpp"
    "/home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_GPU_BSD/HemePure_GPU_BSD/dep/build/ALL-prefix/src/ALL/include/ALL_Tensor.hpp"
    "/home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_GPU_BSD/HemePure_GPU_BSD/dep/build/ALL-prefix/src/ALL/include/ALL_Unstructured.hpp"
    "/home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_GPU_BSD/HemePure_GPU_BSD/dep/build/ALL-prefix/src/ALL/include/ALL_Voronoi.hpp"
    "/home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_GPU_BSD/HemePure_GPU_BSD/dep/build/ALL-prefix/src/ALL/include/ALL_Point.hpp"
    "/home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_GPU_BSD/HemePure_GPU_BSD/dep/build/ALL-prefix/src/ALL/include/ALL_CustomExceptions.hpp"
    )
endif()


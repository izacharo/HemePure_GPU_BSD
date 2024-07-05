# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

if("/home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_GPU_BSD/HemePure_GPU_BSD/dep/src/libtirpc-1.2.6.tar.bz2" STREQUAL "")
  message(FATAL_ERROR "LOCAL can't be empty")
endif()

if(NOT EXISTS "/home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_GPU_BSD/HemePure_GPU_BSD/dep/src/libtirpc-1.2.6.tar.bz2")
  message(FATAL_ERROR "File not found: /home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_GPU_BSD/HemePure_GPU_BSD/dep/src/libtirpc-1.2.6.tar.bz2")
endif()

if("" STREQUAL "")
  message(WARNING "File will not be verified since no URL_HASH specified")
  return()
endif()

if("" STREQUAL "")
  message(FATAL_ERROR "EXPECT_VALUE can't be empty")
endif()

message(STATUS "verifying file...
     file='/home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_GPU_BSD/HemePure_GPU_BSD/dep/src/libtirpc-1.2.6.tar.bz2'")

file("" "/home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_GPU_BSD/HemePure_GPU_BSD/dep/src/libtirpc-1.2.6.tar.bz2" actual_value)

if(NOT "${actual_value}" STREQUAL "")
  message(FATAL_ERROR "error:  hash of
  /home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_GPU_BSD/HemePure_GPU_BSD/dep/src/libtirpc-1.2.6.tar.bz2
does not match expected value
  expected: ''
    actual: '${actual_value}'
")
endif()

message(STATUS "verifying file... done")

cmake_minimum_required(VERSION 2.8.11)

project(VCG)

#set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

file(GLOB header *.h)
file(GLOB source *.cpp)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

#include_directories(
#  ${CMAKE_CURRENT_SOURCE_DIR}/complex
#  ${CMAKE_CURRENT_SOURCE_DIR}/connectors
#  ${CMAKE_CURRENT_SOURCE_DIR}/container
#  ${CMAKE_CURRENT_SOURCE_DIR}/math
#  ${CMAKE_CURRENT_SOURCE_DIR}/simplex
#  ${CMAKE_CURRENT_SOURCE_DIR}/space  
#)
add_definitions(-D_SILENCE_STDEXT_HASH_DEPRECATION_WARNINGS)
add_definitions(-D_CRT_SECURE_NO_WARNINGS)

#add_executable(VCG ${source} ${header})


add_library(vcg STATIC ${source})

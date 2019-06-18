cmake_minimum_required(VERSION 3.1.0)
project(RobustPureHexMeshing)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Build geogram
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern)
include(geogram.cmake)
set_property(TARGET geogram PROPERTY FOLDER "dependencies")

#find_package(LIBIGL QUIET)
#
#if (NOT LIBIGL_FOUND)
#   message(FATAL_ERROR "libigl not found --- You can download it using: \n git clone --recursive https://github.com/libigl/libigl.git ${PROJECT_SOURCE_DIR}/../libigl")
#endif()
#
# Compilation flags: adapt to your needs
if(MSVC)
  # Enable parallel compilation
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /bigobj /openmp")
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR} )
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR} )
else()
  # Libigl requires a modern C++ compiler that supports c++11
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "." )
endif()

if(WIN32)
  LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}") # Add the directory where FindSuiteSparse.cmake module can be 
  set(SuiteSparse_USE_LAPACK_BLAS ON)
  find_package(SuiteSparse QUIET NO_MODULE)  # 1st: Try to locate the *config.cmake file.
  if(NOT SuiteSparse_FOUND)
          find_package(SuiteSparse REQUIRED) # 2nd: Use FindSuiteSparse.cmake module
          include_directories(${SuiteSparse_INCLUDE_DIRS})
  else()
          message(STATUS "Find SuiteSparse : include(${USE_SuiteSparse})")
          include(${USE_SuiteSparse})
  endif()
  MESSAGE(STATUS "SuiteSparse_LIBS: ${SuiteSparse_LIBRARIES}")
  # ------------------------------------------------------------------
  #   End of SuiteSparse detection
  # ------------------------------------------------------------------
else()
  # ------------------------------------------------------------------
  # Detect SuiteSparse libraries:
  # If not found automatically, set SuiteSparse_DIR in CMake to the 
  # directory where SuiteSparse was built.
  # ------------------------------------------------------------------
  LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/..") # Add the directory where FindSuiteSparse.cmake module can be found.

  find_package(SuiteSparse REQUIRED)  # 1st: Try to locate the *config.cmake file.
  if(CSPARSE_FOUND AND CHOLMOD_FOUND)
          include_directories(${CHOLMOD_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR})
  else()
          #message(FATAL_ERROR "UNABLE to Find SuiteSpars, pass SUITE_SPARSE_ROOT to cmake")
          message("UNABLE to Find SuiteSpars, pass SUITE_SPARSE_ROOT to cmake")
  endif()
  MESSAGE(STATUS "SuiteSparse_LIBS: ${CHOLMOD_LIBRARY} ${CHOLMOD_INCLUDE_DIR}")
  # ------------------------------------------------------------------
  #   End of SuiteSparse detection
  # ------------------------------------------------------------------
endif()

# libigl options: choose between header only and compiled static library
# Header-only is preferred for small projects. For larger projects the static build
# considerably reduces the compilation times
option(LIBIGL_USE_STATIC_LIBRARY "Use LibIGL as static library" ON)

# add a customizable menu bar
option(LIBIGL_WITH_NANOGUI     "Use Nanogui menu"   ON)

# libigl options: choose your dependencies (by default everything is OFF except opengl)
option(LIBIGL_WITH_VIEWER      "Use OpenGL viewer"  ON)
option(LIBIGL_WITH_OPENGL      "Use OpenGL"         ON)
option(LIBIGL_WITH_GLFW        "Use GLFW"           ON)
option(LIBIGL_WITH_BBW         "Use BBW"            OFF)
option(LIBIGL_WITH_EMBREE      "Use Embree"         ON)
option(LIBIGL_WITH_PNG         "Use PNG"            ON)
option(LIBIGL_WITH_TETGEN      "Use Tetgen"         OFF)
option(LIBIGL_WITH_TRIANGLE    "Use Triangle"       ON)
option(LIBIGL_WITH_XML         "Use XML"            OFF)
option(LIBIGL_WITH_LIM         "Use LIM"            OFF)
option(LIBIGL_WITH_COMISO      "Use CoMiso"         OFF)
option(LIBIGL_WITH_MATLAB      "Use Matlab"         OFF) # This option is not supported yet
option(LIBIGL_WITH_MOSEK       "Use MOSEK"          OFF) # This option is not supported yet
option(LIBIGL_WITH_CGAL        "Use CGAL"           OFF)
if(LIBIGL_WITH_CGAL) # Do not remove or move this block, the cgal build system fails without it
  find_package(CGAL REQUIRED)
  set(CGAL_DONT_OVERRIDE_CMAKE_FLAGS TRUE CACHE BOOL "CGAL's CMAKE Setup is super annoying ")
  include(${CGAL_USE_FILE})
endif()

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/extern/libigl/shared/cmake" "libigl")

# libigl information
message("libigl includes: ${LIBIGL_INCLUDE_DIRS}")
message("libigl libraries: ${LIBIGL_LIBRARIES}")
message("libigl extra sources: ${LIBIGL_EXTRA_SOURCES}")
message("libigl extra libraries: ${LIBIGL_EXTRA_LIBRARIES}")
message("libigl definitions: ${LIBIGL_DEFINITIONS}")

# Build TBB
set(TBB_BUILD_STATIC ON CACHE BOOL " " FORCE)
set(TBB_BUILD_SHARED OFF CACHE BOOL " " FORCE)
set(TBB_BUILD_TBBMALLOC OFF CACHE BOOL " " FORCE)
set(TBB_BUILD_TBBMALLOC_PROXY OFF CACHE BOOL " " FORCE)
set(TBB_BUILD_TESTS OFF CACHE BOOL " " FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/tbb extern_build/tbb)
set_property(TARGET tbb_static tbb_def_files PROPERTY FOLDER "dependencies")

# Build VCG
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/vcg extern_build/vcg)
set_property(TARGET vcg PROPERTY FOLDER "dependencies")

# Prepare the build environment
include_directories(
${LIBIGL_INCLUDE_DIRS}
  # Intel Thread Building Blocks
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/tbb/include
  # VCG
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/vcg
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/geogram
  )
add_definitions(${LIBIGL_DEFINITIONS})
add_definitions(-DENABLE_SERIALIZATION)

add_definitions(-DNOMINMAX)


file(GLOB header *.h grid_meshing/*.h)
file(GLOB source *.cpp grid_meshing/*.cpp)


add_executable(${PROJECT_NAME} ${source} ${header})

# Build geogram
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern)
include(geogram.cmake)
set_property(TARGET geogram PROPERTY FOLDER "dependencies")

target_link_libraries(${PROJECT_NAME} PUBLIC geogram tbb_static vcg ${LIBIGL_LIBRARIES} ${LIBIGL_EXTRA_LIBRARIES})


cmake_minimum_required(VERSION 3.18)
project(nvcomp LANGUAGES C CUDA CXX)

option(BUILD_TESTS "Build unit and end-to-end tests." OFF)
option(BUILD_BENCHMARKS "Build benchmarks." OFF)
option(BUILD_EXAMPLES "Build examples." OFF)
option(BUILD_STATIC "Build a static library." OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)

list( APPEND CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/cmake
)

find_package(CUDAToolkit REQUIRED)

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" OR
    "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR
    "${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")

    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
endif()

# Using '-G' can cause the register count to overflow for some kernels
# and cause launch failures
# set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG};-G")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g")

if(CMAKE_BUILD_TYPE STREQUAL "DEBUG" OR CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Debug build.")
else()
  message("Release build.")
  # this is required, as CMake sends CMAKE_CXX_FLAGS_DEBUG to nvcc
  # if no build type is set (but not to g++ for some reason).
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(NVCOMP_FIND_QUIET "QUIET")
if(DEFINED DEVEL AND NOT DEVEL EQUAL 0)
  unset(NVCOMP_FIND_QUIET )
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    # g++ warnings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weffc++")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
    # this doesn't work with old style CMakeLists and cuda
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpedantic")

    # g++ warnings
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wextra")
    # this doesn't work with old style CMakeLists and cuda
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wpedantic")
  endif()

  
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --Werror=all-warnings")
  endif()

  # TODO: add clang warnings
endif()

  if (MSVC)
    # Use "/permissive-" in order to be closer to GCC/Clang in what the MSVC compiler accepts.
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/W3>" "$<$<COMPILE_LANGUAGE:CXX>:/permissive->")
  endif()

if(NOT DEFINED CUB_DIR AND DEFINED ENV{CUB_DIR})
  set(CUB_DIR ENV{CUB_DIR})
endif()

message(STATUS "Finding CUB")
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11" AND NOT DEFINED CUB_DIR)
  message(FATAL_ERROR "When using a CUDA prior to CUDA 11, you must specify "
      "the path of cub via the 'CUB_DIR' variable (e.g., "
      "'cmake -DCUB_DIR=...' or 'export CUB_DIR=...'). You can download cub "
      "from 'https://github.com/thrust/cub'.")
endif()
find_path(CUB_HEADER NAMES "cub/cub.cuh" PATHS "${CUB_DIR}" "${CUDAToolkit_INCLUDE_DIRS}")
if(NOT CUB_HEADER)
  message(FATAL_ERROR "Unable to locate 'cub/cub.cuh'. Please specify a path "
      "to cub via 'CUB_DIR'")
endif()

if(NOT DEFINED NVCOMP_EXTS_ROOT AND DEFINED ENV{NVCOMP_EXTS_ROOT})
  set(NVCOMP_EXTS_ROOT "$ENV{NVCOMP_EXTS_ROOT}")
endif()

if( DEFINED NVCOMP_EXTS_ROOT )
  unset( NVCOMP_FIND_QUIET )
  set( ENV{ans_ROOT} ${NVCOMP_EXTS_ROOT} )
  set( ENV{gdeflate_ROOT} ${NVCOMP_EXTS_ROOT} )
  set( ENV{bitcomp_ROOT} ${NVCOMP_EXTS_ROOT} )
endif()

find_package(bitcomp ${NVCOMP_FIND_QUIET})
if (bitcomp_FOUND)
  message (STATUS "Found bitcomp")
  add_definitions(-DENABLE_BITCOMP)
endif()

find_package(gdeflate ${NVCOMP_FIND_QUIET})
if (gdeflate_FOUND)
  message (STATUS "Found gdeflate")
  add_definitions(-DENABLE_GDEFLATE)

  # Find ZLIB for CPU compression
  find_package(ZLIB)
endif()

find_package(ans ${NVCOMP_FIND_QUIET})
if (ans_FOUND)
  message (STATUS "Found ans")
  add_definitions(-DENABLE_ANS)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
file(COPY include/ DESTINATION ${CMAKE_BINARY_DIR}/include)

# determine version
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/include/nvcomp.h" MAIN_HEADER)
string(REGEX MATCH "#define NVCOMP_MAJOR_VERSION ([0-9]+)" _ "${MAIN_HEADER}")
set(VER_MAJOR "${CMAKE_MATCH_1}")
string(REGEX MATCH "#define NVCOMP_MINOR_VERSION ([0-9]+)" _ "${MAIN_HEADER}")
set(VER_MINOR "${CMAKE_MATCH_1}")
string(REGEX MATCH "#define NVCOMP_PATCH_VERSION ([0-9]+)" _ "${MAIN_HEADER}")
set(VER_PATCH "${CMAKE_MATCH_1}")

set(NVCOMP_VERSION "${VER_MAJOR}.${VER_MINOR}.${VER_PATCH}")

message("Build nvCOMP version ${NVCOMP_VERSION}")

if (BUILD_TESTS)
  enable_testing()
endif()

if (UNIX)
  include(GNUInstallDirs)
endif()


add_subdirectory(src)

if (BUILD_TESTS)
  add_subdirectory(tests)
endif()
if (BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
if (BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/nvcomp)


#------
# Build directory exporting
include(CMakePackageConfigHelpers)
configure_package_config_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/nvcomp-config.cmake.in"
                              "${CMAKE_CURRENT_BINARY_DIR}/nvcomp-config.cmake"
                              INSTALL_DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
export(TARGETS nvcomp
       FILE "${CMAKE_CURRENT_BINARY_DIR}/nvcomp-targets.cmake"
       )


#------
# Install directory setup
configure_package_config_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/nvcomp-config.cmake.in"
                              "${CMAKE_CURRENT_BINARY_DIR}/cmake/nvcomp-config.cmake"
                              INSTALL_DESTINATION "${INSTALL_CONFIGDIR}")
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/nvcompConfigVersion.cmake
    VERSION "${NVCOMP_VERSION}"
    COMPATIBILITY SameMinorVersion)

install(TARGETS nvcomp EXPORT nvcomp-exports DESTINATION lib)
install(EXPORT nvcomp-exports
        FILE          nvcomp-targets.cmake
        NAMESPACE     nvcomp::
        DESTINATION   "${INSTALL_CONFIGDIR}")


install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/nvcomp/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nvcomp)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/nvcomp.hpp
              ${CMAKE_CURRENT_SOURCE_DIR}/include/nvcomp.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cmake/nvcomp-config.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/cmake/nvcompConfigVersion.cmake"
        DESTINATION "${INSTALL_CONFIGDIR}")

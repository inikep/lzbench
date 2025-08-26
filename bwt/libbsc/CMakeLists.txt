cmake_minimum_required(VERSION 3.24)

#-------------------------------------------------------------------------------
# Prefer Clang over GCC if available
#-------------------------------------------------------------------------------
if(UNIX AND NOT CMAKE_GENERATOR MATCHES "Visual Studio")
  find_program(CLANG_CC clang)
  find_program(CLANG_CXX clang++)
  if(CLANG_CC AND CLANG_CXX)
    if(NOT DEFINED CMAKE_C_COMPILER)
      set(CMAKE_C_COMPILER ${CLANG_CC} CACHE PATH "C compiler")
    endif()
    if(NOT DEFINED CMAKE_CXX_COMPILER)
      set(CMAKE_CXX_COMPILER ${CLANG_CXX} CACHE PATH "C++ compiler")
    endif()
  endif()
endif()

project(bsc VERSION 3.3.11 LANGUAGES C CXX)

#-------------------------------------------------------------------------------
# Options
#-------------------------------------------------------------------------------
option(BSC_ENABLE_NATIVE_COMPILATION "Enable platform-native optimizations" ON)
option(BSC_ENABLE_OPENMP "Enable OpenMP support" ON)
option(BSC_ENABLE_CUDA "Enable CUDA support (requires OpenMP)" ON)
option(BSC_BUILD_SHARED_LIB "Build libbsc as a shared library" OFF)

#-------------------------------------------------------------------------------
# Language standards
#-------------------------------------------------------------------------------
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#-------------------------------------------------------------------------------
# Default build type
#-------------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

#-------------------------------------------------------------------------------
# Detect architecture specific optimizations support
#-------------------------------------------------------------------------------
if(BSC_ENABLE_NATIVE_COMPILATION)
  include(CheckCCompilerFlag)
  include(CheckCXXCompilerFlag)
  include(CheckCSourceRuns)

  check_c_compiler_flag(-march=native COMPILER_SUPPORTS_MARCH_NATIVE_C)
  check_cxx_compiler_flag(-march=native COMPILER_SUPPORTS_MARCH_NATIVE_CXX)

  if(MSVC AND NOT (COMPILER_SUPPORTS_MARCH_NATIVE_C AND COMPILER_SUPPORTS_MARCH_NATIVE_CXX))
    check_c_source_runs(
      "#include<intrin.h>\\nint main() { int r[4]; __cpuid(r, 7); return (r[1] & (1 << 30)) && ((_xgetbv(0) & 0xe0) == 0xe0) ? 0 : 1; }"
      COMPILER_SUPPORTS_MSVC_ARCH_AVX512
    )
    check_c_source_runs(
      "#include<intrin.h>\\nint main() { int r[4]; __cpuid(r, 7); return (r[1] & (1 <<  5)) && ((_xgetbv(0) & 0x06) == 0x06) ? 0 : 1; }"
      COMPILER_SUPPORTS_MSVC_ARCH_AVX2
    )
    check_c_source_runs(
      "#include<intrin.h>\\nint main() { int r[4]; __cpuid(r, 1); return (r[2] & (1 << 20)) ? 0 : 1; }"
      COMPILER_SUPPORTS_MSVC_ARCH_SSE42
    )
  else()
    set(COMPILER_SUPPORTS_MSVC_ARCH_AVX512 OFF)
    set(COMPILER_SUPPORTS_MSVC_ARCH_AVX2 OFF)
    set(COMPILER_SUPPORTS_MSVC_ARCH_SSE42 OFF)
  endif()

  set(COMPILER_CUDA_ARCHITECTURES "native")
else()
  set(COMPILER_SUPPORTS_MARCH_NATIVE_C OFF)
  set(COMPILER_SUPPORTS_MARCH_NATIVE_CXX OFF)
  set(COMPILER_SUPPORTS_MSVC_ARCH_AVX512 OFF)
  set(COMPILER_SUPPORTS_MSVC_ARCH_AVX2 OFF)
  set(COMPILER_SUPPORTS_MSVC_ARCH_SSE42 OFF)
  set(COMPILER_CUDA_ARCHITECTURES "all")
endif()

#-------------------------------------------------------------------------------
# Detect OpenMP support
#-------------------------------------------------------------------------------
if(BSC_ENABLE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_FOUND)
    message(STATUS "OpenMP found; enabling OpenMP support.")
  else()
    message(WARNING "OpenMP not found; disabling OpenMP support.")
    set(BSC_ENABLE_OPENMP OFF)
  endif()
endif()

#-------------------------------------------------------------------------------
# Detect CUDA support
#-------------------------------------------------------------------------------
if(BSC_ENABLE_CUDA)
  if(NOT BSC_ENABLE_OPENMP)
    message(WARNING "OpenMP is required for CUDA; disabling CUDA support.")
    set(BSC_ENABLE_CUDA OFF)
  elseif(MSVC AND NOT (CMAKE_CXX_COMPILER_ID MATCHES "MSVC"))
    message(WARNING "${CMAKE_CXX_COMPILER_ID} is not supported by nvcc on Windows; disabling CUDA support.")
    set(BSC_ENABLE_CUDA OFF)
  else()
    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
      include(CheckLanguage)
      check_language(CUDA)
      if (CMAKE_CUDA_COMPILER)
        message(STATUS "CUDA Toolkit and CUDA Compiler found; enabling CUDA support.")
        set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
        enable_language(CUDA)
      else()
        message(WARNING "CUDA Compiler not found; disabling CUDA support.")
        set(BSC_ENABLE_CUDA OFF)
      endif()
    else()
      message(WARNING "CUDA Toolkit not found; disabling CUDA support.")
      set(BSC_ENABLE_CUDA OFF)
    endif()
  endif()
endif()

#-------------------------------------------------------------------------------
# libbsc library target
#-------------------------------------------------------------------------------
if(BSC_BUILD_SHARED_LIB)
    set(BSC_LIBRARY_TYPE SHARED)
else()
    set(BSC_LIBRARY_TYPE STATIC)
endif()
 
add_library(libbsc ${BSC_LIBRARY_TYPE})

set_target_properties(libbsc PROPERTIES PREFIX "" IMPORT_PREFIX "")
add_library(bsc::libbsc ALIAS libbsc)

target_sources(libbsc PRIVATE
  libbsc/adler32/adler32.cpp
  libbsc/bwt/bwt.cpp
  libbsc/bwt/libsais/libsais.c
  libbsc/coder/coder.cpp
  libbsc/coder/qlfc/qlfc.cpp
  libbsc/coder/qlfc/qlfc_model.cpp
  libbsc/filters/detectors.cpp
  libbsc/filters/preprocessing.cpp
  libbsc/libbsc/libbsc.cpp
  libbsc/lzp/lzp.cpp
  libbsc/platform/platform.cpp
  libbsc/st/st.cpp
)

if(BSC_BUILD_SHARED_LIB)
  target_compile_definitions(libbsc PUBLIC LIBBSC_SHARED)
  target_compile_definitions(libbsc PRIVATE LIBBSC_EXPORTS)
endif()

#-------------------------------------------------------------------------------
# Enable architecture specific optimizations
#-------------------------------------------------------------------------------
if(COMPILER_SUPPORTS_MARCH_NATIVE_C)
  target_compile_options(libbsc PRIVATE $<$<COMPILE_LANGUAGE:C>:-march=native>)
elseif(COMPILER_SUPPORTS_MSVC_ARCH_AVX512)
  target_compile_options(libbsc PRIVATE $<$<COMPILE_LANGUAGE:C>:/arch:AVX512>)
elseif(COMPILER_SUPPORTS_MSVC_ARCH_AVX2)
  target_compile_options(libbsc PRIVATE $<$<COMPILE_LANGUAGE:C>:/arch:AVX2>)
elseif(COMPILER_SUPPORTS_MSVC_ARCH_SSE42)
  target_compile_options(libbsc PRIVATE $<$<COMPILE_LANGUAGE:C>:/arch:SSE4.2>)
endif()

if(COMPILER_SUPPORTS_MARCH_NATIVE_CXX)
  target_compile_options(libbsc PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
elseif(COMPILER_SUPPORTS_MSVC_ARCH_AVX512)
  target_compile_options(libbsc PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
elseif(COMPILER_SUPPORTS_MSVC_ARCH_AVX2)
  target_compile_options(libbsc PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
elseif(COMPILER_SUPPORTS_MSVC_ARCH_SSE42)
  target_compile_options(libbsc PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:SSE4.2>)
endif()

#-------------------------------------------------------------------------------
# Enable OpenMP support
#-------------------------------------------------------------------------------
if(BSC_ENABLE_OPENMP)
  target_compile_definitions(libbsc PRIVATE LIBSAIS_OPENMP)
  target_compile_definitions(libbsc PUBLIC LIBBSC_OPENMP_SUPPORT)
  target_link_libraries(libbsc PUBLIC OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
endif()

#-------------------------------------------------------------------------------
# Enable CUDA support
#-------------------------------------------------------------------------------
if(BSC_ENABLE_CUDA)
  set_target_properties(libbsc PROPERTIES CUDA_ARCHITECTURES ${COMPILER_CUDA_ARCHITECTURES})
  target_sources(libbsc PRIVATE
    libbsc/st/st.cu
    libbsc/bwt/libcubwt/libcubwt.cu
  )
  target_compile_options(libbsc PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS};-Wno-deprecated-gpu-targets>
  )
  target_compile_definitions(libbsc PUBLIC LIBBSC_CUDA_SUPPORT)
  target_link_libraries(libbsc PUBLIC CUDA::cudart)
endif()

#-------------------------------------------------------------------------------
# bsc executable target
#-------------------------------------------------------------------------------
add_executable(bsc bsc.cpp)

target_link_libraries(bsc PUBLIC bsc::libbsc)

#-------------------------------------------------------------------------------
# Installation directives
#-------------------------------------------------------------------------------
include(GNUInstallDirs)

install(FILES
  libbsc/filters.h
  libbsc/libbsc.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libbsc
)

install(TARGETS libbsc
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libbsc
)

install(TARGETS bsc
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
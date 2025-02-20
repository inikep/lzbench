file(GLOB EXAMPLE_SOURCES *.cpp *.cu)

if(NOT ans_FOUND)
  list(FILTER EXAMPLE_SOURCES EXCLUDE REGEX "benchmark_ans_chunked.cu")
endif()

if(NOT bitcomp_FOUND)
  list(FILTER EXAMPLE_SOURCES EXCLUDE REGEX "benchmark_bitcomp_chunked.cu")
endif()

foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
  # cut off suffixes
  string(REPLACE ".cpp" "" EXAMPLE_NAME ${EXAMPLE_SOURCE})
  string(REPLACE ".cu" "" EXAMPLE_NAME ${EXAMPLE_NAME})
  get_filename_component(BARE_NAME ${EXAMPLE_NAME} NAME)
  add_executable(${BARE_NAME} ${EXAMPLE_SOURCE})
  if (NOT MSVC)
    target_link_libraries(${BARE_NAME} PRIVATE nvcomp CUDA::cudart CUDA::nvml)
  else()
    # Including CUDA::cudart on Windows resulted in multiple definition
    # linker errors for gdeflate and bitcomp benchmarks, since
    # cudart_static.lib is already included.
    target_link_libraries(${BARE_NAME} PRIVATE nvcomp CUDA::nvml)
  endif()
  target_include_directories(${BARE_NAME} PRIVATE
      "$<BUILD_INTERFACE:${nvcomp_SOURCE_DIR}/include>")
endforeach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})

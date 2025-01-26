file(GLOB EXAMPLE_SOURCES *.cpp *.c)

if(NOT gdeflate_FOUND)
list(FILTER EXAMPLE_SOURCES EXCLUDE REGEX "test_gdeflate_batch_c_api.c")
list(FILTER EXAMPLE_SOURCES EXCLUDE REGEX "test_gdeflate.cpp")
endif()
if(NOT ans_FOUND)
list(FILTER EXAMPLE_SOURCES EXCLUDE REGEX "test_ans_batch_c_api.c")
endif()

if(NOT bitcomp_FOUND)
list(FILTER EXAMPLE_SOURCES EXCLUDE REGEX "test_bitcomp.cpp")
endif()

function(add_test_file EXAMPLE_SOURCE)
  get_filename_component(BARE_NAME "${EXAMPLE_SOURCE}" NAME_WE)
  add_executable(${BARE_NAME} ${EXAMPLE_SOURCE})
  if (NOT MSVC)
    target_link_libraries(${BARE_NAME} PRIVATE nvcomp CUDA::cudart)
  else()
    # Including CUDA::cudart on Windows resulted in multiple definition
    # linker errors for gdeflate and bitcomp tests, since
    # cudart_static.lib is already included.
    target_link_libraries(${BARE_NAME} PRIVATE nvcomp)
  endif()
  add_test(NAME ${BARE_NAME} COMMAND ${BARE_NAME})
endfunction()

foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
  get_filename_component(BARE_NAME "${EXAMPLE_NAME}" NAME_WE)
  add_test_file(${EXAMPLE_SOURCE} ${EXAMPLE_NAME})
endforeach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})

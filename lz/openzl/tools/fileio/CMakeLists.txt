# Copyright (c) Meta Platforms, Inc. and affiliates.

# if (OPENZL_BUILD_TOOLS)
  add_library(fileio
    ${CMAKE_CURRENT_LIST_DIR}/fileio.c
  )
  target_include_directories(fileio PUBLIC ${PROJECT_SOURCE_DIR})
  apply_openzl_compile_options_to_target(fileio)
  target_link_libraries(fileio PUBLIC openzl)
  add_dependencies(fileio openzl)
# endif()

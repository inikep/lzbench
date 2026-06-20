# Copyright (c) Meta Platforms, Inc. and affiliates.

if (OPENZL_BUILD_TOOLS)
  add_executable(stream_dump2
    ${CMAKE_CURRENT_LIST_DIR}/stream_dump2.h
    ${CMAKE_CURRENT_LIST_DIR}/stream_dump2.c
    ${CMAKE_CURRENT_LIST_DIR}/stream_dump2_noop_shim.c
  )
  target_include_directories(stream_dump2 PUBLIC ${PROJECT_SOURCE_DIR})
  apply_openzl_compile_options_to_target(stream_dump2)
  target_link_libraries(stream_dump2
    PRIVATE
      openzl
      fileio
  )
  add_dependencies(stream_dump2 openzl fileio)

  # There's no need to build stream_dump2_mc or stream_dump2_xldb.
  # They're prod only!
endif()

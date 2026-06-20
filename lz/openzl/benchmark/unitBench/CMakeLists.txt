# Copyright (c) Meta Platforms, Inc. and affiliates.

file(
    GLOB_RECURSE unitBench_sources CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_LIST_DIR}/*.c"
)
add_executable(unitBench ${unitBench_sources})
target_link_libraries(unitBench PUBLIC openzl fileio timefn)
add_dependencies(unitBench openzl fileio timefn)
target_include_directories(unitBench PUBLIC ${PROJECT_SOURCE_DIR})
apply_openzl_compile_options_to_target(unitBench)

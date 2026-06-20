# Copyright (c) Meta Platforms, Inc. and affiliates.

file(
    GLOB parquet_graph_sources
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/*.c")
file(
    GLOB parquet_graph_headers
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_LIST_DIR}/*.h")

add_library(parquet_graph
    ${parquet_graph_sources}
)
target_include_directories(parquet_graph PRIVATE
    ${PROJECT_BINARY_DIR}/include
    ${PROJECT_SOURCE_DIR})

target_link_libraries(parquet_graph
    openzl
)
add_dependencies(parquet_graph openzl)
apply_openzl_compile_options_to_target(parquet_graph)

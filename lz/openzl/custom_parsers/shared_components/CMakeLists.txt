# Copyright (c) Meta Platforms, Inc. and affiliates.

file(
    GLOB shared_components_sources
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/*.c")
file(
    GLOB shared_components_headers
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_LIST_DIR}/*.h")

add_library(shared_components
    ${shared_components_sources}
)
target_include_directories(shared_components PUBLIC
    ${PROJECT_BINARY_DIR}/include
    ${PROJECT_SOURCE_DIR})

target_link_libraries(shared_components PUBLIC
    openzl
    openzl_cpp
)
add_dependencies(shared_components openzl openzl_cpp)
apply_openzl_compile_options_to_target(shared_components)

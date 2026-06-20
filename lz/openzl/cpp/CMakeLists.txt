# Copyright (c) Meta Platforms, Inc. and affiliates.

# Find C++ files
file(GLOB_RECURSE cpp_sources CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/cpp/src/*.cpp")
file(GLOB_RECURSE cpp_private_headers CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/cpp/src/*.hpp")
file(GLOB_RECURSE cpp_public_headers CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/cpp/include/*.hpp")

add_library(
    openzl_cpp
    ${cpp_sources}
)

# We need PIC to build shared libraries. For now just enable it for all builds.
# Later we could build both PIC & non-PIC versions of the library.
set_property(TARGET openzl_cpp PROPERTY POSITION_INDEPENDENT_CODE ON)

apply_openzl_compile_options_to_target(openzl_cpp)

source_group(
    TREE "${PROJECT_SOURCE_DIR}/cpp"
    PREFIX "openzl-cpp"
    FILES ${cpp_sources} ${cpp_private_headers} ${cpp_public_headers}
)

target_include_directories(
    openzl_cpp
    BEFORE
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/cpp/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp/include>
)

target_include_directories(
    openzl_cpp
    PRIVATE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp/src>
)

target_link_libraries(
    openzl_cpp
    openzl
)
add_dependencies(openzl_cpp openzl)

add_subdirectory(tests)

# Copyright (c) Meta Platforms, Inc. and affiliates.

file(
    GLOB sddl_profile_sources
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/*.c"
)
add_library(sddl_profile
    ${sddl_profile_sources}
)
target_include_directories(sddl_profile PRIVATE
    ${PROJECT_BINARY_DIR}/include
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(sddl_profile
    openzl
    shared_components
)
add_dependencies(sddl_profile
    openzl
    shared_components
)
apply_openzl_compile_options_to_target(sddl_profile)

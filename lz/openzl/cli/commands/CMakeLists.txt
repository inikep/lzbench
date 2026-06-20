# Copyright (c) Meta Platforms, Inc. and affiliates.

file(GLOB commands_sources
    CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

file(GLOB commands_headers
    CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

add_library(commands ${commands_sources})

target_include_directories(commands
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR})

target_link_libraries(commands
    PUBLIC
    args
    utils
    openzl_cpp
    custom_parsers
    tools_io
    tools_training
    logger
)

add_dependencies(commands
    args
    utils
    openzl_cpp
    custom_parsers
    tools_io
    tools_training
    logger
)

apply_openzl_compile_options_to_target(commands)

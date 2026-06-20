# Copyright (c) Meta Platforms, Inc. and affiliates.

# datagen
file(GLOB_RECURSE datagen_srcs CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
list(FILTER datagen_srcs EXCLUDE REGEX "${CMAKE_CURRENT_LIST_DIR}/test/.*\\.cpp")
list(FILTER datagen_srcs EXCLUDE REGEX "${CMAKE_CURRENT_LIST_DIR}/test_registry/.*\\.cpp")

file(GLOB_RECURSE datagen_headers CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_LIST_DIR}/*.h")
list(FILTER datagen_headers EXCLUDE REGEX "${CMAKE_CURRENT_LIST_DIR}/test/.*\\.h")
list(FILTER datagen_headers EXCLUDE REGEX "${CMAKE_CURRENT_LIST_DIR}/test_registry/.*\\.h")

add_library(datagen ${datagen_srcs} ${datagen_headers})
target_include_directories(datagen PUBLIC ${PROJECT_SOURCE_DIR})
apply_openzl_compile_options_to_target(datagen)
target_link_libraries(datagen PUBLIC openzl openzl_cpp openzl_test_support)
add_dependencies(datagen openzl openzl_cpp openzl_test_support)

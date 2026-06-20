# Copyright (c) Meta Platforms, Inc. and affiliates.

if (OPENZL_BUILD_TOOLS)
    add_library(timefn
      ${CMAKE_CURRENT_LIST_DIR}/timefn.c
    )
    target_include_directories(timefn PUBLIC ${PROJECT_SOURCE_DIR})
    apply_openzl_compile_options_to_target(timefn)
    target_link_libraries(timefn PUBLIC openzl)
    add_dependencies(timefn openzl)

    if (OPENZL_BUILD_TESTS)
        add_executable(test_timefn ${CMAKE_CURRENT_LIST_DIR}/tests/test_time.cpp)
        target_link_libraries(test_timefn
            PRIVATE
                timefn
                openzl_test_support
                GTest::gtest_main
        )
        add_dependencies(test_timefn timefn openzl_test_support GTest::gtest_main)
        apply_openzl_compile_options_to_target(test_timefn)
        gtest_discover_tests(test_timefn)
    endif()
endif()

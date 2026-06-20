// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "tools/io/InputBuffer.h"
#include "tools/io/InputSet.h"
#include "tools/io/InputSetMulti.h"
#include "tools/io/InputSetStatic.h"

using namespace testing;

namespace openzl::tools::io::tests {

namespace {

std::unique_ptr<InputSet> make_static_input_set(
        const std::vector<std::string>& vals)
{
    std::vector<std::shared_ptr<Input>> inputs;
    inputs.reserve(vals.size());
    for (const auto& val : vals) {
        inputs.push_back(std::make_shared<InputBuffer>(val, val));
    }
    return std::make_unique<InputSetStatic>(std::move(inputs));
}

std::unique_ptr<InputSet> make_multi_input_set(
        std::vector<std::unique_ptr<InputSet>> input_sets)
{
    return std::make_unique<InputSetMulti>(std::move(input_sets));
}

void check_set(
        const std::unique_ptr<InputSet>& set,
        const std::vector<std::string>& expected)
{
    std::vector<std::string> contents;
    for (const auto& input_ptr : *set) {
        ASSERT_TRUE(input_ptr);
        contents.emplace_back(input_ptr->contents());
    }
    ASSERT_EQ(contents, expected);
}
} // anonymous namespace

TEST(TestInputSet, StaticSetEmpty)
{
    const auto expected = std::vector<std::string>{};
    const auto set      = make_static_input_set(expected);
    check_set(set, expected);
}

TEST(TestInputSet, StaticSetOneElt)
{
    const auto expected = std::vector<std::string>{ "foo" };
    const auto set      = make_static_input_set(expected);
    check_set(set, expected);
}

TEST(TestInputSet, StaticSetTwoElts)
{
    const auto expected = std::vector<std::string>{ "foo", "bar" };
    const auto set      = make_static_input_set(expected);
    check_set(set, expected);
}

TEST(TestInputSet, StaticSetRepeatedDeref)
{
    const auto expected = std::vector<std::string>{ "foo" };
    const auto set      = make_static_input_set(expected);
    auto it             = set->begin();
    const auto ptr1     = *it;
    const auto ptr2     = *it;
    EXPECT_EQ(ptr1, ptr2);
}

TEST(TestInputSet, StaticSetIteratorEquality)
{
    const auto expected = std::vector<std::string>{ "foo", "bar" };
    const auto set      = make_static_input_set(expected);
    auto it1            = set->begin();
    auto it2            = set->begin();
    auto end1           = set->end();
    auto end2           = set->end();

    EXPECT_EQ(end1, end2);

    EXPECT_EQ(it1, it2);
    EXPECT_NE(it1, end1);
    EXPECT_NE(it2, end1);
    ++it1;
    EXPECT_NE(it1, it2);
    EXPECT_NE(it1, end1);
    EXPECT_NE(it2, end1);
    ++it2;
    EXPECT_EQ(it1, it2);
    EXPECT_NE(it1, end1);
    EXPECT_NE(it2, end1);
    ++it1;
    EXPECT_NE(it1, it2);
    EXPECT_EQ(it1, end1);
    EXPECT_NE(it2, end1);
    ++it2;
    EXPECT_EQ(it1, it2);
    EXPECT_EQ(it1, end1);
    EXPECT_EQ(it2, end1);
}

TEST(TestInputSet, MultiSetEmpty)
{
    std::vector<std::unique_ptr<InputSet>> input_sets;
    const auto multi    = make_multi_input_set(std::move(input_sets));
    const auto expected = std::vector<std::string>{};
    check_set(multi, expected);
}

TEST(TestInputSet, MultiSetOneSubset)
{
    std::vector<std::unique_ptr<InputSet>> input_sets;
    input_sets.push_back(make_static_input_set({ "foo", "bar" }));
    const auto multi    = make_multi_input_set(std::move(input_sets));
    const auto expected = std::vector<std::string>{ "foo", "bar" };
    check_set(multi, expected);
}

TEST(TestInputSet, MultiSetTwoSubsets)
{
    std::vector<std::unique_ptr<InputSet>> input_sets;
    input_sets.push_back(make_static_input_set({ "foo", "bar" }));
    input_sets.push_back(make_static_input_set({ "baz", "xyzzy" }));
    const auto multi = make_multi_input_set(std::move(input_sets));
    const auto expected =
            std::vector<std::string>{ "foo", "bar", "baz", "xyzzy" };
    check_set(multi, expected);
}

TEST(TestInputSet, MultiSetOneEmptySubset)
{
    std::vector<std::unique_ptr<InputSet>> input_sets;
    input_sets.push_back(make_static_input_set({}));
    const auto multi    = make_multi_input_set(std::move(input_sets));
    const auto expected = std::vector<std::string>{};
    check_set(multi, expected);
}

TEST(TestInputSet, MultiSetTwoEmptySubsets)
{
    std::vector<std::unique_ptr<InputSet>> input_sets;
    input_sets.push_back(make_static_input_set({}));
    input_sets.push_back(make_static_input_set({}));
    const auto multi    = make_multi_input_set(std::move(input_sets));
    const auto expected = std::vector<std::string>{};
    check_set(multi, expected);
}

TEST(TestInputSet, MultiSetThreeEmptySubsets)
{
    std::vector<std::unique_ptr<InputSet>> input_sets;
    input_sets.push_back(make_static_input_set({}));
    input_sets.push_back(make_static_input_set({}));
    input_sets.push_back(make_static_input_set({}));
    const auto multi    = make_multi_input_set(std::move(input_sets));
    const auto expected = std::vector<std::string>{};
    check_set(multi, expected);
}

TEST(TestInputSet, MultiSetBeginningEmptySubset)
{
    std::vector<std::unique_ptr<InputSet>> input_sets;
    input_sets.push_back(make_static_input_set({}));
    input_sets.push_back(make_static_input_set({ "foo" }));
    input_sets.push_back(make_static_input_set({ "bar" }));
    const auto multi    = make_multi_input_set(std::move(input_sets));
    const auto expected = std::vector<std::string>{ "foo", "bar" };
    check_set(multi, expected);
}

TEST(TestInputSet, MultiSetBeginningTwoEmptySubsets)
{
    std::vector<std::unique_ptr<InputSet>> input_sets;
    input_sets.push_back(make_static_input_set({}));
    input_sets.push_back(make_static_input_set({}));
    input_sets.push_back(make_static_input_set({ "foo" }));
    input_sets.push_back(make_static_input_set({ "bar" }));
    const auto multi    = make_multi_input_set(std::move(input_sets));
    const auto expected = std::vector<std::string>{ "foo", "bar" };
    check_set(multi, expected);
}

TEST(TestInputSet, MultiSetMiddleEmptySubset)
{
    std::vector<std::unique_ptr<InputSet>> input_sets;
    input_sets.push_back(make_static_input_set({ "foo" }));
    input_sets.push_back(make_static_input_set({}));
    input_sets.push_back(make_static_input_set({ "bar" }));
    const auto multi    = make_multi_input_set(std::move(input_sets));
    const auto expected = std::vector<std::string>{ "foo", "bar" };
    check_set(multi, expected);
}

TEST(TestInputSet, MultiSetMiddleTwoEmptySubsets)
{
    std::vector<std::unique_ptr<InputSet>> input_sets;
    input_sets.push_back(make_static_input_set({ "foo" }));
    input_sets.push_back(make_static_input_set({}));
    input_sets.push_back(make_static_input_set({}));
    input_sets.push_back(make_static_input_set({ "bar" }));
    const auto multi    = make_multi_input_set(std::move(input_sets));
    const auto expected = std::vector<std::string>{ "foo", "bar" };
    check_set(multi, expected);
}

TEST(TestInputSet, MultiSetEndEmptySubset)
{
    std::vector<std::unique_ptr<InputSet>> input_sets;
    input_sets.push_back(make_static_input_set({ "foo" }));
    input_sets.push_back(make_static_input_set({ "bar" }));
    input_sets.push_back(make_static_input_set({}));
    const auto multi    = make_multi_input_set(std::move(input_sets));
    const auto expected = std::vector<std::string>{ "foo", "bar" };
    check_set(multi, expected);
}

TEST(TestInputSet, MultiSetEndTwoEmptySubsets)
{
    std::vector<std::unique_ptr<InputSet>> input_sets;
    input_sets.push_back(make_static_input_set({ "foo" }));
    input_sets.push_back(make_static_input_set({ "bar" }));
    input_sets.push_back(make_static_input_set({}));
    input_sets.push_back(make_static_input_set({}));
    const auto multi    = make_multi_input_set(std::move(input_sets));
    const auto expected = std::vector<std::string>{ "foo", "bar" };
    check_set(multi, expected);
}

TEST(TestInputSet, MultiSetStacked)
{
    std::vector<std::unique_ptr<InputSet>> outer_input_sets;
    {
        std::vector<std::unique_ptr<InputSet>> inner_input_sets;
        inner_input_sets.push_back(make_static_input_set({ "foo" }));
        inner_input_sets.push_back(make_static_input_set({ "bar" }));
        outer_input_sets.push_back(
                make_multi_input_set(std::move(inner_input_sets)));
    }
    outer_input_sets.push_back(make_static_input_set({ "beep" }));
    {
        std::vector<std::unique_ptr<InputSet>> inner_input_sets;
        inner_input_sets.push_back(make_static_input_set({ "baz" }));
        inner_input_sets.push_back(make_static_input_set({ "xyzzy" }));
        outer_input_sets.push_back(
                make_multi_input_set(std::move(inner_input_sets)));
    }
    const auto multi = make_multi_input_set(std::move(outer_input_sets));
    const auto expected =
            std::vector<std::string>{ "foo", "bar", "beep", "baz", "xyzzy" };
    check_set(multi, expected);
}

TEST(TestInputSet, MultiSetRepeatedDeref)
{
    std::vector<std::unique_ptr<InputSet>> input_sets;
    input_sets.push_back(make_static_input_set({ "foo", "bar" }));
    input_sets.push_back(make_static_input_set({}));
    input_sets.push_back(make_static_input_set({ "baz", "xyzzy" }));
    const auto set = make_multi_input_set(std::move(input_sets));
    auto it        = set->begin();
    EXPECT_EQ(*it, *it);
    ++it;
    EXPECT_EQ(*it, *it);
}

TEST(TestInputSet, MultiSetIteratorEquality)
{
    std::vector<std::unique_ptr<InputSet>> outer_input_sets;
    {
        std::vector<std::unique_ptr<InputSet>> inner_input_sets;
        inner_input_sets.push_back(make_static_input_set({ "foo" }));
        inner_input_sets.push_back(make_static_input_set({ "bar" }));
        outer_input_sets.push_back(
                make_multi_input_set(std::move(inner_input_sets)));
    }
    outer_input_sets.push_back(make_static_input_set({ "beep" }));
    outer_input_sets.push_back(make_static_input_set({}));
    outer_input_sets.push_back(make_static_input_set({ "hello", "world" }));
    {
        std::vector<std::unique_ptr<InputSet>> inner_input_sets;
        inner_input_sets.push_back(make_static_input_set({ "baz" }));
        inner_input_sets.push_back(make_static_input_set({}));
        inner_input_sets.push_back(make_static_input_set({ "xyzzy" }));
        outer_input_sets.push_back(
                make_multi_input_set(std::move(inner_input_sets)));
    }
    const auto set = make_multi_input_set(std::move(outer_input_sets));

    const size_t elt_count = 7;
    auto it1               = set->begin();
    for (size_t i = 0;; i++) {
        auto it2 = set->begin();
        for (size_t j = 0;; j++) {
            if (i == j) {
                EXPECT_EQ(it1, it2);
            } else {
                EXPECT_NE(it1, it2);
            }
            if (j == elt_count) {
                break;
            }
            ++it2;
        }

        if (i == elt_count) {
            EXPECT_EQ(it1, set->end());
            break;
        } else {
            EXPECT_NE(it1, set->end());
        }
        ++it1;
    }
}
} // namespace openzl::tools::io::tests

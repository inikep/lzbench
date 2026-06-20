// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <functional>
#include <string>

#include <gtest/gtest.h>

#include "openzl/common/debug.h"

#include "tests/utils.h"

namespace {
using namespace std::string_literals;

ZL_STATIC_ASSERT(true, "Test static assert succeeds outside function!");

TEST(DebugTest, staticAssertSucceeds)
{
    ZL_STATIC_ASSERT(true, "Test static assert succeeds inside function!");
}

TEST(DebugTest, argPadding)
{
    EXPECT_EQ(ZS_MACRO_QUOTE(ZS_MACRO_PAD1_SUFFIX(-1)), "_NOMSG"s);
    EXPECT_EQ(ZS_MACRO_QUOTE(ZS_MACRO_PAD1_SUFFIX(-1, -2)), "_FIXED"s);
    EXPECT_EQ(ZS_MACRO_QUOTE(ZS_MACRO_PAD1_SUFFIX(-1, -2, -3)), "_MSG"s);

    EXPECT_EQ(ZS_MACRO_QUOTE(ZS_MACRO_PAD2_SUFFIX(-1)), "_NOT_ENOUGH_ARGS"s);
    EXPECT_EQ(ZS_MACRO_QUOTE(ZS_MACRO_PAD2_SUFFIX(-1, -2)), "_NOMSG"s);
    EXPECT_EQ(ZS_MACRO_QUOTE(ZS_MACRO_PAD2_SUFFIX(-1, -2, -3)), "_FIXED"s);
    EXPECT_EQ(ZS_MACRO_QUOTE(ZS_MACRO_PAD2_SUFFIX(-1, -2, -3, -4)), "_MSG"s);

    EXPECT_EQ(ZS_MACRO_QUOTE(ZS_MACRO_PAD3_SUFFIX(-1)), "_NOT_ENOUGH_ARGS"s);
    EXPECT_EQ(
            ZS_MACRO_QUOTE(ZS_MACRO_PAD3_SUFFIX(-1, -2)), "_NOT_ENOUGH_ARGS"s);
    EXPECT_EQ(ZS_MACRO_QUOTE(ZS_MACRO_PAD3_SUFFIX(-1, -2, -3)), "_NOMSG"s);
    EXPECT_EQ(ZS_MACRO_QUOTE(ZS_MACRO_PAD3_SUFFIX(-1, -2, -3, -4)), "_FIXED"s);
    EXPECT_EQ(
            ZS_MACRO_QUOTE(ZS_MACRO_PAD3_SUFFIX(-1, -2, -3, -4, -5)), "_MSG"s);
}

TEST(DebugTest, assertSucceeds)
{
    ZL_ASSERT(true);
    ZL_ASSERT(true, "foo");
    ZL_ASSERT(true, "foo %d", 1);
}

TEST(DebugTest, assertFails)
{
    ZS_CHECK_ASSERT_FIRES(ZL_ASSERT(false));
}

TEST(DebugTest, requireSucceeds)
{
    ZL_REQUIRE(true);
    ZL_REQUIRE(true, "foo");
    ZL_REQUIRE(true, "foo %d", 1);
}

TEST(DebugTest, requireFails)
{
    ZS_CHECK_REQUIRE_FIRES(ZL_REQUIRE(false));
}

TEST(DebugTest, assertOpSucceeds)
{
    uint64_t u64a     = 0;
    uint64_t u64b     = 0;
    int64_t i64a      = 0;
    int64_t i64b      = 0;
    uint32_t u32a     = 0;
    uint32_t u32b     = 0;
    int32_t i32a      = 0;
    int32_t i32b      = 0;
    uint16_t u16a     = 0;
    uint16_t u16b     = 0;
    int16_t i16a      = 0;
    int16_t i16b      = 0;
    uint8_t u8a       = 0;
    uint8_t u8b       = 0;
    int8_t i8a        = 0;
    int8_t i8b        = 0;
    const void* vptra = nullptr;
    const void* vptrb = nullptr;
    const char* cptra = "foo";
    const char* cptrb = cptra + 1;

    (void)u64a;
    (void)u64b;
    (void)i64a;
    (void)i64b;
    (void)u32a;
    (void)u32b;
    (void)i32a;
    (void)i32b;
    (void)u16a;
    (void)u16b;
    (void)i16a;
    (void)i16b;
    (void)u8a;
    (void)u8b;
    (void)i8a;
    (void)i8b;
    (void)vptra;
    (void)vptrb;
    (void)cptra;
    (void)cptrb;

    u64a = 5;
    i64a = 5;
    u32a = 5;
    i32a = 5;
    u16a = 5;
    i16a = 5;
    u8a  = 5;
    i8a  = 5;

    // var & var
    // u & u
    ZL_ASSERT_EQ(u64a, u64a);
    ZL_ASSERT_EQ(u64a, u32a);
    ZL_ASSERT_EQ(u64a, u16a);
    ZL_ASSERT_EQ(u64a, u8a);

    ZL_ASSERT_EQ(u32a, u64a);
    ZL_ASSERT_EQ(u32a, u32a);
    ZL_ASSERT_EQ(u32a, u16a);
    ZL_ASSERT_EQ(u32a, u8a);

    ZL_ASSERT_EQ(u16a, u64a);
    ZL_ASSERT_EQ(u16a, u32a);
    ZL_ASSERT_EQ(u16a, u16a);
    ZL_ASSERT_EQ(u16a, u8a);

    ZL_ASSERT_EQ(u8a, u64a);
    ZL_ASSERT_EQ(u8a, u32a);
    ZL_ASSERT_EQ(u8a, u16a);
    ZL_ASSERT_EQ(u8a, u8a);

    // i & i
    ZL_ASSERT_EQ(i64a, i64a);
    ZL_ASSERT_EQ(i64a, i32a);
    ZL_ASSERT_EQ(i64a, i16a);
    ZL_ASSERT_EQ(i64a, i8a);

    ZL_ASSERT_EQ(i32a, i64a);
    ZL_ASSERT_EQ(i32a, i32a);
    ZL_ASSERT_EQ(i32a, i16a);
    ZL_ASSERT_EQ(i32a, i8a);

    ZL_ASSERT_EQ(i16a, i64a);
    ZL_ASSERT_EQ(i16a, i32a);
    ZL_ASSERT_EQ(i16a, i16a);
    ZL_ASSERT_EQ(i16a, i8a);

    ZL_ASSERT_EQ(i8a, i64a);
    ZL_ASSERT_EQ(i8a, i32a);
    ZL_ASSERT_EQ(i8a, i16a);
    ZL_ASSERT_EQ(i8a, i8a);

    // u & i
    // ZL_ASSERT_EQ(u64a, i64a);
    // ZL_ASSERT_EQ(u64a, i32a);
    // ZL_ASSERT_EQ(u64a, i16a);
    // ZL_ASSERT_EQ(u64a, i8a);

    ZL_ASSERT_EQ(u32a, i64a);
    // ZL_ASSERT_EQ(u32a, i32a);
    // ZL_ASSERT_EQ(u32a, i16a);
    // ZL_ASSERT_EQ(u32a, i8a);

    ZL_ASSERT_EQ(u16a, i64a);
    ZL_ASSERT_EQ(u16a, i32a);
    ZL_ASSERT_EQ(u16a, i16a);
    ZL_ASSERT_EQ(u16a, i8a);

    ZL_ASSERT_EQ(u8a, i64a);
    ZL_ASSERT_EQ(u8a, i32a);
    ZL_ASSERT_EQ(u8a, i16a);
    ZL_ASSERT_EQ(u8a, i8a);

    // i & u
    // ZL_ASSERT_EQ(i64a, u64a);
    ZL_ASSERT_EQ(i64a, u32a);
    ZL_ASSERT_EQ(i64a, u16a);
    ZL_ASSERT_EQ(i64a, u8a);

    // ZL_ASSERT_EQ(i32a, u64a);
    // ZL_ASSERT_EQ(i32a, u32a);
    ZL_ASSERT_EQ(i32a, u16a);
    ZL_ASSERT_EQ(i32a, u8a);

    // ZL_ASSERT_EQ(i16a, u64a);
    // ZL_ASSERT_EQ(i16a, u32a);
    ZL_ASSERT_EQ(i16a, u16a);
    ZL_ASSERT_EQ(i16a, u8a);

    // ZL_ASSERT_EQ(i8a, u64a);
    // ZL_ASSERT_EQ(i8a, u32a);
    ZL_ASSERT_EQ(i8a, u16a);
    ZL_ASSERT_EQ(i8a, u8a);

    // var & lit
    ZL_ASSERT_EQ(u64a, 5);
    ZL_ASSERT_EQ(i64a, 5);
    ZL_ASSERT_EQ(u32a, 5);
    ZL_ASSERT_EQ(i32a, 5);
    ZL_ASSERT_EQ(u16a, 5);
    ZL_ASSERT_EQ(i16a, 5);
    ZL_ASSERT_EQ(u8a, 5);
    ZL_ASSERT_EQ(i8a, 5);

    ZL_ASSERT_EQ(u64a, 5u);
    ZL_ASSERT_EQ(i64a, 5u);
    ZL_ASSERT_EQ(u32a, 5u);
    // ZL_ASSERT_EQ(i32a, 5u);
    ZL_ASSERT_EQ(u16a, 5u);
    // ZL_ASSERT_EQ(i16a, 5u);
    ZL_ASSERT_EQ(u8a, 5u);
    // ZL_ASSERT_EQ(i8a, 5u);

    ZL_ASSERT_EQ(u64a, 5l);
    ZL_ASSERT_EQ(i64a, 5l);
    ZL_ASSERT_EQ(u32a, 5l);
    ZL_ASSERT_EQ(i32a, 5l);
    ZL_ASSERT_EQ(u16a, 5l);
    ZL_ASSERT_EQ(i16a, 5l);
    ZL_ASSERT_EQ(u8a, 5l);
    ZL_ASSERT_EQ(i8a, 5l);

    ZL_ASSERT_EQ(u64a, 5lu);
    // ZL_ASSERT_EQ(i64a, 5lu);
    ZL_ASSERT_EQ(u32a, 5lu);
    // ZL_ASSERT_EQ(i32a, 5lu);
    ZL_ASSERT_EQ(u16a, 5lu);
    // ZL_ASSERT_EQ(i16a, 5lu);
    ZL_ASSERT_EQ(u8a, 5lu);
    // ZL_ASSERT_EQ(i8a, 5lu);

    ZL_ASSERT_EQ(u64a, 5ll);
    ZL_ASSERT_EQ(i64a, 5ll);
    ZL_ASSERT_EQ(u32a, 5ll);
    ZL_ASSERT_EQ(i32a, 5ll);
    ZL_ASSERT_EQ(u16a, 5ll);
    ZL_ASSERT_EQ(i16a, 5ll);
    ZL_ASSERT_EQ(u8a, 5ll);
    ZL_ASSERT_EQ(i8a, 5ll);

    ZL_ASSERT_EQ(u64a, 5llu);
    // ZL_ASSERT_EQ(i64a, 5llu);
    ZL_ASSERT_EQ(u32a, 5llu);
    // ZL_ASSERT_EQ(i32a, 5llu);
    ZL_ASSERT_EQ(u16a, 5llu);
    // ZL_ASSERT_EQ(i16a, 5llu);
    ZL_ASSERT_EQ(u8a, 5llu);
    // ZL_ASSERT_EQ(i8a, 5llu);

    ZL_ASSERT_EQ(5, u64a);
    ZL_ASSERT_EQ(5, i64a);
    ZL_ASSERT_EQ(5, u32a);
    ZL_ASSERT_EQ(5, i32a);
    ZL_ASSERT_EQ(5, u16a);
    ZL_ASSERT_EQ(5, i16a);
    ZL_ASSERT_EQ(5, u8a);
    ZL_ASSERT_EQ(5, i8a);

    ZL_ASSERT_EQ(5u, u64a);
    ZL_ASSERT_EQ(5u, i64a);
    ZL_ASSERT_EQ(5u, u32a);
    // ZL_ASSERT_EQ(5u, i32a);
    ZL_ASSERT_EQ(5u, u16a);
    // ZL_ASSERT_EQ(5u, i16a);
    ZL_ASSERT_EQ(5u, u8a);
    // ZL_ASSERT_EQ(5u, i8a);

    ZL_ASSERT_EQ(5l, u64a);
    ZL_ASSERT_EQ(5l, i64a);
    ZL_ASSERT_EQ(5l, u32a);
    ZL_ASSERT_EQ(5l, i32a);
    ZL_ASSERT_EQ(5l, u16a);
    ZL_ASSERT_EQ(5l, i16a);
    ZL_ASSERT_EQ(5l, u8a);
    ZL_ASSERT_EQ(5l, i8a);

    ZL_ASSERT_EQ(5lu, u64a);
    // ZL_ASSERT_EQ(5lu, i64a);
    ZL_ASSERT_EQ(5lu, u32a);
    // ZL_ASSERT_EQ(5lu, i32a);
    ZL_ASSERT_EQ(5lu, u16a);
    // ZL_ASSERT_EQ(5lu, i16a);
    ZL_ASSERT_EQ(5lu, u8a);
    // ZL_ASSERT_EQ(5lu, i8a);

    ZL_ASSERT_EQ(5ll, u64a);
    ZL_ASSERT_EQ(5ll, i64a);
    ZL_ASSERT_EQ(5ll, u32a);
    ZL_ASSERT_EQ(5ll, i32a);
    ZL_ASSERT_EQ(5ll, u16a);
    ZL_ASSERT_EQ(5ll, i16a);
    ZL_ASSERT_EQ(5ll, u8a);
    ZL_ASSERT_EQ(5ll, i8a);

    ZL_ASSERT_EQ(5llu, u64a);
    // ZL_ASSERT_EQ(5llu, i64a);
    ZL_ASSERT_EQ(5llu, u32a);
    // ZL_ASSERT_EQ(5llu, i32a);
    ZL_ASSERT_EQ(5llu, u16a);
    // ZL_ASSERT_EQ(5llu, i16a);
    ZL_ASSERT_EQ(5llu, u8a);
    // ZL_ASSERT_EQ(5llu, i8a);

    // lit & lit
    ZL_ASSERT_EQ(5, 5);
    ZL_ASSERT_EQ(5, 5u);
    ZL_ASSERT_EQ(5u, 5);
    ZL_ASSERT_EQ(5u, 5u);

    ZL_ASSERT_EQ(5l, 5);
    ZL_ASSERT_EQ(5l, 5u);
    ZL_ASSERT_EQ(5lu, 5);
    ZL_ASSERT_EQ(5lu, 5u);

    ZL_ASSERT_EQ(5ll, 5);
    ZL_ASSERT_EQ(5ll, 5u);
    ZL_ASSERT_EQ(5llu, 5);
    ZL_ASSERT_EQ(5llu, 5u);

    ZL_ASSERT_EQ(5, 5l);
    ZL_ASSERT_EQ(5, 5lu);
    ZL_ASSERT_EQ(5u, 5l);
    ZL_ASSERT_EQ(5u, 5lu);

    ZL_ASSERT_EQ(5l, 5l);
    ZL_ASSERT_EQ(5l, 5lu);
    ZL_ASSERT_EQ(5lu, 5l);
    ZL_ASSERT_EQ(5lu, 5lu);

    ZL_ASSERT_EQ(5ll, 5l);
    ZL_ASSERT_EQ(5ll, 5lu);
    ZL_ASSERT_EQ(5llu, 5l);
    ZL_ASSERT_EQ(5llu, 5lu);

    ZL_ASSERT_EQ(5, 5ll);
    ZL_ASSERT_EQ(5, 5llu);
    ZL_ASSERT_EQ(5u, 5ll);
    ZL_ASSERT_EQ(5u, 5llu);

    ZL_ASSERT_EQ(5l, 5ll);
    ZL_ASSERT_EQ(5l, 5llu);
    ZL_ASSERT_EQ(5lu, 5ll);
    ZL_ASSERT_EQ(5lu, 5llu);

    ZL_ASSERT_EQ(5ll, 5ll);
    ZL_ASSERT_EQ(5ll, 5llu);
    ZL_ASSERT_EQ(5llu, 5ll);
    ZL_ASSERT_EQ(5llu, 5llu);

    // zero-extension

    u64a = 0xF0F0F0F0F0F0F0F0ull;
    u32a = 0xF0F0F0F0u;
    u16a = 0xF0F0;
    u8a  = 0xF0;

    ZL_ASSERT_NE(u64a, u32a);
    ZL_ASSERT_NE(u64a, u16a);
    ZL_ASSERT_NE(u64a, u8a);
    ZL_ASSERT_NE(u32a, u64a);
    ZL_ASSERT_NE(u32a, u16a);
    ZL_ASSERT_NE(u32a, u8a);
    ZL_ASSERT_NE(u16a, u64a);
    ZL_ASSERT_NE(u16a, u32a);
    ZL_ASSERT_NE(u16a, u8a);
    ZL_ASSERT_NE(u8a, u64a);
    ZL_ASSERT_NE(u8a, u32a);
    ZL_ASSERT_NE(u8a, u16a);

    // sign-extension

    i64a = -1;
    i32a = -1;
    i16a = -1;
    i8a  = -1;

    ZL_ASSERT_EQ(i64a, i64a);
    ZL_ASSERT_EQ(i64a, i32a);
    ZL_ASSERT_EQ(i64a, i16a);
    ZL_ASSERT_EQ(i64a, i8a);

    ZL_ASSERT_EQ(i32a, i64a);
    ZL_ASSERT_EQ(i32a, i32a);
    ZL_ASSERT_EQ(i32a, i16a);
    ZL_ASSERT_EQ(i32a, i8a);

    ZL_ASSERT_EQ(i16a, i64a);
    ZL_ASSERT_EQ(i16a, i32a);
    ZL_ASSERT_EQ(i16a, i16a);
    ZL_ASSERT_EQ(i16a, i8a);

    ZL_ASSERT_EQ(i8a, i64a);
    ZL_ASSERT_EQ(i8a, i32a);
    ZL_ASSERT_EQ(i8a, i16a);
    ZL_ASSERT_EQ(i8a, i8a);

    u8a  = 0xFF;
    u16a = 0xFFFF;
    u32a = 0xFFFFFFFFu;
    u64a = 0xFFFFFFFFFFFFFFFFull;

    ZL_ASSERT_NE(i8a, u8a);
    ZL_ASSERT_NE(i16a, u16a);
    // ZL_ASSERT_NE(i32a, u32a);
    // ZL_ASSERT_NE(i64a, u64a);

    ZL_ASSERT_NE(u8a, i8a);
    ZL_ASSERT_NE(u16a, i16a);
    // ZL_ASSERT_NE(u32a, i32a);
    // ZL_ASSERT_NE(u64a, i64a);

    // pointers

    cptra = nullptr;
    cptrb = "foo";

    ZL_ASSERT_NULL(nullptr); // although why would you want to do this???
    ZL_ASSERT_NULL(nullptr); // although why would you want to do this???
    ZL_ASSERT_NN("foo");     // although why would you want to do this???
    ZL_ASSERT_NULL(cptra);
    ZL_ASSERT_NN(cptrb);
    ZL_ASSERT_EQ(cptra, cptra);
    ZL_ASSERT_NE(cptra, cptrb);

    // vptra = nullptr;
    // vptrb = "foo";

    // ZL_ASSERT_NULL(vptra);
    // ZL_ASSERT_NN(vptrb);
    // ZL_ASSERT_EQ(vptra, vptra);
    // ZL_ASSERT_NE(vptra, vptrb);

    // pointer arithmetic expressions

    cptra = "foo";
    cptrb = cptra + 1;

    ZL_ASSERT_EQ(cptra + 2, cptrb + 1);

    ZL_ASSERT_NE(&u8a, &u8b);
}

TEST(DebugTest, assertOpFails)
{
    ZS_CHECK_ASSERT_FIRES(ZL_ASSERT_EQ(5, 6, "foo"));
}

TEST(DebugTest, requireOpSucceeds)
{
    ZL_REQUIRE_EQ(5, 5);
    ZL_REQUIRE_NE(5, 6);
    ZL_REQUIRE_GE(5, 5);
    ZL_REQUIRE_LE(5, 5);
    ZL_REQUIRE_GT(6, 5);
    ZL_REQUIRE_LT(5, 6);
}

TEST(DebugTest, requireOpSucceedsSign)
{
    ZL_REQUIRE_LT((size_t)1, (size_t)0xffffffffffffffff);
    ZL_REQUIRE_LT((int64_t)0xffffffffffffffff, (int64_t)1);
}

TEST(DebugTest, requireOpFails)
{
    ZS_CHECK_REQUIRE_FIRES(ZL_REQUIRE_EQ(5, 6, "foo"));
}

TEST(DebugTest, assertNNSucceeds)
{
    ZL_ASSERT_NN("foo");
    ZL_ASSERT_NN("foo", "yay!");
}

TEST(DebugTest, assertNNFails)
{
    ZS_CHECK_ASSERT_FIRES(ZL_ASSERT_NN(nullptr));
}

TEST(DebugTest, requireNNSucceeds)
{
    ZL_REQUIRE_NN("foo");
    ZL_REQUIRE_NN("foo", "yay!");
}

TEST(DebugTest, requireNNFails)
{
    ZS_CHECK_REQUIRE_FIRES(ZL_REQUIRE_NN(nullptr));
}

TEST(DebugTest, assertEvaluatesArguments)
{
    int x = 0;
    ZL_ASSERT(x == 0);
}

TEST(DebugTest, assertOpEvaluatesArguments)
{
    int x = 0;
    int y = 0;
    ZL_ASSERT_EQ(x, y);
}

TEST(DebugTest, abort)
{
    ASSERT_DEATH({ ZL_ABORT(); }, "");
}

TEST(DebugTest, log)
{
    int oldLogLevel = ZL_g_logLevel;
    ZL_g_logLevel   = ZL_LOG_ALL;

    ZL_LOG(V5, "%d", 5);
    ZL_LOG(V4, "%d", 4);
    ZL_LOG(V3, "foo");
    ZL_LOG(V2, "bar");
    ZL_LOG(V1, "baz");
    ZL_LOG(V, "quux");
    ZL_LOG(DEBUG, "xyzzy");
    ZL_LOG(WARN, "zappy");
    ZL_LOG(ERROR, "zoro");

    ZL_LOG(OBJ, "OBJ");
    ZL_LOG(FRAME, "FRAME");
    ZL_LOG(BLOCK, "BLOCK");
    ZL_LOG(SEQ, "SEQ");
    ZL_LOG(POS, "POS");

    ZL_g_logLevel = oldLogLevel;
}

void vlogHelper(
        std::function<void(const char*, va_list)> func,
        const char* fmt,
        ...)
{
    va_list args;
    va_start(args, fmt);
    func(fmt, args);
    va_end(args);
}

TEST(DebugTest, logVariants)
{
    int oldLogLevel = ZL_g_logLevel;
    ZL_g_logLevel   = ZL_LOG_ALL;

    ZL_LOG(ALWAYS, "ZL_LOG");
    ZL_DLOG(ALWAYS, "ZL_DLOG");
    ZL_RLOG(ALWAYS, "ZL_RLOG\n");
    ZL_RDLOG(ALWAYS, "ZL_RDLOG\n");

    ZL_FLOG(ALWAYS, "file", "func", 1, "ZL_FLOG");
    ZL_FDLOG(ALWAYS, "file", "func", 1, "ZL_FDLOG");
    ZL_FRLOG(ALWAYS, "file", "func", 1, "ZL_FRLOG\n");
    ZL_FRDLOG(ALWAYS, "file", "func", 1, "ZL_FRDLOG\n");

    vlogHelper(
            [](const char* fmt, va_list args) { ZL_VLOG(ALWAYS, fmt, args); },
            "ZL_VLOG %d",
            1234);
    vlogHelper(
            [](const char* fmt, va_list args) { ZL_VDLOG(ALWAYS, fmt, args); },
            "ZL_VDLOG %d",
            12345);
    vlogHelper(
            [](const char* fmt, va_list args) { ZL_VRLOG(ALWAYS, fmt, args); },
            "ZL_VRLOG %d\n",
            123456);
    vlogHelper(
            [](const char* fmt, va_list args) { ZL_VRDLOG(ALWAYS, fmt, args); },
            "ZL_VRDLOG %d\n",
            1234567);

    vlogHelper(
            [](const char* fmt, va_list args) {
                ZL_VFLOG(ALWAYS, "file", "func", 1, fmt, args);
            },
            "ZL_VFLOG %d",
            1234);
    vlogHelper(
            [](const char* fmt, va_list args) {
                ZL_VFDLOG(ALWAYS, "file", "func", 1, fmt, args);
            },
            "ZL_VFDLOG %d",
            12345);
    vlogHelper(
            [](const char* fmt, va_list args) {
                ZL_VFRLOG(ALWAYS, "file", "func", 1, fmt, args);
            },
            "ZL_VFRLOG %d\n",
            123456);
    vlogHelper(
            [](const char* fmt, va_list args) {
                ZL_VFRDLOG(ALWAYS, "file", "func", 1, fmt, args);
            },
            "ZL_VFRDLOG %d\n",
            1234567);

    ZL_g_logLevel = oldLogLevel;
}

TEST(DebugTest, logEvalutesArguments)
{
    int x = 0;
    ZL_LOG(V9, "%d", x);
}

TEST(DebugTest, rlog)
{
    ZL_RLOG(V9, "%d\n", 1);
}

TEST(DebugTest, rlogEvalutesArguments)
{
    int x = 0;
    ZL_RLOG(V9, "%d", x);
}

TEST(DebugTest, dlog)
{
    ZL_DLOG(V9, "%d", 1);
}

TEST(DebugTest, dlogEvalutesArguments)
{
    int x = 0;
    ZL_DLOG(V9, "%d", x);
}

TEST(DebugTest, rdlog)
{
    ZL_RDLOG(V9, "%d\n", 1);
}

TEST(DebugTest, rdlogEvalutesArguments)
{
    int x = 0;
    ZL_RDLOG(V9, "%d", x);
}

static bool throwAnException()
{
    throw std::runtime_error("Exception thrown!");
}

TEST(DebugTest, logIfDisabled)
{
#if ZL_ENABLE_LOG
    ASSERT_THROW(ZL_LOG_IF(throwAnException(), V9, "Foo!"), std::runtime_error);
#else
    ZL_LOG_IF(throwAnException(), V9, "Foo!");
#endif
}

TEST(DebugTest, rlogIfDisabled)
{
#if ZL_ENABLE_LOG
    ASSERT_THROW(
            ZL_RLOG_IF(throwAnException(), V9, "Foo!"), std::runtime_error);
#else
    ZL_RLOG_IF(throwAnException(), V9, "Foo!");
#endif
}

TEST(DebugTest, dlogIfDisabled)
{
#if ZL_ENABLE_DLOG
    ASSERT_THROW(
            ZL_DLOG_IF(throwAnException(), V9, "Foo!"), std::runtime_error);
#else
    ZL_DLOG_IF(throwAnException(), V9, "Foo!");
#endif
}

TEST(DebugTest, rdlogIfDisabled)
{
#if ZL_ENABLE_DLOG
    ASSERT_THROW(
            ZL_RDLOG_IF(throwAnException(), V9, "Foo!"), std::runtime_error);
#else
    ZL_RDLOG_IF(throwAnException(), V9, "Foo!");
#endif
}

TEST(DebugTest, logFatal)
{
    ASSERT_DEATH({ ZL_REQUIRE_FAIL("Aiiieeeeeee!!!"); }, "");
}

TEST(DebugTest, logDFatal)
{
#if ZL_ENABLE_ASSERT
    ASSERT_DEATH({ ZL_ASSERT_FAIL("Aiiieeeeeee!!!"); }, "");
#else
    // doesn't abort
    ZL_ASSERT_FAIL("Aiiieeeeeee!!!");
#endif
}
} // anonymous namespace

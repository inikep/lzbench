// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "openzl/common/allocation.h"
#include "openzl/common/assertion.h"
#include "openzl/common/errors_internal.h"
#include "openzl/common/refcount.h"

using namespace ::testing;

namespace {

class Refcount {
   public:
    explicit Refcount(size_t size)
    {
        ZL_REQUIRE_SUCCESS(ZL_Refcount_initMalloc(&rc_, malloc(size)));
    }

    explicit Refcount(std::unique_ptr<char> unique)
    {
        ALLOC_CustomAllocation const ca = {
            [](void*, size_t size) noexcept -> void* {
                return ::operator new(size);
            },
            [](void*, void* ptr) noexcept { ::operator delete(ptr); },
            nullptr
        };
        ZL_REQUIRE_SUCCESS(ZL_Refcount_init(
                &rc_,
                unique.release(),
                &ca,
                [](void*, void* ptr) noexcept { delete (char*)ptr; },
                nullptr));
    }

    Refcount(Refcount const&) = delete;

    explicit Refcount(ZL_Refcount rc) : rc_(rc) {}

    explicit Refcount(void const* ref)
    {
        ZL_REQUIRE_SUCCESS(ZL_Refcount_initConstRef(&rc_, ref));
    }

    void* getMut()
    {
        return ZL_Refcount_getMut(&rc_);
    }

    void const* get() const
    {
        return ZL_Refcount_get(&rc_);
    }

    bool null() const
    {
        return ZL_Refcount_null(&rc_);
    }

    bool mut() const
    {
        return ZL_Refcount_mutable(&rc_);
    }

    Refcount copy() const
    {
        return Refcount(ZL_Refcount_copy(&rc_));
    }

    Refcount alias(void* ptr) const
    {
        return Refcount(ZL_Refcount_aliasPtr(&rc_, ptr));
    }

    Refcount alias(size_t offset) const
    {
        return Refcount(ZL_Refcount_aliasOffset(&rc_, offset));
    }

    void constify()
    {
        ZL_Refcount_constify(&rc_);
    }

    ZL_Refcount& operator*()
    {
        return rc_;
    }

    void clear()
    {
        ZL_Refcount_destroy(&rc_);
    }

    ~Refcount()
    {
        ZL_Refcount_destroy(&rc_);
    }

   private:
    ZL_Refcount rc_;
};

TEST(RefcountTest, Basic)
{
    Refcount rc(5);
    ASSERT_TRUE(rc.mut());
    ASSERT_FALSE(rc.null());
    {
        auto cp = rc.copy();
        ASSERT_FALSE(rc.mut());
        ASSERT_FALSE(cp.mut());
        cp.constify();
    }

    ASSERT_TRUE(rc.mut());
    rc.constify();
    ASSERT_FALSE(rc.mut());
    Refcount cp = rc.copy();
    ASSERT_FALSE(cp.mut());

    rc.clear();
    ASSERT_TRUE(rc.null());
    ASSERT_FALSE(cp.mut());
}

TEST(RefcountTest, inArena)
{
    Arena* arena = ALLOC_HeapArena_create();
    ZL_Refcount rc;
    void* buffer = ZL_Refcount_inArena(&rc, arena, 100);
    memset(buffer, 0, 100);
    ZL_Refcount_destroy(&rc);
    ALLOC_Arena_freeArena(arena);
}

TEST(RefcountTest, ConstRef)
{
    char const x{};
    Refcount rc(&x);
    ASSERT_FALSE(rc.mut());
    {
        Refcount cp = rc.copy();
        ASSERT_FALSE(cp.mut());
        ASSERT_EQ(cp.get(), &x);
    }
    ASSERT_EQ(rc.get(), &x);
}

TEST(RefcountTest, CustomFree)
{
    auto unique = std::make_unique<char>(5);
    auto ptr    = unique.get();
    Refcount rc(std::move(unique));
    ASSERT_TRUE(rc.mut());
    ASSERT_EQ(rc.getMut(), ptr);
    ASSERT_EQ(rc.get(), ptr);
}

TEST(RefcountTest, Alias)
{
    Refcount rc(10);
    Refcount ap = rc.alias((char*)rc.getMut() + 5);
    Refcount ao = rc.alias(5);
    ASSERT_EQ((char const*)rc.get() + 5, ap.get());
    ASSERT_EQ(ap.get(), ao.get());
}

} // namespace

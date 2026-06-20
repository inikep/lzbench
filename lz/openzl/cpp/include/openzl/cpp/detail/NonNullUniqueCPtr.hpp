// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>
#include <type_traits>

#include "openzl/cpp/Exception.hpp"

namespace openzl {
namespace detail {

/**
 * Owns a T* object and guarantees that it is non-null.
 * This is useful for owning C context objects because:
 * 1. It ensures that if the constructor fails an exception is thrown.
 * 2. It specializes the deleter to be a C-style free function.
 * 3. It ensures that an exception is thrown on use-after-move.
 * 4. It handles const ownership of an object that needs to be passed as
 *    non-const to a deleter.
 */
template <typename T>
class NonNullUniqueCPtr {
   public:
    using NonConstT = typename std::remove_const<T>::type;
    using DeleterFn = void (*)(NonConstT*);

    /**
     * Takes ownership of @p obj, which must be non-null.
     * @p deleter is called on @p obj when the NonNullUniqueCPtr is destroyed.
     * @p deleter may be nullptr, in which case the object is not deleted.
     * Throws an exception if @p obj is nullptr.
     */
    explicit NonNullUniqueCPtr(NonConstT* obj, DeleterFn deleter)
            : obj_(obj, deleter)
    {
        if (obj_ == nullptr) {
            throw Exception("NonNullUniqueCPtr obj is null");
        }
    }

    /**
     * Creates a NonNullUniqueCPtr that references @p obj, which must be
     * non-null.
     */
    explicit NonNullUniqueCPtr(const T* obj) : obj_(obj, nullptr)
    {
        if (obj_ == nullptr) {
            throw Exception("NonNullUniqueCPtr obj is null");
        }
    }

    T* get() const
    {
        if (obj_ == nullptr) {
            throw Exception("NonNullUniqueCPtr use-after-move");
        }
        return obj_.get();
    }

    T& operator*() const
    {
        if (obj_ == nullptr) {
            throw Exception("NonNullUniqueCPtr use-after-move");
        }
        return *obj_;
    }

   private:
    struct Deleter {
       public:
        /* implicit */ Deleter(DeleterFn deleter) : deleter_(deleter) {}

        void operator()(T* obj) const
        {
            if (deleter_) {
                deleter_(const_cast<NonConstT*>(obj));
            }
        }

       private:
        DeleterFn deleter_;
    };
    std::unique_ptr<T, Deleter> obj_;
};

} // namespace detail

} // namespace openzl

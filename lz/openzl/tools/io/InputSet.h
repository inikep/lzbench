// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <stddef.h>
#include <iterator>
#include <memory>

#include "tools/io/Input.h"

namespace openzl::tools::io {

/**
 * Abstract interface representing an iterable set of @ref Input objects.
 *
 * Implementations include:
 *
 * - @ref InputSetStatic   : which just implements this interface over a vector
 *                           of Inputs it's given.
 * - @ref InputSetMulti    : which composes multiple InputSets into one
 *                           chained iterable.
 * - @ref InputSetDir      : which finds all the files in a directory,
 *                           optionally recursively.
 * - @ref InputSetFileOrDir: which resolves either to that single file or to
 *                           an InputSetDir of that path if it's a directory.
 *
 * Iterating over the input set multiple times is not guaranteed to produce the
 * same Inputs. Use @ref InputSetStatic::from_input_set() to freeze the Inputs,
 * allowing free repeat iterations. This requires materializing the whole list
 * and all the Inputs in the list though.
 *
 * Targeted towards uses like:
 *
 * ```c
 * size_t total_size(const InputSet& inputs) {
 *   size_t t = 0;
 *   for (const auto& input : inputs) {
 *     t += input->contents().size();
 *   }
 *   return t;
 * }
 * ```
 */
class InputSet {
   protected:
    /// Private internal abstract interface that backs the public Iterator.
    /// This is what InputSet implementations must implement.
    class IteratorState {
       public:
        virtual ~IteratorState() = default;

        virtual std::unique_ptr<IteratorState> copy() const = 0;

        virtual IteratorState& operator++() = 0;

        /// Should return `nullptr` iff the iterator has reached the end of the
        /// set. Returning `nullptr` earlier will result in truncated iteration.
        virtual const std::shared_ptr<Input>& operator*() const = 0;

        virtual bool operator==(const IteratorState& o) const = 0;
    };

   public:
    /// Public Iterator object.
    class Iterator;

   public:
    virtual ~InputSet() = default;

    Iterator begin() const;
    Iterator end() const;

   protected:
    virtual std::unique_ptr<IteratorState> begin_state() const = 0;
};

class InputSet::Iterator {
   public:
    // Implements LegacyInputIterator.
    using difference_type   = std::ptrdiff_t;
    using value_type        = std::shared_ptr<Input>;
    using pointer           = value_type*;
    using reference         = value_type&;
    using iterator_category = std::input_iterator_tag;

   public:
    // end()
    Iterator();

    // begin()
    explicit Iterator(std::unique_ptr<IteratorState> state);

    Iterator(const Iterator& o);
    Iterator(Iterator&& o) = default;

    Iterator& operator=(const Iterator& o);
    Iterator& operator=(Iterator&& o) = default;

    const value_type& operator*() const;

    Iterator& operator++();

    Iterator operator++(int) const;

    bool operator==(const Iterator& o) const;

    bool operator!=(const Iterator& o) const;

   private:
    std::unique_ptr<IteratorState> state_;
};

} // namespace openzl::tools::io

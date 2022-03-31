/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "qpl/cpp_api/results/execution_result.hpp"

namespace qpl {

template <class result_t>
template <class present_statement_t, class absent_statement_t>
void execution_result<result_t, sync>::handle(present_statement_t present_statement,
                                              absent_statement_t absent_statement) const {
    static_assert(std::is_invocable<present_statement_t, result_t>::value,
                  "Wrong arguments type for present statement");
    static_assert(std::is_invocable<absent_statement_t, uint32_t>::value,
                  "Absent statement can only be called with uint32_t");

    if (status_ == 0) {
        present_statement(std::move(result_));
    } else {
        absent_statement(status_);
    }
}

template <class result_t>
template <class present_statement_t, class absent_statement_t>
void execution_result<result_t, async>::handle(present_statement_t present_statement,
                                               absent_statement_t absent_statement) const {
    static_assert(std::is_invocable<absent_statement_t, uint32_t>::value,
                  "Absent statement can only be called with uint32_t");

    if (future_.valid()) {
        future_.wait();
        auto future_result = future_.get();
        future_result.handle(present_statement, absent_statement);
    }
}

template <class result_t>
template <class absent_statement_t>
void execution_result<result_t, sync>::if_absent(absent_statement_t absent_statement) const {
    static_assert(std::is_invocable<absent_statement_t, uint32_t>::value,
                  "Absent statement can only be called with uint32_t");

    if (status_ != 0) {
        absent_statement(status_);
    }
}

template <class result_t>
template <class present_statement_t>
void execution_result<result_t, sync>::if_present(present_statement_t present_statement) const {
    static_assert(std::is_invocable<present_statement_t, result_t>::value,
                  "Wrong arguments type for present statement");

    if (status_ == 0) {
        present_statement(std::move(result_));
    }
}

template <class result_t>
template <class absent_statement_t>
void execution_result<result_t, async>::if_absent(absent_statement_t absent_statement) const {
    if (future_.valid()) {
        future_.wait();
        auto future_result = future_.get();
        future_result.if_absent(absent_statement);
    }
}

template <class result_t>
template <class present_statement_t>
void execution_result<result_t, async>::if_present(present_statement_t present_statement) const {
    if (future_.valid()) {
        future_.wait();
        auto future_result = future_.get();
        future_result.if_present(present_statement);
    }
}

} // namespace qpl

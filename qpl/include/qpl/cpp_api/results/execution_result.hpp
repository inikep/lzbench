/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_EXECUTION_RESULT_HPP
#define QPL_EXECUTION_RESULT_HPP

#include <future>

namespace qpl {

/**
 * @addtogroup HL_PUBLIC
 * @{
 */

/**
 * @brief Contains supported execution modes
 */
enum execution_mode : uint32_t {
    sync,    /**< Represents synchronous execution */
    async    /**< Represents asynchronous execution */
};

/**
 * @brief Class specialization for parameterized execution_result.
 *
 * @tparam  result_t  type of the kept value
 * @tparam  mode      execution path
 */
template <class result_t, execution_mode mode = sync>
class execution_result;

/**
 * @brief Class that represents the result of synchronous execution
 *
 * @tparam  result_t  type of the kept value
 */
template <class result_t>
class execution_result<result_t, sync> {
public:
    /**
     * @brief Default constructor
     */
    execution_result() = default;

    /**
     * @brief Default copy constructor
     */
    constexpr execution_result(const execution_result &other) noexcept = default;

    /**
     * @brief Default move constructor
     */
    constexpr execution_result(execution_result &&other) noexcept = default;

    /**
     * @brief Default assignment operator
     */
    constexpr auto operator=(const execution_result &other) -> execution_result & = default;

    /**
     * @brief Method that performs the given actions depending on status
     *
     * @tparam  present_statement_t  type of the present statement
     * @tparam  absent_statement_t   type of the absent statement
     *
     * @param   present_statement    action to be performed if the status is zero
     * @param   absent_statement     action to be performed if a status is not zero
     */
    template <class present_statement_t, class absent_statement_t>
    void handle(present_statement_t present_statement, absent_statement_t absent_statement) const;

    /**
     * @brief Method that performs the given action if a result was present
     *
     * @tparam  present_statement_t  type of the present statement
     *
     * @param   present_statement    action to be performed if the status is zero
     */
    template <class present_statement_t>
    void if_present(present_statement_t present_statement) const;

    /**
     * @brief Method that performs the given action if a result was absent
     *
     * @tparam  absent_statement_t  type of the absent statement
     *
     * @param   absent_statement    action to be performed if a status is not zero
     */
    template <class absent_statement_t>
    void if_absent(absent_statement_t absent_statement) const;

    /**
     * @brief Constructor that accepts the status code and the result
     */
    constexpr explicit execution_result(uint32_t status, result_t result)
            : status_(status),
              result_(std::move(result)) {
        // Empty constructor
    }

private:
    uint32_t status_ = 0;    /**< Status code */
    result_t result_{};      /**< Execution result */
};

/**
 * @brief Class that represents the future result of asynchronous execution
 *
 * @tparam  result_t  type of the kept value
 */
template <class result_t>
class execution_result<result_t, async> {
public:
    /**
     * @brief Deleted default constructor
     */
    execution_result() = delete;

    /**
     * @brief Default copy constructor
     */
    constexpr execution_result(const execution_result &other) noexcept = default;

    /**
     * @brief Default move constructor
     */
    constexpr execution_result(execution_result &&other) noexcept = default;

    /**
     * @brief Default assignment operator
     */
    constexpr auto operator=(const execution_result &other) -> execution_result & = default;

    /**
     * @brief Method that performs the given actions depending on status
     *
     * @tparam  present_statement_t  type of the present statement
     * @tparam  absent_statement_t   type of the absent statement
     *
     * @param   present_statement    action to be performed if the status is zero
     * @param   absent_statement     action to be performed if a status is not zero
     */
    template <class present_statement_t, class absent_statement_t>
    void handle(present_statement_t present_statement, absent_statement_t absent_statement) const;

    /**
     * @brief Method that performs the given action if a result was present
     *
     * @tparam  present_statement_t  type of the present statement
     *
     * @param   present_statement    action to be performed if the status is zero
     */
    template <class present_statement_t>
    void if_present(present_statement_t present_statement) const;

    /**
     * @brief Method that performs the given action if a result was absent
     *
     * @tparam  absent_statement_t  type of the absent statement
     *
     * @param   absent_statement    action to be performed if a status is not zero
     */
    template <class absent_statement_t>
    void if_absent(absent_statement_t absent_statement) const;

    /**
     * @brief Constructor that accepts the @code std::shared_future @endcode with the result
     */
    constexpr explicit execution_result(const std::shared_future<result_t> &future)
            : future_(future) {
        // Empty constructor
    }

private:
    /**
     * @code std::shared_future @endcode containing execution result
     */
    std::shared_future<result_t> future_{};
};

/** @} */

} // namespace qpl

#include "execution_result.cxx"

#endif // QPL_EXECUTION_RESULT_HPP

/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_EXCEPTIONS_HPP
#define QPL_EXCEPTIONS_HPP

#include <exception>

namespace qpl {

/**
 * @addtogroup HL_PUBLIC
 * @{
 */

/**
 * @brief Base library exception
 */
class exception : public std::exception {
public:
    /**
     * @brief Constructor that accepts exception description
     */
    explicit exception(const char *message) noexcept;

    /**
     * @brief Returns string message with detailed exception explanation
     */
    [[nodiscard]] auto what() const noexcept -> const char * override;

    /**
     * @brief Default destructor
     */
    ~exception() noexcept override = default;

private:
    const char *message_;    /**< Detailed message about exception */
};

/**
 * @brief exception for invalid arguments
 */
class invalid_argument_exception : public exception {
public:
    /**
     * @brief Constructor that accepts exception description
     */
    explicit invalid_argument_exception(const char *message) noexcept;
};

/**
 * @brief exception for runtime errors
 */
class operation_process_exception : public exception {
public:
    /**
     * @brief Constructor that accepts exception description
     */
    explicit operation_process_exception(const char *message) noexcept;
};

/**
 * @brief exception for memory overflow issues
 */
class memory_overflow_exception final : public invalid_argument_exception {
public:
    /**
     * @brief Constructor that accepts exception description
     */
    explicit memory_overflow_exception(const char *message) noexcept;
};

/**
 * @brief exception for memory underflow issues
 */
class memory_underflow_exception : public operation_process_exception {
public:
    /**
     * @brief Constructor that accepts exception description
     */
    explicit memory_underflow_exception(const char *message) noexcept;
};

/**
 * @brief exception for invalid passed input data (like header parsing errors, etc)
 */
class invalid_data_exception final : public operation_process_exception {
public:
    /**
     * @brief Constructor that accepts exception description
     */
    explicit invalid_data_exception(const char *message) noexcept;
};

/**
 * @brief exception for compression invalid properties
 */
class invalid_compression_parameter_exception final : public operation_process_exception {
public:
    /**
     * @brief Constructor that accepts exception description
     */
    explicit invalid_compression_parameter_exception(const char *message) noexcept;
};

/**
 * @brief exception for memory underflow in destination buffer
 */
class short_destination_exception final : public memory_underflow_exception {
public:
    /**
     * @brief Constructor that accepts exception description
     */
    explicit short_destination_exception(const char *message) noexcept;
};

/** @} */

} // namespace qpl

#endif // QPL_EXCEPTIONS_HPP

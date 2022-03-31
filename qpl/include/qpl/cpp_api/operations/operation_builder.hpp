/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

/*
 *  Intel® Query Processing Library (Intel® QPL)
 *  High Level API (public C++ API)
 */

#ifndef QPL_OPERATION_BUILDER_HPP
#define QPL_OPERATION_BUILDER_HPP

#include "qpl/cpp_api/util/exceptions.hpp"

namespace qpl {

/**
 * @addtogroup HL_PUBLIC
 * @{
 */

namespace traits {

template <class compression_operation_t, class compression_operation_builder_t>
struct common_type_for_compression_operation;

template <class analytic_operation_t, class analytic_operation_builder_t>
struct common_type_for_analytic_operation;

template <class operation_with_mask_t, class operation_with_mask_builder_t>
struct common_type_for_operation_with_mask;

} // namespace traits

/**
 * @brief Contains supported input formats
 */
enum parsers {
    little_endian_packed_array = 0u,    /**< Little-endian bit order */
    big_endian_packed_array    = 1u,    /**< Big-endian bit order */
    parquet_rle                = 2u     /**< Apache parquet format */
};

namespace internal {

/**
 * @addtogroup HL_PRIVATE
 * @{
 */

template <class compression_operation_t, class compression_operation_builder_t = std::nullptr_t>
class compression_operation_builder;

template <class analytic_operation_t, class analytic_operation_builder_t = std::nullptr_t>
class analytic_operation_builder;

template <class operation_with_mask_t, class operation_with_mask_builder_t = std::nullptr_t>
class operation_with_mask_builder;

/**
 * @brief Checks if given output width is supported by the library
 *
 * @param  value  width of the element in bits
 *
 * @return either @code true @endcode or throws @ref invalid_argument_exception
 */
static inline auto check_qpl_output_format(uint32_t value) -> bool {
    switch (value) {
        case 1:
        case 8:
        case 16:
        case 32: {
            return true;
        }
        default: {
            throw invalid_argument_exception("Intel QPL doesn't support such output format");
        }
    }
}

/**
 * @brief Checks if given input width is supported by the library
 *
 * @param  value  width of the element in bits
 *
 * @return either @code true @endcode or throws @ref invalid_argument_exception
 */
static inline auto check_qpl_input_format(uint32_t value) -> bool {
    if (value < 1 || value > 32) {
        throw invalid_argument_exception("Intel QPL doesn't support such input format");
    }

    return true;
}

/**
 * @brief Common class that is parent for all builders
 *
 * @tparam  operation_t  type of the operation that can be built with usage of the builder
 */
template <class operation_t>
class operation_builder {
public:
    /**
     * @brief Completes the build of the operation
     *
     * @return built instance of required operation
     */
    [[nodiscard]] auto build() const -> operation_t {
        return operation_;
    }

protected:
    /**
     * @brief Accepts a copy of operation that should be modified
     */
    constexpr explicit operation_builder(operation_t operation)
            : operation_(std::move(operation)) {
        // Empty constructor
    }

    operation_t operation_;    /**< Instance of currently configurable operation */
};

/**
 * @brief Basic builder for compression operations
 *
 * @tparam  compression_operation_t          type of compression operation that should be built
 * @tparam  compression_operation_builder_t  type of the builder that used for operation building
 */
template <class compression_operation_t,
          class compression_operation_builder_t>
class compression_operation_builder : public operation_builder<compression_operation_t> {
    /**
     * @brief Return type for builder methods
     */
    using common_type = typename traits::common_type_for_compression_operation<compression_operation_t,
                                                                               compression_operation_builder_t>::type;

public:
    /**
     * @brief Simple default constructor
     */
    constexpr explicit compression_operation_builder()
            : operation_builder<compression_operation_t>(compression_operation_t()) {
        // Empty constructor
    }

    /**
     * @brief Constructor that accepts already existing operation for re-configuration
     */
    constexpr explicit compression_operation_builder(compression_operation_t operation)
            : operation_builder<compression_operation_t>(std::move(operation)) {
        // Empty constructor
    }

    /**
     * @brief Sets new GZIP mode
     */
    [[nodiscard]] auto gzip_mode(bool value) -> common_type & {
        operation_builder<compression_operation_t>::operation_.properties_.gzip_mode_ = value;

        return *reinterpret_cast<common_type *>(this);
    }
};

/**
 * @brief Basic builder type for analytic operations
 *
 * @tparam  analytic_operation_t          type of analytic operation that should be built
 * @tparam  analytic_operation_builder_t  type of the builder that used for operation building
 */
template <class analytic_operation_t,
          class analytic_operation_builder_t>
class analytic_operation_builder : public operation_builder<analytic_operation_t> {
    /**
     * @brief Return type for builder methods
     */
    using common_type =
    typename traits::common_type_for_analytic_operation<analytic_operation_t,
                                                        analytic_operation_builder_t>::type;

public:
    /**
     * @brief Constructor that accepts already existing operation for re-configuration
     */
    constexpr explicit analytic_operation_builder(analytic_operation_t operation)
            : operation_builder<analytic_operation_t>(std::move(operation)) {
        // Empty constructor
    }

    /**
     * @brief Sets new bit width of one output element
     */
    [[nodiscard]] auto output_vector_width(uint32_t value) -> common_type & {
        check_qpl_output_format(value);
        operation_builder<analytic_operation_t>::operation_.output_vector_bit_width_ = value;

        return *reinterpret_cast<common_type *>(this);
    }

    /**
     * @brief Sets new bit width of one input element
     */
    [[nodiscard]] auto input_vector_width(uint32_t value) -> common_type & {
        check_qpl_input_format(value);
        operation_builder<analytic_operation_t>::operation_.input_vector_bit_width_ = value;

        return *reinterpret_cast<common_type *>(this);
    }

    /**
     * @brief Sets the number of input elements to process
     */
    [[nodiscard]] auto number_of_input_elements(uint32_t value) -> common_type & {
        operation_builder<analytic_operation_t>::operation_.number_of_input_elements_ = value;

        return *reinterpret_cast<common_type *>(this);
    }

    /**
     * @brief Sets new input parser (little_endian_packed_array big_endian_packed_array
     *        or parquet_rle)
     */
    template <parsers current_parser>
    [[nodiscard]] auto parser(uint32_t number_of_input_elements) -> common_type & {
        static_assert((current_parser == parsers::big_endian_packed_array ||
                       current_parser == parsers::little_endian_packed_array ||
                       current_parser == parsers::parquet_rle
                      ), "Intel QPL doesn't support such parser");

        operation_builder<analytic_operation_t>::operation_.parser_                   = current_parser;
        operation_builder<analytic_operation_t>::operation_.number_of_input_elements_ = number_of_input_elements;

        return *reinterpret_cast<common_type *>(this);
    }

    /**
     * @brief Sets the initial index output starts with
     */
    [[nodiscard]] auto initial_output_index(size_t value) -> common_type &{
        operation_builder<analytic_operation_t>::operation_.initial_output_index_ = (uint32_t)value;

        return *reinterpret_cast<common_type *>(this);
    }
};

/**
 * @brief Basic builder type for analytic operations with mask
 *
 * @tparam  operation_with_mask_t          type of analytic operation that should be built
 * @tparam  operation_with_mask_builder_t  type of the builder that used for operation building
 */
template <class operation_with_mask_t,
          class operation_with_mask_builder_t>
class operation_with_mask_builder : public analytic_operation_builder<operation_with_mask_t> {
    /**
     * @brief Return type for builder methods
     */
    using common_type = typename traits::common_type_for_operation_with_mask<operation_with_mask_t,
                                                                             operation_with_mask_builder_t>::type;

public:
    /**
     * @brief Simple default constructor
     */
    constexpr explicit operation_with_mask_builder()
            : analytic_operation_builder<operation_with_mask_t>(operation_with_mask_t()) {
        // Empty constructor
    }

    /**
     * @brief Duplicates any operation with mask constructor that accepts the mask
     */
    constexpr explicit operation_with_mask_builder(const uint8_t *mask, size_t mask_byte_length)
            : analytic_operation_builder<operation_with_mask_t>(operation_with_mask_t(mask, mask_byte_length)) {
        // Empty constructor
    }

    /**
     * @brief Constructor that accepts already existing operation for re-configuration
     */
    constexpr explicit operation_with_mask_builder(operation_with_mask_t operation)
            : analytic_operation_builder<operation_with_mask_t>(std::move(operation)) {
        // Empty constructor
    }

    /**
     * @brief Sets new mask to the operation
     *
     * @param  mask              pointer to the mask
     * @param  mask_byte_length  length of the mask in bytes
     */
    auto mask(const uint8_t *mask, size_t mask_byte_length) -> common_type & {
        analytic_operation_builder<operation_with_mask_t>::operation_.reset_mask(mask, mask_byte_length);

        return *reinterpret_cast<common_type *>(this);
    }

    /**
     * @brief Sets new bit width of one input element
     */
    [[nodiscard]] auto input_vector_width(uint32_t value) -> common_type & {
        check_qpl_input_format(value);
        analytic_operation_builder<operation_with_mask_t>::operation_.input_vector_bit_width_ = value;

        return *reinterpret_cast<common_type *>(this);
    }

    /**
     * @brief Sets new bit width of one output element
     */
    [[nodiscard]] auto output_vector_width(uint32_t value) -> common_type & {
        check_qpl_output_format(value);
        analytic_operation_builder<operation_with_mask_t>::operation_.output_vector_bit_width_ = value;

        return *reinterpret_cast<common_type *>(this);
    }

    /**
     * @brief Sets the number of input elements to process
     */
    [[nodiscard]] auto number_of_input_elements(uint32_t value) -> common_type & {
        analytic_operation_builder<operation_with_mask_t>::operation_.number_of_input_elements_ = value;

        return *reinterpret_cast<common_type *>(this);
    }

    /**
     * @brief Sets new input parser (little_endian_packed_array, big_endian_packed_array or parquet_rle)
     */
    template <parsers current_parser>
    [[nodiscard]] auto parser(uint32_t number_of_input_elements) -> common_type & {
        auto obj = analytic_operation_builder<operation_with_mask_t>::template parser<current_parser>(
                number_of_input_elements);

        return *reinterpret_cast<common_type *>(this);
    }

    /**
     * @brief Not supported
     */
    [[nodiscard]] auto initial_output_index(size_t value) -> common_type & = delete;
};

/** @} */

} // namespace internal

namespace traits {
/**
 * @addtogroup HL_TRAITS
 * @{
 */

/**
 * @brief Defines the return type for @ref internal.analytic_operation_builder
 */
template <class analytic_operation_t,
          class analytic_operation_builder_t>
struct common_type_for_analytic_operation {
    /**
     * Return type for @ref internal.analytic_operation_builder
     */
    using type = analytic_operation_builder_t;
};

/**
 * @brief Defines the return type for @ref internal.analytic_operation_builder
 */
template <class analytic_operation_t>
struct common_type_for_analytic_operation<analytic_operation_t, std::nullptr_t> {
    /**
     * Return type for @ref internal.analytic_operation_builder
     */
    using type = internal::analytic_operation_builder<analytic_operation_t>;
};

/**
 * @brief Defines the return type for @ref internal.operation_with_mask_builder
 */
template <class operation_with_mask_t,
          class operation_with_mask_builder_t>
struct common_type_for_operation_with_mask {
    /**
     * Return type for @ref internal.operation_with_mask_builder
     */
    using type = operation_with_mask_builder_t;
};

/**
 * @brief Defines the return type for @ref internal.operation_with_mask_builder
 */
template <class operation_with_mask_t>
struct common_type_for_operation_with_mask<operation_with_mask_t, std::nullptr_t> {
    /**
     * Return type for @ref internal.operation_with_mask_builder
     */
    using type = internal::operation_with_mask_builder<operation_with_mask_t>;
};

/**
 * @brief Defines the return type for @ref internal.compression_operation_builder
 */
template <class compression_operation_t,
          class compression_operation_builder_t>
struct common_type_for_compression_operation {
    /**
     * Return type for @ref internal.compression_operation_builder
     */
    using type = compression_operation_builder_t;
};

/**
 * @brief Defines the return type for @ref internal.compression_operation_builder
 */
template <class compression_operation_t>
struct common_type_for_compression_operation<compression_operation_t, std::nullptr_t> {
    /**
     * Return type for @ref internal.compression_operation_builder
     */
    using type = internal::compression_operation_builder<compression_operation_t>;
};

/** @} */

} // namespace traits

/** @} */

} // namespace qpl

#endif // QPL_OPERATION_BUILDER_HPP

/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_ANALYTIC_OPERATION_HPP
#define QPL_ANALYTIC_OPERATION_HPP

#include "qpl/cpp_api/operations/operation.hpp"

#if defined(__linux__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#endif

namespace qpl {

/**
 * @defgroup HL_ANALYTICS Analytics API
 * @ingroup HL_PUBLIC
 * @{
 * @brief Analytics entities and functions of high-level API
 */

class compression_operation;

/**
 * @brief Interface that contains analytics operations specific properties and methods
 */
class analytics_operation : public operation {
public:
    /**
     * @brief Simple constructor that defines inclusivity for operation
     *
     * @param  is_inclusive  sets an operation into inclusive or exclusive mode
     */
    constexpr explicit analytics_operation(bool is_inclusive)
            : is_inclusive_(is_inclusive) {
        // Empty constructor
    }

    /**
     * @brief Friend function for composing decompression and analytic operations
     */
    template <execution_path path,
            template <class> class allocator_t,
                             class input_iterator_t,
                             class output_iterator_t,
                             class operation_t>
    friend auto compose(compression_operation &decompress_operation,
                        operation_t &operation,
                        const input_iterator_t &source_begin,
                        const input_iterator_t &source_end,
                        const output_iterator_t &destination_begin,
                        const output_iterator_t &destination_end,
                        int32_t numa_id,
                        uint8_t *job_buffer,
                        uint32_t number_of_decompressed_elements) -> execution_result<uint32_t, sync>;

    /**
     * @brief Friend function for composing two analytic operations
     */
    template <execution_path path,
            template <class> class allocator_t,
                             class mask_builder_t,
                             class mask_applier_t,
                             class input_iterator_t,
                             class output_iterator_t>
    friend auto compose(mask_builder_t &mask_builder,
                        mask_applier_t &mask_applier,
                        const input_iterator_t &source_begin,
                        const input_iterator_t &source_end,
                        const output_iterator_t &destination_begin,
                        const output_iterator_t &destination_end,
                        int32_t numa_id,
                        uint8_t *job_buffer) -> execution_result<uint32_t, sync>;

    /**
     * @brief Friend function to make two analytic operations compatible
     */
    template <class operation1_t, class operation2_t>
    friend void internal::connect_two(const operation1_t &a, operation2_t &b);

    /**
     * @brief Returns width in bits for one element in output buffer
     */
    [[nodiscard]] virtual auto get_output_vector_width() const noexcept -> uint32_t {
        return (output_vector_bit_width_ == 1 || output_vector_bit_width_ == 0)
               ? input_vector_bit_width_
               : output_vector_bit_width_;
    }

    /**
     * @brief Returns width in bits for one element in input buffer
     */
    [[nodiscard]] auto get_input_vector_width() const noexcept -> uint32_t {
        return input_vector_bit_width_;
    }

protected:
    /**
     * @brief Enables decompression functionality
     *
     * @param  number_of_decompressed_elements  number of decompressed elements to be processed
     * @param  gzip_mode                        flag for gzip compressed data
     */
    virtual void enable_decompression(uint32_t number_of_decompressed_elements, bool gzip_mode) {
        is_decompression_enabled_        = true;
        gzip_mode_                       = gzip_mode;
        number_of_decompressed_elements_ = number_of_decompressed_elements;
    }

    parsers  parser_                   = little_endian_packed_array; /**< Input format */
    uint32_t   number_of_input_elements_ = 0; /**< Number of elements that should be processed */
    bool     is_inclusive_             = false; /**< Is operation inclusive or exclusive */
    uint32_t input_vector_bit_width_   = byte_bit_length; /**< Bit-width of one input element */
    uint32_t output_vector_bit_width_  = 1;               /**< Bit-width of one output element */
    uint32_t initial_output_index_     = 0u; /**< Initial index value for output (with index output mode) */

    // These three fields are used to enable analytics with decompression

    bool is_decompression_enabled_ = false;    /**< Is decompression enabled for analytics */
    bool gzip_mode_                = false;    /**< Is GZIP mode enabled for decompression */

    /**
     * Number of elements that should be decompressed before analytics
     */
    size_t number_of_decompressed_elements_ = 0;
};

/**
 * @brief Interface for specific analytics operations that uses two source buffers
 *        (one is source and another one is bit mask)
 */
class operation_with_mask : public analytics_operation {
public:
    /**
     * @brief Simple default constructor
     */
    constexpr operation_with_mask() : analytics_operation(false) {
        // Empty constructor
    }

    /**
     * @brief Constructor that accepts bitmask for the operation
     *
     * @param  mask              pointer to mask
     * @param  mask_byte_length  length of the mask in bytes
     */
    constexpr operation_with_mask(const uint8_t *mask, size_t mask_byte_length)
            : analytics_operation(false),
              mask_(mask),
              mask_byte_length_(mask_byte_length) {
        // Empty constructor
    }

protected:
    const uint8_t *mask_            = nullptr;    /**< Pointer to mask */
    size_t        mask_byte_length_ = 0;          /**< Size of the mask */

    /**
     * @brief Resets current bit-mask for operation
     *
     * @param  mask              pointer to buffer with mask
     * @param  mask_byte_length  length in bytes of the mask
     */
    virtual void reset_mask(const uint8_t *mask, size_t mask_byte_length) noexcept = 0;
};

/** @} */

} // namespace qpl

#if defined(__linux__)
#pragma GCC diagnostic pop
#endif

#endif // QPL_ANALYTIC_OPERATION_HPP

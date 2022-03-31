/*******************************************************************************
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef QPL_TRAITS_HPP
#define QPL_TRAITS_HPP

namespace qpl {

class inflate_operation;

class expand_operation;

class find_unique_operation;

class set_membership_operation;

class merge;

class operation;

class analytics_operation;

class operation_with_mask;

namespace traits {

/**
 * @defgroup HL_TRAITS Type traits
 * @ingroup HL_PRIVATE
 * @{
 * @brief Contains service type traits entities and rules
 */

/**
 * @brief Checks whether any of its types are equal to the first type
 */
template <class param_t, class... params_t>
inline constexpr bool is_any = std::disjunction<std::is_same<param_t, params_t>...>{};

/**
 * @brief Calculates number of operations considering merged ones
 */
template <class param_t,
          class... params_t>
constexpr auto actual_number_of_operations() -> uint32_t;

/**
 * @brief Checks if two operations should be merged or not
 *
 * @note Two operations should be merged if they may be combined as a pair of
 *       "mask builder" and "mask applier"
 */
template <class operation1_t,
          class operation2_t>
constexpr auto should_be_merged() -> bool {
    return std::is_base_of<operation_with_mask, operation2_t>::value &&
           !std::is_same<operation2_t, expand_operation>::value &&
           !std::is_same<operation2_t, set_membership_operation>::value;
}

/**
 * @brief Checks if passed list of types contains only children of @ref operation
 */
template <class param_t,
          class... params_t>
constexpr auto are_qpl_operations() -> bool {
    if constexpr (sizeof...(params_t) > 0) {
        return std::is_base_of<operation, param_t>::value && are_qpl_operations<params_t...>();
    }

    return std::is_base_of<operation, param_t>::value;
}

/**
 * @brief Checks if passed list of types contains only children of @ref analytics_operation
 */
template <class param_t,
          class... params_t>
constexpr auto are_analytics_operations() {
    if constexpr (sizeof...(params_t) > 0) {
        return std::is_base_of<analytics_operation, param_t>::value &&
               are_analytics_operations<params_t...>();
    }

    return std::is_base_of<analytics_operation, param_t>::value;
}

/**
 * @brief Checks if passed operation type is a child of @ref operation_with_mask
 */
template <class param_t,
          class... params_t>
constexpr auto is_the_following_mask_operation() -> bool {
    return std::is_base_of<operation_with_mask, param_t>::value;
}

/**
 * @brief Checks if @ref merge is applied correctly in the chain
 */
template <class param_t,
          class... params_t>
constexpr auto is_the_following_object_merge_manipulator() -> bool {
    if constexpr (std::is_same<merge, param_t>::value) {
        static_assert(sizeof...(params_t) > 0, "Merge cannot be applied there");
    }

    return std::is_same<merge, param_t>::value;
}

/**
 * @brief Checks if two operations should be merged as pair decompress + analyze
 */
template <class param1_t,
          class param2_t>
constexpr auto decompression_should_be_enabled() -> bool {
    return std::is_same<inflate_operation, param1_t>::value
           && are_analytics_operations<param2_t>();
}

/**
 * @brief Calculates number of operations considering merged ones
 */
template <class param1_t,
          class param2_t,
          class... params_t>
constexpr auto actual_number_of_operations_after_merge() -> uint32_t {
    static_assert(std::is_base_of<analytics_operation, param2_t>::value,
                  "Merge cannot be applied there");
    static_assert(!should_be_merged<std::nullptr_t, param2_t>(),
                  "Merge cannot be applied there");

    if constexpr (sizeof...(params_t) > 0) {
        static_assert(!is_the_following_mask_operation<params_t...>(),
                      "Merge cannot be applied there");

        return 1 + actual_number_of_operations<params_t...>();
    } else {
        return 1;
    }
}

template <class param_t, class... params_t>
constexpr auto actual_number_of_operations() -> uint32_t {
    constexpr bool is_merged = should_be_merged<std::nullptr_t, param_t>();

    if constexpr (sizeof...(params_t) == 0) {
        return is_merged ? 0 : 1;
    } else {
        if constexpr (is_the_following_object_merge_manipulator<params_t...>()) {
            static_assert(std::is_same<inflate_operation, param_t>::value, "Merge cannot be applied there");
            return actual_number_of_operations_after_merge<params_t...>();
        } else {
            return (is_merged ? 0 : 1) + actual_number_of_operations<params_t...>();
        }
    }
}

/**
 * @brief Checks whether the number of elements for the operation should be set
 */
template <class param_t>
constexpr auto need_number_of_elements() -> bool {
    return are_analytics_operations<param_t>() &&
           !std::is_same<expand_operation, param_t>::value;
}


/** @} */

} // namespace traits
} // namespace qpl

#endif // QPL_TRAITS_HPP

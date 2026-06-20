// Copyright (c) Meta Platforms, Inc. and affiliates.

/**
 * \file
 *
 * This file, `base_types.h`, contains implementation details regarding
 * low-level public (?) types. While these implementation details are exposed to
 * the compiler, they should not be invoked by end-users, hence they are moved
 * away from `zstrong.h`.
 */

#ifndef ZSTRONG_COMMON_BASE_TYPES_H
#define ZSTRONG_COMMON_BASE_TYPES_H

#include "openzl/zl_errors.h" // ZL_Report

#include <stddef.h> // size_t

typedef unsigned int ZL_StreamID;
typedef unsigned int ZL_TransformID;

// Grammar related
typedef unsigned ZL_tagID;
typedef struct {
    ZL_tagID tag;
} ZL_TypeTag;

/**
 * A non-owning list of ZS_TypeTags.
 */
typedef struct {
    const ZL_TypeTag* tags;
    size_t nbTags;
} ZL_TypeTagList;

/**
 * A non-owning list of ZS_StreamIDs.
 */
typedef struct {
    const ZL_StreamID* ids;
    size_t nbIds;
} ZL_StreamIDList;

/* Generate list of basic types
 * Note : Base types properties are defined in grammar.c
 * */

// This method uses a list of enums to define ID of basic types
#define ZL_BASETYPE_NAME(basename) ZS_t_##basename

#define ZL_ENUM_NAME(basename) ZS_##basename##_v,

#define ZL_LIST_BASETYPES(OP) \
    OP(u8)                    \
    OP(u16le)                 \
    OP(u32le)                 \
    OP(u64le)                 \
    OP(unstructuredLengths)   \
    OP(unstructuredData)      \
    OP(nbBaseTypes) // must be at the end

typedef enum { ZL_LIST_BASETYPES(ZL_ENUM_NAME) } ZL_baseTypes_values_e;

#define ZL_TYPE_CONSTANT(basename) \
    (ZL_TypeTag)                   \
    {                              \
        ZL_ENUM_NAME(basename)     \
    }

#if 0

// This method uses the list of enums to generate the same list of constants.
// Unfortunately, in C syntax, a `static const` variable is not a "constant"
// This status is reserved to literals and enums,
// and macros using literals and enums.

#    define ZL_CREATE_TYPE_CONSTANT(basename)                \
        static ZL_TypeTag const ZL_BASETYPE_NAME(basename) = \
                ZL_TYPE_CONSTANT(basename);

// Create all base type constants, such as,
// ZL_t_u8, ZL_t_u16le, ... , ZL_t_unstructuredData
ZL_LIST_BASETYPES(ZL_CREATE_TYPE_CONSTANT)

#else

// Declare basic types with the preprocessor
// As macros can't generate macros,
// this requires one line per base type.
// But at least, the resulting base type name is considered a constant.

#    define ZL_t_u8 ZL_TYPE_CONSTANT(u8)
#    define ZL_t_u16le ZL_TYPE_CONSTANT(u16le)
#    define ZL_t_u32le ZL_TYPE_CONSTANT(u32le)
#    define ZL_t_u64le ZL_TYPE_CONSTANT(u64le)
#    define ZL_t_unstructuredLengths ZL_TYPE_CONSTANT(unstructuredLengths)
#    define ZL_t_unstructuredData ZL_TYPE_CONSTANT(unstructuredData)
#    define ZL_t_nbBaseTypes ZL_TYPE_CONSTANT(nbBaseTypes)

#endif

#endif // ZSTRONG_COMMON_BASE_TYPES_H

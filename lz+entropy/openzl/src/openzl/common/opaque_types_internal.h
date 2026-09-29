// Copyright (c) Meta Platforms, Inc. and affiliates.

/**
 * \file
 *
 * This file declares result types that wrap public but opaque types. Normally
 * we would declare the result type alongside the main type's definition. For
 * now though we want to keep these declarations private. So they're declared
 * here.
 */

#ifndef ZSTRONG_COMMON_OPAQUE_TYPES_INTERNAL_H
#define ZSTRONG_COMMON_OPAQUE_TYPES_INTERNAL_H

#include <stdarg.h> // va_list
#include <stddef.h> // size_t

#include "openzl/shared/portability.h" // ZL_BEGIN_C_DECLS, ZL_END_C_DECLS
#include "openzl/zl_errors.h"          // ZL_Report
#include "openzl/zl_opaque_types.h"    // ZL_DataID, ZL_NodeID, ZL_GraphID

ZL_BEGIN_C_DECLS

ZL_RESULT_DECLARE_TYPE(ZL_IDType);
ZL_RESULT_DECLARE_TYPE(ZL_DataID);

ZL_END_C_DECLS

#endif // ZSTRONG_COMMON_OPAQUE_TYPES_INTERNAL_H

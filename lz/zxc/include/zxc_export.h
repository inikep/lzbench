/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ZXC_EXPORT_H
#define ZXC_EXPORT_H

#ifdef ZXC_STATIC_DEFINE
#define ZXC_EXPORT
#define ZXC_NO_EXPORT
#else
#ifndef ZXC_EXPORT
#ifdef zxc_lib_EXPORTS
// We are building this library
#ifdef _WIN32
#define ZXC_EXPORT __declspec(dllexport)
#else
#define ZXC_EXPORT __attribute__((visibility("default")))
#endif
#else
// We are using this library
#ifdef _WIN32
#define ZXC_EXPORT __declspec(dllimport)
#else
#define ZXC_EXPORT __attribute__((visibility("default")))
#endif
#endif
#endif

#ifndef ZXC_NO_EXPORT
#ifdef _WIN32
#define ZXC_NO_EXPORT
#else
#define ZXC_NO_EXPORT __attribute__((visibility("hidden")))
#endif
#endif
#endif

#ifndef ZXC_DEPRECATED
#ifdef _WIN32
#define ZXC_DEPRECATED __declspec(deprecated)
#else
#define ZXC_DEPRECATED __attribute__((__deprecated__))
#endif
#endif

#ifndef ZXC_DEPRECATED_EXPORT
#define ZXC_DEPRECATED_EXPORT ZXC_EXPORT ZXC_DEPRECATED
#endif

#ifndef ZXC_DEPRECATED_NO_EXPORT
#define ZXC_DEPRECATED_NO_EXPORT ZXC_NO_EXPORT ZXC_DEPRECATED
#endif

#endif /* ZXC_EXPORT_H */

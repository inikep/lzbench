/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file zxc_export.h
 * @brief Platform-specific symbol visibility macros.
 *
 * `ZXC_EXPORT`, `ZXC_NO_EXPORT` and `ZXC_DEPRECATED` decide which symbols
 * leave the shared library.
 *
 * - Define @c ZXC_STATIC_DEFINE when building or consuming ZXC as a **static**
 *   library: it drops the import/export annotations entirely.
 * - Building the shared library, the CMake target defines @c zxc_lib_EXPORTS
 *   for you, which selects `dllexport` / `visibility("default")`.
 * - Consuming it, neither macro is defined, so the header picks
 *   `visibility("default")` on ELF and **no annotation** on Windows.
 *
 * On Windows `dllimport` is opt-in via @c ZXC_DLL_IMPORT, because an
 * unannotated declaration still links against a DLL (through a call thunk),
 * while `dllimport` against a static library fails on unresolved
 * `__imp_zxc_*`. CMake, Meson and pkg-config set the right macro for you.
 */

#ifndef ZXC_EXPORT_H
#define ZXC_EXPORT_H

/**
 * @defgroup export Symbol Visibility
 * @brief Macros controlling DLL export/import and deprecation attributes.
 * @{
 */

#ifdef ZXC_STATIC_DEFINE

/**
 * @def ZXC_EXPORT
 * @brief Marks a symbol as part of the public shared-library API.
 *
 * Nothing for a static library (@c ZXC_STATIC_DEFINE);
 * `__declspec(dllexport)` when building the Windows DLL;
 * `__declspec(dllimport)` when consuming it with @c ZXC_DLL_IMPORT;
 * `__attribute__((visibility("default")))` on GCC/Clang.
 */
#define ZXC_EXPORT

/**
 * @def ZXC_NO_EXPORT
 * @brief Marks a symbol as hidden (not exported from the shared library).
 *
 * Nothing for static builds or Windows;
 * `__attribute__((visibility("hidden")))` on GCC/Clang.
 */
#define ZXC_NO_EXPORT

#else /* shared library */

#ifndef ZXC_EXPORT
#ifdef zxc_lib_EXPORTS
/* Building the library */
#ifdef _WIN32
#define ZXC_EXPORT __declspec(dllexport)
#else
#define ZXC_EXPORT __attribute__((visibility("default")))
#endif
#else
/* Consuming the library */
#if defined(_WIN32) && defined(ZXC_DLL_IMPORT)
#define ZXC_EXPORT __declspec(dllimport)
#elif defined(_WIN32)
/* Static library, vendored sources, or a DLL consumed without ZXC_DLL_IMPORT:
   no annotation links in every case. */
#define ZXC_EXPORT
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

#endif /* ZXC_STATIC_DEFINE */

#ifndef ZXC_DEPRECATED
/**
 * @def ZXC_DEPRECATED
 * @brief Marks a symbol as deprecated, so referencing it warns.
 *
 * `__declspec(deprecated)` on MSVC, `__attribute__((__deprecated__))` on
 * GCC/Clang.
 */
#ifdef _WIN32
#define ZXC_DEPRECATED __declspec(deprecated)
#else
#define ZXC_DEPRECATED __attribute__((__deprecated__))
#endif
#endif

/**
 * @def ZXC_DEPRECATED_EXPORT
 * @brief Combines `ZXC_EXPORT` with `ZXC_DEPRECATED`.
 */
#ifndef ZXC_DEPRECATED_EXPORT
#define ZXC_DEPRECATED_EXPORT ZXC_EXPORT ZXC_DEPRECATED
#endif

/**
 * @def ZXC_DEPRECATED_NO_EXPORT
 * @brief Combines `ZXC_NO_EXPORT` with `ZXC_DEPRECATED`.
 */
#ifndef ZXC_DEPRECATED_NO_EXPORT
#define ZXC_DEPRECATED_NO_EXPORT ZXC_NO_EXPORT ZXC_DEPRECATED
#endif

/** @} */ /* end of export */

#endif /* ZXC_EXPORT_H */

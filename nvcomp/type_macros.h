/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef NVCOMP_TYPEMACROS_H
#define NVCOMP_TYPEMACROS_H

#include <stdexcept>
#include <string>

/******************************************************************************
 * DEFINES ********************************************************************
 *****************************************************************************/

#define NVCOMP_TYPE_TWO_SWITCH_FIRST_ONLY(type_var, second_type, func, ...)    \
  do {                                                                         \
    switch (type_var) {                                                        \
    case NVCOMP_TYPE_CHAR:                                                     \
      func<int8_t, second_type>(__VA_ARGS__);                                  \
      break;                                                                   \
    case NVCOMP_TYPE_UCHAR:                                                    \
      func<uint8_t, second_type>(__VA_ARGS__);                                 \
      break;                                                                   \
    case NVCOMP_TYPE_SHORT:                                                    \
      func<int16_t, second_type>(__VA_ARGS__);                                 \
      break;                                                                   \
    case NVCOMP_TYPE_USHORT:                                                   \
      func<uint16_t, second_type>(__VA_ARGS__);                                \
      break;                                                                   \
    case NVCOMP_TYPE_INT:                                                      \
      func<int32_t, second_type>(__VA_ARGS__);                                 \
      break;                                                                   \
    case NVCOMP_TYPE_UINT:                                                     \
      func<uint32_t, second_type>(__VA_ARGS__);                                \
      break;                                                                   \
    case NVCOMP_TYPE_LONGLONG:                                                 \
      func<int64_t, second_type>(__VA_ARGS__);                                 \
      break;                                                                   \
    case NVCOMP_TYPE_ULONGLONG:                                                \
      func<uint64_t, second_type>(__VA_ARGS__);                                \
      break;                                                                   \
    default:                                                                   \
      throw std::runtime_error("Unknown type: " + std::to_string(type_var));   \
    }                                                                          \
  } while (0)

#define NVCOMP_TYPE_TWO_SWITCH(type1_var, type2_var, func, ...)                \
  do {                                                                         \
    switch (type2_var) {                                                       \
    case NVCOMP_TYPE_CHAR:                                                     \
      NVCOMP_TYPE_TWO_SWITCH_FIRST_ONLY(type1_var, int8_t, func, __VA_ARGS__); \
      break;                                                                   \
    case NVCOMP_TYPE_UCHAR:                                                    \
      NVCOMP_TYPE_TWO_SWITCH_FIRST_ONLY(                                       \
          type1_var, uint8_t, func, __VA_ARGS__);                              \
      break;                                                                   \
    case NVCOMP_TYPE_SHORT:                                                    \
      NVCOMP_TYPE_TWO_SWITCH_FIRST_ONLY(                                       \
          type1_var, int16_t, func, __VA_ARGS__);                              \
      break;                                                                   \
    case NVCOMP_TYPE_USHORT:                                                   \
      NVCOMP_TYPE_TWO_SWITCH_FIRST_ONLY(                                       \
          type1_var, uint16_t, func, __VA_ARGS__);                             \
      break;                                                                   \
    case NVCOMP_TYPE_INT:                                                      \
      NVCOMP_TYPE_TWO_SWITCH_FIRST_ONLY(                                       \
          type1_var, int32_t, func, __VA_ARGS__);                              \
      break;                                                                   \
    case NVCOMP_TYPE_UINT:                                                     \
      NVCOMP_TYPE_TWO_SWITCH_FIRST_ONLY(                                       \
          type1_var, uint32_t, func, __VA_ARGS__);                             \
      break;                                                                   \
    case NVCOMP_TYPE_LONGLONG:                                                 \
      NVCOMP_TYPE_TWO_SWITCH_FIRST_ONLY(                                       \
          type1_var, int64_t, func, __VA_ARGS__);                              \
      break;                                                                   \
    case NVCOMP_TYPE_ULONGLONG:                                                \
      NVCOMP_TYPE_TWO_SWITCH_FIRST_ONLY(                                       \
          type1_var, uint64_t, func, __VA_ARGS__);                             \
      break;                                                                   \
    default:                                                                   \
      throw std::runtime_error("Unknown type: " + std::to_string(type2_var));  \
    }                                                                          \
  } while (0)

#define NVCOMP_TYPE_TWO_SWITCH_FIRST_ONLY_RETURN(                              \
    type_var, second_type, func, ...)                                          \
  do {                                                                         \
    switch (type_var) {                                                        \
    case NVCOMP_TYPE_CHAR:                                                     \
      return func<int8_t, second_type>(__VA_ARGS__);                           \
    case NVCOMP_TYPE_UCHAR:                                                    \
      return func<uint8_t, second_type>(__VA_ARGS__);                          \
    case NVCOMP_TYPE_SHORT:                                                    \
      return func<int16_t, second_type>(__VA_ARGS__);                          \
    case NVCOMP_TYPE_USHORT:                                                   \
      return func<uint16_t, second_type>(__VA_ARGS__);                         \
    case NVCOMP_TYPE_INT:                                                      \
      return func<int32_t, second_type>(__VA_ARGS__);                          \
    case NVCOMP_TYPE_UINT:                                                     \
      return func<uint32_t, second_type>(__VA_ARGS__);                         \
    case NVCOMP_TYPE_LONGLONG:                                                 \
      return func<int64_t, second_type>(__VA_ARGS__);                          \
    case NVCOMP_TYPE_ULONGLONG:                                                \
      return func<uint64_t, second_type>(__VA_ARGS__);                         \
    default:                                                                   \
      throw std::runtime_error("Unknown type: " + std::to_string(type_var));   \
    }                                                                          \
  } while (0)

#define NVCOMP_TYPE_TWO_SWITCH_RETURN(type1_var, type2_var, func, ...)         \
  do {                                                                         \
    switch (type2_var) {                                                       \
    case NVCOMP_TYPE_CHAR:                                                     \
      NVCOMP_TYPE_TWO_SWITCH_FIRST_ONLY_RETURN(                                \
          type1_var, int8_t, func, __VA_ARGS__);                               \
    case NVCOMP_TYPE_UCHAR:                                                    \
      NVCOMP_TYPE_TWO_SWITCH_FIRST_ONLY_RETURN(                                \
          type1_var, uint8_t, func, __VA_ARGS__);                              \
    case NVCOMP_TYPE_SHORT:                                                    \
      NVCOMP_TYPE_TWO_SWITCH_FIRST_ONLY_RETURN(                                \
          type1_var, int16_t, func, __VA_ARGS__);                              \
    case NVCOMP_TYPE_USHORT:                                                   \
      NVCOMP_TYPE_TWO_SWITCH_FIRST_ONLY_RETURN(                                \
          type1_var, uint16_t, func, __VA_ARGS__);                             \
    case NVCOMP_TYPE_INT:                                                      \
      NVCOMP_TYPE_TWO_SWITCH_FIRST_ONLY_RETURN(                                \
          type1_var, int32_t, func, __VA_ARGS__);                              \
    case NVCOMP_TYPE_UINT:                                                     \
      NVCOMP_TYPE_TWO_SWITCH_FIRST_ONLY_RETURN(                                \
          type1_var, uint32_t, func, __VA_ARGS__);                             \
    case NVCOMP_TYPE_LONGLONG:                                                 \
      NVCOMP_TYPE_TWO_SWITCH_FIRST_ONLY_RETURN(                                \
          type1_var, int64_t, func, __VA_ARGS__);                              \
    case NVCOMP_TYPE_ULONGLONG:                                                \
      NVCOMP_TYPE_TWO_SWITCH_FIRST_ONLY_RETURN(                                \
          type1_var, uint64_t, func, __VA_ARGS__);                             \
    default:                                                                   \
      throw std::runtime_error("Unknown type: " + std::to_string(type2_var));  \
    }                                                                          \
  } while (0)

#define NVCOMP_TYPE_ONE_SWITCH(type_var, func, ...)                            \
  do {                                                                         \
    switch (type_var) {                                                        \
    case NVCOMP_TYPE_CHAR:                                                     \
      func<int8_t>(__VA_ARGS__);                                               \
      break;                                                                   \
    case NVCOMP_TYPE_UCHAR:                                                    \
      func<uint8_t>(__VA_ARGS__);                                              \
      break;                                                                   \
    case NVCOMP_TYPE_SHORT:                                                    \
      func<int16_t>(__VA_ARGS__);                                              \
      break;                                                                   \
    case NVCOMP_TYPE_USHORT:                                                   \
      func<uint16_t>(__VA_ARGS__);                                             \
      break;                                                                   \
    case NVCOMP_TYPE_INT:                                                      \
      func<int32_t>(__VA_ARGS__);                                              \
      break;                                                                   \
    case NVCOMP_TYPE_UINT:                                                     \
      func<uint32_t>(__VA_ARGS__);                                             \
      break;                                                                   \
    case NVCOMP_TYPE_LONGLONG:                                                 \
      func<int64_t>(__VA_ARGS__);                                              \
      break;                                                                   \
    case NVCOMP_TYPE_ULONGLONG:                                                \
      func<uint64_t>(__VA_ARGS__);                                             \
      break;                                                                   \
    default:                                                                   \
      throw std::runtime_error("Unknown type: " + std::to_string(type_var));   \
    }                                                                          \
  } while (0)

#define NVCOMP_TYPE_ONE_SWITCH_RETURN(type_var, func, ...)                     \
  do {                                                                         \
    switch (type_var) {                                                        \
    case NVCOMP_TYPE_CHAR:                                                     \
      return func<int8_t>(__VA_ARGS__);                                        \
    case NVCOMP_TYPE_UCHAR:                                                    \
      return func<uint8_t>(__VA_ARGS__);                                       \
    case NVCOMP_TYPE_SHORT:                                                    \
      return func<int16_t>(__VA_ARGS__);                                       \
    case NVCOMP_TYPE_USHORT:                                                   \
      return func<uint16_t>(__VA_ARGS__);                                      \
    case NVCOMP_TYPE_INT:                                                      \
      return func<int32_t>(__VA_ARGS__);                                       \
    case NVCOMP_TYPE_UINT:                                                     \
      return func<uint32_t>(__VA_ARGS__);                                      \
    case NVCOMP_TYPE_LONGLONG:                                                 \
      return func<int64_t>(__VA_ARGS__);                                       \
    case NVCOMP_TYPE_ULONGLONG:                                                \
      return func<uint64_t>(__VA_ARGS__);                                      \
    default:                                                                   \
      throw std::runtime_error("Unknown type: " + std::to_string(type_var));   \
    }                                                                          \
  } while (0)

#endif

/***********************************************************************

Copyright 2014-2025 Kennon Conrad

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

***********************************************************************/

// GLZAformat.c
//   Adds 0xFF after all 0xFE and 0xFF bytes to support compressions insert and define symbols.
//   Replaces 'A' - 'Z' with 'C' followed by the corresponding lower case letter or
//     'B' followed by a series of lower case letters when text detected.
//   For non-text files, checks order 1 entropy of standard coding vs. delta coding for strides
//   1 - 100 (global).  Delta transforms data when appropriate.


#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "GLZA.h"

void clear_counts(uint32_t symbol_counts[0x100], uint32_t order_1_counts[0x100][0x100]) {
  uint8_t i = 0xFF;
  do {
    symbol_counts[i] = 0;
    uint8_t j = 0xFF;
    do {
      order_1_counts[i][j] = 0;
    } while (j-- != 0);
  } while (i-- != 0);
  return;
}


double calculate_order_1_entropy(uint32_t symbol_counts[0x100], uint32_t order_1_counts[0x100][0x100]) {
  uint16_t i, j;
  uint16_t num_symbols = 0;
  double entropy = 0.0;
  for (i = 0 ; i < 0x100 ; i++) {
    if (symbol_counts[i] != 0) {
      num_symbols++;
      entropy += (double)symbol_counts[i] * log2((double)symbol_counts[i]);
      for (j = 0 ; j < 0x100 ; j++) {
        if (order_1_counts[i][j]) {
          double d_count = (double)order_1_counts[i][j];
          entropy -= d_count * log2(d_count);
        }
      }
    }
  }
  entropy += (double)num_symbols * (log2((double)num_symbols) + 11.0);
  return(entropy);
}


uint8_t GLZAformat(int32_t insize, uint8_t * inbuf, int32_t * outsize_ptr, uint8_t ** outbuf, struct param_data * params) {
  const uint32_t CHARS_TO_WRITE = 0x40000;
  uint8_t this_char, prev_char, next_char, cap_encoded, cap_lock_disabled, delta_disabled, stride;
  uint8_t *inbuf2, *in_char_ptr, *end_char_ptr, *out_char_ptr;
  uint32_t i, j, k;
  uint32_t num_AZ, num_az_pre_AZ, num_az_post_AZ, num_spaces;
  uint32_t order_1_counts[0x100][0x100];
  uint32_t symbol_counts[0x100];
  double order_1_entropy, best_stride_entropy, saved_entropy[4];

  // format byte: B0: cap encoded, B3:B1 = stride (0 - 4), B5:B4 = log2 delta length (0 - 2), B6: little endian

  cap_encoded = 0;
  cap_lock_disabled = 0;
  delta_disabled = 0;
  if (params != 0) {
    cap_encoded = params->cap_encoded;
    cap_lock_disabled = params->cap_lock_disabled;
    delta_disabled = params->delta_disabled;
  }

  inbuf2 = (uint8_t *)malloc(insize);
  for (i = 0 ; i < insize ; i++)
    inbuf2[i] = inbuf[i];
  *outbuf = (uint8_t *)malloc(2 * insize + 1);
  if (*outbuf == 0)
    return(0);

  end_char_ptr = inbuf2 + insize;
  num_AZ = 0;
  num_az_pre_AZ = 0;
  num_az_post_AZ = 0;
  num_spaces = 0;

  if (insize > 4) {
    in_char_ptr = inbuf2;
    this_char = *in_char_ptr++;
    if (this_char == 0x20)
      num_spaces++;
    if ((this_char >= 'A') && (this_char <= 'Z')) {
      num_AZ++;
      next_char = *in_char_ptr & 0xDF;
      if ((next_char >= 'A') && (next_char <= 'Z'))
        num_az_post_AZ++;
    }

    while (in_char_ptr != end_char_ptr) {
      this_char = *in_char_ptr++;
      if (this_char == 0x20)
        num_spaces++;
      if ((this_char >= 'A') && (this_char <= 'Z')) {
        num_AZ++;
        prev_char = *(in_char_ptr - 2) & 0xDF;
        next_char = *in_char_ptr & 0xDF;
        if ((next_char >= 'A') && (next_char <= 'Z'))
          num_az_post_AZ++;
        if ((prev_char >= 'A') && (prev_char <= 'Z'))
          num_az_pre_AZ++;
      }
    }
  }

  out_char_ptr = *outbuf;


  if (((4 * num_az_post_AZ > num_AZ) && (num_az_post_AZ > num_az_pre_AZ) && (num_spaces > 1 + (insize / 50))
      && (cap_encoded != 2)) || (cap_encoded == 1)) {
#ifdef PRINTON
    fprintf(stderr, "Converting textual data\n");
#endif

    *out_char_ptr++ = 1;
    in_char_ptr = inbuf2;
    while (in_char_ptr != end_char_ptr) {
      if ((*in_char_ptr >= 'A') && (*in_char_ptr <= 'Z')) {
        if ((in_char_ptr + 1 < end_char_ptr)
            && ((*(in_char_ptr + 1) >= 'A') && (*(in_char_ptr + 1) <= 'Z') && (cap_lock_disabled == 0))
            && ((in_char_ptr + 1 == end_char_ptr) || (*(in_char_ptr + 2) < 'a') || (*(in_char_ptr + 2) > 'z'))) {
          *out_char_ptr++ = 'B';
          *out_char_ptr++ = *in_char_ptr++ + 0x20;
          *out_char_ptr++ = *in_char_ptr++ + 0x20;
          while ((in_char_ptr < end_char_ptr) && (*in_char_ptr >= 'A') && (*in_char_ptr <= 'Z'))
            *out_char_ptr++ = *in_char_ptr++ + 0x20;
          if ((in_char_ptr < end_char_ptr) && (*in_char_ptr >= 'a') && (*in_char_ptr <= 'z'))
            *out_char_ptr++ = 'C';
        }
        else {
          *out_char_ptr++ = 'C';
          *out_char_ptr++ = *in_char_ptr++ + 0x20;
        }
      }
      else if (*in_char_ptr == 0xA) {
        in_char_ptr++;
        *out_char_ptr++ = 0xA;
        *out_char_ptr++ = ' ';
      }
      else
        *out_char_ptr++ = *in_char_ptr++;
    }
  }
  else if ((delta_disabled != 1) && (insize > 4)) {
    clear_counts(symbol_counts, order_1_counts);
    for (i = 0 ; i < insize - 1 ; i++) {
      symbol_counts[inbuf2[i]]++;
      order_1_counts[inbuf2[i]][inbuf2[i + 1]]++;
    }
    symbol_counts[inbuf2[insize - 1]]++;
    order_1_counts[inbuf2[insize - 1]][0x80]++;
    order_1_entropy = calculate_order_1_entropy(symbol_counts, order_1_counts);
    best_stride_entropy = order_1_entropy;
    stride = 0;

    j = insize < 101 ? insize - 1 : 100;

    for (k = 1 ; k <= j ; k++) {
      clear_counts(symbol_counts, order_1_counts);
      if ((k == 2) | (k == 4)) {
        i = 0;
        while (i < k) {
          symbol_counts[inbuf2[i]]++;
          order_1_counts[inbuf2[i]][0xFF & (inbuf2[i + k] - inbuf2[i])]++;
          i++;
        }
        while (i < (uint32_t)insize - k) {
          symbol_counts[0xFF & (inbuf2[i] - inbuf2[i - k])]++;
          order_1_counts[0xFF & (inbuf2[i] - inbuf2[i - k])][0xFF & (inbuf2[i + k] - inbuf2[i])]++;
          i++;
        }
        while (i < insize) {
          symbol_counts[0xFF & (inbuf2[i] - inbuf2[i - k])]++;
          order_1_counts[0xFF & (inbuf2[i] - inbuf2[i - k])][0x80]++;
          i++;
        }
        order_1_entropy = calculate_order_1_entropy(symbol_counts, order_1_counts);
        if ((order_1_entropy < 0.95 * best_stride_entropy)
            || ((stride != 0) && (order_1_entropy < best_stride_entropy))) {
          stride = k;
          best_stride_entropy = order_1_entropy;
        }
      }
      else {
        for (i = 0 ; i < k - 1 ; i++) {
          symbol_counts[inbuf2[i]]++;
          order_1_counts[inbuf2[i]][inbuf2[i + 1]]++;
        }
        symbol_counts[inbuf2[k - 1]]++;
        order_1_counts[inbuf2[k - 1]][0xFF & (inbuf2[k] - inbuf2[0])]++;
        uint8_t failed_test = 0;
        i = k;
        if (insize > 100000) {
          uint32_t initial_test_size = 100000 + ((insize - 100000) >> 3);
          if (initial_test_size > insize)
            initial_test_size = insize - 1;
          while (i < initial_test_size) {
            symbol_counts[0xFF & (inbuf2[i] - inbuf2[i - k])]++;
            order_1_counts[0xFF & (inbuf2[i] - inbuf2[i - k])][0xFF & (inbuf2[i + 1] - inbuf2[i + 1 - k])]++;
            i++;
          }
          order_1_entropy = calculate_order_1_entropy(symbol_counts, order_1_counts);
          if (order_1_entropy >= 1.05 * best_stride_entropy * (double)initial_test_size / (double)insize)
            failed_test = 1;
        }
        if (failed_test == 0) {
          while (i < insize - 1) {
            symbol_counts[0xFF & (inbuf2[i] - inbuf2[i - k])]++;
            order_1_counts[0xFF & (inbuf2[i] - inbuf2[i - k])][0xFF & (inbuf2[i + 1] - inbuf2[i + 1 - k])]++;
            i++;
          }
          symbol_counts[0xFF & (inbuf2[insize - 1] - inbuf2[insize - 1 - k])]++;
          order_1_counts[0xFF & (inbuf2[insize - 1] - inbuf2[insize - 1- k])][0x80]++;
          order_1_entropy = calculate_order_1_entropy(symbol_counts, order_1_counts);
          if ((order_1_entropy < 0.9 * best_stride_entropy)
              || ((stride != 0) && (order_1_entropy < best_stride_entropy))) {
            stride = k;
            best_stride_entropy = order_1_entropy;
          }
        }
      }
    }

    double min_entropy = best_stride_entropy;

#ifdef PRINTON
    if (stride != 0)
      fprintf(stderr, "Applying %u byte delta transformation\n", (unsigned int)stride);
    else
      fprintf(stderr, "Converting data\n");
#endif

    if (stride == 0)
      *out_char_ptr++ = 0;
    else if (stride == 1) {
      *out_char_ptr++ = 2;
      in_char_ptr = end_char_ptr - 1;
      while (--in_char_ptr >= inbuf2)
        *(in_char_ptr + 1) -= *in_char_ptr;
    }
    else if (stride == 2) {
      for (j = 0 ; j < 2 ; j++) {
        clear_counts(symbol_counts, order_1_counts);
        uint8_t delta_symbol;
        uint8_t prior_delta_symbol = inbuf2[j];
        for (i = j ; i < (insize & ~1) - 2 ; i += 2) {
          delta_symbol = inbuf2[i + 2] - inbuf2[i];
          symbol_counts[prior_delta_symbol]++;
          order_1_counts[prior_delta_symbol][delta_symbol]++;
          prior_delta_symbol = delta_symbol;
        }
        symbol_counts[prior_delta_symbol]++;
        order_1_counts[prior_delta_symbol][0]++;
        saved_entropy[j] = calculate_order_1_entropy(symbol_counts, order_1_counts);
      }

      clear_counts(symbol_counts, order_1_counts);
      if (saved_entropy[0] < saved_entropy[1]) {
        // big endian
        uint16_t symbol, delta_symbol;
        uint16_t prior_symbol = (inbuf2[0] << 8) + inbuf2[1];
        uint16_t prior_delta_symbol = prior_symbol;
        for (i = 0 ; i < insize - 3 ; i += 2) {
          symbol = (inbuf2[i + 2] << 8) + inbuf2[i + 3];
          delta_symbol = symbol - prior_symbol + 0x8080;
          symbol_counts[prior_delta_symbol >> 8]++;
          order_1_counts[prior_delta_symbol >> 8][delta_symbol >> 8]++;
          symbol_counts[0xFF & prior_delta_symbol]++;
          order_1_counts[0xFF & prior_delta_symbol][0xFF & delta_symbol]++;
          prior_symbol = symbol;
          prior_delta_symbol = delta_symbol;
        }
        if (i == insize - 3) {
          symbol = (inbuf2[i + 2] << 8);
          delta_symbol = (inbuf2[i + 2] << 8) - prior_symbol + 0x8080;
          symbol_counts[delta_symbol >> 8]++;
          order_1_counts[delta_symbol >> 8][0]++;
        }
        else
          delta_symbol = 0;
        symbol_counts[prior_delta_symbol >> 8]++;
        order_1_counts[prior_delta_symbol >> 8][delta_symbol >> 8]++;
        symbol_counts[0xFF & prior_delta_symbol]++;
        order_1_counts[0xFF & prior_delta_symbol][0]++;
        order_1_entropy = calculate_order_1_entropy(symbol_counts, order_1_counts);
        if (order_1_entropy < best_stride_entropy) {
#ifdef PRINTON
          fprintf(stderr, "Big endian\n");
#endif
          *out_char_ptr++ = 0x14;
          in_char_ptr = inbuf2 + ((end_char_ptr - inbuf2 - 4) & ~1);
          uint16_t value = (*(in_char_ptr + 2) << 8) + *(in_char_ptr + 3);
          while (in_char_ptr >= inbuf2) {
            uint16_t prior_value = (*in_char_ptr << 8) + *(in_char_ptr + 1);
            uint16_t delta_value = value - prior_value + 0x80;
            *(in_char_ptr + 2) = delta_value >> 8;
            *(in_char_ptr + 3) = delta_value & 0xFF;
            value = prior_value;
            in_char_ptr -= 2;
          }
        }
        else {
#ifdef PRINTON
          fprintf(stderr, "No carry\n");
#endif
          *out_char_ptr++ = 4;
          in_char_ptr = end_char_ptr - 2;
          while (--in_char_ptr >= inbuf2)
            *(in_char_ptr + 2) -= *in_char_ptr;
        }
      }
      else {
        uint16_t symbol, delta_symbol;
        uint16_t prior_symbol = (inbuf2[1] << 8) + inbuf2[0];
        uint16_t prior_delta_symbol = prior_symbol;
        for (i = 0 ; i < insize - 3 ; i += 2) {
          symbol = (inbuf2[i + 3] << 8) + inbuf2[i + 2];
          delta_symbol = symbol - prior_symbol + 0x8080;
          symbol_counts[0xFF & prior_delta_symbol]++;
          order_1_counts[0xFF & prior_delta_symbol][0xFF & delta_symbol]++;
          symbol_counts[prior_delta_symbol >> 8]++;
          order_1_counts[prior_delta_symbol >> 8][delta_symbol >> 8]++;
          prior_symbol = symbol;
          prior_delta_symbol = delta_symbol;
        }
        if (i == insize - 3) {
          delta_symbol = inbuf2[i + 2] - prior_symbol + 0x8080;
          symbol_counts[0xFF & delta_symbol]++;
          order_1_counts[0xFF & delta_symbol][0]++;
        }
        else
          delta_symbol = 0;
        symbol_counts[0xFF & prior_delta_symbol]++;
        order_1_counts[0xFF & prior_delta_symbol][0xFF & delta_symbol]++;
        symbol_counts[prior_delta_symbol >> 8]++;
        order_1_counts[prior_delta_symbol >> 8][0]++;
        order_1_entropy = calculate_order_1_entropy(symbol_counts, order_1_counts);
        if (order_1_entropy < best_stride_entropy) {
#ifdef PRINTON
          fprintf(stderr, "Little endian\n");
#endif
          *out_char_ptr++ = 0x34;
          in_char_ptr = inbuf2 + ((end_char_ptr - inbuf2 - 4) & ~1);
          uint16_t value = (*(in_char_ptr + 3) << 8) + *(in_char_ptr + 2);
          while (in_char_ptr >= inbuf2) {
            uint16_t prior_value = (*(in_char_ptr + 1) << 8) + *in_char_ptr;
            uint16_t delta_value = value - prior_value + 0x80;
            *(in_char_ptr + 2) = delta_value & 0xFF;
            *(in_char_ptr + 3) = (delta_value >> 8);
            value = prior_value;
            in_char_ptr -= 2;
          }
        }
        else {
#ifdef PRINTON
          fprintf(stderr, "No carry\n");
#endif
          *out_char_ptr++ = 4;
          in_char_ptr = end_char_ptr - 2;
          while (--in_char_ptr >= inbuf2)
            *(in_char_ptr + 2) -= *in_char_ptr;
        }
      }
    }
    else if (stride == 4) {
      for (k = 0 ; k < 4 ; k++) {
        clear_counts(symbol_counts, order_1_counts);
        symbol_counts[inbuf2[k]]++;
        order_1_counts[inbuf2[k]][0xFF & (inbuf2[k + stride] - inbuf2[k])]++;
        i = k + stride;
        while (i < insize - stride) {
          symbol_counts[0xFF & (inbuf2[i] - inbuf2[i - stride])]++;
          order_1_counts[0xFF & (inbuf2[i] - inbuf2[i - stride])][0xFF & (inbuf2[i + stride] - inbuf2[i])]++;
          i += stride;
        }
        symbol_counts[0xFF & (inbuf2[i] - inbuf2[i - stride])]++;
        order_1_counts[0xFF & (inbuf2[i] - inbuf2[i - stride])][0]++;
        saved_entropy[k] = calculate_order_1_entropy(symbol_counts, order_1_counts);
      }
      double best_entropy[4];
      uint8_t best_entropy_position[4];
      for (i = 0 ; i < 4 ; i++) {
        best_entropy[i] = saved_entropy[i];
        best_entropy_position[i] = i;
        int8_t j;
        for (j = i - 1 ; j >= 0 ; j--) {
          if (saved_entropy[i] < best_entropy[j]) {
            best_entropy[j + 1] = best_entropy[j];
            best_entropy_position[j + 1] = best_entropy_position[j];
            best_entropy[j] = saved_entropy[i];
            best_entropy_position[j] = i;
          }
        }
      }

      if (best_entropy[3] > 1.05 * best_entropy[0]) {
        if ((3.0 * best_entropy[1] < best_entropy[0] + best_entropy[2] + best_entropy[3])
            && (((best_entropy_position[0] - best_entropy_position[1]) & 3) == 2)) {
          clear_counts(symbol_counts, order_1_counts);
          if (best_entropy[0] + best_entropy[2] < best_entropy[1] + best_entropy[3]) {
            // big endian
            uint16_t symbol1, symbol2, delta_symbol1, delta_symbol2;
            uint16_t prior_symbol1 = (inbuf2[0] << 8) + inbuf2[1];
            uint16_t prior_symbol2 = (inbuf2[2] << 8) + inbuf2[3];
            uint16_t prior_delta_symbol1 = prior_symbol1;
            uint16_t prior_delta_symbol2 = prior_symbol2;
            for (i = 0 ; i < insize - 7 ; i += 4) {
              symbol1 = (inbuf2[i + 4] << 8) + inbuf2[i + 5];
              symbol2 = (inbuf2[i + 6] << 8) + inbuf2[i + 7];
              delta_symbol1 = symbol1 - prior_symbol1 + 0x8080;
              delta_symbol2 = symbol2 - prior_symbol2 + 0x8080;
              symbol_counts[prior_delta_symbol1 >> 8]++;
              order_1_counts[prior_delta_symbol1 >> 8][delta_symbol1 >> 8]++;
              symbol_counts[0xFF & prior_delta_symbol1]++;
              order_1_counts[0xFF & prior_delta_symbol1][0xFF & delta_symbol1]++;
              symbol_counts[prior_delta_symbol2 >> 8]++;
              order_1_counts[prior_delta_symbol2 >> 8][delta_symbol2 >> 8]++;
              symbol_counts[0xFF & prior_delta_symbol2]++;
              order_1_counts[0xFF & prior_delta_symbol2][0xFF & delta_symbol2]++;
              prior_symbol1 = symbol1;
              prior_symbol2 = symbol2;
              prior_delta_symbol1 = delta_symbol1;
              prior_delta_symbol2 = delta_symbol2;
            }
            if (i == insize - 7) {
              delta_symbol1 = (inbuf2[i + 4] << 8) + inbuf2[i + 5] - prior_symbol1 + 0x8080;
              delta_symbol2 = (inbuf2[i + 6] << 8) - prior_symbol2 + 0x8080;
              symbol_counts[delta_symbol1 >> 8]++;
              order_1_counts[delta_symbol1 >> 8][0x80]++;
              symbol_counts[0xFF & delta_symbol1]++;
              order_1_counts[0xFF & delta_symbol1][0x80]++;
              symbol_counts[delta_symbol2 >> 8]++;
              order_1_counts[delta_symbol2 >> 8][0x80]++;
            }
            else if (i == insize - 6) {
              delta_symbol1 = (inbuf2[i + 4] << 8) + inbuf2[i + 5] - prior_symbol1 + 0x8080;
              delta_symbol2 = 0x8080;
              symbol_counts[delta_symbol1 >> 8]++;
              order_1_counts[delta_symbol1 >> 8][0x80]++;
              symbol_counts[0xFF & delta_symbol1]++;
              order_1_counts[0xFF & delta_symbol1][0x80]++;
            }
            else if (i == insize - 5) {
              delta_symbol1 = (inbuf2[i + 4] << 8) - prior_symbol1 + 0x8080;
              delta_symbol2 = 0x8080;
              symbol_counts[delta_symbol1 >> 8]++;
              order_1_counts[delta_symbol1 >> 8][0x80]++;
            }
            else {
              delta_symbol1 = 0x8080;
              delta_symbol2 = 0x8080;
            }
            symbol_counts[prior_delta_symbol1 >> 8]++;
            order_1_counts[prior_delta_symbol1 >> 8][delta_symbol1 >> 8]++;
            symbol_counts[0xFF & prior_delta_symbol1]++;
            order_1_counts[0xFF & prior_delta_symbol1][0xFF & delta_symbol1]++;
            symbol_counts[prior_delta_symbol2 >> 8]++;
            order_1_counts[prior_delta_symbol2 >> 8][delta_symbol2 >> 8]++;
            symbol_counts[0xFF & prior_delta_symbol2]++;
            order_1_counts[0xFF & prior_delta_symbol2][0x80]++;
            order_1_entropy = calculate_order_1_entropy(symbol_counts, order_1_counts);
            if (order_1_entropy < min_entropy) {
#ifdef PRINTON
              fprintf(stderr, "Two channel big endian\n");
#endif
              *out_char_ptr++ = 0x58;
              in_char_ptr = inbuf2 + ((end_char_ptr - inbuf2 - 6) & ~1);
              while (in_char_ptr >= inbuf2) {
                uint16_t delta_value = (*(in_char_ptr + 4) << 8) + *(in_char_ptr + 5)
                    - ((*in_char_ptr << 8) + *(in_char_ptr + 1)) + 0x80;
                *(in_char_ptr + 4) = delta_value >> 8;
                *(in_char_ptr + 5) = delta_value & 0xFF;
                in_char_ptr -= 2;
              }
            }
            else {
#ifdef PRINTON
              fprintf(stderr, "No carry\n");
#endif
              *out_char_ptr++ = 8;
              in_char_ptr = end_char_ptr - 4;
              while (--in_char_ptr >= inbuf2)
                *(in_char_ptr + 4) -= *in_char_ptr;
            }
          }
          else {
            // little endian
            uint16_t symbol1, symbol2, delta_symbol1, delta_symbol2;
            uint16_t prior_symbol1 = (inbuf2[1] << 8) + inbuf2[0];
            uint16_t prior_symbol2 = (inbuf2[3] << 8) + inbuf2[2];
            uint16_t prior_delta_symbol1 = prior_symbol1;
            uint16_t prior_delta_symbol2 = prior_symbol2;
            for (i = 0 ; i < insize - 7 ; i += 4) {
              symbol1 = (inbuf2[i + 5] << 8) + inbuf2[i + 4];
              symbol2 = (inbuf2[i + 7] << 8) + inbuf2[i + 6];
              delta_symbol1 = symbol1 - prior_symbol1 + 0x8080;
              delta_symbol2 = symbol2 - prior_symbol2 + 0x8080;
              symbol_counts[0xFF & prior_delta_symbol1]++;
              order_1_counts[0xFF & prior_delta_symbol1][0xFF & delta_symbol1]++;
              symbol_counts[prior_delta_symbol1 >> 8]++;
              order_1_counts[prior_delta_symbol1 >> 8][delta_symbol1 >> 8]++;
              symbol_counts[0xFF & prior_delta_symbol2]++;
              order_1_counts[0xFF & prior_delta_symbol2][0xFF & delta_symbol2]++;
              symbol_counts[prior_delta_symbol2 >> 8]++;
              order_1_counts[prior_delta_symbol2 >> 8][delta_symbol2 >> 8]++;
              prior_symbol1 = symbol1;
              prior_symbol2 = symbol2;
              prior_delta_symbol1 = delta_symbol1;
              prior_delta_symbol2 = delta_symbol2;
            }
            if (i == insize - 7) {
              delta_symbol1 = (inbuf2[i + 5] << 8) + inbuf2[i + 4] - prior_symbol1 + 0x8080;
              delta_symbol2 = inbuf2[i + 6] - prior_symbol2 + 0x8080;
              symbol_counts[0xFF & delta_symbol1]++;
              order_1_counts[0xFF & delta_symbol1][0x80]++;
              symbol_counts[delta_symbol1 >> 8]++;
              order_1_counts[delta_symbol1 >> 8][0x80]++;
              symbol_counts[0xFF & delta_symbol2]++;
              order_1_counts[0xFF & delta_symbol2][0x80]++;
            }
            else if (i == insize - 6) {
              delta_symbol1 = (inbuf2[i + 4] << 8) + inbuf2[i + 5] - prior_symbol1 + 0x8080;
              delta_symbol2 = 0x8080;
              symbol_counts[0xFF & delta_symbol1]++;
              order_1_counts[0xFF & delta_symbol1][0x80]++;
              symbol_counts[delta_symbol1 >> 8]++;
              order_1_counts[delta_symbol1 >> 8][0x80]++;
            }
            else if (i == insize - 5) {
              delta_symbol1 = (inbuf2[i + 4] << 8) - prior_symbol1 + 0x8080;
              delta_symbol2 = 0x8080;
              symbol_counts[delta_symbol1 >> 8]++;
              order_1_counts[delta_symbol1 >> 8][0x80]++;
            }
            else {
              delta_symbol1 = 0x8080;
              delta_symbol2 = 0x8080;
            }
            symbol_counts[0xFF & prior_delta_symbol1]++;
            order_1_counts[0xFF & prior_delta_symbol1][0xFF & delta_symbol1]++;
            symbol_counts[prior_delta_symbol1 >> 8]++;
            order_1_counts[prior_delta_symbol1 >> 8][delta_symbol1 >> 8]++;
            symbol_counts[0xFF & prior_delta_symbol2]++;
            order_1_counts[0xFF & prior_delta_symbol2][0xFF & delta_symbol1]++;
            symbol_counts[prior_delta_symbol2 >> 8]++;
            order_1_counts[prior_delta_symbol2 >> 8][0x80]++;
            order_1_entropy = calculate_order_1_entropy(symbol_counts, order_1_counts);
            if (order_1_entropy < min_entropy) {
#ifdef PRINTON
              fprintf(stderr, "Two channel little endian\n");
#endif
              *out_char_ptr++ = 0x78;
              in_char_ptr = inbuf2 + ((end_char_ptr - inbuf2 - 6) & ~1);
              while (in_char_ptr >= inbuf2) {
                uint16_t delta_value = (*(in_char_ptr + 5) << 8) + *(in_char_ptr + 4)
                    - ((*(in_char_ptr + 1) << 8) + *in_char_ptr) + 0x80;
                *(in_char_ptr + 4) = delta_value & 0xFF;
                *(in_char_ptr + 5) = (delta_value >> 8) & 0xFF;
                in_char_ptr -= 2;
              }
            }
            else {
#ifdef PRINTON
              fprintf(stderr, "No carry\n");
#endif
              *out_char_ptr++ = 8;
              in_char_ptr = end_char_ptr - 4;
              while (--in_char_ptr >= inbuf2)
                *(in_char_ptr + 4) -= *in_char_ptr;
            }
          }
        }
        else {
          // try big endian first
          clear_counts(symbol_counts, order_1_counts);
          uint32_t symbol, delta_symbol;
          uint32_t prior_symbol = (inbuf2[0] << 24) + (inbuf2[1] << 16) + (inbuf2[2] << 8) + inbuf2[3];
          uint32_t prior_delta_symbol = prior_symbol;
          for (i = 0 ; i < insize - 7 ; i += 4) {
            symbol = (inbuf2[i + 4] << 24) + (inbuf2[i + 5] << 16) + (inbuf2[i + 6] << 8) + inbuf2[i + 7];
            delta_symbol = symbol - prior_symbol + 0x80808080;
            symbol_counts[prior_delta_symbol >> 24]++;
            order_1_counts[prior_delta_symbol >> 24][delta_symbol >> 24]++;
            symbol_counts[0xFF & (prior_delta_symbol >> 16)]++;
            order_1_counts[0xFF & (prior_delta_symbol >> 16)][0xFF & (delta_symbol >> 16)]++;
            symbol_counts[0xFF & (prior_delta_symbol >> 8)]++;
            order_1_counts[0xFF & (prior_delta_symbol >> 8)][0xFF & (delta_symbol >> 8)]++;
            symbol_counts[0xFF & prior_delta_symbol]++;
            order_1_counts[0xFF & prior_delta_symbol][0xFF & delta_symbol]++;
            prior_symbol = symbol;
            prior_delta_symbol = delta_symbol;
          }
          if (i == insize - 7) {
            delta_symbol = (inbuf2[i + 4] << 24) + (inbuf2[i + 5] << 16) + (inbuf2[i + 6] << 8) - prior_symbol + 0x80808080;
            symbol_counts[delta_symbol >> 24]++;
            order_1_counts[delta_symbol >> 24][0x80]++;
            symbol_counts[0xFF & (delta_symbol >> 16)]++;
            order_1_counts[0xFF & (delta_symbol >> 16)][0x80]++;
            symbol_counts[0xFF & (delta_symbol >> 8)]++;
            order_1_counts[0xFF & (delta_symbol >> 8)][0x80]++;
          }
          else if (i == insize - 6) {
            delta_symbol = (inbuf2[i + 4] << 24) + (inbuf2[i + 5] << 16) - prior_symbol + 0x80808080;
            symbol_counts[delta_symbol >> 24]++;
            order_1_counts[delta_symbol >> 24][0x80]++;
            symbol_counts[0xFF & (delta_symbol >> 16)]++;
            order_1_counts[0xFF & (delta_symbol >> 16)][0x80]++;
          }
          else if (i == insize - 5) {
            delta_symbol = (inbuf2[i + 4] << 24) - prior_symbol + 0x80808080;
            symbol_counts[delta_symbol >> 24]++;
            order_1_counts[delta_symbol >> 24][0x80]++;
          }
          else
            delta_symbol = 0x80808080;
          symbol_counts[prior_delta_symbol >> 24]++;
          order_1_counts[prior_delta_symbol >> 24][delta_symbol >> 24]++;
          symbol_counts[0xFF & (prior_delta_symbol >> 16)]++;
          order_1_counts[0xFF & (prior_delta_symbol >> 16)][0xFF & (delta_symbol >> 16)]++;
          symbol_counts[0xFF & (prior_delta_symbol >> 8)]++;
          order_1_counts[0xFF & (prior_delta_symbol >> 8)][0xFF & (delta_symbol >> 8)]++;
          symbol_counts[0xFF & prior_delta_symbol]++;
          order_1_counts[0xFF & prior_delta_symbol][0x80]++;
          saved_entropy[0] = calculate_order_1_entropy(symbol_counts, order_1_counts);

          clear_counts(symbol_counts, order_1_counts);
          prior_symbol = (inbuf2[3] << 24) + (inbuf2[2] << 16) + (inbuf2[1] << 8) + inbuf2[0];
          prior_delta_symbol = prior_symbol;
          for (i = 0 ; i < insize - 7 ; i += 4) {
            symbol = (inbuf2[i + 7] << 24) + (inbuf2[i + 6] << 16) + (inbuf2[i + 5] << 8) + inbuf2[i + 4];
            delta_symbol = symbol - prior_symbol + 0x80808080;
            symbol_counts[0xFF & prior_delta_symbol]++;
            order_1_counts[0xFF & prior_delta_symbol][0xFF & delta_symbol]++;
            symbol_counts[0xFF & (prior_delta_symbol >> 8)]++;
            order_1_counts[0xFF & (prior_delta_symbol >> 8)][0xFF & (delta_symbol >> 8)]++;
            symbol_counts[0xFF & (prior_delta_symbol >> 16)]++;
            order_1_counts[0xFF & (prior_delta_symbol >> 16)][0xFF & (delta_symbol >> 16)]++;
            symbol_counts[prior_delta_symbol >> 24]++;
            order_1_counts[prior_delta_symbol >> 24][delta_symbol >> 24]++;
            prior_symbol = symbol;
            prior_delta_symbol = delta_symbol;
          }
          if (i == insize - 7) {
            delta_symbol = (inbuf2[i + 6] << 16) + (inbuf2[i + 5] << 8) + inbuf2[i + 4] - prior_symbol + 0x80808080;
            symbol_counts[0xFF & delta_symbol]++;
            order_1_counts[0xFF & delta_symbol][0]++;
            symbol_counts[0xFF & (delta_symbol >> 8)]++;
            order_1_counts[0xFF & (delta_symbol >> 8)][0]++;
            symbol_counts[0xFF & (delta_symbol >> 16)]++;
            order_1_counts[0xFF & (delta_symbol >> 16)][0]++;
          }
          else if (i == insize - 6) {
            delta_symbol = (inbuf2[i + 5] << 8) + inbuf2[i + 4] - prior_symbol + 0x80808080;
            symbol_counts[0xFF & delta_symbol]++;
            order_1_counts[0xFF & delta_symbol][0]++;
            symbol_counts[0xFF & (delta_symbol >> 8)]++;
            order_1_counts[0xFF & (delta_symbol >> 8)][0]++;
          }
          else if (i == insize - 5) {
            delta_symbol = inbuf2[i + 4] - prior_symbol + 0x80808080;
            symbol_counts[0xFF & delta_symbol]++;
            order_1_counts[0xFF & delta_symbol][0]++;
          }
          else
            delta_symbol = 0x80808080;
          symbol_counts[0xFF & prior_delta_symbol]++;
          order_1_counts[0xFF & prior_delta_symbol][0xFF & delta_symbol]++;
          symbol_counts[0xFF & (prior_delta_symbol >> 8)]++;
          order_1_counts[0xFF & (prior_delta_symbol >> 8)][0xFF & (delta_symbol >> 8)]++;
          symbol_counts[0xFF & (prior_delta_symbol >> 16)]++;
          order_1_counts[0xFF & (prior_delta_symbol >> 16)][0xFF & (delta_symbol >> 16)]++;
          symbol_counts[prior_delta_symbol >> 24]++;
          order_1_counts[prior_delta_symbol >> 24][0]++;
          order_1_entropy = calculate_order_1_entropy(symbol_counts, order_1_counts);

          if ((saved_entropy[0] < min_entropy) && (saved_entropy[0] < order_1_entropy)) {
#ifdef PRINTON
            fprintf(stderr, "Big endian\n");
#endif
            *out_char_ptr++ = 0x18;
            in_char_ptr = inbuf2 + ((end_char_ptr - inbuf2 - 8) & ~3);
            uint32_t value = (*(in_char_ptr + 4) << 24) + (*(in_char_ptr + 5) << 16)
                + (*(in_char_ptr + 6) << 8) + *(in_char_ptr + 7);
            while (in_char_ptr >= inbuf2) {
              uint32_t prior_value = (*in_char_ptr << 24) + (*(in_char_ptr + 1) << 16)
                  + (*(in_char_ptr + 2) << 8) + *(in_char_ptr + 3);
              uint32_t delta_value = value - prior_value + 0x808080;
              *(in_char_ptr + 4) = delta_value >> 24;
              *(in_char_ptr + 5) = (delta_value >> 16) & 0xFF;
              *(in_char_ptr + 6) = (delta_value >> 8) & 0xFF;
              *(in_char_ptr + 7) = delta_value & 0xFF;
              value = prior_value;
              in_char_ptr -= 4;
            }
          }
          else if (order_1_entropy < min_entropy) {
#ifdef PRINTON
            fprintf(stderr, "Little endian\n");
#endif
            *out_char_ptr++ = 0x38;
            in_char_ptr = inbuf2 + ((end_char_ptr - inbuf2 - 8) & ~3);
            uint32_t value = (*(in_char_ptr + 7) << 24) + (*(in_char_ptr + 6) << 16)
                + (*(in_char_ptr + 5) << 8) + *(in_char_ptr + 4);
            while (in_char_ptr >= inbuf2) {
              uint32_t prior_value = (*(in_char_ptr + 3) << 24) + (*(in_char_ptr + 2) << 16)
                  + (*(in_char_ptr + 1) << 8) + *in_char_ptr;
              uint32_t delta_value = value - prior_value + 0x808080;
              *(in_char_ptr + 7) = delta_value >> 24;
              *(in_char_ptr + 6) = (delta_value >> 16) & 0xFF;
              *(in_char_ptr + 5) = (delta_value >> 8) & 0xFF;
              *(in_char_ptr + 4) = delta_value & 0xFF;
              value = prior_value;
              in_char_ptr -= 4;
            }
          }
          else {
#ifdef PRINTON
            fprintf(stderr, "No carry\n");
#endif
            *out_char_ptr++ = 8;
            in_char_ptr = end_char_ptr - 4;
            while (--in_char_ptr >= inbuf2)
              *(in_char_ptr + 4) -= *in_char_ptr;
          }
        }
      }
      else {
#ifdef PRINTON
        fprintf(stderr, "No carry\n");
#endif
        *out_char_ptr++ = 8;
        in_char_ptr = end_char_ptr - 4;
        while (--in_char_ptr >= inbuf2)
          *(in_char_ptr + 4) -= *in_char_ptr;
      }
    }
    else if (stride == 3) {
      *out_char_ptr++ = 6;
      in_char_ptr = end_char_ptr - 3;
      while (--in_char_ptr >= inbuf2)
        *(in_char_ptr + 3) -= *in_char_ptr;
    }
    else {
      *out_char_ptr++ = 0x80 + stride;
      in_char_ptr = end_char_ptr - stride;
      while (--in_char_ptr >= inbuf2)
        *(in_char_ptr + stride) -= *in_char_ptr;
      in_char_ptr = inbuf2 + stride - 1;
      while (--in_char_ptr >= inbuf2)
        *(in_char_ptr + 1) -= *in_char_ptr;
    }

    if ((stride == 2) || (stride == 4)) {
      uint8_t * in_char2 = (uint8_t *)malloc(CHARS_TO_WRITE);
      uint8_t * in_char2_ptr;
      uint8_t * start_block_ptr = inbuf2;
      uint8_t * end_block_ptr = start_block_ptr + CHARS_TO_WRITE;
      if (stride == 2) {
        while (end_block_ptr < end_char_ptr) {
          in_char2_ptr = in_char2;
          in_char_ptr = start_block_ptr + 1;
          while (in_char_ptr < end_block_ptr) {
            *in_char2_ptr++ = *in_char_ptr;
            in_char_ptr += 2;
          }
          in_char2_ptr = start_block_ptr;
          in_char_ptr = start_block_ptr;
          while (in_char_ptr < end_block_ptr) {
            *in_char2_ptr++ = *in_char_ptr;
            in_char_ptr += 2;
          }
          in_char_ptr = in_char2;
          while (in_char2_ptr < end_block_ptr)
            *in_char2_ptr++ = *in_char_ptr++;
          start_block_ptr = end_block_ptr;
          end_block_ptr += CHARS_TO_WRITE;
        }
        in_char2_ptr = in_char2;
        in_char_ptr = start_block_ptr + 1;
        while (in_char_ptr < end_char_ptr) {
          *in_char2_ptr++ = *in_char_ptr;
          in_char_ptr += 2;
        }
        in_char2_ptr = start_block_ptr;
        in_char_ptr = start_block_ptr;
        while (in_char_ptr < end_char_ptr) {
          *in_char2_ptr++ = *in_char_ptr;
          in_char_ptr += 2;
        }
        in_char_ptr = in_char2;
        while (in_char2_ptr < end_char_ptr)
          *in_char2_ptr++ = *in_char_ptr++;
      }
      else {
        while (end_block_ptr < end_char_ptr) {
          in_char2_ptr = in_char2;
          in_char_ptr = start_block_ptr + 1;
          while (in_char_ptr < end_block_ptr) {
            *in_char2_ptr++ = *in_char_ptr;
            in_char_ptr += 4;
          }
          in_char_ptr = start_block_ptr + 2;
          while (in_char_ptr < end_block_ptr) {
            *in_char2_ptr++ = *in_char_ptr;
            in_char_ptr += 4;
          }
          in_char_ptr = start_block_ptr + 3;
          while (in_char_ptr < end_block_ptr) {
            *in_char2_ptr++ = *in_char_ptr;
            in_char_ptr += 4;
          }
          in_char2_ptr = start_block_ptr;
          in_char_ptr = start_block_ptr;
          while (in_char_ptr < end_block_ptr) {
            *in_char2_ptr++ = *in_char_ptr;
            in_char_ptr += 4;
          }
          in_char_ptr = in_char2;
          while (in_char2_ptr < end_block_ptr)
            *in_char2_ptr++ = *in_char_ptr++;
          start_block_ptr = end_block_ptr;
          end_block_ptr += CHARS_TO_WRITE;
        }
        in_char2_ptr = in_char2;
        in_char_ptr = start_block_ptr + 1;
        while (in_char_ptr < end_char_ptr) {
          *in_char2_ptr++ = *in_char_ptr;
          in_char_ptr += 4;
        }
        in_char_ptr = start_block_ptr + 2;
        while (in_char_ptr < end_char_ptr) {
          *in_char2_ptr++ = *in_char_ptr;
          in_char_ptr += 4;
        }
        in_char_ptr = start_block_ptr + 3;
        while (in_char_ptr < end_char_ptr) {
          *in_char2_ptr++ = *in_char_ptr;
          in_char_ptr += 4;
        }
        in_char_ptr = start_block_ptr;
        in_char2_ptr = start_block_ptr;
        while (in_char_ptr < end_char_ptr) {
          *in_char2_ptr++ = *in_char_ptr;
          in_char_ptr += 4;
        }
        in_char_ptr = in_char2;
        while (in_char2_ptr < end_char_ptr)
          *in_char2_ptr++ = *in_char_ptr++;
      }

      in_char_ptr = inbuf2;
      while (in_char_ptr != end_char_ptr)
        *out_char_ptr++ = *in_char_ptr++;
      free(in_char2);
    }
    else {
      in_char_ptr = inbuf2;
      while (in_char_ptr != end_char_ptr)
        *out_char_ptr++ = *in_char_ptr++;
    }
  }
  else {
#ifdef PRINTON
    fprintf(stderr, "Converting data\n");
#endif
    *out_char_ptr++ = 0;
    in_char_ptr = inbuf2;
    while (in_char_ptr != end_char_ptr)
      *out_char_ptr++ = *in_char_ptr++;
  }

  *outsize_ptr = out_char_ptr - *outbuf;
  if ((*outbuf = (uint8_t *)realloc(*outbuf, *outsize_ptr)) == 0) {
    fprintf(stderr, "ERROR - Compressed output buffer memory reallocation failed\n");
    return(0);
  }
  free(inbuf2);
  return(1);
}

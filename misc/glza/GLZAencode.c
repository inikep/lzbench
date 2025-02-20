/***********************************************************************

Copyright 2014-2016 Kennon Conrad

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

// GLZAencode.c
//   Encodes files created by GLZAcompress
//
// Usage:
//   GLZAencode [-m#] [-v#] <infilename> <outfilename>, where
//       -m# overrides the programs decision on whether to use mtf.  -m0 disables mtf, -m1 enables mtf
//       -v0 prints the dictionary to stdout, most frequent first
//       -v1 prints the dicitonary to stdout, simple symbols followed by complex symbols in the order they were created


#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "GLZAmodel.h"

#define START_UTF8_2BYTE_SYMBOLS 0x80
#define START_UTF8_3BYTE_SYMBOLS 0x800
#define START_UTF8_4BYTE_SYMBOLS 0x10000
#define MAX_INSTANCES_FOR_MTF_QUEUE 15
#define MTF_QUEUE_SIZE 64

const uint8_t MAX_BITS_IN_CODE = 25;
const uint32_t UNIQUE_CHAR = 0xFFFFFFFF;
const uint32_t READ_SIZE = 0x80000;

#define nsob g_nsob
#define nbob g_nbob
#define fbob g_fbob
#define sum_nbob g_sum_nbob
#define max_code_length g_max_code_length
#define prior_is_cap g_prior_is_cap
#define prior_end g_prior_end
#define UTF8_compliant g_UTF8_compliant
#define max_regular_code_length g_max_regular_code_length
#define num_base_symbols g_num_base_symbols
#define symbol_lengths g_symbol_lengths
#define cap_symbol_defined g_cap_symbol_defined
#define cap_lock_symbol_defined g_cap_lock_symbol_defined
#define end_char_ptr g_end_char_ptr
#define in_char_ptr g_in_char_ptr
#define mtfg_queue_16_offset g_mtfg_queue_16_offset
#define mtfg_queue_16 g_mtfg_queue_16
#define mtfg_queue_32_offset g_mtfg_queue_32_offset
#define mtfg_queue_32 g_mtfg_queue_32
#define mtfg_queue_64_offset g_mtfg_queue_64_offset
#define mtfg_queue_64 g_mtfg_queue_64
#define mtfg_queue_128_offset g_mtfg_queue_128_offset
#define mtfg_queue_128 g_mtfg_queue_128
#define mtfg_queue_192_offset g_mtfg_queue_192_offset
#define mtfg_queue_192 g_mtfg_queue_192
#define mtfg_queue_8 g_mtfg_queue_8
#define mtfg_queue_8_offset g_mtfg_queue_8_offset
#define mtfg_queue_0 g_mtfg_queue_0
#define cap_encoded g_cap_encoded
#define sym_list_bits g_sym_list_bits
#define sym_list_ptrs g_sym_list_ptrs
#define CodeLength g_CodeLength
#define symbol_count g_symbol_count
#define mtf_queue_number g_mtf_queue_number
#define mtf_queue_size g_mtf_queue_size
#define mtf_queue g_mtf_queue
#define mtf_queue_ptr g_mtf_queue_ptr
#define min_extra_reduce_index g_min_extra_reduce_index
#define BinNum g_BinNum
#define end_symbol g_end_symbol
#define SIDSymbol g_SIDSymbol
#define base_bits g_base_bits
#define use_mtf g_use_mtf
#define use_mtfg g_use_mtfg
#define cap_encoded g_cap_encoded
#define symbol g_symbol
#define symbol_to_move g_symbol_to_move
#define symbol_index g_symbol_index

uint8_t UTF8_compliant, base_bits, cap_encoded, prior_is_cap, use_mtf, use_mtfg, format, CodeLength, SIDSymbol;
uint8_t symbol_code_length, symbol_bits, temp_bits, max_code_length, max_regular_code_length, found_first_symbol, end_symbol;
uint8_t cap_symbol_defined, cap_lock_symbol_defined, mtf_queue_number;
uint8_t mtfg_queue_8_offset, mtfg_queue_16_offset, mtfg_queue_32_offset;
uint8_t mtfg_queue_64_offset, mtfg_queue_128_offset, mtfg_queue_192_offset;
uint8_t symbol_lengths[0x100], nbob_shift[0x101], sym_list_bits[0x100][26], mtf_queue_overflow_code_length[16];
uint8_t mtf_queue_size[16];
uint8_t *in_char_ptr, *end_char_ptr;
uint16_t BinNum, fbob[0x100][26];
uint32_t num_define_symbols_written, num_grammar_rules, num_symbols_to_code, mtf_queue_miss_code_space, min_extra_reduce_index;
uint32_t symbol_index, end_symbols, symbol_to_move, prior_end, rules_reduced, start_my_symbols, num_base_symbols;
uint32_t BinCode;
uint32_t mtfg_queue_symbol_7, mtfg_queue_symbol_15, mtfg_queue_symbol_31;
uint32_t mtfg_queue_symbol_63, mtfg_queue_symbol_127, mtfg_queue_symbol_191;
uint32_t nsob[0x100][26], nbob[0x100][26], sum_nbob[0x100];
uint32_t mtf_queue[16][65], mtf_queue_hit_count[16];
uint32_t mtf_queue_started[16], mtf_queue_done[16], mtf_queue_peak[16];
uint32_t mtfg_queue_0[8], mtfg_queue_8[8], mtfg_queue_16[0x10], mtfg_queue_32[0x20];
uint32_t mtfg_queue_64[0x40], mtfg_queue_128[0x40], mtfg_queue_192[0x40];
uint32_t *mtf_queue_ptr, *mtf_queue_end_ptr;
uint32_t *symbol, *symbol_ptr, *ranked_symbols, *first_define_ptr;
uint32_t *sym_list_ptrs[0x100][26];
int32_t prior_symbol;


// type:  bit 0: string ends 'C' or 'B' (cap symbol/cap lock symbol), bit1: string starts a-z, bit 2: non-ergodic, bit 3: in queue
// bit 4: "word"ness determined, bit 5: "word", bit 6: >15 instance word, bit 7: >15 instance word likely to be followed by ' '

struct symbol_data {
  uint8_t starts, ends, code_length, type;
  uint32_t count, inst_found, array_index, mtfg_hits, hit_score, define_symbol_start_index;
  int32_t space_score;
} *sd;


uint32_t z1[0x100], z2[0x100];


void print_string(uint32_t symbol_number) {
  uint32_t *symbol_ptr, *next_symbol_ptr;
  if (symbol_number < start_my_symbols) {
    if (UTF8_compliant != 0) {
      if (symbol_number < START_UTF8_2BYTE_SYMBOLS)
        printf("%c",(unsigned char)symbol_number);
      else if (symbol_number < START_UTF8_3BYTE_SYMBOLS) {
        printf("%c",(unsigned char)(symbol_number >> 6) + 0xC0);
        printf("%c",(unsigned char)(symbol_number & 0x3F) + 0x80);
      }
      else if (symbol_number < START_UTF8_4BYTE_SYMBOLS) {
        printf("%c",(unsigned char)(symbol_number >> 12) + 0xE0);
        printf("%c",(unsigned char)((symbol_number >> 6) & 0x3F) + 0x80);
        printf("%c",(unsigned char)(symbol_number & 0x3F) + 0x80);
      }
      else {
        printf("%c",(unsigned char)(symbol_number >> 18) + 0xF0);
        printf("%c",(unsigned char)((symbol_number >> 12) & 0x3F) + 0x80);
        printf("%c",(unsigned char)((symbol_number >> 6) & 0x3F) + 0x80);
        printf("%c",(unsigned char)(symbol_number & 0x3F) + 0x80);
      }
    }
    else
      printf("%c",(unsigned char)symbol_number);
  }
  else {
    symbol_ptr = symbol + sd[symbol_number].define_symbol_start_index;
    next_symbol_ptr = symbol + sd[symbol_number+1].define_symbol_start_index - 1;
    while (symbol_ptr != next_symbol_ptr)
      print_string(*symbol_ptr++);
  }
  return;
}


void print_string2(uint32_t symbol_number) {
  uint32_t *symbol_ptr, *next_symbol_ptr;
  if (symbol_number < start_my_symbols) {
    if (UTF8_compliant != 0) {
      if (symbol_number < START_UTF8_2BYTE_SYMBOLS)
        fprintf(stderr,"%c",(unsigned char)symbol_number);
      else if (symbol_number < START_UTF8_3BYTE_SYMBOLS) {
        fprintf(stderr,"%c",(unsigned char)(symbol_number >> 6) + 0xC0);
        fprintf(stderr,"%c",(unsigned char)(symbol_number & 0x3F) + 0x80);
      }
      else if (symbol_number < START_UTF8_4BYTE_SYMBOLS) {
        fprintf(stderr,"%c",(unsigned char)(symbol_number >> 12) + 0xE0);
        fprintf(stderr,"%c",(unsigned char)((symbol_number >> 6) & 0x3F) + 0x80);
        fprintf(stderr,"%c",(unsigned char)(symbol_number & 0x3F) + 0x80);
      }
      else {
        fprintf(stderr,"%c",(unsigned char)(symbol_number >> 18) + 0xF0);
        fprintf(stderr,"%c",(unsigned char)((symbol_number >> 12) & 0x3F) + 0x80);
        fprintf(stderr,"%c",(unsigned char)((symbol_number >> 6) & 0x3F) + 0x80);
        fprintf(stderr,"%c",(unsigned char)(symbol_number & 0x3F) + 0x80);
      }
    }
    else
      fprintf(stderr,"%c",(unsigned char)symbol_number);
  }
  else {
    symbol_ptr = symbol + sd[symbol_number].define_symbol_start_index;
    next_symbol_ptr = symbol + sd[symbol_number+1].define_symbol_start_index - 1;
    while (symbol_ptr != next_symbol_ptr)
      print_string2(*symbol_ptr++);
  }
  return;
}


void get_symbol_category(uint32_t symbol_number, uint8_t *sym_type_ptr) {
  if (symbol_number >= start_my_symbols) {
    if (sd[symbol_number].type & 0x20) {
      *sym_type_ptr |= 0x30;
      return;
    }
    uint32_t * string_ptr = symbol + sd[symbol_number+1].define_symbol_start_index - 2;
    get_symbol_category(*string_ptr, sym_type_ptr);
    while (((*sym_type_ptr & 0x10) == 0) && (string_ptr != symbol + sd[symbol_number].define_symbol_start_index))
      get_symbol_category(*--string_ptr, sym_type_ptr);
    if ((sd[symbol_number].type & 0x10) == 0)
      sd[symbol_number].type |= *sym_type_ptr & 0x30;
  }
  else if (symbol_number == (uint32_t)' ')
    *sym_type_ptr |= 0x30;
  return;
}


uint8_t find_first(uint32_t symbol_number) {
  uint8_t first_char;
  uint32_t first_symbol = symbol[sd[symbol_number].define_symbol_start_index];
  if (first_symbol >= start_my_symbols) {
    if ((first_char = sd[first_symbol].starts) == 0) {
      first_char = find_first(first_symbol);
      sd[first_symbol].starts = first_char;
    }
    return(first_char);
  }
  return((uint8_t)first_symbol);
}


uint8_t find_first_UTF8(uint32_t symbol_number) {
  uint8_t first_char;
  uint32_t first_symbol = symbol[sd[symbol_number].define_symbol_start_index];
  if (first_symbol >= start_my_symbols) {
    if ((first_char = sd[first_symbol].starts) == 0) {
      first_char = find_first_UTF8(first_symbol);
      sd[first_symbol].starts = first_char;
    }
    return(first_char);
  }
  if (first_symbol < START_UTF8_2BYTE_SYMBOLS)
    return((uint8_t)first_symbol);
  else if (first_symbol < 0x250)
    return(0x80);
  else if (first_symbol < 0x370)
    return(0x81);
  else if (first_symbol < 0x400)
    return(0x82);
  else if (first_symbol < 0x530)
    return(0x83);
  else if (first_symbol < 0x590)
    return(0x84);
  else if (first_symbol < 0x600)
    return(0x85);
  else if (first_symbol < 0x700)
    return(0x86);
  else if (first_symbol < START_UTF8_3BYTE_SYMBOLS)
    return(0x87);
  else if (first_symbol < 0x1000)
    return(0x88);
  else if (first_symbol < 0x2000)
    return(0x89);
  else if (first_symbol < 0x3000)
    return(0x8A);
  else if (first_symbol < 0x3040)
    return(0x8B);
  else if (first_symbol < 0x30A0)
    return(0x8C);
  else if (first_symbol < 0x3100)
    return(0x8D);
  else if (first_symbol < 0x3200)
    return(0x8E);
  else if (first_symbol < 0xA000)
    return(0x8F);
  else if (first_symbol < START_UTF8_4BYTE_SYMBOLS)
    return(0x8E);
  else
    return(0x90);
}


uint8_t find_last(uint32_t symbol_number) {
  uint8_t last_char;
  uint32_t last_symbol = symbol[sd[symbol_number+1].define_symbol_start_index - 2];
  if (last_symbol >= start_my_symbols) {
    if ((last_char = sd[last_symbol].ends) == 0) {
      last_char = find_last(last_symbol);
      sd[last_symbol].ends = last_char;
    }
    return(last_char);
  }
  if ((cap_encoded != 0) && (last_symbol == 'B'))
    return('C');
  return((uint8_t)last_symbol);
}


uint8_t find_last_UTF8(uint32_t symbol_number) {
  uint8_t last_char;
  uint32_t last_symbol = symbol[sd[symbol_number+1].define_symbol_start_index - 2];
  if (last_symbol >= start_my_symbols) {
    if ((last_char = sd[last_symbol].ends) == 0) {
      last_char = find_last_UTF8(last_symbol);
      sd[last_symbol].ends = last_char;
    }
    return(last_char);
  }
  if ((cap_encoded != 0) && (last_symbol == 'B'))
    return('C');
  if (last_symbol < START_UTF8_2BYTE_SYMBOLS)
    return((uint8_t)last_symbol);
  else if (last_symbol < 0x250)
    return(0x80);
  else if (last_symbol < 0x370)
    return(0x81);
  else if (last_symbol < 0x400)
    return(0x82);
  else if (last_symbol < 0x530)
    return(0x83);
  else if (last_symbol < 0x590)
    return(0x84);
  else if (last_symbol < 0x600)
    return(0x85);
  else if (last_symbol < 0x700)
    return(0x86);
  else if (last_symbol < START_UTF8_3BYTE_SYMBOLS)
    return(0x87);
  else if (last_symbol < 0x1000)
    return(0x88);
  else if (last_symbol < 0x2000)
    return(0x89);
  else if (last_symbol < 0x3000)
    return(0x8A);
  else if (last_symbol < 0x3040)
    return(0x8B);
  else if (last_symbol < 0x30A0)
    return(0x8C);
  else if (last_symbol < 0x3100)
    return(0x8D);
  else if (last_symbol < 0x3200)
    return(0x8E);
  else if (last_symbol < 0xA000)
    return(0x8F);
  else if (last_symbol < START_UTF8_4BYTE_SYMBOLS)
    return(0x8E);
  else
    return(0x90);
}


void remove_mtfg_queue_symbol_16(uint8_t mtfg_queue_position) {
  while (mtfg_queue_position != 31) {
    *(mtfg_queue_16 + ((mtfg_queue_16_offset + mtfg_queue_position) & 0xF))
        = *(mtfg_queue_16 + ((mtfg_queue_16_offset + mtfg_queue_position + 1) & 0xF));
    mtfg_queue_position++;
  }
  *(mtfg_queue_16 + ((mtfg_queue_16_offset - 1) & 0xF)) = *(mtfg_queue_32 + mtfg_queue_32_offset);
  *(mtfg_queue_32 + mtfg_queue_32_offset) = *(mtfg_queue_64 + mtfg_queue_64_offset);
  mtfg_queue_32_offset = (mtfg_queue_32_offset + 1) & 0x1F;
  *(mtfg_queue_64 + mtfg_queue_64_offset) = *(mtfg_queue_128 + mtfg_queue_128_offset);
  mtfg_queue_64_offset = (mtfg_queue_64_offset + 1) & 0x3F;
  *(mtfg_queue_128 + mtfg_queue_128_offset) = *(mtfg_queue_192 + mtfg_queue_192_offset);
  mtfg_queue_128_offset = (mtfg_queue_128_offset + 1) & 0x3F;
  *(mtfg_queue_192 + mtfg_queue_192_offset) = end_symbols;
  mtfg_queue_192_offset = (mtfg_queue_192_offset + 1) & 0x3F;
  return;
}


void remove_mtfg_queue_symbol_32(uint8_t mtfg_queue_position) {
  while (mtfg_queue_position != 63) {
    *(mtfg_queue_32 + ((mtfg_queue_32_offset + mtfg_queue_position) & 0x1F))
        = *(mtfg_queue_32 + ((mtfg_queue_32_offset + mtfg_queue_position + 1) & 0x1F));
    mtfg_queue_position++;
  }
  *(mtfg_queue_32 + ((mtfg_queue_32_offset - 1) & 0x1F)) = *(mtfg_queue_64 + mtfg_queue_64_offset);
  *(mtfg_queue_64 + mtfg_queue_64_offset) = *(mtfg_queue_128 + mtfg_queue_128_offset);
  mtfg_queue_64_offset = (mtfg_queue_64_offset + 1) & 0x3F;
  *(mtfg_queue_128 + mtfg_queue_128_offset) = *(mtfg_queue_192 + mtfg_queue_192_offset);
  mtfg_queue_128_offset = (mtfg_queue_128_offset + 1) & 0x3F;
  *(mtfg_queue_192 + mtfg_queue_192_offset) = end_symbols;
  mtfg_queue_192_offset = (mtfg_queue_192_offset + 1) & 0x3F;
  return;
}


void remove_mtfg_queue_symbol_64(uint8_t mtfg_queue_position) {
  while (mtfg_queue_position != 127) {
    *(mtfg_queue_64 + ((mtfg_queue_64_offset + mtfg_queue_position) & 0x3F))
        = *(mtfg_queue_64 + ((mtfg_queue_64_offset + mtfg_queue_position + 1) & 0x3F));
    mtfg_queue_position++;
  }
  *(mtfg_queue_64 + ((mtfg_queue_64_offset - 1) & 0x3F)) = *(mtfg_queue_128 + mtfg_queue_128_offset);
  *(mtfg_queue_128 + mtfg_queue_128_offset) = *(mtfg_queue_192 + mtfg_queue_192_offset);
  mtfg_queue_128_offset = (mtfg_queue_128_offset + 1) & 0x3F;
  *(mtfg_queue_192 + mtfg_queue_192_offset) = end_symbols;
  mtfg_queue_192_offset = (mtfg_queue_192_offset + 1) & 0x3F;
  return;
}


void remove_mtfg_queue_symbol_128(uint8_t mtfg_queue_position) {
  while (mtfg_queue_position != 191) {
    *(mtfg_queue_128 + ((mtfg_queue_128_offset + mtfg_queue_position) & 0x3F))
        = *(mtfg_queue_128 + ((mtfg_queue_128_offset + mtfg_queue_position + 1) & 0x3F));
    mtfg_queue_position++;
  }
  *(mtfg_queue_128 + ((mtfg_queue_128_offset - 1) & 0x3F)) = *(mtfg_queue_192 + mtfg_queue_192_offset);
  *(mtfg_queue_192 + mtfg_queue_192_offset) = end_symbols;
  mtfg_queue_192_offset = (mtfg_queue_192_offset + 1) & 0x3F;
  return;
}


void remove_mtfg_queue_symbol_192(uint8_t mtfg_queue_position) {
  while (mtfg_queue_position != 255) {
    *(mtfg_queue_192 + ((mtfg_queue_192_offset + mtfg_queue_position) & 0x3F))
        = *(mtfg_queue_192 + ((mtfg_queue_192_offset + mtfg_queue_position + 1) & 0x3F));
    mtfg_queue_position++;
  }
  *(mtfg_queue_192 + ((mtfg_queue_192_offset - 1) & 0x3F)) = end_symbols;
  return;
}


void increment_mtfg_queue_0(uint32_t symbol_number) {
  mtfg_queue_symbol_7 = mtfg_queue_0[7];
  mtfg_queue_0[7] = mtfg_queue_0[6];
  mtfg_queue_0[6] = mtfg_queue_0[5];
  mtfg_queue_0[5] = mtfg_queue_0[4];
  mtfg_queue_0[4] = mtfg_queue_0[3];
  mtfg_queue_0[3] = mtfg_queue_0[2];
  mtfg_queue_0[2] = mtfg_queue_0[1];
  mtfg_queue_0[1] = mtfg_queue_0[0];
  mtfg_queue_0[0] = symbol_number;
  return;
}


void increment_mtfg_queue_8() {
  mtfg_queue_8_offset = (mtfg_queue_8_offset - 1) & 7;
  mtfg_queue_symbol_15 = *(mtfg_queue_8 + mtfg_queue_8_offset);
  *(mtfg_queue_8 + mtfg_queue_8_offset) = mtfg_queue_symbol_7;
  return;
}


void increment_mtfg_queue_16() { \
  mtfg_queue_16_offset = (mtfg_queue_16_offset - 1) & 0xF;
  mtfg_queue_symbol_31 = *(mtfg_queue_16 + mtfg_queue_16_offset);
  *(mtfg_queue_16 + mtfg_queue_16_offset) = mtfg_queue_symbol_15;
  return;
}


void increment_mtfg_queue_32() { \
  mtfg_queue_32_offset = (mtfg_queue_32_offset - 1) & 0x1F;
  mtfg_queue_symbol_63 = *(mtfg_queue_32 + mtfg_queue_32_offset);
  *(mtfg_queue_32 + mtfg_queue_32_offset) = mtfg_queue_symbol_31;
  return;
}


void increment_mtfg_queue_64() {
  mtfg_queue_64_offset = (mtfg_queue_64_offset - 1) & 0x3F;
  mtfg_queue_symbol_127 = *(mtfg_queue_64 + mtfg_queue_64_offset);
  *(mtfg_queue_64 + mtfg_queue_64_offset) = mtfg_queue_symbol_63;
  return;
}


void increment_mtfg_queue_128() {
  mtfg_queue_128_offset = (mtfg_queue_128_offset - 1) & 0x3F;
  mtfg_queue_symbol_191 = *(mtfg_queue_128 + mtfg_queue_128_offset);
  *(mtfg_queue_128 + mtfg_queue_128_offset) = mtfg_queue_symbol_127;
  return;
}


void increment_mtfg_queue_192() {
  mtfg_queue_192_offset = (mtfg_queue_192_offset - 1) & 0x3F;
  sd[*(mtfg_queue_192 + mtfg_queue_192_offset)].type &= 0xF7;
  *(mtfg_queue_192 + mtfg_queue_192_offset) = mtfg_queue_symbol_191;
  return;
}


void add_symbol_to_mtfg_queue(uint32_t symbol_number) {
  sd[symbol_number].type |= 8;
  increment_mtfg_queue_0(symbol_number);
  increment_mtfg_queue_8();
  if (sd[mtfg_queue_symbol_15].code_length > 12) {
    increment_mtfg_queue_16();
    if (sd[mtfg_queue_symbol_31].code_length != 13) {
      increment_mtfg_queue_32();
      if (sd[mtfg_queue_symbol_63].code_length != 14) {
        increment_mtfg_queue_64();
        if (sd[mtfg_queue_symbol_127].code_length != 15) {
          increment_mtfg_queue_128();
          if (sd[mtfg_queue_symbol_191].code_length != 16)
            increment_mtfg_queue_192();
          else
            sd[mtfg_queue_symbol_191].type &= 0xF7;
        }
        else
          sd[mtfg_queue_symbol_127].type &= 0xF7;
      }
      else
        sd[mtfg_queue_symbol_63].type &= 0xF7;
    }
    else
      sd[mtfg_queue_symbol_31].type &= 0xF7;
  } 
  else
    sd[mtfg_queue_symbol_15].type &= 0xF7;
  return;
}


void manage_mtfg_queue1(uint32_t symbol_number) {
  uint8_t mtfg_queue_position;
  uint32_t subqueue_position;
  mtfg_queue_position = 0;
  do {
    if (symbol_number == mtfg_queue_0[mtfg_queue_position]) {
      sd[symbol_number].hit_score += 61 - 3 * mtfg_queue_position;
      while (mtfg_queue_position) {
        mtfg_queue_0[mtfg_queue_position] = mtfg_queue_0[mtfg_queue_position-1];
        mtfg_queue_position--;
      }
      mtfg_queue_0[0] = symbol_number;
      break;
    }
  } while (++mtfg_queue_position != 5);
  if (mtfg_queue_position == 5) {
    do {
      if (symbol_number == mtfg_queue_0[mtfg_queue_position]) {
        sd[symbol_number].hit_score += 56 - 2 * mtfg_queue_position;
        while (mtfg_queue_position != 5) {
          mtfg_queue_0[mtfg_queue_position] = mtfg_queue_0[mtfg_queue_position-1];
          mtfg_queue_position--;
        }
        mtfg_queue_0[5] = mtfg_queue_0[4];
        mtfg_queue_0[4] = mtfg_queue_0[3];
        mtfg_queue_0[3] = mtfg_queue_0[2];
        mtfg_queue_0[2] = mtfg_queue_0[1];
        mtfg_queue_0[1] = mtfg_queue_0[0];
        mtfg_queue_0[0] = symbol_number;
        break;
      }
    } while (++mtfg_queue_position != 8);
    if (mtfg_queue_position == 8) {
      increment_mtfg_queue_0(symbol_number);
      do {
        if (symbol_number == *(mtfg_queue_8 + ((mtfg_queue_position + mtfg_queue_8_offset) & 7))) {
          sd[symbol_number].hit_score += 48 - mtfg_queue_position;
          subqueue_position = mtfg_queue_position - 8;
          while (subqueue_position) {
            *(mtfg_queue_8 + ((mtfg_queue_8_offset + subqueue_position) & 7))
                = *(mtfg_queue_8 + ((mtfg_queue_8_offset + subqueue_position - 1) & 7));
            subqueue_position--;
          }
          *(mtfg_queue_8 + mtfg_queue_8_offset) = mtfg_queue_symbol_7;
          break;
        }
      } while (++mtfg_queue_position != 16);
      if (mtfg_queue_position == 16) {
        increment_mtfg_queue_8();
        do {
          if (symbol_number == *(mtfg_queue_16 + ((mtfg_queue_position + mtfg_queue_16_offset) & 0xF))) {
            sd[symbol_number].hit_score += 40 - (mtfg_queue_position >> 1);
            if (sd[mtfg_queue_symbol_15].code_length <= 12) {
              sd[mtfg_queue_symbol_15].type &= 0xF7;
              remove_mtfg_queue_symbol_16(mtfg_queue_position);
            }
            else {
              subqueue_position = mtfg_queue_position - 16;
              while (subqueue_position) {
                *(mtfg_queue_16 + ((mtfg_queue_16_offset + subqueue_position) & 0xF))
                    = *(mtfg_queue_16 + ((mtfg_queue_16_offset + subqueue_position - 1) & 0xF));
                subqueue_position--;
              }
              *(mtfg_queue_16 + mtfg_queue_16_offset) = mtfg_queue_symbol_15;
            }
            break;
          }
        } while (++mtfg_queue_position != 32);
        if (mtfg_queue_position == 32) {
          do {
            if (symbol_number == *(mtfg_queue_32 + ((mtfg_queue_position + mtfg_queue_32_offset) & 0x1F))) {
              sd[symbol_number].hit_score += 28 - (mtfg_queue_position >> 3);
              if (sd[mtfg_queue_symbol_15].code_length <= 12) {
                sd[mtfg_queue_symbol_15].type &= 0xF7;
                remove_mtfg_queue_symbol_32(mtfg_queue_position);
              }
              else {
                increment_mtfg_queue_16();
                if (sd[mtfg_queue_symbol_31].code_length == 13) {
                  sd[mtfg_queue_symbol_31].type &= 0xF7;
                  remove_mtfg_queue_symbol_32(mtfg_queue_position);
                }
                else {
                  subqueue_position = mtfg_queue_position - 32;
                  while (subqueue_position) {
                    *(mtfg_queue_32 + ((mtfg_queue_32_offset + subqueue_position) & 0x1F))
                        = *(mtfg_queue_32 + ((mtfg_queue_32_offset + subqueue_position - 1) & 0x1F));
                    subqueue_position--;
                  }
                  *(mtfg_queue_32 + mtfg_queue_32_offset) = mtfg_queue_symbol_31;
                }
              }
              break;
            }
          } while (++mtfg_queue_position != 64);
          if (mtfg_queue_position == 64) {
            do {
              if (symbol_number == *(mtfg_queue_64 + ((mtfg_queue_position + mtfg_queue_64_offset) & 0x3F))) {
                sd[symbol_number].hit_score += 22 - (mtfg_queue_position >> 5);
                if (sd[mtfg_queue_symbol_15].code_length <= 12) {
                  sd[mtfg_queue_symbol_15].type &= 0xF7;
                  remove_mtfg_queue_symbol_64(mtfg_queue_position);
                }
                else {
                  increment_mtfg_queue_16();
                  if (sd[mtfg_queue_symbol_31].code_length == 13) {
                    sd[mtfg_queue_symbol_31].type &= 0xF7;
                    remove_mtfg_queue_symbol_64(mtfg_queue_position);
                  }
                  else {
                    increment_mtfg_queue_32();
                    if (sd[mtfg_queue_symbol_63].code_length == 14) {
                      sd[mtfg_queue_symbol_63].type &= 0xF7;
                      remove_mtfg_queue_symbol_64(mtfg_queue_position);
                    }
                    else {
                      subqueue_position = mtfg_queue_position - 64;
                      while (subqueue_position) {
                        *(mtfg_queue_64 + ((mtfg_queue_64_offset + subqueue_position) & 0x3F))
                            = *(mtfg_queue_64 + ((mtfg_queue_64_offset + subqueue_position - 1) & 0x3F));
                        subqueue_position--;
                      }
                      *(mtfg_queue_64 + mtfg_queue_64_offset) = mtfg_queue_symbol_63;
                    }
                  }
                }
                break;
              }
            } while (++mtfg_queue_position != 128);
            if (mtfg_queue_position == 128) {
              do {
                if (symbol_number == *(mtfg_queue_128 + ((mtfg_queue_position + mtfg_queue_128_offset) & 0x3F))) {
                  sd[symbol_number].hit_score += 20 - (mtfg_queue_position >> 6);
                  if (sd[mtfg_queue_symbol_15].code_length <= 12) {
                    sd[mtfg_queue_symbol_15].type &= 0xF7;
                    remove_mtfg_queue_symbol_128(mtfg_queue_position);
                  }
                  else {
                    increment_mtfg_queue_16();
                    if (sd[mtfg_queue_symbol_31].code_length == 13) {
                      sd[mtfg_queue_symbol_31].type &= 0xF7;
                      remove_mtfg_queue_symbol_128(mtfg_queue_position);
                    }
                    else {
                      increment_mtfg_queue_32();
                      if (sd[mtfg_queue_symbol_63].code_length == 14) {
                        sd[mtfg_queue_symbol_63].type &= 0xF7;
                        remove_mtfg_queue_symbol_128(mtfg_queue_position);
                      }
                      else {
                        increment_mtfg_queue_64();
                        if (sd[mtfg_queue_symbol_127].code_length == 15) {
                          sd[mtfg_queue_symbol_127].type &= 0xF7;
                          remove_mtfg_queue_symbol_128(mtfg_queue_position);
                        }
                        else {
                          subqueue_position = mtfg_queue_position - 128;
                          while (subqueue_position) {
                            *(mtfg_queue_128 + ((mtfg_queue_128_offset + subqueue_position) & 0x3F))
                                = *(mtfg_queue_128 + ((mtfg_queue_128_offset + subqueue_position - 1) & 0x3F));
                            subqueue_position--;
                          }
                          *(mtfg_queue_128 + mtfg_queue_128_offset) = mtfg_queue_symbol_127;
                        }
                      }
                    }
                  }
                  break;
                }
              } while (++mtfg_queue_position != 192);
              if (mtfg_queue_position == 192) {
                while (symbol_number != *(mtfg_queue_192 + ((mtfg_queue_position + mtfg_queue_192_offset) & 0x3F)))
                  mtfg_queue_position++;
                sd[symbol_number].hit_score += 20 - (mtfg_queue_position >> 6);
                if (sd[mtfg_queue_symbol_15].code_length <= 12) {
                  sd[mtfg_queue_symbol_15].type &= 0xF7;
                  remove_mtfg_queue_symbol_192(mtfg_queue_position);
                }
                else {
                  increment_mtfg_queue_16();
                  if (sd[mtfg_queue_symbol_31].code_length == 13) {
                    sd[mtfg_queue_symbol_31].type &= 0xF7;
                    remove_mtfg_queue_symbol_192(mtfg_queue_position);
                  }
                  else {
                    increment_mtfg_queue_32();
                    if (sd[mtfg_queue_symbol_63].code_length == 14) {
                      sd[mtfg_queue_symbol_63].type &= 0xF7;
                      remove_mtfg_queue_symbol_192(mtfg_queue_position);
                    }
                    else {
                      increment_mtfg_queue_64();
                      if (sd[mtfg_queue_symbol_127].code_length == 15) {
                        sd[mtfg_queue_symbol_127].type &= 0xF7;
                        remove_mtfg_queue_symbol_192(mtfg_queue_position);
                      }
                      else {
                        increment_mtfg_queue_128();
                        if (sd[mtfg_queue_symbol_191].code_length == 16) {
                          sd[mtfg_queue_symbol_191].type &= 0xF7;
                          remove_mtfg_queue_symbol_192(mtfg_queue_position);
                        }
                        else {
                          subqueue_position = mtfg_queue_position - 192;
                          while (subqueue_position) {
                            *(mtfg_queue_192 + ((mtfg_queue_192_offset + subqueue_position) & 0x3F))
                                = *(mtfg_queue_192 + ((mtfg_queue_192_offset + subqueue_position - 1) & 0x3F));
                            subqueue_position--;
                          }
                          *(mtfg_queue_192 + mtfg_queue_192_offset) = mtfg_queue_symbol_191;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return;
}


void manage_mtfg_queue(uint32_t symbol_number, uint8_t in_definition) {
  uint8_t mtfg_queue_position = 0;
  uint8_t cap_queue_position = 0;
  uint8_t subqueue_position;
  do {
    if (symbol_number == mtfg_queue_0[mtfg_queue_position]) {
      if (in_definition == 0)
        EncodeMtfgType(LEVEL0);
      else
        EncodeMtfgType(LEVEL1);
      EncodeMtfgQueuePos(NOT_CAP, mtfg_queue_position);
      while (mtfg_queue_position) {
        mtfg_queue_0[mtfg_queue_position] = mtfg_queue_0[mtfg_queue_position-1];
        mtfg_queue_position--;
      }
      mtfg_queue_0[0] = symbol_number;
      break;
    }
  } while (++mtfg_queue_position != 8);
  if (mtfg_queue_position == 8) {
    increment_mtfg_queue_0(symbol_number);
    do {
      if (symbol_number == *(mtfg_queue_8 + ((mtfg_queue_position + mtfg_queue_8_offset) & 7))) {
        if (in_definition == 0)
          EncodeMtfgType(LEVEL0);
        else
          EncodeMtfgType(LEVEL1);
        EncodeMtfgQueuePos(NOT_CAP, mtfg_queue_position);
        subqueue_position = mtfg_queue_position - 8;
        while (subqueue_position) {
          *(mtfg_queue_8 + ((mtfg_queue_8_offset + subqueue_position) & 7))
              = *(mtfg_queue_8 + ((mtfg_queue_8_offset + subqueue_position - 1) & 7));
          subqueue_position--;
        }
        *(mtfg_queue_8 + mtfg_queue_8_offset) = mtfg_queue_symbol_7;
        break;
      }
    } while (++mtfg_queue_position != 16);
    if (mtfg_queue_position == 16) {
      increment_mtfg_queue_8();
      do {
        if (symbol_number == *(mtfg_queue_16 + ((mtfg_queue_position + mtfg_queue_16_offset) & 0xF))) {
          if (in_definition == 0)
            EncodeMtfgType(LEVEL0);
          else
            EncodeMtfgType(LEVEL1);
          EncodeMtfgQueuePos(NOT_CAP, mtfg_queue_position);
          if (sd[mtfg_queue_symbol_15].code_length <= 12) {
            sd[mtfg_queue_symbol_15].type &= 0xF7;
            remove_mtfg_queue_symbol_16(mtfg_queue_position);
          }
          else {
            subqueue_position = mtfg_queue_position - 16;
            while (subqueue_position) {
              *(mtfg_queue_16 + ((mtfg_queue_16_offset + subqueue_position) & 0xF))
                  = *(mtfg_queue_16 + ((mtfg_queue_16_offset + subqueue_position - 1) & 0xF));
              subqueue_position--;
            }
            *(mtfg_queue_16 + mtfg_queue_16_offset) = mtfg_queue_symbol_15;
          }
          break;
        }
      } while (++mtfg_queue_position != 32);
      if (mtfg_queue_position == 32) {
        do {
          if (symbol_number == *(mtfg_queue_32 + ((mtfg_queue_position + mtfg_queue_32_offset) & 0x1F))) {
            if (in_definition == 0)
              EncodeMtfgType(LEVEL0);
            else
              EncodeMtfgType(LEVEL1);
            EncodeMtfgQueuePos(NOT_CAP, mtfg_queue_position);
            if (sd[mtfg_queue_symbol_15].code_length <= 12) {
              sd[mtfg_queue_symbol_15].type &= 0xF7;
              remove_mtfg_queue_symbol_32(mtfg_queue_position);
            }
            else {
              increment_mtfg_queue_16();
              if (sd[mtfg_queue_symbol_31].code_length == 13) {
                sd[mtfg_queue_symbol_31].type &= 0xF7;
                remove_mtfg_queue_symbol_32(mtfg_queue_position);
              }
              else {
                subqueue_position = mtfg_queue_position - 32;
                while (subqueue_position) {
                  *(mtfg_queue_32 + ((mtfg_queue_32_offset + subqueue_position) & 0x1F))
                      = *(mtfg_queue_32 + ((mtfg_queue_32_offset + subqueue_position - 1) & 0x1F));
                  subqueue_position--;
                }
                *(mtfg_queue_32 + mtfg_queue_32_offset) = mtfg_queue_symbol_31;
              }
            }
            break;
          }
        } while (++mtfg_queue_position != 64);
        if (mtfg_queue_position == 64) {
          do {
            if (symbol_number == *(mtfg_queue_64 + ((mtfg_queue_position + mtfg_queue_64_offset) & 0x3F))) {
              if (in_definition == 0)
                EncodeMtfgType(LEVEL0);
              else
                EncodeMtfgType(LEVEL1);
              EncodeMtfgQueuePos(NOT_CAP, mtfg_queue_position);
              if (sd[mtfg_queue_symbol_15].code_length <= 12) {
                sd[mtfg_queue_symbol_15].type &= 0xF7;
                remove_mtfg_queue_symbol_64(mtfg_queue_position);
              }
              else {
                increment_mtfg_queue_16();
                if (sd[mtfg_queue_symbol_31].code_length == 13) {
                  sd[mtfg_queue_symbol_31].type &= 0xF7;
                  remove_mtfg_queue_symbol_64(mtfg_queue_position);
                }
                else {
                  increment_mtfg_queue_32();
                  if (sd[mtfg_queue_symbol_63].code_length == 14) {
                    sd[mtfg_queue_symbol_63].type &= 0xF7;
                    remove_mtfg_queue_symbol_64(mtfg_queue_position);
                  }
                  else {
                    subqueue_position = mtfg_queue_position - 64;
                    while (subqueue_position) {
                      *(mtfg_queue_64 + ((mtfg_queue_64_offset + subqueue_position) & 0x3F))
                          = *(mtfg_queue_64 + ((mtfg_queue_64_offset + subqueue_position - 1) & 0x3F));
                      subqueue_position--;
                    }
                    *(mtfg_queue_64 + mtfg_queue_64_offset) = mtfg_queue_symbol_63;
                  }
                }
              }
              break;
            }
          } while (++mtfg_queue_position != 128);
          if (mtfg_queue_position == 128) {
            do {
              if (symbol_number == *(mtfg_queue_128 + ((mtfg_queue_position + mtfg_queue_128_offset) & 0x3F))) {
                if (in_definition == 0)
                  EncodeMtfgType(LEVEL0);
                else
                  EncodeMtfgType(LEVEL1);
                EncodeMtfgQueuePos(NOT_CAP, mtfg_queue_position);
                if (sd[mtfg_queue_symbol_15].code_length <= 12) {
                  sd[mtfg_queue_symbol_15].type &= 0xF7;
                  remove_mtfg_queue_symbol_128(mtfg_queue_position);
                }
                else {
                  increment_mtfg_queue_16();
                  if (sd[mtfg_queue_symbol_31].code_length == 13) {
                    sd[mtfg_queue_symbol_31].type &= 0xF7;
                    remove_mtfg_queue_symbol_128(mtfg_queue_position);
                  }
                  else {
                    increment_mtfg_queue_32();
                    if (sd[mtfg_queue_symbol_63].code_length == 14) {
                      sd[mtfg_queue_symbol_63].type &= 0xF7;
                      remove_mtfg_queue_symbol_128(mtfg_queue_position);
                    }
                    else {
                      increment_mtfg_queue_64();
                      if (sd[mtfg_queue_symbol_127].code_length == 15) {
                        sd[mtfg_queue_symbol_127].type &= 0xF7;
                        remove_mtfg_queue_symbol_128(mtfg_queue_position);
                      }
                      else {
                        subqueue_position = mtfg_queue_position - 128;
                        while (subqueue_position) {
                          *(mtfg_queue_128 + ((mtfg_queue_128_offset + subqueue_position) & 0x3F))
                              = *(mtfg_queue_128 + ((mtfg_queue_128_offset + subqueue_position - 1) & 0x3F));
                          subqueue_position--;
                        }
                        *(mtfg_queue_128 + mtfg_queue_128_offset) = mtfg_queue_symbol_127;
                      }
                    }
                  }
                }
                break;
              }
            } while (++mtfg_queue_position != 192);
            if (mtfg_queue_position == 192) {
              while (symbol_number != *(mtfg_queue_192 + ((mtfg_queue_position + mtfg_queue_192_offset) & 0x3F))) {
                if (sd[*(mtfg_queue_192 + ((mtfg_queue_position + mtfg_queue_192_offset) & 0x3F))].type & 2)
                  cap_queue_position++;
                mtfg_queue_position++;
              }
              if (in_definition == 0)
                EncodeMtfgType(LEVEL0);
              else
                EncodeMtfgType(LEVEL1);
              EncodeMtfgQueuePos(NOT_CAP, mtfg_queue_position);
              if (sd[mtfg_queue_symbol_15].code_length <= 12) {
                sd[mtfg_queue_symbol_15].type &= 0xF7;
                remove_mtfg_queue_symbol_192(mtfg_queue_position);
              }
              else {
                increment_mtfg_queue_16();
                if (sd[mtfg_queue_symbol_31].code_length == 13) {
                  sd[mtfg_queue_symbol_31].type &= 0xF7;
                  remove_mtfg_queue_symbol_192(mtfg_queue_position);
                }
                else {
                  increment_mtfg_queue_32();
                  if (sd[mtfg_queue_symbol_63].code_length == 14) {
                    sd[mtfg_queue_symbol_63].type &= 0xF7;
                    remove_mtfg_queue_symbol_192(mtfg_queue_position);
                  }
                  else {
                    increment_mtfg_queue_64();
                    if (sd[mtfg_queue_symbol_127].code_length == 15) {
                      sd[mtfg_queue_symbol_127].type &= 0xF7;
                      remove_mtfg_queue_symbol_192(mtfg_queue_position);
                    }
                    else {
                      increment_mtfg_queue_128();
                      if (sd[mtfg_queue_symbol_191].code_length == 16) {
                        sd[mtfg_queue_symbol_191].type &= 0xF7;
                        remove_mtfg_queue_symbol_192(mtfg_queue_position);
                      }
                      else {
                        subqueue_position = mtfg_queue_position - 192;
                        while (subqueue_position) {
                          *(mtfg_queue_192 + ((mtfg_queue_192_offset + subqueue_position) & 0x3F))
                              = *(mtfg_queue_192 + ((mtfg_queue_192_offset + subqueue_position - 1) & 0x3F));
                          subqueue_position--;
                        }
                        *(mtfg_queue_192 + mtfg_queue_192_offset) = mtfg_queue_symbol_191;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return;
}


void manage_mtfg_queue_prior_cap(uint32_t symbol_number, uint8_t in_definition) {
  uint8_t mtfg_queue_position = 0;
  uint8_t cap_queue_position = 0;
  uint8_t subqueue_position;
  do {
    if (symbol_number == mtfg_queue_0[mtfg_queue_position]) {
      if (in_definition == 0)
        EncodeMtfgType(LEVEL0_CAP);
      else
        EncodeMtfgType(LEVEL1_CAP);
      uint8_t saved_qp = mtfg_queue_position;
      mtfg_queue_position = cap_queue_position;
      EncodeMtfgQueuePos(CAP, mtfg_queue_position);
      mtfg_queue_position = saved_qp;
      while (mtfg_queue_position) {
        mtfg_queue_0[mtfg_queue_position] = mtfg_queue_0[mtfg_queue_position-1];
        mtfg_queue_position--;
      }
      mtfg_queue_0[0] = symbol_number;
      break;
    }
    else if (sd[mtfg_queue_0[mtfg_queue_position]].type & 2)
      cap_queue_position++;
  } while (++mtfg_queue_position != 8);
  if (mtfg_queue_position == 8) {
    increment_mtfg_queue_0(symbol_number);
    do {
      if (symbol_number == *(mtfg_queue_8 + ((mtfg_queue_position + mtfg_queue_8_offset) & 7))) {
        if (in_definition == 0)
          EncodeMtfgType(LEVEL0_CAP);
        else
          EncodeMtfgType(LEVEL1_CAP);
        uint8_t saved_qp = mtfg_queue_position;
        mtfg_queue_position = cap_queue_position;
        EncodeMtfgQueuePos(CAP, mtfg_queue_position);
        mtfg_queue_position = saved_qp;
        subqueue_position = mtfg_queue_position - 8;
        while (subqueue_position) {
          *(mtfg_queue_8 + ((mtfg_queue_8_offset + subqueue_position) & 7))
              = *(mtfg_queue_8 + ((mtfg_queue_8_offset + subqueue_position - 1) & 7));
          subqueue_position--;
        }
        *(mtfg_queue_8 + mtfg_queue_8_offset) = mtfg_queue_symbol_7;
        break;
      }
      else if (sd[*(mtfg_queue_8 + ((mtfg_queue_position + mtfg_queue_8_offset) & 7))].type & 2)
        cap_queue_position++;
    } while (++mtfg_queue_position != 16);
    if (mtfg_queue_position == 16) {
      increment_mtfg_queue_8();
      do {
        if (symbol_number == *(mtfg_queue_16 + ((mtfg_queue_position + mtfg_queue_16_offset) & 0xF))) {
          if (in_definition == 0)
            EncodeMtfgType(LEVEL0_CAP);
          else
            EncodeMtfgType(LEVEL1_CAP);
          uint8_t saved_qp = mtfg_queue_position;
          mtfg_queue_position = cap_queue_position;
          EncodeMtfgQueuePos(CAP, mtfg_queue_position);
          mtfg_queue_position = saved_qp;
          if (sd[mtfg_queue_symbol_15].code_length <= 12) {
            sd[mtfg_queue_symbol_15].type &= 0xF7;
            remove_mtfg_queue_symbol_16(mtfg_queue_position);
          }
          else {
            subqueue_position = mtfg_queue_position - 16;
            while (subqueue_position) {
              *(mtfg_queue_16 + ((mtfg_queue_16_offset + subqueue_position) & 0xF))
                  = *(mtfg_queue_16 + ((mtfg_queue_16_offset + subqueue_position - 1) & 0xF));
              subqueue_position--;
            }
            *(mtfg_queue_16 + mtfg_queue_16_offset) = mtfg_queue_symbol_15;
          }
          break;
        }
        else if (sd[*(mtfg_queue_16 + ((mtfg_queue_position + mtfg_queue_16_offset) & 0xF))].type & 2)
          cap_queue_position++;
      } while (++mtfg_queue_position != 32);
      if (mtfg_queue_position == 32) {
        do {
          if (symbol_number == *(mtfg_queue_32 + ((mtfg_queue_position + mtfg_queue_32_offset) & 0x1F))) {
            if (in_definition == 0)
              EncodeMtfgType(LEVEL0_CAP);
            else
              EncodeMtfgType(LEVEL1_CAP);
            uint8_t saved_qp = mtfg_queue_position;
            mtfg_queue_position = cap_queue_position;
            EncodeMtfgQueuePos(CAP, mtfg_queue_position);
            mtfg_queue_position = saved_qp;
            if (sd[mtfg_queue_symbol_15].code_length <= 12) {
              sd[mtfg_queue_symbol_15].type &= 0xF7;
              remove_mtfg_queue_symbol_32(mtfg_queue_position);
            }
            else {
              increment_mtfg_queue_16();
              if (sd[mtfg_queue_symbol_31].code_length == 13) {
                sd[mtfg_queue_symbol_31].type &= 0xF7;
                remove_mtfg_queue_symbol_32(mtfg_queue_position);
              }
              else {
                subqueue_position = mtfg_queue_position - 32;
                while (subqueue_position) {
                  *(mtfg_queue_32 + ((mtfg_queue_32_offset + subqueue_position) & 0x1F))
                      = *(mtfg_queue_32 + ((mtfg_queue_32_offset + subqueue_position - 1) & 0x1F));
                  subqueue_position--;
                }
                *(mtfg_queue_32 + mtfg_queue_32_offset) = mtfg_queue_symbol_31;
              }
            }
            break;
          }
          else if (sd[*(mtfg_queue_32 + ((mtfg_queue_position + mtfg_queue_32_offset) & 0x1F))].type & 2)
            cap_queue_position++;
        } while (++mtfg_queue_position != 64);
        if (mtfg_queue_position == 64) {
          do {
            if (symbol_number == *(mtfg_queue_64 + ((mtfg_queue_position + mtfg_queue_64_offset) & 0x3F))) {
              if (in_definition == 0)
                EncodeMtfgType(LEVEL0_CAP);
              else
                EncodeMtfgType(LEVEL1_CAP);
              uint8_t saved_qp = mtfg_queue_position;
              mtfg_queue_position = cap_queue_position;
              EncodeMtfgQueuePos(CAP, mtfg_queue_position);
              mtfg_queue_position = saved_qp;
              if (sd[mtfg_queue_symbol_15].code_length <= 12) {
                sd[mtfg_queue_symbol_15].type &= 0xF7;
                remove_mtfg_queue_symbol_64(mtfg_queue_position);
              }
              else {
                increment_mtfg_queue_16();
                if (sd[mtfg_queue_symbol_31].code_length == 13) {
                  sd[mtfg_queue_symbol_31].type &= 0xF7;
                  remove_mtfg_queue_symbol_64(mtfg_queue_position);
                }
                else {
                  increment_mtfg_queue_32();
                  if (sd[mtfg_queue_symbol_63].code_length == 14) {
                    sd[mtfg_queue_symbol_63].type &= 0xF7;
                    remove_mtfg_queue_symbol_64(mtfg_queue_position);
                  }
                  else {
                    subqueue_position = mtfg_queue_position - 64;
                    while (subqueue_position) {
                      *(mtfg_queue_64 + ((mtfg_queue_64_offset + subqueue_position) & 0x3F))
                          = *(mtfg_queue_64 + ((mtfg_queue_64_offset + subqueue_position - 1) & 0x3F));
                      subqueue_position--;
                    }
                    *(mtfg_queue_64 + mtfg_queue_64_offset) = mtfg_queue_symbol_63;
                  }
                }
              }
              break;
            }
            else if (sd[*(mtfg_queue_64 + ((mtfg_queue_position + mtfg_queue_64_offset) & 0x3F))].type & 2)
              cap_queue_position++;
          } while (++mtfg_queue_position != 128);
          if (mtfg_queue_position == 128) {
            do {
              if (symbol_number == *(mtfg_queue_128 + ((mtfg_queue_position + mtfg_queue_128_offset) & 0x3F))) {
                if (in_definition == 0)
                  EncodeMtfgType(LEVEL0_CAP);
                else
                  EncodeMtfgType(LEVEL1_CAP);
                uint8_t saved_qp = mtfg_queue_position;
                mtfg_queue_position = cap_queue_position;
                EncodeMtfgQueuePos(CAP, mtfg_queue_position);
                mtfg_queue_position = saved_qp;
                if (sd[mtfg_queue_symbol_15].code_length <= 12) {
                  sd[mtfg_queue_symbol_15].type &= 0xF7;
                  remove_mtfg_queue_symbol_128(mtfg_queue_position);
                }
                else {
                  increment_mtfg_queue_16();
                  if (sd[mtfg_queue_symbol_31].code_length == 13) {
                    sd[mtfg_queue_symbol_31].type &= 0xF7;
                    remove_mtfg_queue_symbol_128(mtfg_queue_position);
                  }
                  else {
                    increment_mtfg_queue_32();
                    if (sd[mtfg_queue_symbol_63].code_length == 14) {
                      sd[mtfg_queue_symbol_63].type &= 0xF7;
                      remove_mtfg_queue_symbol_128(mtfg_queue_position);
                    }
                    else {
                      increment_mtfg_queue_64();
                      if (sd[mtfg_queue_symbol_127].code_length == 15) {
                        sd[mtfg_queue_symbol_127].type &= 0xF7;
                        remove_mtfg_queue_symbol_128(mtfg_queue_position);
                      }
                      else {
                        subqueue_position = mtfg_queue_position - 128;
                        while (subqueue_position) {
                          *(mtfg_queue_128 + ((mtfg_queue_128_offset + subqueue_position) & 0x3F))
                              = *(mtfg_queue_128 + ((mtfg_queue_128_offset + subqueue_position - 1) & 0x3F));
                          subqueue_position--;
                        }
                        *(mtfg_queue_128 + mtfg_queue_128_offset) = mtfg_queue_symbol_127;
                      }
                    }
                  }
                }
                break;
              }
              else if (sd[*(mtfg_queue_128 + ((mtfg_queue_position + mtfg_queue_128_offset) & 0x3F))].type & 2)
                cap_queue_position++;
            } while (++mtfg_queue_position != 192);
            if (mtfg_queue_position == 192) {
              while (symbol_number != *(mtfg_queue_192 + ((mtfg_queue_position + mtfg_queue_192_offset) & 0x3F))) {
                if (sd[*(mtfg_queue_192 + ((mtfg_queue_position + mtfg_queue_192_offset) & 0x3F))].type & 2)
                  cap_queue_position++;
                mtfg_queue_position++;
              }
              if (in_definition == 0)
                EncodeMtfgType(LEVEL0_CAP);
              else
                EncodeMtfgType(LEVEL1_CAP);
              uint8_t saved_qp = mtfg_queue_position;
              mtfg_queue_position = cap_queue_position;
              EncodeMtfgQueuePos(CAP, mtfg_queue_position);
              mtfg_queue_position = saved_qp;
              if (sd[mtfg_queue_symbol_15].code_length <= 12) {
                sd[mtfg_queue_symbol_15].type &= 0xF7;
                remove_mtfg_queue_symbol_192(mtfg_queue_position);
              }
              else {
                increment_mtfg_queue_16();
                if (sd[mtfg_queue_symbol_31].code_length == 13) {
                  sd[mtfg_queue_symbol_31].type &= 0xF7;
                  remove_mtfg_queue_symbol_192(mtfg_queue_position);
                }
                else {
                  increment_mtfg_queue_32();
                  if (sd[mtfg_queue_symbol_63].code_length == 14) {
                    sd[mtfg_queue_symbol_63].type &= 0xF7;
                    remove_mtfg_queue_symbol_192(mtfg_queue_position);
                  }
                  else {
                    increment_mtfg_queue_64();
                    if (sd[mtfg_queue_symbol_127].code_length == 15) {
                      sd[mtfg_queue_symbol_127].type &= 0xF7;
                      remove_mtfg_queue_symbol_192(mtfg_queue_position);
                    }
                    else {
                      increment_mtfg_queue_128();
                      if (sd[mtfg_queue_symbol_191].code_length == 16) {
                        sd[mtfg_queue_symbol_191].type &= 0xF7;
                        remove_mtfg_queue_symbol_192(mtfg_queue_position);
                      }
                      else {
                        subqueue_position = mtfg_queue_position - 192;
                        while (subqueue_position) {
                          *(mtfg_queue_192 + ((mtfg_queue_192_offset + subqueue_position) & 0x3F))
                              = *(mtfg_queue_192 + ((mtfg_queue_192_offset + subqueue_position - 1) & 0x3F));
                          subqueue_position--;
                        }
                        *(mtfg_queue_192 + mtfg_queue_192_offset) = mtfg_queue_symbol_191;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return;
}


void encode_dictionary_symbol(uint32_t dsymbol) {
  uint8_t first_char = sd[dsymbol].starts;
  symbol_index = sd[dsymbol].array_index;
  if (cap_encoded != 0) {
    if (prior_end != 0xA) {
      if (sd[prior_symbol].type & 0x20) {
        if (sd[prior_symbol].type & 0x80)
          EncodeFirstChar(first_char, 2, prior_end);
        else if (sd[prior_symbol].type & 0x40)
          EncodeFirstChar(first_char, 3, prior_end);
        else
          EncodeFirstChar(first_char, 1, prior_end);
      }
      else
        EncodeFirstChar(first_char, 0, prior_end);
    }
  }
  else if (UTF8_compliant != 0)
    EncodeFirstChar(first_char, 0, prior_end);
  else 
    EncodeFirstCharBinary(first_char, prior_end);
  if (CodeLength > 12 + nbob_shift[first_char]) {
    uint32_t max_codes_in_bins, mcib;
    uint8_t reduce_bits = 0;
    max_codes_in_bins = nbob[first_char][CodeLength] << (CodeLength - (12 + nbob_shift[first_char]));
    mcib = max_codes_in_bins >> 1;
    while (mcib >= nsob[first_char][CodeLength]) {
      reduce_bits++;
      mcib = mcib >> 1;
    }
    if (CodeLength - reduce_bits > 12 + nbob_shift[first_char]) {
      BinNum = fbob[first_char][CodeLength];
      min_extra_reduce_index = 2 * nsob[first_char][CodeLength] - (max_codes_in_bins >> reduce_bits);
      if (symbol_index >= min_extra_reduce_index) {
        uint16_t symbol_bins = 2;
        BinCode = 2 * symbol_index - min_extra_reduce_index;
        uint16_t code_bin = (uint16_t)(BinCode >> (CodeLength - (12 + nbob_shift[first_char]) - reduce_bits));
        BinNum += code_bin;
        BinCode -= (uint32_t)code_bin << (CodeLength - (12 + nbob_shift[first_char]) - reduce_bits);
        while (BinCode && (sd[sym_list_ptrs[first_char][CodeLength][--symbol_index]].type & 8)) {
          if (symbol_index >= min_extra_reduce_index) {
            symbol_bins += 2;
            BinCode -= 2;
          }
          else {
            symbol_bins++;
            BinCode--;
          }
        }
        CodeLength -= reduce_bits + nbob_shift[first_char];
        EncodeLongDictionarySymbol(BinCode, BinNum, sum_nbob[first_char], CodeLength, symbol_bins);
      }
      else {
        BinCode = symbol_index;
        uint16_t symbol_bins = 1;
        while ((BinCode & ((1 << (CodeLength - (12 + nbob_shift[first_char]) - reduce_bits)) - 1))
            && (sd[sym_list_ptrs[first_char][CodeLength][BinCode - 1]].type & 8)) {
          symbol_bins++;
          BinCode--;
        }
        CodeLength -= reduce_bits + nbob_shift[first_char];
        uint16_t code_bin = (uint16_t)(symbol_index >> (CodeLength - 12));
        BinNum += code_bin;
        BinCode -= (uint32_t)code_bin << (CodeLength - 12);
        EncodeLongDictionarySymbol(BinCode, BinNum, sum_nbob[first_char], CodeLength, symbol_bins);
      }
    }
    else {
      uint16_t symbol_bins = 1;
      while (symbol_index && (sd[sym_list_ptrs[first_char][CodeLength][symbol_index - 1]].type & 8)) {
        symbol_bins++;
        symbol_index--;
      }
      BinNum = fbob[first_char][CodeLength] + symbol_index;
      EncodeShortDictionarySymbol(12, BinNum, sum_nbob[first_char], symbol_bins);
    }
  }
  else {
    uint16_t symbol_bins = 1;
    while (symbol_index && (sd[sym_list_ptrs[first_char][CodeLength][symbol_index - 1]].type & 8)) {
      symbol_bins++;
      symbol_index--;
    }
    BinNum = fbob[first_char][CodeLength] + (symbol_index << (12 + nbob_shift[first_char] - CodeLength));
    EncodeShortDictionarySymbol(CodeLength - nbob_shift[first_char], BinNum, sum_nbob[first_char], symbol_bins);
  }
  return;
}


void update_mtf_queue(uint32_t this_symbol, uint32_t symbol_inst, uint32_t this_symbol_count) {
  uint8_t i1;
  if (symbol_inst != this_symbol_count - 1) { // not the last instance
    if (sd[this_symbol].type & 8) { // symbol in queue
      i1 = mtf_queue_size[this_symbol_count] - 1;
      while (mtf_queue[this_symbol_count][i1] != this_symbol)
        i1--;
      mtf_queue_hit_count[this_symbol_count]++;
      while (i1 < mtf_queue_size[this_symbol_count] - 1) {
        mtf_queue[this_symbol_count][i1] = mtf_queue[this_symbol_count][i1+1];
        i1++;
      }
      mtf_queue[this_symbol_count][i1] = this_symbol;
    }
    else { // symbol not in mtf queue, move it back into the queue
      sd[this_symbol].type |= 8;
      if (mtf_queue_size[this_symbol_count] < MTF_QUEUE_SIZE)
        mtf_queue[this_symbol_count][mtf_queue_size[this_symbol_count]++] = this_symbol;
      else { // move the queue elements down
        sd[mtf_queue[this_symbol_count][0]].type &= 0xF7;
        i1 = 0;
        while (i1 < mtf_queue_size[this_symbol_count] - 1) {
          mtf_queue[this_symbol_count][i1] = mtf_queue[this_symbol_count][i1+1];
          i1++;
        }
        mtf_queue[this_symbol_count][i1] = this_symbol;
      }
    }
  }
  else { // last instance
    mtf_queue_done[this_symbol_count]++;
    if (sd[this_symbol].type & 8) {
      i1 = --mtf_queue_size[this_symbol_count];
      while (mtf_queue[this_symbol_count][i1] != this_symbol)
        i1--;
      mtf_queue_hit_count[this_symbol_count]++;
      while (i1 < mtf_queue_size[this_symbol_count]) {
        mtf_queue[this_symbol_count][i1] = mtf_queue[this_symbol_count][i1+1];
        i1++;
      }
    }
  }
  return;
}


uint8_t add_dictionary_symbol(uint32_t symbol, uint8_t bits) {
  uint8_t first_char = sd[symbol].starts;
  if (nsob[first_char][bits] == ((uint32_t)1 << sym_list_bits[first_char][bits])) {
    sym_list_bits[first_char][bits]++;
    if (0 == (sym_list_ptrs[first_char][bits]
        = (uint32_t *)realloc(sym_list_ptrs[first_char][bits], sizeof(uint32_t) * (1 << sym_list_bits[first_char][bits])))) {
      fprintf(stderr,"FATAL ERROR - symbol list realloc failure\n");
      return(0);
    }
  }
  sd[symbol].array_index = nsob[first_char][bits];
  sym_list_ptrs[first_char][bits][nsob[first_char][bits]++] = symbol;
  if ((nsob[first_char][bits] << (32 - bits)) > (nbob[first_char][bits] << (20 - nbob_shift[first_char]))) {
    if (bits >= 12 + nbob_shift[first_char]) {
      nbob[first_char][bits]++;
      sum_nbob[first_char]++;
      for (temp_bits = bits + 1 ; temp_bits <= max_code_length ; temp_bits++)
        fbob[first_char][temp_bits]++;
    }
    else {
      nbob[first_char][bits] += 1 << (12 + nbob_shift[first_char] - bits);
      sum_nbob[first_char] += 1 << (12 + nbob_shift[first_char] - bits);
      for (temp_bits = bits + 1 ; temp_bits <= max_code_length ; temp_bits++)
        fbob[first_char][temp_bits] += 1 << (12 + nbob_shift[first_char] - bits);
    }
    if (sum_nbob[first_char] > 0x1000) {
      do {
        nbob_shift[first_char]--;
        uint8_t code_length;
        sum_nbob[first_char] = 0;
        for (code_length = 1 ; code_length <= max_code_length ; code_length++)
          sum_nbob[first_char] += (nbob[first_char][code_length] = (nbob[first_char][code_length] + 1) >> 1);
      } while (sum_nbob[first_char] > 0x1000);
      uint16_t bin = nbob[first_char][1];
      for (temp_bits = 2 ; temp_bits <= max_code_length ; temp_bits++) {
        fbob[first_char][temp_bits] = bin;
        bin += nbob[first_char][temp_bits];
      }
    }
  }
  return(1);
}


void remove_dictionary_symbol(uint32_t symbol, uint8_t bits) {
  uint8_t first_char = sd[symbol].starts;
  sym_list_ptrs[first_char][bits][sd[symbol].array_index] = sym_list_ptrs[first_char][bits][--nsob[first_char][bits]];
  sd[sym_list_ptrs[first_char][bits][nsob[first_char][bits]]].array_index = sd[symbol].array_index;
  return;
}


uint8_t manage_mtf_queue(uint32_t this_symbol, uint32_t symbol_inst, uint32_t this_symbol_count, uint8_t in_definition) {
  uint8_t i1, mtf_queue_position;
  mtf_queue_number = (uint8_t)this_symbol_count - 2;
  if (symbol_inst != this_symbol_count - 1) { // not the last instance
    if (sd[this_symbol].type & 8) {
      i1 = mtf_queue_size[this_symbol_count];
      while (i1 != 0) {
        i1--;
        if (mtf_queue[this_symbol_count][i1] == this_symbol) { // return the instance hit code and instance hit code bits
          mtf_queue_position = mtf_queue_size[this_symbol_count] - i1 - 1;
          if (prior_is_cap == 0) {
            if (in_definition == 0)
              EncodeMtfType(LEVEL0);
            else
              EncodeMtfType(LEVEL1);
            EncodeMtfQueueNum(NOT_CAP, mtf_queue_number);
            EncodeMtfQueuePos(NOT_CAP, mtf_queue_number, mtf_queue_size, mtf_queue_position);
          }
          else {
            if (in_definition == 0)
              EncodeMtfType(LEVEL0_CAP);
            else
              EncodeMtfType(LEVEL1_CAP);
            EncodeMtfQueueNum(CAP, mtf_queue_number);
            if (mtf_queue_position) {
              uint32_t *end_mtf_queue_ptr = &mtf_queue[this_symbol_count][mtf_queue_size[this_symbol_count] - 1];
              uint32_t *mtf_queue_ptr = end_mtf_queue_ptr - mtf_queue_position + 1;
              do {
                if ((sd[*mtf_queue_ptr].type & 2) == 0)
                  mtf_queue_position--;
              } while (mtf_queue_ptr++ != end_mtf_queue_ptr);
            }
            EncodeMtfQueuePos(CAP, mtf_queue_number, mtf_queue_size, mtf_queue_position);
          }
          while (i1 < mtf_queue_size[this_symbol_count] - 1) {
            mtf_queue[this_symbol_count][i1] = mtf_queue[this_symbol_count][i1+1];
            i1++;
          }
          mtf_queue[this_symbol_count][i1] = this_symbol;
          prior_is_cap = cap_encoded & sd[this_symbol].type;
          return(1);
        }
      }
    }
    // symbol not in mtf queue, return the symbol code and length
    sd[this_symbol].type |= 8;
    CodeLength = sd[this_symbol].code_length;
    if (prior_is_cap == 0) {
      UpFreqMtfQueueNum(NOT_CAP, mtf_queue_number);
      if (in_definition == 0)
        EncodeDictType(LEVEL0);
      else
        EncodeDictType(LEVEL1);
    }
    else {
      UpFreqMtfQueueNum(CAP, mtf_queue_number);
      if (in_definition == 0)
        EncodeDictType(LEVEL0_CAP);
      else
        EncodeDictType(LEVEL1_CAP);
    }
    encode_dictionary_symbol(this_symbol);
    // move the symbol back into the mtf queue
    symbol_bits = mtf_queue_overflow_code_length[this_symbol_count];
    if (mtf_queue_size[this_symbol_count] < MTF_QUEUE_SIZE) {
      mtf_queue[this_symbol_count][mtf_queue_size[this_symbol_count]++] = this_symbol;
      remove_dictionary_symbol(this_symbol, symbol_bits);
    }
    else {
      sd[mtf_queue[this_symbol_count][0]].type &= 0xF7;
      symbol_to_move = mtf_queue[this_symbol_count][0];
      remove_dictionary_symbol(this_symbol, symbol_bits);
      if (add_dictionary_symbol(symbol_to_move, symbol_bits) == 0)
        return(0);
      // move the queue elements down
      while (i1 < mtf_queue_size[this_symbol_count] - 1) {
        mtf_queue[this_symbol_count][i1] = mtf_queue[this_symbol_count][i1+1];
        i1++;
      }
      mtf_queue[this_symbol_count][i1] = this_symbol;
    }
    prior_is_cap = cap_encoded & sd[this_symbol].type;
  }
  else { // last instance
    // default is to return the symbol code and length if no match found
    if (sd[this_symbol].type & 8) {
      i1 = mtf_queue_size[this_symbol_count];
      while (i1-- != 0) {
        if (mtf_queue[this_symbol_count][i1] == this_symbol) { // return the mtf queue code and length
          mtf_queue_position = mtf_queue_size[this_symbol_count] - i1 - 1;
          if (prior_is_cap == 0) {
            if (in_definition == 0)
              EncodeMtfType(LEVEL0);
            else
              EncodeMtfType(LEVEL1);
            EncodeMtfQueueNumLastSymbol(NOT_CAP, mtf_queue_number);
            EncodeMtfQueuePos(NOT_CAP, mtf_queue_number, mtf_queue_size, mtf_queue_position);
          }
          else {
            if (in_definition == 0)
              EncodeMtfType(LEVEL0_CAP);
            else
              EncodeMtfType(LEVEL1_CAP);
            EncodeMtfQueueNumLastSymbol(CAP, mtf_queue_number);
            if (mtf_queue_position) {
              uint32_t *end_mtf_queue_ptr = &mtf_queue[this_symbol_count][mtf_queue_size[this_symbol_count] - 1];
              uint32_t *mtf_queue_ptr = end_mtf_queue_ptr - mtf_queue_position + 1;
              do {
                if ((sd[*mtf_queue_ptr].type & 2) == 0)
                  mtf_queue_position--;
              } while (mtf_queue_ptr++ != end_mtf_queue_ptr);
            }
            EncodeMtfQueuePos(CAP, mtf_queue_number, mtf_queue_size, mtf_queue_position);
          }
          mtf_queue_size[this_symbol_count]--;
          mtf_queue_ptr = &mtf_queue[this_symbol_count][i1];
          mtf_queue_end_ptr = &mtf_queue[this_symbol_count][mtf_queue_size[this_symbol_count]];
          while (mtf_queue_ptr != mtf_queue_end_ptr)
          {
            *mtf_queue_ptr = *(mtf_queue_ptr+1);
            mtf_queue_ptr++;
          }
          prior_is_cap = cap_encoded & sd[this_symbol].type;
          return(1);
        }
      }
    }
    // symbol not in mtf queue, return the symbol code and length
    CodeLength = sd[this_symbol].code_length;
    if (prior_is_cap == 0) {
      if (in_definition == 0)
        EncodeDictType(LEVEL0);
      else
        EncodeDictType(LEVEL1);
    }
    else {
      if (in_definition == 0)
        EncodeDictType(LEVEL0_CAP);
      else
        EncodeDictType(LEVEL1_CAP);
    }
    encode_dictionary_symbol(this_symbol);
    symbol_bits = mtf_queue_overflow_code_length[this_symbol_count];
    remove_dictionary_symbol(this_symbol, symbol_bits);
    prior_is_cap = cap_encoded & sd[this_symbol].type;
  }
  return(1);
}


void manage_mtf_symbol(uint32_t this_symbol, uint32_t symbol_inst, uint32_t this_symbol_count, uint8_t in_definition) {
  CodeLength = sd[this_symbol].code_length;
  if (prior_is_cap == 0) {
    if (in_definition == 0)
      EncodeDictType(LEVEL0);
    else
      EncodeDictType(LEVEL1);
  }
  else {
    if (in_definition == 0)
      EncodeDictType(LEVEL0_CAP);
    else
      EncodeDictType(LEVEL1_CAP);
  }
  encode_dictionary_symbol(this_symbol);
  prior_is_cap = cap_encoded & sd[this_symbol].type;
  if (symbol_inst == this_symbol_count - 1) { // last instance
    symbol_bits = mtf_queue_overflow_code_length[this_symbol_count];
    remove_dictionary_symbol(this_symbol, symbol_bits);
  }
  return;
}


uint32_t count_symbols(uint32_t this_symbol) {
  uint32_t string_symbols, *symbol_string_ptr, *end_symbol_string_ptr;
  if (this_symbol < start_my_symbols)
    return(1);
  symbol_string_ptr = symbol + sd[this_symbol].define_symbol_start_index;
  end_symbol_string_ptr = symbol + sd[this_symbol+1].define_symbol_start_index - 1;

  string_symbols = 0;
  while (symbol_string_ptr != end_symbol_string_ptr) {
    if ((sd[*symbol_string_ptr].count == 1) && (*symbol_string_ptr >= start_my_symbols))
      string_symbols += count_symbols(*symbol_string_ptr);
    else
      string_symbols++;
    symbol_string_ptr++;
  }
  return(string_symbols);
}


void count_embedded_definition_symbols(uint32_t define_symbol) {
  uint32_t *define_string_ptr, *define_string_end_ptr;
  uint32_t define_symbol_instances, symbol_inst, i1, this_symbol, this_symbol_count;

  if ((sd[define_symbol].count == 1) && (define_symbol >= start_my_symbols)) {
    // count the symbols in the string instead of creating a single instance symbol (artifacts of TreeCompress)
    define_string_ptr = symbol + sd[define_symbol].define_symbol_start_index;
    define_string_end_ptr = symbol + sd[define_symbol+1].define_symbol_start_index - 1;
    do {
      this_symbol = *define_string_ptr++;
      if (define_string_ptr != symbol + sd[define_symbol].define_symbol_start_index) {
        if (sd[prior_symbol].type & 0x20) {
          if (sd[this_symbol].starts == 0x20)
            sd[prior_symbol].space_score += 2;
          else
            sd[prior_symbol].space_score -= 9;
        }
      }
      symbol_inst = sd[this_symbol].inst_found++;
      define_symbol_instances = sd[this_symbol].count;
      if (symbol_inst == 0)
        count_embedded_definition_symbols(this_symbol);
      else if (define_symbol_instances <= MAX_INSTANCES_FOR_MTF_QUEUE) {
        update_mtf_queue(this_symbol, symbol_inst, define_symbol_instances);
        prior_symbol = this_symbol;
      }
      else {
        CodeLength = sd[this_symbol].code_length;
        if (CodeLength >= 11) {
          if (sd[this_symbol].type & 8) {
            manage_mtfg_queue1(this_symbol);
            sd[this_symbol].mtfg_hits++;
          }
          else
            add_symbol_to_mtfg_queue(this_symbol);
        }
        prior_symbol = this_symbol;
      }
    } while (define_string_ptr != define_string_end_ptr);
    define_string_ptr--;
    sd[define_symbol].type |= sd[this_symbol].type & 0x30;
    while (((sd[define_symbol].type & 0x10) == 0)
        && (define_string_ptr-- != symbol + sd[define_symbol].define_symbol_start_index))
      get_symbol_category(*define_string_ptr, &sd[define_symbol].type);
    return;
  }

  // get the symbol code length
  define_symbol_instances = sd[define_symbol].count;
  if (define_symbol_instances != 1) { // calculate the new code
    if (define_symbol_instances <= MAX_INSTANCES_FOR_MTF_QUEUE)
      symbol_code_length = mtf_queue_overflow_code_length[define_symbol_instances];
    else
      symbol_code_length = sd[define_symbol].code_length;
  }

  // count the symbols in the definition
  if (define_symbol >= start_my_symbols) {
    define_string_ptr = symbol + sd[define_symbol].define_symbol_start_index;
    define_string_end_ptr = symbol + sd[define_symbol+1].define_symbol_start_index - 1;
    do {
      this_symbol = *define_string_ptr;
      if (define_string_ptr != symbol + sd[define_symbol].define_symbol_start_index) {
        if (sd[prior_symbol].type & 0x20) {
          if (sd[this_symbol].starts == 0x20)
            sd[prior_symbol].space_score += 2;
          else
            sd[prior_symbol].space_score -= 9;
        }
      }
      define_string_ptr++;
      symbol_inst = sd[this_symbol].inst_found++;
      this_symbol_count = sd[this_symbol].count;
      if (symbol_inst == 0)
        count_embedded_definition_symbols(this_symbol);
      else if (this_symbol_count <= MAX_INSTANCES_FOR_MTF_QUEUE) {
        update_mtf_queue(this_symbol, symbol_inst, this_symbol_count);
        prior_symbol = this_symbol;
      }
      else {
        CodeLength = sd[this_symbol].code_length;
        if (CodeLength >= 11) {
          if (sd[this_symbol].type & 8) {
            manage_mtfg_queue1(this_symbol);
            sd[this_symbol].mtfg_hits++;
          }
          else
            add_symbol_to_mtfg_queue(this_symbol);
        }
        prior_symbol = this_symbol;
      }
    } while (define_string_ptr != define_string_end_ptr);
    define_string_ptr--;
    sd[define_symbol].type |= sd[this_symbol].type & 0x30;
    while (((sd[define_symbol].type & 0x10) == 0)
        && (define_string_ptr-- != symbol + sd[define_symbol].define_symbol_start_index))
      get_symbol_category(*define_string_ptr, &sd[define_symbol].type);
  }
  else if ((define_symbol == (uint32_t)' ') || (define_symbol == (uint32_t)'C') || (define_symbol == (uint32_t)'B'))
    sd[define_symbol].type |= 0x10;
  prior_symbol = define_symbol;

  if (define_symbol_instances != 1) { // assign symbol code
    if (define_symbol_instances <= MAX_INSTANCES_FOR_MTF_QUEUE) { // Handle initial mtf instance
      sd[define_symbol].type |= 8;
      mtf_queue_started[define_symbol_instances]++;
      if (mtf_queue_started[define_symbol_instances] - mtf_queue_done[define_symbol_instances]
          > mtf_queue_peak[define_symbol_instances])
        mtf_queue_peak[define_symbol_instances]++;
      if (mtf_queue_size[define_symbol_instances] < MTF_QUEUE_SIZE)
        mtf_queue[define_symbol_instances][mtf_queue_size[define_symbol_instances]++] = define_symbol;
      else {
        sd[mtf_queue[define_symbol_instances][0]].type &= 0xF7;
        for (i1=0 ; i1<63 ; i1++)
          mtf_queue[define_symbol_instances][i1] = mtf_queue[define_symbol_instances][i1+1];
        mtf_queue[define_symbol_instances][63] = define_symbol;
      }
    }
    else {
      sd[define_symbol].mtfg_hits = 0;
      sd[define_symbol].hit_score = 0;
      CodeLength = sd[define_symbol].code_length;
      if (CodeLength >= 11)
        add_symbol_to_mtfg_queue(define_symbol);
    }
  }
  num_define_symbols_written++;
  return;
}


uint8_t embed_define_binary(uint32_t define_symbol, uint8_t in_definition) {
  uint32_t *define_string_ptr, *this_define_symbol_start_ptr, *define_string_end_ptr;
  uint32_t define_symbol_instances, symbols_in_definition, symbol_inst, i1, this_symbol, this_symbol_count;
  uint8_t new_symbol_code_length;

  if ((sd[define_symbol].count == 1) && (define_symbol >= start_my_symbols)) {
    // write the symbol string instead of creating a single instance symbol (artifacts of TreeCompress)
    rules_reduced++;
    define_string_ptr = symbol + sd[define_symbol].define_symbol_start_index;
    define_string_end_ptr = symbol + sd[define_symbol+1].define_symbol_start_index - 1;
    while (define_string_ptr != define_string_end_ptr) {
      this_symbol = *define_string_ptr++;
      symbol_inst = sd[this_symbol].inst_found++;
      define_symbol_instances = sd[this_symbol].count;
      if (symbol_inst == 0) {
        if (embed_define_binary(this_symbol, in_definition) == 0)
          return(0);
      }
      else if (define_symbol_instances <= MAX_INSTANCES_FOR_MTF_QUEUE) {
        if (use_mtf) {
          if (manage_mtf_queue(this_symbol, symbol_inst, define_symbol_instances, in_definition) == 0)
            return(0);
        }
        else
          manage_mtf_symbol(this_symbol, symbol_inst, define_symbol_instances, in_definition);
      }
      else {
        if (sd[this_symbol].type & 8) {
          if (prior_is_cap == 0)
            manage_mtfg_queue(this_symbol, in_definition);
          else
            manage_mtfg_queue_prior_cap(this_symbol, in_definition);
        }
        else {
          CodeLength = sd[this_symbol].code_length;
          if (in_definition == 0)
            EncodeDictType(LEVEL0);
          else
            EncodeDictType(LEVEL1);
          encode_dictionary_symbol(this_symbol);
          if (sd[this_symbol].type & 4)
            add_symbol_to_mtfg_queue(this_symbol);
        }
      }
      prior_end = sd[this_symbol].ends;
    }
    return(1);
  }

  // write the define code
  if (in_definition == 0)
    EncodeNewType(LEVEL0);
  else
    EncodeNewType(LEVEL1);

  define_symbol_instances = sd[define_symbol].count;
  if (define_symbol_instances != 1)
    new_symbol_code_length = sd[define_symbol].code_length;
  else
    new_symbol_code_length = 0x20;

  // send symbol length, instances and ergodicity bit
  if (define_symbol < start_my_symbols) {
    symbol_lengths[define_symbol] = new_symbol_code_length;
    SIDSymbol = 0;
    EncodeSID(NOT_CAP, SIDSymbol);
    if (define_symbol_instances == 1)
      EncodeINST(NOT_CAP, SIDSymbol, MAX_INSTANCES_FOR_MTF_QUEUE - 1);
    else if (define_symbol_instances <= MAX_INSTANCES_FOR_MTF_QUEUE)
      EncodeINST(NOT_CAP, SIDSymbol, define_symbol_instances - 2);
    else
      EncodeINST(NOT_CAP, SIDSymbol, MAX_INSTANCES_FOR_MTF_QUEUE + max_regular_code_length - new_symbol_code_length);
    EncodeBaseSymbol(define_symbol, 8, 0x100);
    if (define_symbol & 1) {
      if (symbol_lengths[define_symbol - 1]) {
        DoubleRangeDown();
      }
    }
    else if (symbol_lengths[define_symbol + 1])
      DoubleRange();

    uint8_t j1 = 0xFF;
    do {
      InitFirstCharBinBinary(j1, (uint8_t)define_symbol, new_symbol_code_length);
    } while (j1-- != 0);
    InitTrailingCharBinary(define_symbol, symbol_lengths);
    prior_end = define_symbol;

    if (found_first_symbol == 0) {
      found_first_symbol = 1;
      end_symbol = prior_end;
      sym_list_ptrs[end_symbol][max_code_length][0] = end_symbols;
      nsob[end_symbol][max_code_length] = 1;
      nbob[end_symbol][max_code_length] = 1;
      if (max_code_length >= 12) {
        nbob_shift[end_symbol] = max_code_length - 12;
        nbob[end_symbol][max_code_length] = 1;
        sum_nbob[end_symbol] = 1;
      }
      else {
        nbob[end_symbol][max_code_length] = 1 << (12 - max_code_length);
        sum_nbob[end_symbol] = 1 << (12 - max_code_length);
      }
    }
  }
  else {
    num_grammar_rules++;
    this_define_symbol_start_ptr = symbol + sd[define_symbol].define_symbol_start_index;
    define_string_ptr = this_define_symbol_start_ptr;
    define_string_end_ptr = symbol + sd[define_symbol+1].define_symbol_start_index - 1;

    // count the symbols in the definition
    symbols_in_definition = 0;
    while (define_string_ptr != define_string_end_ptr) {
      if ((sd[*define_string_ptr].count != 1) || (*define_string_ptr < start_my_symbols))
        symbols_in_definition++;
      else
        symbols_in_definition += count_symbols(*define_string_ptr);
      define_string_ptr++;
    }
    if (symbols_in_definition < 16) {
      SIDSymbol = symbols_in_definition - 1;;
      EncodeSID(NOT_CAP, SIDSymbol);
    }
    else {
      SIDSymbol = 15;
      EncodeSID(NOT_CAP, SIDSymbol);
      int32_t extra_symbols = symbols_in_definition - 16;
      int32_t temp2 = extra_symbols;
      uint8_t data_bits = 1;
      while (temp2 >= (1 << data_bits))
        temp2 -= (1 << data_bits++);
      temp2 = (int32_t)data_bits;
      while (temp2 > 2) {
        temp2 -= 2;
        EncodeExtraLength(3);
      }
      extra_symbols += 2 - (1 << data_bits);
      if (temp2 == 2)
        EncodeExtraLength(2);
      else
        data_bits++;
      while (data_bits) {
        data_bits -= 2;
        EncodeExtraLength((extra_symbols >> data_bits) & 3);
      }
    }

    if (define_symbol_instances <= MAX_INSTANCES_FOR_MTF_QUEUE)
      EncodeINST(NOT_CAP, SIDSymbol, define_symbol_instances - 2);
    else
      EncodeINST(NOT_CAP, SIDSymbol, MAX_INSTANCES_FOR_MTF_QUEUE + max_regular_code_length - new_symbol_code_length);

    // write the symbol string
    define_string_ptr = this_define_symbol_start_ptr;
    while (define_string_ptr != define_string_end_ptr) {
      this_symbol = *define_string_ptr++;
      symbol_inst = sd[this_symbol].inst_found++;
      this_symbol_count = sd[this_symbol].count;
      if (symbol_inst == 0) {
        if (embed_define_binary(this_symbol, 1) == 0)
          return(0);
      }
      else if (this_symbol_count <= MAX_INSTANCES_FOR_MTF_QUEUE) {
        if (use_mtf) {
          if (manage_mtf_queue(this_symbol, symbol_inst, this_symbol_count, 1) == 0)
            return(0);
        }
        else
          manage_mtf_symbol(this_symbol, symbol_inst, this_symbol_count, 1);
      }
      else {
        CodeLength = sd[this_symbol].code_length;
        if (sd[this_symbol].type & 8) {
          if (prior_is_cap == 0)
            manage_mtfg_queue(this_symbol, 1);
          else
            manage_mtfg_queue_prior_cap(this_symbol, 1);
        }
        else {
          EncodeDictType(LEVEL1);
          encode_dictionary_symbol(this_symbol);
          if (sd[this_symbol].type & 4)
            add_symbol_to_mtfg_queue(this_symbol);
        }
      }
      prior_end = sd[this_symbol].ends;
    }
  }

  if (define_symbol_instances != 1) { // assign symbol code
    if (define_symbol_instances <= MAX_INSTANCES_FOR_MTF_QUEUE) {
      if (use_mtf) {
        mtf_queue_number = define_symbol_instances - 2;
        UpFreqMtfQueueNum(NOT_CAP, mtf_queue_number);
        // Handle initial mtf symbol instance
        sd[define_symbol].type |= 8;
        if (mtf_queue_size[define_symbol_instances] < MTF_QUEUE_SIZE)
          mtf_queue[define_symbol_instances][mtf_queue_size[define_symbol_instances]++] = define_symbol;
        else {
          symbol_to_move = mtf_queue[define_symbol_instances][0];
          sd[symbol_to_move].type &= 0xF7;
          if (add_dictionary_symbol(symbol_to_move, new_symbol_code_length) == 0)
            return(0);
          for (i1=0 ; i1<63 ; i1++)
            mtf_queue[define_symbol_instances][i1] = mtf_queue[define_symbol_instances][i1+1];
          mtf_queue[define_symbol_instances][63] = define_symbol;
        }
      }
      else if (add_dictionary_symbol(define_symbol, new_symbol_code_length) == 0)
        return(0);
    }
    else {
      if (use_mtfg && (new_symbol_code_length >= 11)) {
        if (sd[define_symbol].type & 4) {
          EncodeERG(0, 1);
          add_symbol_to_mtfg_queue(define_symbol);
        }
        else
          EncodeERG(0, 0);
      }
      if (add_dictionary_symbol(define_symbol, new_symbol_code_length) == 0)
        return(0);
    }
  }
  num_define_symbols_written++;
  return (1);
}


uint8_t embed_define(uint32_t define_symbol, uint8_t in_definition) {
  uint32_t *define_string_ptr, *this_define_symbol_start_ptr, *define_string_end_ptr;
  uint32_t define_symbol_instances, symbols_in_definition, symbol_inst, i1, this_symbol, this_symbol_count;
  uint8_t new_symbol_code_length, char_before_define_is_cap;

  if ((sd[define_symbol].count == 1) && (define_symbol >= start_my_symbols)) {
    // write the symbol string instead of creating a single instance symbol (artifacts of TreeCompress)
    rules_reduced++;
    define_string_ptr = symbol + sd[define_symbol].define_symbol_start_index;
    define_string_end_ptr = symbol + sd[define_symbol+1].define_symbol_start_index - 1;
    while (define_string_ptr != define_string_end_ptr) {
      this_symbol = *define_string_ptr++;
      symbol_inst = sd[this_symbol].inst_found++;
      define_symbol_instances = sd[this_symbol].count;
      if (symbol_inst == 0) {
        if (embed_define(this_symbol, in_definition) == 0)
          return(0);
      }
      else if (define_symbol_instances <= MAX_INSTANCES_FOR_MTF_QUEUE) {
        if (use_mtf) {
          if (manage_mtf_queue(this_symbol, symbol_inst, define_symbol_instances, in_definition) == 0)
            return(0);
        }
        else
          manage_mtf_symbol(this_symbol, symbol_inst, define_symbol_instances, in_definition);
        prior_symbol = this_symbol;
      }
      else {
        if (sd[this_symbol].type & 8) {
          if (prior_is_cap == 0)
            manage_mtfg_queue(this_symbol, in_definition);
          else
            manage_mtfg_queue_prior_cap(this_symbol, in_definition);
          prior_is_cap = cap_encoded & sd[this_symbol].type;
        }
        else {
          CodeLength = sd[this_symbol].code_length;
          if (prior_is_cap == 0) {
            if (in_definition == 0)
              EncodeDictType(LEVEL0);
            else
              EncodeDictType(LEVEL1);
            prior_is_cap = cap_encoded & sd[this_symbol].type;
          }
          else {
            if (in_definition == 0)
              EncodeDictType(LEVEL0_CAP);
            else
              EncodeDictType(LEVEL1_CAP);
            prior_is_cap = sd[this_symbol].type & 1;
          }
          encode_dictionary_symbol(this_symbol);
          if (sd[this_symbol].type & 4)
            add_symbol_to_mtfg_queue(this_symbol);
        }
        prior_symbol = this_symbol;
      }
      prior_end = sd[this_symbol].ends;
    }
    if ((sd[define_symbol].type & 0x40) == 0)
      sd[define_symbol].type |= sd[symbol[sd[define_symbol+1].define_symbol_start_index - 2]].type & 0xC0;
    return(1);
  }

  // write the define code
  if (prior_is_cap == 0) {
    if (in_definition == 0)
      EncodeNewType(LEVEL0);
    else
      EncodeNewType(LEVEL1);
  }
  else {
    if (in_definition == 0)
      EncodeNewType(LEVEL0_CAP);
    else
      EncodeNewType(LEVEL1_CAP);
  }

  define_symbol_instances = sd[define_symbol].count;
  if (define_symbol_instances != 1)
    new_symbol_code_length = sd[define_symbol].code_length;
  else
    new_symbol_code_length = 0x20;

  // send symbol length, instances and ergodicity bit
  if (define_symbol < start_my_symbols) {
    SIDSymbol = 0;
    EncodeSID(prior_is_cap, 0);
    if (define_symbol_instances == 1)
      EncodeINST(prior_is_cap, 0, MAX_INSTANCES_FOR_MTF_QUEUE - 1);
    else if (define_symbol_instances <= MAX_INSTANCES_FOR_MTF_QUEUE)
      EncodeINST(prior_is_cap, 0, define_symbol_instances - 2);
    else
      EncodeINST(prior_is_cap, 0, MAX_INSTANCES_FOR_MTF_QUEUE + max_regular_code_length - new_symbol_code_length);
    uint32_t new_symbol = define_symbol;
    if (cap_encoded != 0) {
      if (new_symbol > 'Z')
        new_symbol -= 24;
      else if (new_symbol > 'A')
        new_symbol -= 1;
    }
    EncodeBaseSymbol(new_symbol, base_bits, num_base_symbols);
    if ((UTF8_compliant == 0) || (define_symbol < START_UTF8_2BYTE_SYMBOLS)) {
      if (define_symbol & 1) {
        if (symbol_lengths[define_symbol - 1])
          DoubleRangeDown();
      }
      else if (symbol_lengths[define_symbol + 1])
        DoubleRange();
    }

    if (cap_encoded != 0) {
      if (UTF8_compliant != 0) {
        if (define_symbol < START_UTF8_2BYTE_SYMBOLS) {
          symbol_lengths[define_symbol] = new_symbol_code_length;
          InitBaseSymbolCap(define_symbol, 0x90, new_symbol_code_length, &cap_symbol_defined, &cap_lock_symbol_defined,
              symbol_lengths);
          prior_end = define_symbol;
          if (prior_end == 'B')
            prior_end = 'C';
        }
        else {
          if (define_symbol < 0x250)
            prior_end = 0x80;
          else if (define_symbol < 0x370)
            prior_end = 0x81;
          else if (define_symbol < 0x400)
            prior_end = 0x82;
          else if (define_symbol < 0x530)
            prior_end = 0x83;
          else if (define_symbol < 0x590)
            prior_end = 0x84;
          else if (define_symbol < 0x600)
            prior_end = 0x85;
          else if (define_symbol < 0x700)
            prior_end = 0x86;
          else if (define_symbol < START_UTF8_3BYTE_SYMBOLS)
            prior_end = 0x87;
          else if (define_symbol < 0x1000)
            prior_end = 0x88;
          else if (define_symbol < 0x2000)
            prior_end = 0x89;
          else if (define_symbol < 0x3000)
            prior_end = 0x8A;
          else if (define_symbol < 0x3040)
            prior_end = 0x8B;
          else if (define_symbol < 0x30A0)
            prior_end = 0x8C;
          else if (define_symbol < 0x3100)
            prior_end = 0x8D;
          else if (define_symbol < 0x3200)
            prior_end = 0x8E;
          else if (define_symbol < 0xA000)
            prior_end = 0x8F;
          else if (define_symbol < START_UTF8_4BYTE_SYMBOLS)
            prior_end = 0x8E;
          else
            prior_end = 0x90;
          if (symbol_lengths[prior_end] == 0) {
            symbol_lengths[prior_end] = new_symbol_code_length;
            uint8_t j1 = 0x90;
            do {
              InitFirstCharBin(j1, prior_end, new_symbol_code_length, cap_symbol_defined, cap_lock_symbol_defined);
            } while (--j1 != 'Z');
            j1 = 'A' - 1;
            do {
              InitFirstCharBin(j1, prior_end, new_symbol_code_length, cap_symbol_defined, cap_lock_symbol_defined);
            } while (j1-- != 0);
            j1 = 0x90;
            do {
              InitSymbolFirstChar(prior_end, j1);
              if (symbol_lengths[j1])
                InitTrailingCharBin(prior_end, j1, symbol_lengths[j1]);
              else if ((j1 == 'C') && cap_symbol_defined)
                InitTrailingCharBin(prior_end, 'C', symbol_lengths[j1]);
              else if ((j1 == 'B') && cap_lock_symbol_defined)
                InitTrailingCharBin(prior_end, 'B', symbol_lengths[j1]);
            } while (j1-- != 0);
          }
        }
      }
      else {
        symbol_lengths[define_symbol] = new_symbol_code_length;
        InitBaseSymbolCap(define_symbol, 0xFF, new_symbol_code_length, &cap_symbol_defined, &cap_lock_symbol_defined,
            symbol_lengths);
        prior_end = define_symbol;
      }
    }
    else {
      if (UTF8_compliant != 0) {
        if (define_symbol < START_UTF8_2BYTE_SYMBOLS) {
          symbol_lengths[define_symbol] = new_symbol_code_length;
          uint8_t j1 = 0x90;
          do {
            InitFirstCharBin(j1, (uint8_t)define_symbol, new_symbol_code_length, cap_symbol_defined,
                cap_lock_symbol_defined);
          } while (j1-- != 0);
          j1 = 0x90;
          do {
            InitSymbolFirstChar(define_symbol, j1);
            if (symbol_lengths[j1])
              InitTrailingCharBin((uint8_t)define_symbol, j1, symbol_lengths[j1]);
          } while (j1-- != 0);
          prior_end = define_symbol;
        }
        else {
          if (define_symbol < 0x250)
            prior_end = 0x80;
          else if (define_symbol < 0x370)
            prior_end = 0x81;
          else if (define_symbol < 0x400)
            prior_end = 0x82;
          else if (define_symbol < 0x530)
            prior_end = 0x83;
          else if (define_symbol < 0x590)
            prior_end = 0x84;
          else if (define_symbol < 0x600)
            prior_end = 0x85;
          else if (define_symbol < 0x700)
            prior_end = 0x86;
          else if (define_symbol < START_UTF8_3BYTE_SYMBOLS)
            prior_end = 0x87;
          else if (define_symbol < 0x1000)
            prior_end = 0x88;
          else if (define_symbol < 0x2000)
            prior_end = 0x89;
          else if (define_symbol < 0x3000)
            prior_end = 0x8A;
          else if (define_symbol < 0x3040)
            prior_end = 0x8B;
          else if (define_symbol < 0x30A0)
            prior_end = 0x8C;
          else if (define_symbol < 0x3100)
            prior_end = 0x8D;
          else if (define_symbol < 0x3200)
            prior_end = 0x8E;
          else if (define_symbol < 0xA000)
            prior_end = 0x8F;
          else if (define_symbol < START_UTF8_4BYTE_SYMBOLS)
            prior_end = 0x8E;
          else
            prior_end = 0x90;
          if (symbol_lengths[prior_end] == 0) {
            symbol_lengths[prior_end] = new_symbol_code_length;
            uint8_t j1 = 0x90;
            do {
              InitFirstCharBin(j1, prior_end, new_symbol_code_length, cap_symbol_defined, cap_lock_symbol_defined);
            } while (j1-- != 0);
            j1 = 0x90;
            do {
              InitSymbolFirstChar(prior_end, j1);
              if (symbol_lengths[j1])
                InitTrailingCharBin(prior_end, j1, symbol_lengths[j1]);
            } while (j1-- != 0);
            InitFreqFirstChar(prior_end, prior_end);
          }
        }
      }
      else {
        symbol_lengths[define_symbol] = new_symbol_code_length;
        uint8_t j1 = 0xFF;
        do {
          InitFirstCharBin(j1, (uint8_t)define_symbol, new_symbol_code_length, cap_symbol_defined, cap_lock_symbol_defined);
        } while (j1-- != 0);
        j1 = 0xFF;
        do {
          InitSymbolFirstChar(define_symbol, j1);
          if (symbol_lengths[j1])
            InitTrailingCharBin((uint8_t)define_symbol, j1, symbol_lengths[j1]);
        } while (j1-- != 0);
        prior_end = define_symbol;
      }
    }
    prior_symbol = define_symbol;

    char_before_define_is_cap = prior_is_cap;
    prior_is_cap = cap_encoded & sd[define_symbol].type;
    if (found_first_symbol == 0) {
      found_first_symbol = 1;
      if (prior_end != 0x43)
        end_symbol = prior_end;
      else
        end_symbol = define_symbol;
      sym_list_ptrs[end_symbol][max_code_length][0] = end_symbols;
      nsob[end_symbol][max_code_length] = 1;
      nbob[end_symbol][max_code_length] = 1;
      if (max_code_length >= 12) {
        nbob_shift[end_symbol] = max_code_length - 12;
        nbob[end_symbol][max_code_length] = 1;
        sum_nbob[end_symbol] = 1;
      }
      else {
        nbob[end_symbol][max_code_length] = 1 << (12 - max_code_length);
        sum_nbob[end_symbol] = 1 << (12 - max_code_length);
      }
    }
  }
  else {
    num_grammar_rules++;
    this_define_symbol_start_ptr = symbol + sd[define_symbol].define_symbol_start_index;
    define_string_ptr = this_define_symbol_start_ptr;
    define_string_end_ptr = symbol + sd[define_symbol+1].define_symbol_start_index - 1;

    // count the symbols in the definition
    symbols_in_definition = 0;
    while (define_string_ptr != define_string_end_ptr) {
      if ((sd[*define_string_ptr].count != 1) || (*define_string_ptr < start_my_symbols))
        symbols_in_definition++;
      else
        symbols_in_definition += count_symbols(*define_string_ptr);
      define_string_ptr++;
    }
    if (symbols_in_definition < 16) {
      SIDSymbol = symbols_in_definition - 1;;
      EncodeSID(prior_is_cap, SIDSymbol);
    }
    else {
      SIDSymbol = 15;
      EncodeSID(prior_is_cap, SIDSymbol);
      int32_t extra_symbols = symbols_in_definition - 16;
      int32_t temp2 = extra_symbols;
      uint8_t data_bits = 1;
      while (temp2 >= (1 << data_bits))
        temp2 -= (1 << data_bits++);
      temp2 = (int32_t)data_bits;
      while (temp2 > 2) {
        temp2 -= 2;
        EncodeExtraLength(3);
      }
      extra_symbols += 2 - (1 << data_bits);
      if (temp2 == 2) {
        EncodeExtraLength(2);
      }
      else
        data_bits++;
      while (data_bits) {
        data_bits -= 2;
        EncodeExtraLength((extra_symbols >> data_bits) & 3);
      }
    }

    if (define_symbol_instances <= MAX_INSTANCES_FOR_MTF_QUEUE)
      EncodeINST(prior_is_cap, SIDSymbol, define_symbol_instances - 2);
    else
      EncodeINST(prior_is_cap, SIDSymbol, MAX_INSTANCES_FOR_MTF_QUEUE + max_regular_code_length - new_symbol_code_length);

    char_before_define_is_cap = prior_is_cap;

    // write the symbol string
    define_string_ptr = this_define_symbol_start_ptr;
    while (define_string_ptr != define_string_end_ptr) {
      this_symbol = *define_string_ptr++;
      symbol_inst = sd[this_symbol].inst_found++;
      this_symbol_count = sd[this_symbol].count;
      if (symbol_inst == 0) {
        if (embed_define(this_symbol, 1) == 0)
          return(0);
      }
      else if (this_symbol_count <= MAX_INSTANCES_FOR_MTF_QUEUE) {
        if (use_mtf) {
          if (manage_mtf_queue(this_symbol, symbol_inst, this_symbol_count, 1) == 0)
            return(0);
        }
        else
          manage_mtf_symbol(this_symbol, symbol_inst, this_symbol_count, 1);
        prior_symbol = this_symbol;
      }
      else {
        CodeLength = sd[this_symbol].code_length;
        if (sd[this_symbol].type & 8) {
          if (prior_is_cap == 0)
            manage_mtfg_queue(this_symbol, 1);
          else
            manage_mtfg_queue_prior_cap(this_symbol, 1);
          prior_is_cap = cap_encoded & sd[this_symbol].type;
        }
        else {
          if (prior_is_cap == 0) {
            EncodeDictType(LEVEL1);
            prior_is_cap = cap_encoded & sd[this_symbol].type;
          }
          else {
            EncodeDictType(LEVEL1_CAP);
            prior_is_cap = sd[this_symbol].type & 1;
          }
          encode_dictionary_symbol(this_symbol);
          if (sd[this_symbol].type & 4)
            add_symbol_to_mtfg_queue(this_symbol);
        }
        prior_symbol = this_symbol;
      }
      prior_end = sd[this_symbol].ends;
    }
    prior_symbol = define_symbol;
  }

  if (define_symbol_instances != 1) { // assign symbol code
    uint8_t tag_type = 0;
    if (cap_encoded != 0) {
      if (sd[define_symbol].type & 0x40) {
        if (sd[define_symbol].type & 0x80) {
          tag_type = 2;
          EncodeWordTag(1);
        }
        else {
          tag_type = 1;
          EncodeWordTag(0);
        }
      }
      else if (define_symbol >= start_my_symbols)
        sd[define_symbol].type |= sd[symbol[sd[define_symbol+1].define_symbol_start_index - 2]].type & 0xC0;
    }
    if (define_symbol_instances <= MAX_INSTANCES_FOR_MTF_QUEUE) {
      if (use_mtf) {
        mtf_queue_number = define_symbol_instances - 2;
        if (char_before_define_is_cap == 0) {
          UpFreqMtfQueueNum(NOT_CAP, mtf_queue_number);
        }
        else {
          UpFreqMtfQueueNum(CAP, mtf_queue_number);
        }
        // Handle initial mtf symbol instance
        sd[define_symbol].type |= 8;
        if (mtf_queue_size[define_symbol_instances] < MTF_QUEUE_SIZE)
          mtf_queue[define_symbol_instances][mtf_queue_size[define_symbol_instances]++] = define_symbol;
        else {
          symbol_to_move = mtf_queue[define_symbol_instances][0];
          sd[symbol_to_move].type &= 0xF7;
          if (add_dictionary_symbol(symbol_to_move, new_symbol_code_length) == 0)
            return(0);
          for (i1=0 ; i1<63 ; i1++)
            mtf_queue[define_symbol_instances][i1] = mtf_queue[define_symbol_instances][i1+1];
          mtf_queue[define_symbol_instances][63] = define_symbol;
        }
      }
      else if (add_dictionary_symbol(define_symbol, new_symbol_code_length) == 0)
        return(0);
    }
    else {
      if (use_mtfg && (new_symbol_code_length >= 11)) {
        if (sd[define_symbol].type & 4) {
          EncodeERG(tag_type, 1);
          add_symbol_to_mtfg_queue(define_symbol);
        }
        else
          EncodeERG(tag_type, 0);
      }
      if (add_dictionary_symbol(define_symbol, new_symbol_code_length) == 0)
        return(0);
    }
  }
  num_define_symbols_written++;
  return(1);
}


uint8_t GLZAencode(size_t in_size, uint8_t * inbuf, size_t * outsize_ptr, uint8_t * outbuf, FILE * fd, size_t file_size) {
  const uint8_t INSERT_SYMBOL_CHAR = 0xFE;
  const uint8_t DEFINE_SYMBOL_CHAR = 0xFF;
  const size_t WRITE_SIZE = 0x40000;
  uint8_t this_char, verbose;
  uint32_t i1, i2, num_symbols, num_symbols_defined, num_definitions_to_code, grammar_size;
  uint32_t UTF8_value, max_UTF8_value, this_symbol, this_symbol_count, symbol_inst, prior_inst, next_symbol;
  uint32_t min_ranked_symbols, ranked_symbols_save, num_ranked_symbols, num_regular_definitions;
  uint32_t mtfg_symbols_reduced, mtf_overflow_symbols_to_code;
  uint32_t *end_symbol_ptr;
  uint32_t *ranked_symbols_ptr, *end_ranked_symbols_ptr, *min_ranked_symbols_ptr, *max_ranked_symbols_ptr;
  uint32_t *min_one_instance_ranked_symbols_ptr, *next_sorted_symbol_ptr;
  int32_t remaining_symbols_to_code, remaining_code_space;
  double d_remaining_symbols_to_code, symbol_inst_factor;


  for (i1 = 2 ; i1 <= MAX_INSTANCES_FOR_MTF_QUEUE ; i1++) {
    mtf_queue_started[i1] = 0;
    mtf_queue_done[i1] = 0;
    mtf_queue_peak[i1] = 0;
    mtf_queue_size[i1] = 0;
    mtf_queue_hit_count[i1] = 0;
  }
  mtfg_queue_8_offset = 0;
  mtfg_queue_16_offset = 0;
  mtfg_queue_32_offset = 0;
  mtfg_queue_64_offset = 0;
  mtfg_queue_128_offset = 0;
  mtfg_queue_192_offset = 0;

  verbose = 0;
  use_mtfg = 0;
  use_mtf = 2;
  num_symbols = 0;

  in_char_ptr = inbuf;
  end_char_ptr = inbuf + in_size;
  format = *in_char_ptr++;
  cap_encoded = (format == 1) ? 1 : 0; 
  UTF8_compliant = 1;
  max_UTF8_value = 0x7F;

  // parse the file to determine UTF8_compliant
  while (in_char_ptr != end_char_ptr) {
    if (*in_char_ptr >= INSERT_SYMBOL_CHAR) {
      if (*(in_char_ptr+1) != DEFINE_SYMBOL_CHAR)
        in_char_ptr += 4;
      else {
        UTF8_compliant = 0;
        break;
      }
    }
    else if (*in_char_ptr >= 0x80) {
      if (*in_char_ptr < 0xC0) {
        UTF8_compliant = 0;
        break;
      }
      else if (*in_char_ptr < 0xE0) {
        if ((*(in_char_ptr+1) < 0x80) || (*(in_char_ptr+1) >= 0xC0)) {
          UTF8_compliant = 0;
          break;
        }
        else {
          UTF8_value = 0x40 * (*in_char_ptr & 0x1F) + (*(in_char_ptr+1) & 0x3F);
          if (UTF8_value > max_UTF8_value)
            max_UTF8_value = UTF8_value;
          in_char_ptr += 2;
        }
      }
      else if (*in_char_ptr < 0xF0) {
        if ((*(in_char_ptr+1) < 0x80) || (*(in_char_ptr+1) >= 0xC0)
            || (*(in_char_ptr+2) < 0x80) || (*(in_char_ptr+2) >= 0xC0)) {
          UTF8_compliant = 0;
          break;
        }
        else {
          UTF8_value = 0x1000 * (*in_char_ptr & 0xF) + 0x40 * (*(in_char_ptr+1) & 0x3F) + (*(in_char_ptr+2) & 0x3F);
          if (UTF8_value > max_UTF8_value)
            max_UTF8_value = UTF8_value;
          in_char_ptr += 3;
        }
      }
      else if (*in_char_ptr < 0xF8) {
        if ((*(in_char_ptr+1) < 0x80) || (*(in_char_ptr+1) >= 0xC0) || (*(in_char_ptr+2) < 0x80)
            || (*(in_char_ptr+2) >= 0xC0) || (*(in_char_ptr+3) < 0x80) || (*(in_char_ptr+3) >= 0xC0)) {
          UTF8_compliant = 0;
          break;
        }
        else {
          UTF8_value = 0x40000 * (*in_char_ptr & 0x7) + 0x1000 * (*(in_char_ptr+1) & 0x3F)
              + 0x40 * (*(in_char_ptr+2) & 0x3F) + (*(in_char_ptr+3) & 0x3F);
          if (UTF8_value > max_UTF8_value)
            max_UTF8_value = UTF8_value;
          in_char_ptr += 4;
        }
      }
      else {
        UTF8_compliant = 0;
        break;
      }
    }
    else
      in_char_ptr++;
    num_symbols++;
  }

  in_char_ptr = inbuf + 1;
  end_char_ptr = inbuf + in_size;
  num_symbols_defined = 0;
  rules_reduced = 0;

  symbol = 0;
  if (UTF8_compliant != 0)
    symbol = (uint32_t *)malloc(sizeof(uint32_t) * (num_symbols + 1));
  else
    symbol = (uint32_t *)malloc(sizeof(uint32_t) * (in_size + 1));
  if (symbol == 0) {
    fprintf(stderr,"Symbol memory allocation failed\n");
    return(0);
  }
  symbol_ptr = symbol;

  start_my_symbols = 0x00080000;
  first_define_ptr = 0;

  if (UTF8_compliant != 0) {
    base_bits = 0;
    while (max_UTF8_value >> base_bits)
      base_bits++;
    start_my_symbols = 1 << base_bits;
    num_base_symbols = start_my_symbols;
    if (cap_encoded != 0)
      num_base_symbols -= 24;
    while (in_char_ptr < end_char_ptr) {
      this_char = *in_char_ptr++;
      if (this_char == INSERT_SYMBOL_CHAR) {
        this_symbol = start_my_symbols;
        this_symbol += 0x10000 * (uint32_t)*in_char_ptr++;
        this_symbol += 0x100 * (uint32_t)*in_char_ptr++;
        this_symbol += (uint32_t)*in_char_ptr++;
        *symbol_ptr++ = this_symbol;
      }
      else if (this_char == DEFINE_SYMBOL_CHAR) {
        if (first_define_ptr == 0)
          first_define_ptr = symbol_ptr;
        in_char_ptr += 3;
        *symbol_ptr++ = ((uint32_t)DEFINE_SYMBOL_CHAR << 24) + num_symbols_defined++;
      }
      else if (this_char < START_UTF8_2BYTE_SYMBOLS)
        *symbol_ptr++ = (uint32_t)this_char;
      else {
        if (this_char >= 0xF8) { // not a UTF-8 character
          fprintf(stderr,"ERROR - non UTF-8 character %x\n",(unsigned char)this_char);
          return(0);
        }
        else if (this_char >= 0xF0) { // 4 byte UTF-8 character
          UTF8_value = 0x40000 * (this_char & 7);
          UTF8_value += 0x1000 * (*in_char_ptr++ & 0x3F);
          UTF8_value += 0x40 * (*in_char_ptr++ & 0x3F);
        }
        else if (this_char >= 0xE0) { // 3 byte UTF-8 character
          UTF8_value = 0x1000 * (this_char & 0xF);
          UTF8_value += 0x40 * (*in_char_ptr++ & 0x3F);
        }
        else // 2 byte UTF-8 character
          UTF8_value = 0x40 * (this_char & 0x1F);
        UTF8_value += *in_char_ptr++ & 0x3F;
        *symbol_ptr++ = UTF8_value;
      }
    }
  }
  else {
    base_bits = 8;
    start_my_symbols = 0x100;
    num_base_symbols = 0x100;
    while (in_char_ptr < end_char_ptr) {
      this_char = *in_char_ptr++;
      if (this_char < INSERT_SYMBOL_CHAR)
        *symbol_ptr++ = (uint32_t)this_char;
      else if (*in_char_ptr == DEFINE_SYMBOL_CHAR) {
        *symbol_ptr++ = (uint32_t)this_char;
        in_char_ptr++;
      }
      else if (this_char == INSERT_SYMBOL_CHAR) {
        this_symbol = start_my_symbols;
        this_symbol += 0x10000 * (uint32_t)*in_char_ptr++;
        this_symbol += 0x100 * (uint32_t)*in_char_ptr++;
        this_symbol += (uint32_t)*in_char_ptr++;
        *symbol_ptr++ = this_symbol;
      }
      else {
        if (first_define_ptr == 0)
          first_define_ptr = symbol_ptr;
        in_char_ptr += 3;
        *symbol_ptr++ = ((uint32_t)DEFINE_SYMBOL_CHAR << 24) + num_symbols_defined++;
      }
    }
  }
#ifdef PRINTON
  fprintf(stderr,"cap encoded %u, UTF8 compliant %u\n",(unsigned int)cap_encoded,(unsigned int)UTF8_compliant);
#endif

  if (first_define_ptr == 0)
    first_define_ptr = symbol_ptr;
  grammar_size = symbol_ptr - symbol;

  *symbol_ptr = UNIQUE_CHAR;
  end_symbol_ptr = symbol_ptr;
  num_symbols = end_symbol_ptr - symbol;
  end_symbols = start_my_symbols + num_symbols_defined;
#ifdef PRINTON
  fprintf(stderr,"Read %u symbols including %u definition symbols\n",(unsigned int)num_symbols,
      (unsigned int)num_symbols_defined);
#endif

  if (0 == (sd = (struct symbol_data *)malloc(sizeof(struct symbol_data) * (end_symbols + 1)))) {
    fprintf(stderr,"Symbol data memory allocation failed\n");
    return(0);
  }

  if (0 == (ranked_symbols = (uint32_t *)malloc(sizeof(uint32_t) * end_symbols))) {
    fprintf(stderr,"Ranked symbol array memory allocation failed\n");
    return(0);
  }

  // count the number of instances of each symbol
  for (i1 = 0 ; i1 < end_symbols ; i1++) {
    sd[i1].count = 0;
    sd[i1].inst_found = 0;
  }
  symbol_ptr = symbol;
  num_symbols_defined = 0;
  while (1) {
    if (*symbol_ptr < (uint32_t)DEFINE_SYMBOL_CHAR << 24)
      sd[*symbol_ptr++].count++;
    else if (*symbol_ptr++ != UNIQUE_CHAR) {
      sd[start_my_symbols + num_symbols_defined++].define_symbol_start_index = symbol_ptr - symbol;
    }
    else
      break;
  }
  sd[start_my_symbols + num_symbols_defined].define_symbol_start_index = symbol_ptr - symbol;

  if (cap_encoded != 0) {
    i1 = 0;
    do {
      sd[i1].type = 0;
    } while (++i1 != 0x61);
    do {
      sd[i1].type = 2;
    } while (++i1 != 0x7B);
    do {
      sd[i1].type = 0;
    } while (++i1 != start_my_symbols);
    sd['B'].type = 1;
    sd['C'].type = 1;
    while (i1 < end_symbols) {
      next_symbol = symbol[sd[i1].define_symbol_start_index];
      while (next_symbol > i1)
        next_symbol = symbol[sd[next_symbol].define_symbol_start_index];
      sd[i1].type = sd[next_symbol].type & 2;
      next_symbol = symbol[sd[i1+1].define_symbol_start_index-2];
      while (next_symbol > i1)
        next_symbol = symbol[sd[next_symbol+1].define_symbol_start_index-2];
      sd[i1].type |= sd[next_symbol].type & 1;
      i1++;
    }
  }
  else {
    i1 = 0;
    while (i1 < end_symbols)
      sd[i1++].type = 0;
  }

  ranked_symbols_ptr = ranked_symbols;
  for (i1=0 ; i1<end_symbols ; i1++)
    if (sd[i1].count)
      *ranked_symbols_ptr++ = i1;
  end_ranked_symbols_ptr = ranked_symbols_ptr;
  min_ranked_symbols_ptr = ranked_symbols_ptr;

  // move single instance symbols to the end of the sorted symbols array
  ranked_symbols_ptr = ranked_symbols;
  while (ranked_symbols_ptr < min_ranked_symbols_ptr) {
    if (sd[*ranked_symbols_ptr].count == 1) { // move this symbol to the top of the moved to end 1 instance symbols
      ranked_symbols_save = *ranked_symbols_ptr;
      *ranked_symbols_ptr = *--min_ranked_symbols_ptr;
      *min_ranked_symbols_ptr = ranked_symbols_save;
    }
    else
      ranked_symbols_ptr++;
  }
  min_one_instance_ranked_symbols_ptr = min_ranked_symbols_ptr;

  // sort symbols with 800 or fewer instances by putting them at the end of the sorted symbols array
  for (i1=2 ; i1<801 ; i1++) {
    ranked_symbols_ptr = ranked_symbols;
    while (ranked_symbols_ptr < min_ranked_symbols_ptr) {
      if (sd[*ranked_symbols_ptr].count == i1) {
        ranked_symbols_save = *ranked_symbols_ptr;
        *ranked_symbols_ptr = *--min_ranked_symbols_ptr;
        *min_ranked_symbols_ptr = ranked_symbols_save;
      }
      else
        ranked_symbols_ptr++;
    }
  }

  // sort the remaining symbols by moving the most frequent symbols to the top of the sorted symbols array
  min_ranked_symbols = min_ranked_symbols_ptr - ranked_symbols;
  for (i1=0 ; i1<min_ranked_symbols ; i1++) {
    uint32_t max_symbol_count = 0;
    ranked_symbols_ptr = &ranked_symbols[i1];
    while (ranked_symbols_ptr < min_ranked_symbols_ptr) {
      if (sd[*ranked_symbols_ptr].count > max_symbol_count) {
        max_symbol_count = sd[*ranked_symbols_ptr].count;
        max_ranked_symbols_ptr = ranked_symbols_ptr;
      }
      ranked_symbols_ptr++;
    }
    if (max_symbol_count > 0) {
      ranked_symbols_save = ranked_symbols[i1];
      ranked_symbols[i1] = *max_ranked_symbols_ptr;
      *max_ranked_symbols_ptr = ranked_symbols_save;
    }
  }
  num_ranked_symbols = end_ranked_symbols_ptr - ranked_symbols;
  num_symbols_to_code = num_symbols - (end_ranked_symbols_ptr - ranked_symbols);
  num_definitions_to_code = min_one_instance_ranked_symbols_ptr - ranked_symbols;
  num_define_symbols_written = 0;

  remaining_symbols_to_code = num_symbols_to_code - num_symbols_defined;
  remaining_code_space = 1 << 30;
  remaining_code_space -= 1 << (30 - MAX_BITS_IN_CODE); // reserve space for EOF

  for (i1 = 2 ; i1 < 5 ; i1++)
    mtf_queue_overflow_code_length[i1] = 25;
  for (i1 = 5 ; i1 < 10 ; i1++)
    mtf_queue_overflow_code_length[i1] = 24;
  for (i1 = 10 ; i1 <= MAX_INSTANCES_FOR_MTF_QUEUE ; i1++)
    mtf_queue_overflow_code_length[i1] = 23;

  remaining_code_space += (remaining_code_space >> 5) - 0x20;
  remaining_symbols_to_code += remaining_symbols_to_code >> 5;
  max_regular_code_length = 1;
  num_regular_definitions = 0;

  prior_inst = 0;
  i1 = 0;
  while (i1 < num_definitions_to_code) {
    symbol_inst = sd[ranked_symbols[i1]].count - 1;
    if (symbol_inst != prior_inst) {
      if (symbol_inst < MAX_INSTANCES_FOR_MTF_QUEUE)
        break;
      prior_inst = symbol_inst;
      symbol_inst_factor = (double)0x5A827999 / (double)symbol_inst; /* 0x40000000 * sqrt(2.0) */
    }
    d_remaining_symbols_to_code = (double)remaining_symbols_to_code;
    symbol_code_length = (uint8_t)(log2(d_remaining_symbols_to_code * symbol_inst_factor / (double)remaining_code_space));
    if (symbol_code_length < 2) // limit so files with less than 2 bit symbols (ideally) work
      symbol_code_length = 2;
    num_regular_definitions++;
    if (symbol_code_length > 24)
      symbol_code_length = 24;
    if (symbol_code_length > max_regular_code_length)
      max_regular_code_length = symbol_code_length;
    sd[ranked_symbols[i1]].code_length = symbol_code_length;
    remaining_code_space -= (1 << (30 - symbol_code_length));
    remaining_symbols_to_code -= symbol_inst;
    i1++;
  }

  i1 = num_definitions_to_code;
  while (i1 < num_ranked_symbols)
    sd[ranked_symbols[i1++]].code_length = 0x20;

  mtfg_queue_8_offset = 0;
  mtfg_queue_16_offset = 0;
  mtfg_queue_32_offset = 0;
  mtfg_queue_64_offset = 0;
  mtfg_queue_128_offset = 0;
  mtfg_queue_192_offset = 0;
  for (i1 = 0 ; i1 < 8 ; i1++)
    mtfg_queue_0[i1] = end_symbols;
  for (i1 = 0 ; i1 < 8 ; i1++)
    mtfg_queue_8[i1] = end_symbols;
  for (i1 = 0 ; i1 < 16 ; i1++)
    mtfg_queue_16[i1] = end_symbols;
  for (i1 = 0 ; i1 < 32 ; i1++)
    mtfg_queue_32[i1] = end_symbols;
  for (i1 = 0 ; i1 < 64 ; i1++)
    mtfg_queue_64[i1] = end_symbols;
  for (i1 = 0 ; i1 < 64 ; i1++)
    mtfg_queue_128[i1] = end_symbols;
  for (i1 = 0 ; i1 < 64 ; i1++)
    mtfg_queue_192[i1] = end_symbols;

  symbol_ptr = symbol;

  for (i1 = start_my_symbols ; i1 < end_symbols ; i1++) {
    sd[i1].starts = 0;
    sd[i1].ends = 0;
  }

  if (UTF8_compliant != 0) {
    i1 = 0;
    while (i1 < 0x80) {
      sd[i1].starts = (uint8_t)i1;
      sd[i1].ends = (uint8_t)i1;
      i1++;
    }
    uint32_t temp_UTF8_limit = 0x250;
    if (max_UTF8_value < temp_UTF8_limit)
      temp_UTF8_limit = max_UTF8_value + 1;
    while (i1 < temp_UTF8_limit) {
      sd[i1].starts = 0x80;
      sd[i1].ends = 0x80;
      i1++;
    }
    temp_UTF8_limit = 0x370;
    if (max_UTF8_value < temp_UTF8_limit)
      temp_UTF8_limit = max_UTF8_value + 1;
    while (i1 < temp_UTF8_limit) {
      sd[i1].starts = 0x81;
      sd[i1].ends = 0x81;
      i1++;
    }
    temp_UTF8_limit = 0x400;
    if (max_UTF8_value < temp_UTF8_limit)
      temp_UTF8_limit = max_UTF8_value + 1;
    while (i1 < temp_UTF8_limit) {
      sd[i1].starts = 0x82;
      sd[i1].ends = 0x82;
      i1++;
    }
    temp_UTF8_limit = 0x530;
    if (max_UTF8_value < temp_UTF8_limit)
      temp_UTF8_limit = max_UTF8_value + 1;
    while (i1 < temp_UTF8_limit) {
      sd[i1].starts = 0x83;
      sd[i1].ends = 0x83;
      i1++;
    }
    temp_UTF8_limit = 0x590;
    if (max_UTF8_value < temp_UTF8_limit)
      temp_UTF8_limit = max_UTF8_value + 1;
    while (i1 < temp_UTF8_limit) {
      sd[i1].starts = 0x84;
      sd[i1].ends = 0x84;
      i1++;
    }
    temp_UTF8_limit = 0x600;
    if (max_UTF8_value < temp_UTF8_limit)
      temp_UTF8_limit = max_UTF8_value + 1;
    while (i1 < temp_UTF8_limit) {
      sd[i1].starts = 0x85;
      sd[i1].ends = 0x85;
      i1++;
    }
    temp_UTF8_limit = 0x700;
    if (max_UTF8_value < temp_UTF8_limit)
      temp_UTF8_limit = max_UTF8_value + 1;
    while (i1 < temp_UTF8_limit) {
      sd[i1].starts = 0x86;
      sd[i1].ends = 0x86;
      i1++;
    }
    temp_UTF8_limit = START_UTF8_3BYTE_SYMBOLS;
    if (max_UTF8_value < temp_UTF8_limit)
      temp_UTF8_limit = max_UTF8_value + 1;
    while (i1 < temp_UTF8_limit) {
      sd[i1].starts = 0x87;
      sd[i1].ends = 0x87;
      i1++;
    }
    temp_UTF8_limit = 0x1000;
    if (max_UTF8_value < temp_UTF8_limit)
      temp_UTF8_limit = max_UTF8_value + 1;
    while (i1 < temp_UTF8_limit) {
      sd[i1].starts = 0x88;
      sd[i1].ends = 0x88;
      i1++;
    }
    temp_UTF8_limit = 0x2000;
    if (max_UTF8_value < temp_UTF8_limit)
      temp_UTF8_limit = max_UTF8_value + 1;
    while (i1 < temp_UTF8_limit) {
      sd[i1].starts = 0x89;
      sd[i1].ends = 0x89;
      i1++;
    }
    temp_UTF8_limit = 0x3000;
    if (max_UTF8_value < temp_UTF8_limit)
      temp_UTF8_limit = max_UTF8_value + 1;
    while (i1 < temp_UTF8_limit) {
      sd[i1].starts = 0x8A;
      sd[i1].ends = 0x8A;
      i1++;
    }
    temp_UTF8_limit = 0x3040;
    if (max_UTF8_value < temp_UTF8_limit)
      temp_UTF8_limit = max_UTF8_value + 1;
    while (i1 < temp_UTF8_limit) {
      sd[i1].starts = 0x8B;
      sd[i1].ends = 0x8B;
      i1++;
    }
    temp_UTF8_limit = 0x30A0;
    if (max_UTF8_value < temp_UTF8_limit)
      temp_UTF8_limit = max_UTF8_value + 1;
    while (i1 < temp_UTF8_limit) {
      sd[i1].starts = 0x8C;
      sd[i1].ends = 0x8C;
      i1++;
    }
    temp_UTF8_limit = 0x3100;
    if (max_UTF8_value < temp_UTF8_limit)
      temp_UTF8_limit = max_UTF8_value + 1;
    while (i1 < temp_UTF8_limit) {
      sd[i1].starts = 0x8D;
      sd[i1].ends = 0x8D;
      i1++;
    }
    temp_UTF8_limit = 0x3200;
    if (max_UTF8_value < temp_UTF8_limit)
      temp_UTF8_limit = max_UTF8_value + 1;
    while (i1 < temp_UTF8_limit) {
      sd[i1].starts = 0x8E;
      sd[i1].ends = 0x8E;
      i1++;
    }
    temp_UTF8_limit = 0xA000;
    if (max_UTF8_value < temp_UTF8_limit)
      temp_UTF8_limit = max_UTF8_value + 1;
    while (i1 < temp_UTF8_limit) {
      sd[i1].starts = 0x8F;
      sd[i1].ends = 0x8F;
      i1++;
    }
    temp_UTF8_limit = START_UTF8_4BYTE_SYMBOLS;
    if (max_UTF8_value < temp_UTF8_limit)
      temp_UTF8_limit = max_UTF8_value + 1;
    while (i1 < temp_UTF8_limit) {
      sd[i1].starts = 0x8E;
      sd[i1].ends = 0x8E;
      i1++;
    }
    while (i1 <= max_UTF8_value) {
      sd[i1].starts = 0x90;
      sd[i1].ends = 0x90;
      i1++;
    }
    if (cap_encoded != 0)
      sd['B'].ends = 'C';
    i1 = start_my_symbols;
    while (i1 < end_symbols) {
      if (sd[i1].starts == 0)
        sd[i1].starts = find_first_UTF8(i1);
      if (sd[i1].ends == 0)
        sd[i1].ends = find_last_UTF8(i1);
      i1++;
    }
  }
  else {
    i1 = 0;
    while (i1 < 0x100) {
      sd[i1].starts = (uint8_t)i1;
      sd[i1].ends = (uint8_t)i1;
      i1++;
    }
    if (cap_encoded != 0)
      sd['B'].ends = 'C';
    i1 = start_my_symbols;
    while (i1 < end_symbols) {
      if (sd[i1].starts == 0)
        sd[i1].starts = find_first(i1);
      if (sd[i1].ends == 0)
        sd[i1].ends = find_last(i1);
      i1++;
    }
  }

  for (i1 = 0 ; i1 < end_symbols; i1++)
    sd[i1].space_score = 0;

  this_symbol = (uint32_t)-1;
  while (symbol_ptr < first_define_ptr) {
#ifdef PRINTON
    if (((symbol_ptr - symbol) & 0x3FFFFF) == 0)
      fprintf(stderr,"Parsed %u of %u level 0 symbols\r",
          (unsigned int)(symbol_ptr - symbol),(unsigned int)(first_define_ptr - symbol));
#endif
    this_symbol = *symbol_ptr++;
    if (prior_symbol >= 0) {
      if (sd[prior_symbol].type & 0x20) {
        if (sd[this_symbol].starts == 0x20)
          sd[prior_symbol].space_score += 2;
        else
          sd[prior_symbol].space_score -= 9;
      }
    }
    this_symbol_count = sd[this_symbol].count;
    symbol_inst = sd[this_symbol].inst_found++;
    if (symbol_inst == 0)
      count_embedded_definition_symbols(this_symbol);
    else if (this_symbol_count <= MAX_INSTANCES_FOR_MTF_QUEUE) {
      update_mtf_queue(this_symbol, symbol_inst, this_symbol_count);
      prior_symbol = this_symbol;
    }
    else {
      CodeLength = sd[this_symbol].code_length;
      if (CodeLength >= 11) {
        if (sd[this_symbol].type & 8) {
          manage_mtfg_queue1(this_symbol);
          sd[this_symbol].mtfg_hits++;
        }
        else
          add_symbol_to_mtfg_queue(this_symbol);
      }
      prior_symbol = this_symbol;
    }
  }
#ifdef PRINTON
  fprintf(stderr,"Parsed %u level 0 symbols              \r",(unsigned int)(first_define_ptr - symbol));
#endif

  mtfg_symbols_reduced = 0;
  if (use_mtf == 2) {
    use_mtf = 0;
    double sum_expected_peak = 0.0;
    double sum_actual_peak = 0.0;
    for (i1 = 2 ; i1 <= 15 ; i1++) {
      sum_expected_peak += (double)(i1 - 1) * (double)mtf_queue_started[i1] * (1.0 - (1.0 / (double)(1 << (i1 - 1))));
      sum_actual_peak += (double)(i1 - 1) * (double)mtf_queue_peak[i1];
    }
    double score1, score2;
    score1 = 5.75 * (double)mtf_queue_started[2] / ((double)mtf_queue_peak[2] * (32.9 - log2((double)num_symbols)));
    score2 = sum_expected_peak / sum_actual_peak;
    if (score1 + score2 > 2.08)
      use_mtf = 1;
  }

  if (use_mtf && (max_regular_code_length >= 11)) {
    use_mtfg = 1;
    for (i1 = 0 ; i1 < end_symbols ; i1++) {
      if ((sd[i1].count > MAX_INSTANCES_FOR_MTF_QUEUE) && (sd[i1].code_length >= 11)
          && (sd[i1].hit_score > sd[i1].count * (165 - 3 * max_regular_code_length) / 50)) {
        sd[i1].type |= 4;
        if (sd[i1].count - sd[i1].mtfg_hits <= MAX_INSTANCES_FOR_MTF_QUEUE) {
          mtfg_symbols_reduced += sd[i1].count - MAX_INSTANCES_FOR_MTF_QUEUE - 1;
          sd[i1].count = MAX_INSTANCES_FOR_MTF_QUEUE + 1;
        }
        else {
          mtfg_symbols_reduced += sd[i1].mtfg_hits;
          sd[i1].count -= sd[i1].mtfg_hits;
        }
      }
    }
  }

  // sort symbols with 800 or fewer instances by putting them at the end of the sorted symbols array
  next_sorted_symbol_ptr = ranked_symbols + num_regular_definitions - 1;
  for (i1 = MAX_INSTANCES_FOR_MTF_QUEUE + 1 ; i1 < 801 ; i1++) {
    while ((next_sorted_symbol_ptr > ranked_symbols) && (sd[*next_sorted_symbol_ptr].count == i1))
      next_sorted_symbol_ptr--;
    ranked_symbols_ptr = next_sorted_symbol_ptr - 1;
    while (ranked_symbols_ptr >= ranked_symbols) {
      if (sd[*ranked_symbols_ptr].count == i1) {
        ranked_symbols_save = *ranked_symbols_ptr;
        *ranked_symbols_ptr-- = *next_sorted_symbol_ptr;
        *next_sorted_symbol_ptr-- = ranked_symbols_save;
      }
      else
        ranked_symbols_ptr--;
    }
  }

  for (i1 = 1 ; i1 < num_regular_definitions ; i1++) {
    uint32_t temp_symbol = ranked_symbols[i1];
    uint32_t temp_symbol_count = sd[temp_symbol].count;
    if (temp_symbol_count > sd[ranked_symbols[i1-1]].count) {
      i2 = i1 - 1;
      ranked_symbols[i1] = ranked_symbols[i2];
      while (i2 && (temp_symbol_count > sd[ranked_symbols[i2-1]].count)) {
        ranked_symbols[i2] = ranked_symbols[i2-1];
        i2--;
      }
      ranked_symbols[i2] = temp_symbol;
    }
  }

  mtfg_queue_8_offset = 0;
  mtfg_queue_16_offset = 0;
  mtfg_queue_32_offset = 0;
  mtfg_queue_64_offset = 0;
  mtfg_queue_128_offset = 0;
  mtfg_queue_192_offset = 0;
  for (i1 = 0 ; i1 < 8 ; i1++)
    mtfg_queue_0[i1] = end_symbols;
  for (i1 = 0 ; i1 < 8 ; i1++)
    mtfg_queue_8[i1] = end_symbols;
  for (i1 = 0 ; i1 < 16 ; i1++)
    mtfg_queue_16[i1] = end_symbols;
  for (i1 = 0 ; i1 < 32 ; i1++)
    mtfg_queue_32[i1] = end_symbols;
  for (i1 = 0 ; i1 < 64 ; i1++)
    mtfg_queue_64[i1] = end_symbols;
  for (i1 = 0 ; i1 < 64 ; i1++)
    mtfg_queue_128[i1] = end_symbols;
  for (i1 = 0 ; i1 < 64 ; i1++)
    mtfg_queue_192[i1] = end_symbols;

  if (mtf_queue_peak[2] > MTF_QUEUE_SIZE) {
    if (use_mtf)
      mtf_queue_overflow_code_length[2] = (uint32_t)(0.5 + log2((double)num_symbols_to_code * 2.0 / 3.0
          * (double)(mtf_queue_peak[2] - MTF_QUEUE_SIZE) / (double)(mtf_queue_started[2] - mtf_queue_hit_count[2])));
    else
      mtf_queue_overflow_code_length[2] = (uint32_t)(0.5 + log2((double)num_symbols_to_code * 2.0 / 3.0
          * (double)mtf_queue_peak[2] / (double)mtf_queue_started[2]));
    if (mtf_queue_overflow_code_length[2] > 25)
      mtf_queue_overflow_code_length[2] = 25;
  }
  else if (mtf_queue_peak[2]) {
    mtf_queue_overflow_code_length[2] = (uint32_t)(0.5 + log2((double)num_symbols_to_code * 2.0 / 3.0
        * (double)mtf_queue_peak[2] / (double)mtf_queue_started[2]));
    if (mtf_queue_overflow_code_length[2] > 25)
      mtf_queue_overflow_code_length[2] = 25;
  }
  else
    mtf_queue_overflow_code_length[2] = 25;

  for (i1 = 3 ; i1 <= MAX_INSTANCES_FOR_MTF_QUEUE ; i1++) {
    if (mtf_queue_peak[i1] > MTF_QUEUE_SIZE) {
      if (use_mtf)
        mtf_queue_overflow_code_length[i1] = (uint32_t)(0.5 + log2((double)num_symbols_to_code
            * (double)i1 / (double)(i1 + 1) * (double)(mtf_queue_peak[i1] - MTF_QUEUE_SIZE)
            / (double)(mtf_queue_started[i1] * (i1 - 1) - mtf_queue_hit_count[i1])));
      else
        mtf_queue_overflow_code_length[i1] = (uint32_t)(0.5 + log2((double)num_symbols_to_code
            * (double)i1 / (double)(i1 + 1) * (double)mtf_queue_peak[i1] / (double)(mtf_queue_started[i1] * (i1 - 1))));
      if (mtf_queue_overflow_code_length[i1] > mtf_queue_overflow_code_length[i1 - 1])
        mtf_queue_overflow_code_length[i1] = mtf_queue_overflow_code_length[i1 - 1];
      else if (mtf_queue_overflow_code_length[i1] < mtf_queue_overflow_code_length[i1 - 1] - 1)
        mtf_queue_overflow_code_length[i1] = mtf_queue_overflow_code_length[i1 - 1] - 1;
    }
    else if (mtf_queue_peak[i1] && (use_mtf == 0)) {
      mtf_queue_overflow_code_length[i1] = (uint32_t)(0.5 + log2((double)num_symbols_to_code * (double)i1 / (double)(i1 + 1)
          * (double)mtf_queue_peak[i1] / (double)(mtf_queue_started[i1] * (i1 - 1))));
      if (mtf_queue_overflow_code_length[i1] > mtf_queue_overflow_code_length[i1 - 1])
        mtf_queue_overflow_code_length[i1] = mtf_queue_overflow_code_length[i1 - 1];
      else if (mtf_queue_overflow_code_length[i1] < mtf_queue_overflow_code_length[i1 - 1] - 1)
        mtf_queue_overflow_code_length[i1] = mtf_queue_overflow_code_length[i1 - 1] - 1;
    }
    else
      mtf_queue_overflow_code_length[i1] = mtf_queue_overflow_code_length[i1 - 1];
  }
  max_code_length = mtf_queue_overflow_code_length[2];

  if (use_mtf) {
    mtf_queue_miss_code_space = 0;
    mtf_overflow_symbols_to_code = 0;
    for (i1 = 2 ; i1 <= MAX_INSTANCES_FOR_MTF_QUEUE ; i1++) {
      if (mtf_queue_peak[i1] > MTF_QUEUE_SIZE)
        mtf_queue_miss_code_space += (1 << (30 - mtf_queue_overflow_code_length[i1]))
            * (mtf_queue_peak[i1] - MTF_QUEUE_SIZE);
      mtf_overflow_symbols_to_code += (i1-1) * mtf_queue_started[i1];
      mtf_queue_size[i1] = 0;
    }
  }
  else {
    mtf_queue_miss_code_space = 0;
    mtf_overflow_symbols_to_code = 0;
    for (i1 = 2 ; i1 <= MAX_INSTANCES_FOR_MTF_QUEUE ; i1++) {
      mtf_queue_miss_code_space += (1 << (30 - mtf_queue_overflow_code_length[i1])) * mtf_queue_peak[i1];
      mtf_overflow_symbols_to_code += (i1-1) * mtf_queue_started[i1];
    }
  }

  // Recalculate code lengths knowing how many symbols are needed for 2 - 15 instance symbols that fall out of mtf queues
  num_define_symbols_written = 0;
  remaining_symbols_to_code = num_symbols_to_code - mtf_overflow_symbols_to_code - num_symbols_defined
      - mtfg_symbols_reduced;
  remaining_code_space = 1 << 30;
  remaining_code_space -= 1 << (30 - max_code_length); // reserve space for EOF
  remaining_code_space -= mtf_queue_miss_code_space; // reserve code space for symbols that overflow mtf queues
  remaining_code_space += remaining_code_space >> 5;
  remaining_symbols_to_code += remaining_symbols_to_code >> 5;
  max_regular_code_length = 1;

  prior_inst = 0;
  for (i1=0 ; i1<num_definitions_to_code ; i1++) {
    sd[ranked_symbols[i1]].type &= 0xF7;
    symbol_inst = sd[ranked_symbols[i1]].count;
    if (symbol_inst <= MAX_INSTANCES_FOR_MTF_QUEUE) {
      symbol_code_length = mtf_queue_overflow_code_length[symbol_inst];
      sd[ranked_symbols[i1]].code_length = symbol_code_length;
    }
    else {
      num_regular_definitions--;
      d_remaining_symbols_to_code = (double)remaining_symbols_to_code;
      if (--symbol_inst != prior_inst) {
        prior_inst = symbol_inst;
        symbol_inst_factor = (double)0x5A827999 / (double)symbol_inst; /* 0x40000000 * sqrt(2.0) */
      }
      symbol_code_length
          = (uint8_t)(log2(d_remaining_symbols_to_code * symbol_inst_factor / (double)(remaining_code_space - 0x20)));
      if (symbol_code_length < 2) // limit so files with less than 2 bit symbols (ideally) work
        symbol_code_length = 2;
      else if (i1 && (symbol_code_length < sd[ranked_symbols[0]].code_length))
        symbol_code_length = sd[ranked_symbols[0]].code_length;
      else if (symbol_code_length > max_code_length)
        symbol_code_length = max_code_length;
      while (remaining_code_space - (1 << (30 - symbol_code_length))
          < (int32_t)(num_regular_definitions * (0x40000000 >> (max_code_length - 1))))
        symbol_code_length++;
      if (symbol_code_length > max_regular_code_length)
        max_regular_code_length = symbol_code_length;
      if (symbol_code_length < 11)
        sd[ranked_symbols[i1]].type &= 0xFB;
      sd[ranked_symbols[i1]].code_length = symbol_code_length;
      remaining_code_space -= (1 << (30 - symbol_code_length));
      remaining_symbols_to_code -= symbol_inst;
      if (i1) {
        if (sd[ranked_symbols[i1 - 1]].code_length > symbol_code_length) {
          i2 = i1 - 1;
          while (i2 && (sd[ranked_symbols[i2 - 1]].code_length > symbol_code_length))
            i2--;
          sd[ranked_symbols[i1]].code_length = sd[ranked_symbols[i2]].code_length;
          sd[ranked_symbols[i2]].code_length = symbol_code_length;
        }
      }
    }
  }
  if (verbose != 0) {
    if (verbose == 1) {
      for (i1 = 0 ; i1 < num_definitions_to_code ; i1++) {
        if ((sd[ranked_symbols[i1]].code_length >= 11)
            && (sd[ranked_symbols[i1]].inst_found > MAX_INSTANCES_FOR_MTF_QUEUE))
          printf("%u: #%u %u L%u D%02x: \"",(unsigned int)i1,(unsigned int)sd[ranked_symbols[i1]].inst_found,
              (unsigned int)sd[ranked_symbols[i1]].count,(unsigned int)sd[ranked_symbols[i1]].code_length,
              (unsigned int)sd[ranked_symbols[i1]].type & 0xF4);
        else
          printf("%u: #%u L%u: \"",(unsigned int)i1,(unsigned int)sd[ranked_symbols[i1]].inst_found,
              (unsigned int)sd[ranked_symbols[i1]].code_length);
        print_string(ranked_symbols[i1]);
        printf("\"\n");
      }
    }
    else {
      uint32_t symbol_limit = num_definitions_to_code;
      for (i1 = 0 ; i1 < symbol_limit ; i1++) {
        if (sd[i1].inst_found == 0)
          symbol_limit++;
        else {
          if ((sd[i1].code_length >= 11) && (sd[i1].inst_found > MAX_INSTANCES_FOR_MTF_QUEUE))
            printf("%u: #%u %u L%u D%02x: \"",(unsigned int)i1,(unsigned int)sd[i1].inst_found,
                (unsigned int)sd[i1].count,(unsigned int)sd[i1].code_length,(unsigned int)sd[i1].type & 0xF4);
          else
            printf("%u: #%u L%u: \"",(unsigned int)i1,(unsigned int)sd[i1].inst_found,(unsigned int)sd[i1].code_length);
          print_string(i1);
          printf("\"\n");
        }
      }
    }
  }
  if (num_definitions_to_code == 0) {
    max_regular_code_length = 24;
    sd[ranked_symbols[0]].code_length = 25;
  }
  else if (sd[ranked_symbols[0]].count <= MAX_INSTANCES_FOR_MTF_QUEUE)
    max_regular_code_length = sd[ranked_symbols[0]].code_length - 1;

  sd[end_symbols].type = 0;
  if (max_code_length >= 14) {
    i1 = 0;
    while ((sd[ranked_symbols[i1]].count > MAX_INSTANCES_FOR_MTF_QUEUE) && (i1 < num_definitions_to_code)) {
      if (sd[ranked_symbols[i1]].type & 0x20) {
        if (sd[ranked_symbols[i1]].space_score > 0)
          sd[ranked_symbols[i1]].type |= 0xC0;
        else
          sd[ranked_symbols[i1]].type |= 0x40;
      }
      i1++;
    }
    for (i1 = start_my_symbols ; i1 < end_symbols; i1++) {
      if (sd[i1].type & 0x40) {
        uint32_t last_symbol = symbol[sd[i1 + 1].define_symbol_start_index - 2];
        while (last_symbol >= start_my_symbols) {
          if (sd[last_symbol].type & 0x80) {
            sd[i1].type &= 0x3F;
            break;
          }
          last_symbol = symbol[sd[last_symbol + 1].define_symbol_start_index - 2];
        }
      }
    }
  }
  else {
    for (i1 = 0 ; i1 < num_definitions_to_code ; i1++)
      sd[ranked_symbols[i1]].type &= 0xF;
  }

  if (use_mtfg != 0) {
    if (max_regular_code_length >= 11) {
      for (i1 = 0 ; i1 < end_symbols ; i1++) {
        if ((sd[i1].count > MAX_INSTANCES_FOR_MTF_QUEUE) && (sd[i1].code_length >= 11)
            && (sd[i1].hit_score > sd[i1].count * (165 - 3 * max_regular_code_length) / 50))
          sd[i1].type |= 4;
        else
          sd[i1].type &= 0xFB;
      }
    }
    else {
      use_mtfg = 0;
      for (i1 = 0 ; i1 < end_symbols ; i1++)
        sd[i1].type &= 0xFB;
    }
  }

  for (i1 = 0 ; i1 < end_symbols ; i1++)
    sd[i1].inst_found = 0;
  symbol_ptr = symbol;
  prior_is_cap = 0;
  InitEncoder(max_regular_code_length, 
      MAX_INSTANCES_FOR_MTF_QUEUE + (uint32_t)(max_regular_code_length - sd[ranked_symbols[0]].code_length) + 1,
      cap_encoded, UTF8_compliant, use_mtf, use_mtfg);

  if (fd != 0) {
    if ((outbuf = (uint8_t *)malloc(in_size * 2 + 100000)) == 0) {
      fprintf(stderr,"Encoded data memory allocation failed\n");
      return(0);
    }
  }

  // HEADER:
  // BYTE 0:  4.0 * log2(file_size)
  // BYTE 1:  7=cap_encoded, 6=UTF8_compliant, 5=use_mtf, 4-0=max_code_length-1
  // BYTE 2:  7=M4D, 6=M3D, 5=use_delta, 4-0=min_code_length-1
  // BYTE 3:  7=M7D, 6=M6D, 5=M5D, 4-0=max_code_length-max_regular_code_length
  // BYTE 4:  7=M15D, 6=M14D, 5=M13D, 4=M12D, 3=M11D, 2=M10D, 1=M9D, 0=M8D
  // if UTF8_compliant
  // BYTE 5:  7-5=unused, 4-0=base_bits
  // else if use_delta
  //   if stride <= 4
  // BYTE 5:  7=0, 6-5=unused, 4=two channel, 3=little endian, 2=any endian, 1-0=stride-1
  //   else
  // BYTE 5:  7=1, 6-0=stride

  SetOutBuffer(outbuf);
  WriteOutCharNum(0);
  WriteOutBuffer((uint8_t)(4.0 * log2((double)file_size) + 1.0));
  WriteOutBuffer((cap_encoded << 7) | (UTF8_compliant << 6) | (use_mtf << 5) | (mtf_queue_overflow_code_length[2] - 1));
  this_char = (((format & 0xFE) != 0) << 5) | (sd[ranked_symbols[0]].code_length - 1);
  if (mtf_queue_overflow_code_length[3] != mtf_queue_overflow_code_length[2])
    this_char |= 0x40;
  if (mtf_queue_overflow_code_length[4] != mtf_queue_overflow_code_length[3])
    this_char |= 0x80;
  WriteOutBuffer(this_char);
  i1 = 7;
  do {
    this_char = (this_char << 1) | (mtf_queue_overflow_code_length[i1] != mtf_queue_overflow_code_length[i1-1]);
  } while (--i1 != 4);
  WriteOutBuffer((this_char << 5) | (mtf_queue_overflow_code_length[2] - max_regular_code_length));
  i1 = 15;
  do {
    this_char = (this_char << 1) | (mtf_queue_overflow_code_length[i1] != mtf_queue_overflow_code_length[i1-1]);
  } while (--i1 != 7);
  WriteOutBuffer(this_char);
  i1 = 0xFF;
  if (UTF8_compliant != 0) {
    WriteOutBuffer(base_bits);
    i1 = 0x90;
  }
  else if ((format & 0xFE) != 0) {
    if ((format & 0x80) == 0)
      WriteOutBuffer(((format & 0xF0) >> 2) | (((format & 0xE) >> 1) - 1));
    else
      WriteOutBuffer(format);
  }
  do {
    for (i2 = 2 ; i2 <= max_code_length ; i2++) {
      sym_list_bits[i1][i2] = 2;
      if (0 == (sym_list_ptrs[i1][i2] = (uint32_t *)malloc(sizeof(uint32_t) * 4))) {
        fprintf(stderr,"FATAL ERROR - symbol list malloc failure\n"); \
        return(0);
      }
      nsob[i1][i2] = 0;
      nbob[i1][i2] = 0;
      fbob[i1][i2] = 0;
    }
    sum_nbob[i1] = 0;
    nbob_shift[i1] = max_code_length - 12;
    symbol_lengths[i1] = 0;
  } while (i1--);
  found_first_symbol = 0;
  prior_end = 0;
  num_grammar_rules = 1;

#ifdef PRINTON
  fprintf(stderr,"\nuse_mtf %u, mcl %u mrcl %u \n",
      (unsigned int)use_mtf,(unsigned int)max_code_length,(unsigned int)max_regular_code_length);
#endif

  if ((UTF8_compliant != 0) || (cap_encoded != 0)) {
    cap_symbol_defined = 0;
    cap_lock_symbol_defined = 0;
    while (symbol_ptr < first_define_ptr) {
      this_symbol = *symbol_ptr++;
      this_symbol_count = sd[this_symbol].count;
      symbol_inst = sd[this_symbol].inst_found++;
      if (symbol_inst == 0) {
        if (embed_define(this_symbol, 0) == 0)
          return(0);
      }
      else if (this_symbol_count <= MAX_INSTANCES_FOR_MTF_QUEUE) {
        if (use_mtf) {
          if (manage_mtf_queue(this_symbol, symbol_inst, this_symbol_count, 0) == 0)
            return(0);
        }
        else
          manage_mtf_symbol(this_symbol, symbol_inst, this_symbol_count, 0);
        prior_symbol = this_symbol;
      }
      else {
        if (sd[this_symbol].type & 8) {
          if (prior_is_cap == 0)
            manage_mtfg_queue(this_symbol, 0);
          else
            manage_mtfg_queue_prior_cap(this_symbol, 0);
          prior_is_cap = cap_encoded & sd[this_symbol].type;
        }
        else {
          CodeLength = sd[this_symbol].code_length;
          if (prior_is_cap == 0) {
            EncodeDictType(LEVEL0);
            prior_is_cap = cap_encoded & sd[this_symbol].type;
          }
          else {
            EncodeDictType(LEVEL0_CAP);
            prior_is_cap = sd[this_symbol].type & 1;
          }
          encode_dictionary_symbol(this_symbol);
          if (sd[this_symbol].type & 4)
            add_symbol_to_mtfg_queue(this_symbol);
        }
        prior_symbol = this_symbol;
      }
      prior_end = sd[this_symbol].ends;
#ifdef PRINTON
      if (((symbol_ptr-symbol)&0x1FFFFF) == 0)
        fprintf(stderr,"Encoded %u of %u level 1 symbols\r",
            (unsigned int)(symbol_ptr - symbol),(unsigned int)(first_define_ptr - symbol));
#endif
    }
  }
  else {
    while (symbol_ptr < first_define_ptr) {
      this_symbol = *symbol_ptr++;
      this_symbol_count = sd[this_symbol].count;
      symbol_inst = sd[this_symbol].inst_found++;
      if (symbol_inst == 0) {
        if (embed_define_binary(this_symbol, 0) == 0)
          return(0);
      }
      else if (this_symbol_count <= MAX_INSTANCES_FOR_MTF_QUEUE) {
        if (use_mtf) {
          if (manage_mtf_queue(this_symbol, symbol_inst, this_symbol_count, 0) == 0)
            return(0);
        }
        else
          manage_mtf_symbol(this_symbol, symbol_inst, this_symbol_count, 0);
      }
      else {
        if (sd[this_symbol].type & 8) {
          if (prior_is_cap == 0)
            manage_mtfg_queue(this_symbol, 0);
          else
            manage_mtfg_queue_prior_cap(this_symbol, 0);
        }
        else {
          CodeLength = sd[this_symbol].code_length;
          EncodeDictType(LEVEL0);
          encode_dictionary_symbol(this_symbol);
          if (sd[this_symbol].type & 4)
            add_symbol_to_mtfg_queue(this_symbol);
        }
      }
      prior_end = sd[this_symbol].ends;
#ifdef PRINTON
      if (((symbol_ptr-symbol)&0x1FFFFF) == 0)
        fprintf(stderr,"Encoded %u of %u level 1 symbols\r",
            (unsigned int)(symbol_ptr - symbol),(unsigned int)(first_define_ptr - symbol));
#endif
    }
  }

  // send EOF and flush output
  CodeLength = max_code_length - nbob_shift[end_symbol];
  BinNum = fbob[end_symbol][max_code_length];
  if (prior_is_cap == 0)
    EncodeDictType(LEVEL0);
  else
    EncodeDictType(LEVEL0_CAP);
  if (cap_encoded != 0) {
    if (sd[prior_symbol].type & 0x20) {
      if (sd[prior_symbol].type & 0x80)
        EncodeFirstChar(end_symbol, 2, prior_end);
      else if (sd[prior_symbol].type & 0x40)
        EncodeFirstChar(end_symbol, 3, prior_end);
      else
        EncodeFirstChar(end_symbol, 1, prior_end);
    }
    else
      EncodeFirstChar(end_symbol, 0, prior_end);
  }
  else if (UTF8_compliant != 0)
    EncodeFirstChar(end_symbol, 0, prior_end);
  else
    EncodeFirstCharBinary(end_symbol, prior_end);
  if (max_code_length - nbob_shift[end_symbol] > 12) {
    EncodeLongDictionarySymbol(0, BinNum, sum_nbob[end_symbol], CodeLength, 1);
  }
  else
    EncodeShortDictionarySymbol(CodeLength, BinNum, sum_nbob[end_symbol], 1);
  FinishEncoder();
  grammar_size -= 2 * rules_reduced;
#ifdef PRINTON
  fprintf(stderr,"Encoded %u level 1 symbols             \n",(unsigned int)(symbol_ptr - symbol));
  fprintf(stderr,"Reduced %u grammar rules\n",rules_reduced);
  fprintf(stderr,"%u grammar rules.  Grammar size: %u symbols\n",(unsigned int)num_grammar_rules,(unsigned int)grammar_size);
#endif
  i1 = 0xFF;
  if (UTF8_compliant != 0)
    i1 = 0x90;
  do {
    for (i2 = 2 ; i2 <= max_code_length ; i2++)
      free(sym_list_ptrs[i1][i2]);
  } while (i1--);
  free(symbol);
  free(sd);
  free(ranked_symbols);
  *outsize_ptr = ReadOutCharNum();
  if (fd != 0) {
    size_t writesize = 0;
    while (*outsize_ptr - writesize > WRITE_SIZE) {
      fwrite(outbuf + writesize, 1, WRITE_SIZE, fd);
      writesize += WRITE_SIZE;
      fflush(fd);
    }
    fwrite(outbuf + writesize, 1, *outsize_ptr - writesize, fd);
    fflush(fd);
    free(outbuf);
  }
  return(1);
}
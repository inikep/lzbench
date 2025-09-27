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

// GLZAdecode.c
//   Decodes files created by GLZAencode

#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "GLZA.h"
#include "GLZAmodel.h"

const uint32_t CHARS_TO_WRITE = 0x40000;
const uint32_t MAX_U32_VALUE = 0xFFFFFFFF;
const uint64_t MAX_U64_VALUE = 0xFFFFFFFFFFFFFFFF;

uint64_t *symbol_buffer_write_ptr, *symbol_buffer_end_write_ptr, symbol_buffer[0x800];
uint32_t symbol, dictionary_size, outbuf_index;
static uint32_t num_base_symbols;
uint16_t out_buffers_sent;
static uint16_t sum_nbob[0x100];
static uint16_t queue_size, queue_size_az, queue_size_space, queue_size_other;
uint8_t min_code_length, find_first_symbol;
uint8_t two_threads, prior_type, write_cap_on, write_cap_lock_on, skip_space_on;
uint8_t delta_format, stride;
uint8_t queue_offset_az, queue_offset_space, queue_offset_other;
uint8_t queue_az[0x100], queue_space[0x100], queue_other[0x100], queue_data_free_list[0x100];
uint8_t lookup_bits[0x100][0x1000];
uint8_t out_char0[0x40064], out_char1[0x40064];
uint8_t *out_char_ptr, *start_char_ptr, *end_outbuf, *outbuf, *symbol_strings;
static uint8_t UTF8_compliant, cap_encoded, prior_is_cap, prior_end, use_mtf, max_code_length;
static uint8_t cap_symbol_defined, cap_lock_symbol_defined;
static uint8_t symbol_lengths[0x100], max_regular_code_length, bin_code_length[0x100];
static uint8_t queue_offset, queue[0x100], queue_miss_code_length[15];
atomic_uchar done_parsing, symbol_buffer_owner[2];
FILE * fd;

// type: bit 0: string starts a-z, bit1: nonergodic, bit2: wordness determined, bit3: sent mtf
//       bits 5 - 4:  0: not a word, 1: word, 2: word & likely followed by ' ', 3: word & >= 15 repeats (ending sub)symbol

struct sym_data {
  uint32_t string_index;
  uint32_t string_length;
  union {
    uint32_t four_bytes;
    struct {
      uint8_t type;
      uint8_t repeats;
      uint8_t remaining;
      uint8_t ends;
    } bytes;
  };
};

struct sym_data2 {
  uint32_t string_index;
  uint32_t string_length;
  union {
    uint32_t four_bytes;
    struct {
      uint8_t type;
      uint8_t repeats;
      uint8_t remaining;
      uint8_t ends;
    } bytes;
  };
  uint8_t starts;
  uint8_t code_length;
} *queue_data;

struct bin_data {
  uint32_t nsob;
  uint16_t nbob;
  uint16_t fbob;
  uint32_t sym_list_size;
  struct sym_data * symbol_data;
} bin_data[0x100][27];


struct sym_data * dadd_dictionary_symbol(uint8_t bits, uint8_t first_char) {
  struct bin_data * bin_info = &bin_data[first_char][bits];
  if (bin_info->nsob == bin_info->sym_list_size) {
    bin_info->sym_list_size <<= 1;
    if (0 == (bin_info->symbol_data
          = (struct sym_data *)realloc(bin_info->symbol_data, sizeof(struct sym_data) * bin_info->sym_list_size))) {
      fprintf(stderr, "ERROR - memory allocation failed\n");
      exit(EXIT_FAILURE);
    }
  }
  if ((bin_info->nsob << (32 - bits)) == ((uint32_t)bin_info->nbob << (32 - bin_code_length[first_char]))) {
    if (bits >= bin_code_length[first_char]) { /* add one bin */
      bin_info->nbob++;
      if (sum_nbob[first_char] < 0x1000) {
        sum_nbob[first_char]++;
        if (bits != max_code_length) {
          lookup_bits[first_char][bin_data[first_char][bits + 1].fbob] = bits;
          while (++bits != max_code_length) {
            if (bin_data[first_char][bits].nbob != 0)
              lookup_bits[first_char][bin_data[first_char][bits + 1].fbob] = bits;
            bin_data[first_char][bits].fbob++;
          }
          bin_data[first_char][max_code_length].fbob++;
        }
      }
      else {
        bin_code_length[first_char]--;
        uint16_t first_max_code_length = bin_data[first_char][max_code_length].fbob;
        sum_nbob[first_char]
            = (bin_data[first_char][min_code_length].nbob = (bin_data[first_char][min_code_length].nbob + 1) >> 1);
        for (bits = min_code_length + 1 ; bits <= max_code_length ; bits++) {
          bin_data[first_char][bits].fbob = sum_nbob[first_char];
          sum_nbob[first_char] += (bin_data[first_char][bits].nbob = (bin_data[first_char][bits].nbob + 1) >> 1);
        }
        uint16_t bin = 0;
        for (bits = min_code_length ; bits < max_code_length ; bits++)
          while (bin < bin_data[first_char][bits + 1].fbob)
            lookup_bits[first_char][bin++] = bits;
        while (bin < first_max_code_length)
          lookup_bits[first_char][bin++] = max_code_length;
      }
    }
    else { /* add multiple bins */
      uint32_t new_bins = 1 << (bin_code_length[first_char] - bits);
      if (sum_nbob[first_char] + new_bins <= 0x1000) {
        bin_info->nbob += new_bins;
        sum_nbob[first_char] += new_bins;
        if (bits != max_code_length) {
          uint8_t code_length = max_code_length;
          do {
            bin_data[first_char][code_length--].fbob += new_bins;
            uint16_t bin;
            if (bin_data[first_char][code_length].nbob >= new_bins)
              for (bin = bin_data[first_char][code_length + 1].fbob - new_bins ;
                  bin < bin_data[first_char][code_length + 1].fbob ; bin++)
                lookup_bits[first_char][bin] = code_length;
            else
              for (bin = bin_data[first_char][code_length].fbob + new_bins ;
                  bin < bin_data[first_char][code_length].fbob + new_bins + bin_data[first_char][code_length].nbob ; bin++)
                lookup_bits[first_char][bin] = code_length;
          } while (code_length > bits);
        }
      }
      else if (new_bins <= 0x1000) {
        bin_info->nbob += new_bins;
        uint16_t first_max_code_length = bin_data[first_char][max_code_length].fbob;
        do {
          bin_code_length[first_char]--;
          sum_nbob[first_char]
              = (bin_data[first_char][min_code_length].nbob = (bin_data[first_char][min_code_length].nbob + 1) >> 1);
          for (bits = min_code_length + 1 ; bits <= max_code_length ; bits++)
            sum_nbob[first_char] += (bin_data[first_char][bits].nbob = (bin_data[first_char][bits].nbob + 1) >> 1);
        } while (sum_nbob[first_char] > 0x1000);
        uint16_t bin = bin_data[first_char][min_code_length].nbob;
        for (bits = min_code_length + 1 ; bits <= max_code_length ; bits++) {
          bin_data[first_char][bits].fbob = bin;
          bin += bin_data[first_char][bits].nbob;
        }
        bin = 0;
        for (bits = min_code_length ; bits < max_code_length ; bits++)
          while (bin < bin_data[first_char][bits + 1].fbob)
            lookup_bits[first_char][bin++] = bits;
        while (bin < first_max_code_length)
          lookup_bits[first_char][bin++] = max_code_length;
      }
      else if (sum_nbob[first_char] == 0) {
        uint8_t bin_shift = bin_code_length[first_char] - 12 - bits;
        bin_code_length[first_char] -= bin_shift;
        bin_info->nbob = (new_bins >>= bin_shift);
        sum_nbob[first_char] = new_bins;
        uint16_t bin = 0;
        while (bin < sum_nbob[first_char])
          lookup_bits[first_char][bin++] = bits;
        while (++bits <= max_code_length)
          bin_data[first_char][bits].fbob = sum_nbob[first_char];
      }
      else {
        uint16_t first_max_code_length = bin_data[first_char][max_code_length].fbob;
        uint8_t bin_shift = bin_code_length[first_char] - 11 - bits;
        bin_code_length[first_char] -= bin_shift;
        bin_data[first_char][min_code_length].nbob = ((bin_data[first_char][min_code_length].nbob - 1) >> bin_shift) + 1;
        sum_nbob[first_char] = bin_data[first_char][min_code_length].nbob;
        uint8_t code_length;
        for (code_length = min_code_length + 1 ; code_length <= max_code_length ; code_length++)
          sum_nbob[first_char]
              += bin_data[first_char][code_length].nbob = ((bin_data[first_char][code_length].nbob - 1) >> bin_shift) + 1;
        bin_info->nbob += (new_bins >>= bin_shift);
        sum_nbob[first_char] += new_bins;
        uint16_t bin = 0;
        for (bits = min_code_length + 1 ; bits <= max_code_length ; bits++)
          bin_data[first_char][bits].fbob = (bin += bin_data[first_char][bits - 1].nbob);
        bin = 0;
        for (bits = min_code_length ; bits < max_code_length ; bits++)
          while (bin < bin_data[first_char][bits + 1].fbob)
            lookup_bits[first_char][bin++] = bits;
        while (bin < first_max_code_length)
          lookup_bits[first_char][bin++] = max_code_length;
      }
    }
  }
  return(&bin_info->symbol_data[bin_info->nsob++]);
}


struct sym_data * dadd_single_dictionary_symbol(uint8_t first_char) {
  struct bin_data * bin_info = &bin_data[first_char][max_code_length + 1];
  if (bin_info->nsob == bin_info->sym_list_size) {
    bin_info->sym_list_size <<= 1;
    if (0 == (bin_info->symbol_data
          = (struct sym_data *)realloc(bin_info->symbol_data, sizeof(struct sym_data) * bin_info->sym_list_size))) {
      fprintf(stderr, "ERROR - memory allocation failed\n");
      exit(EXIT_FAILURE);
    }
  }
  return(&bin_info->symbol_data[bin_info->nsob++]);
}


void dremove_dictionary_symbol(struct bin_data * bin_info, uint32_t index) {
  bin_info->symbol_data[index].string_index = bin_info->symbol_data[--bin_info->nsob].string_index;
  bin_info->symbol_data[index].string_length = bin_info->symbol_data[bin_info->nsob].string_length;
  bin_info->symbol_data[index].four_bytes = bin_info->symbol_data[bin_info->nsob].four_bytes;
  return;
}


struct sym_data2 * dadd_symbol_to_queue(struct sym_data *sym_data_ptr, uint8_t code_length, uint8_t first_char) {
  uint8_t queue_data_index = queue_data_free_list[queue_size];
  queue[(uint8_t)--queue_offset] = queue_data_index;
  queue_size++;
  sym_data_ptr->bytes.type |= 8;
  struct sym_data2 * queue_data_ptr = &queue_data[queue_data_index];
  queue_data_ptr->string_index = sym_data_ptr->string_index;
  queue_data_ptr->string_length = sym_data_ptr->string_length;
  queue_data_ptr->four_bytes = sym_data_ptr->four_bytes;
  queue_data_ptr->starts = first_char;
  queue_data_ptr->code_length = code_length;
  return(queue_data_ptr);
}


struct sym_data2 * dadd_symbol_to_queue_cap_encoded(struct sym_data *sym_data_ptr, uint8_t code_length,
    uint8_t first_char) {
  uint8_t queue_data_index = queue_data_free_list[queue_size];
  queue_size++;
  if ((sym_data_ptr->bytes.type & 1) != 0) {
    queue_az[(uint8_t)--queue_offset_az] = queue_data_index;
    queue_size_az++;
  }
  else if (first_char == 0x20) {
    queue_space[(uint8_t)--queue_offset_space] = queue_data_index;
    queue_size_space++;
  }
  else {
    queue_other[(uint8_t)--queue_offset_other] = queue_data_index;
    queue_size_other++;
  }
  sym_data_ptr->bytes.type |= 8;
  struct sym_data2 * queue_data_ptr = &queue_data[queue_data_index];
  queue_data_ptr->string_index = sym_data_ptr->string_index;
  queue_data_ptr->string_length = sym_data_ptr->string_length;
  queue_data_ptr->four_bytes = sym_data_ptr->four_bytes;
  queue_data_ptr->starts = first_char;
  queue_data_ptr->code_length = code_length;
  return(queue_data_ptr);
}


struct sym_data * dupdate_queue(uint8_t queue_position) {
  struct sym_data2 * queue_data_ptr;
  uint8_t queue_data_index = queue[(uint8_t)(queue_position + queue_offset)];
  queue_data_ptr = &queue_data[queue_data_index];
  if ((queue_data_ptr->bytes.remaining < MAX_INSTANCES_FOR_REMOVE) && (--queue_data_ptr->bytes.remaining == 0)) {
    queue_size--;
    queue_data_free_list[queue_size] = queue_data_index;
    if (queue_position <= (queue_size >> 1)) {
      while (queue_position != 0) {
        *(queue + (uint8_t)(queue_offset + queue_position)) = *(queue + (uint8_t)(queue_offset + queue_position - 1));
        queue_position--;
      }
      queue_offset++;
    }
    else {
      while (queue_position != queue_size) {
        *(queue + (uint8_t)(queue_offset + queue_position)) = *(queue + (uint8_t)(queue_offset + queue_position + 1));
        queue_position++;
      }
    }
  }
  else {
    uint16_t context = 6 * queue_data_ptr->bytes.repeats + prior_is_cap + (queue_data_ptr->bytes.type & 1)
        + 3 * ((queue_data_ptr->bytes.type >> 4) == 2);
    if (DecodeGoMtf(context, 1) == 0) {
      queue_size--;
      queue_data_free_list[queue_size] = queue_data_index;
      struct sym_data * dict_data_ptr = dadd_dictionary_symbol(queue_data_ptr->code_length, queue_data_ptr->starts);
      dict_data_ptr->string_index = queue_data_ptr->string_index;
      dict_data_ptr->string_length = queue_data_ptr->string_length;
      dict_data_ptr->four_bytes = queue_data_ptr->four_bytes;
      if (queue_position <= (queue_size >> 1)) {
        queue_position += queue_offset;
        while (queue_position != queue_offset) {
          *(queue + queue_position) = *(queue + (uint8_t)(queue_position - 1));
          queue_position--;
        }
        queue_offset++;
      }
      else {
        queue_position += queue_offset;
        while (queue_position != (uint8_t)(queue_offset + queue_size)) {
          *(queue + queue_position) = *(queue + (uint8_t)(queue_position + 1));
          queue_position++;
        }
      }
    }
    else {
      if (queue_position <= (queue_size >> 1)) {
        queue_position += queue_offset;
        while (queue_position != queue_offset) {
          *(queue + queue_position) = *(queue + (uint8_t)(queue_position - 1));
          queue_position--;
        }
        *(queue + queue_offset) = queue_data_index;
      }
      else {
        queue_position += queue_offset;
        while (queue_position != (uint8_t)(queue_offset + queue_size)) {
          *(queue + queue_position) = *(queue + (uint8_t)(queue_position + 1));
          queue_position++;
        }
        *(queue + --queue_offset) = queue_data_index;
      }
    }
  }
  return((struct sym_data *)queue_data_ptr);
}


struct sym_data * dupdate_az_queue(uint8_t queue_position) {
  struct sym_data2 * queue_data_ptr;
  uint8_t queue_data_index = queue_az[(uint8_t)(queue_position + queue_offset_az)];
  queue_data_ptr = &queue_data[queue_data_index];
  if ((queue_data_ptr->bytes.remaining < MAX_INSTANCES_FOR_REMOVE) && (--queue_data_ptr->bytes.remaining == 0)) {
    queue_size--;
    queue_size_az--;
    queue_data_free_list[queue_size] = queue_data_index;
    if (queue_position <= (queue_size_az >> 1)) {
      while (queue_position != 0) {
        *(queue_az + (uint8_t)(queue_offset_az + queue_position))
            = *(queue_az + (uint8_t)(queue_offset_az + queue_position - 1));
        queue_position--;
      }
      queue_offset_az++;
    }
    else {
      while (queue_position != queue_size_az) {
        *(queue_az + (uint8_t)(queue_offset_az + queue_position))
            = *(queue_az + (uint8_t)(queue_offset_az + queue_position + 1));
        queue_position++;
      }
    }
  }
  else {
    uint16_t context = 6 * queue_data_ptr->bytes.repeats + prior_is_cap + (queue_data_ptr->bytes.type & 1)
        + 3 * ((queue_data_ptr->bytes.type >> 4) == 2);
    if (DecodeGoMtf(context, 1) == 0) {
      queue_size--;
      queue_size_az--;
      queue_data_free_list[queue_size] = queue_data_index;
      struct sym_data * dict_data_ptr = dadd_dictionary_symbol(queue_data_ptr->code_length, queue_data_ptr->starts);
      dict_data_ptr->string_index = queue_data_ptr->string_index;
      dict_data_ptr->string_length = queue_data_ptr->string_length;
      dict_data_ptr->four_bytes = queue_data_ptr->four_bytes;
      if (queue_position <= (queue_size_az >> 1)) {
        queue_position += queue_offset_az;
        while (queue_position != queue_offset_az) {
          *(queue_az + queue_position) = *(queue_az + (uint8_t)(queue_position - 1));
          queue_position--;
        }
        queue_offset_az++;
      }
      else {
        queue_position += queue_offset_az;
        while (queue_position != (uint8_t)(queue_offset_az + queue_size_az)) {
          *(queue_az + queue_position) = *(queue_az + (uint8_t)(queue_position + 1));
          queue_position++;
        }
      }
    }
    else {
      if (queue_position <= (queue_size_az >> 1)) {
        queue_position += queue_offset_az;
        while (queue_position != queue_offset_az) {
          *(queue_az + queue_position) = *(queue_az + (uint8_t)(queue_position - 1));
          queue_position--;
        }
        *(queue_az + queue_offset_az) = queue_data_index;
      }
      else {
        queue_position += queue_offset_az;
        while (queue_position != (uint8_t)(queue_offset_az + queue_size_az)) {
          *(queue_az + queue_position) = *(queue_az + (uint8_t)(queue_position + 1));
          queue_position++;
        }
        *(queue_az + --queue_offset_az) = queue_data_index;
      }
    }
  }
  return((struct sym_data *)queue_data_ptr);
}


struct sym_data * dupdate_space_queue(uint8_t queue_position) {
  struct sym_data2 * queue_data_ptr;
  uint8_t queue_data_index = queue_space[(uint8_t)(queue_position + queue_offset_space)];
  queue_data_ptr = &queue_data[queue_data_index];
  if ((queue_data_ptr->bytes.remaining < MAX_INSTANCES_FOR_REMOVE) && (--queue_data_ptr->bytes.remaining == 0)) {
    queue_size--;
    queue_size_space--;
    queue_data_free_list[queue_size] = queue_data_index;
    if (queue_position <= (queue_size_space >> 1)) {
      while (queue_position != 0) {
        *(queue_space + (uint8_t)(queue_offset_space + queue_position))
            = *(queue_space + (uint8_t)(queue_offset_space + queue_position - 1));
        queue_position--;
      }
      queue_offset_space++;
    }
    else {
      while (queue_position != queue_size_space) {
        *(queue_space + (uint8_t)(queue_offset_space + queue_position))
            = *(queue_space + (uint8_t)(queue_offset_space + queue_position + 1));
        queue_position++;
      }
    }
  }
  else {
    uint16_t context = 6 * queue_data_ptr->bytes.repeats + prior_is_cap + (queue_data_ptr->bytes.type & 1)
        + 3 * ((queue_data_ptr->bytes.type >> 4) == 2);
    if (DecodeGoMtf(context, 1) == 0) {
      queue_size--;
      queue_size_space--;
      queue_data_free_list[queue_size] = queue_data_index;
      struct sym_data * dict_data_ptr = dadd_dictionary_symbol(queue_data_ptr->code_length, queue_data_ptr->starts);
      dict_data_ptr->string_index = queue_data_ptr->string_index;
      dict_data_ptr->string_length = queue_data_ptr->string_length;
      dict_data_ptr->four_bytes = queue_data_ptr->four_bytes;
      if (queue_position <= (queue_size_space >> 1)) {
        queue_position += queue_offset_space;
        while (queue_position != queue_offset_space) {
          *(queue_space + queue_position) = *(queue_space + (uint8_t)(queue_position - 1));
          queue_position--;
        }
        queue_offset_space++;
      }
      else {
        queue_position += queue_offset_space;
        while (queue_position != (uint8_t)(queue_offset_space + queue_size_space)) {
          *(queue_space + queue_position) = *(queue_space + (uint8_t)(queue_position + 1));
          queue_position++;
        }
      }
    }
    else {
      if (queue_position <= (queue_size_space >> 1)) {
        queue_position += queue_offset_space;
        while (queue_position != queue_offset_space) {
          *(queue_space + queue_position) = *(queue_space + (uint8_t)(queue_position - 1));
          queue_position--;
        }
        *(queue_space + queue_offset_space) = queue_data_index;
      }
      else {
        queue_position += queue_offset_space;
        while (queue_position != (uint8_t)(queue_offset_space + queue_size_space)) {
          *(queue_space + queue_position) = *(queue_space + (uint8_t)(queue_position + 1));
          queue_position++;
        }
        *(queue_space + --queue_offset_space) = queue_data_index;
      }
    }
  }
  return((struct sym_data *)queue_data_ptr);
}


struct sym_data * dupdate_other_queue(uint8_t queue_position) {
  struct sym_data2 * queue_data_ptr;
  uint8_t queue_data_index = queue_other[(uint8_t)(queue_position + queue_offset_other)];
  queue_data_ptr = &queue_data[queue_data_index];
  if ((queue_data_ptr->bytes.remaining < MAX_INSTANCES_FOR_REMOVE) && (--queue_data_ptr->bytes.remaining == 0)) {
    queue_size--;
    queue_size_other--;
    queue_data_free_list[queue_size] = queue_data_index;
    if (queue_position <= (queue_size_other >> 1)) {
      while (queue_position != 0) {
        *(queue_other + (uint8_t)(queue_offset_other + queue_position))
            = *(queue_other + (uint8_t)(queue_offset_other + queue_position - 1));
        queue_position--;
      }
      queue_offset_other++;
    }
    else {
      while (queue_position != queue_size_other) {
        *(queue_other + (uint8_t)(queue_offset_other + queue_position))
            = *(queue_other + (uint8_t)(queue_offset_other + queue_position + 1));
        queue_position++;
      }
    }
  }
  else {
    uint16_t context = 6 * queue_data_ptr->bytes.repeats + prior_is_cap + (queue_data_ptr->bytes.type & 1)
        + 3 * ((queue_data_ptr->bytes.type >> 4) == 2);
    if (DecodeGoMtf(context, 1) == 0) {
      queue_size--;
      queue_size_other--;
      queue_data_free_list[queue_size] = queue_data_index;
      struct sym_data * dict_data_ptr = dadd_dictionary_symbol(queue_data_ptr->code_length, queue_data_ptr->starts);
      dict_data_ptr->string_index = queue_data_ptr->string_index;
      dict_data_ptr->string_length = queue_data_ptr->string_length;
      dict_data_ptr->four_bytes = queue_data_ptr->four_bytes;
      if (queue_position <= (queue_size_other >> 1)) {
        queue_position += queue_offset_other;
        while (queue_position != queue_offset_other) {
          *(queue_other + queue_position) = *(queue_other + (uint8_t)(queue_position - 1));
          queue_position--;
        }
        queue_offset_other++;
      }
      else {
        queue_position += queue_offset_other;
        while (queue_position != (uint8_t)(queue_offset_other + queue_size_other)) {
          *(queue_other + queue_position) = *(queue_other + (uint8_t)(queue_position + 1));
          queue_position++;
        }
      }
    }
    else {
      if (queue_position <= (queue_size_other >> 1)) {
        queue_position += queue_offset_other;
        while (queue_position != queue_offset_other) {
          *(queue_other + queue_position) = *(queue_other + (uint8_t)(queue_position - 1));
          queue_position--;
        }
        *(queue_other + queue_offset_other) = queue_data_index;
      }
      else {
        queue_position += queue_offset_other;
        while (queue_position != (uint8_t)(queue_offset_other + queue_size_other)) {
          *(queue_other + queue_position) = *(queue_other + (uint8_t)(queue_position + 1));
          queue_position++;
        }
        *(queue_other + --queue_offset_other) = queue_data_index;
      }
    }
  }
  return((struct sym_data *)queue_data_ptr);
}


uint32_t get_dictionary_symbol(uint16_t bin_num, uint8_t code_length, uint8_t first_char) {
  uint32_t temp_index;
  uint16_t bins_per_symbol, extra_bins, end_extra_index;
  struct bin_data * bin_info = &bin_data[first_char][code_length];
  uint32_t num_symbols = bin_info->nsob;
  uint16_t num_bins = bin_info->nbob;
  uint32_t index = bin_num - bin_info->fbob;
  if (code_length > bin_code_length[first_char]) {
    uint32_t min_extra_reduce_index;
    int8_t index_bits = code_length - bin_code_length[first_char];
    uint32_t shifted_max_symbols = num_bins << (index_bits - 1);
    if (shifted_max_symbols >= num_symbols) {
      shifted_max_symbols >>= 1;
      while (shifted_max_symbols >= num_symbols) {
        shifted_max_symbols >>= 1;
        index_bits--;
      }
      if (--index_bits <= 0) {
        if (index_bits == 0) {
          extra_bins = num_bins - num_symbols;
          if (index >= 2 * extra_bins)
            index -= extra_bins;
          else {
            IncreaseRange(index & 1, 2);
            index >>= 1;
          }
        }
        else {
          bins_per_symbol = num_bins / num_symbols;
          extra_bins = num_bins - num_symbols * bins_per_symbol;
          end_extra_index = extra_bins * (bins_per_symbol + 1);
          if (index >= end_extra_index) {
            temp_index = index - end_extra_index;
            index = temp_index / bins_per_symbol;
            IncreaseRange(temp_index - index * bins_per_symbol, bins_per_symbol);
            index += extra_bins;
          }
          else {
            temp_index = index;
            index = temp_index / ++bins_per_symbol;
            IncreaseRange(temp_index - index * bins_per_symbol, bins_per_symbol);
          }
        }
        return(index);
      }
    }
    min_extra_reduce_index = (num_symbols - shifted_max_symbols) << 1;
    index <<= index_bits;
    uint32_t bin_code = DecodeBinCode(index_bits);
    index += bin_code;
    if (index >= min_extra_reduce_index) {
      index = (index + min_extra_reduce_index) >> 1;
      IncreaseRange(bin_code & 1, 2);
    }
    return(index);
  }
  uint8_t bin_shift = bin_code_length[first_char] - code_length;
  if ((num_symbols << bin_shift) == num_bins) {  // the bins are full
    temp_index = index;
    IncreaseRange(temp_index - ((index >>= bin_shift) << bin_shift), 1 << bin_shift);
    return(index);
  }
  if (num_bins < 2 * num_symbols) {
    extra_bins = num_bins - num_symbols;
    if (index >= 2 * extra_bins) {
      index -= extra_bins;
      return(index);
    }
    IncreaseRange(index & 1, 2);
    return(index >> 1);
  }
  bins_per_symbol = num_bins / num_symbols;
  extra_bins = num_bins - num_symbols * bins_per_symbol;
  end_extra_index = extra_bins * (bins_per_symbol + 1);
  if (index >= end_extra_index) {
    temp_index = index - end_extra_index;
    index = temp_index / bins_per_symbol;
    IncreaseRange(temp_index - index * bins_per_symbol, bins_per_symbol);
    index += extra_bins;
    return(index);
  }
  else {
    temp_index = index;
    index /= ++bins_per_symbol;
    IncreaseRange(temp_index - index * bins_per_symbol, bins_per_symbol);
  }
  return(index);
}


uint32_t get_extra_length() {
  uint8_t extras = 0;
  uint32_t SymsInDef;
  uint8_t code;
  do {
    extras++;
    code = DecodeExtraLength();
  } while (code == 3);
  if (code == 2) {
    extras++;
    SymsInDef = 1;
  }
  else
    SymsInDef = 2 + code;
  while (--extras != 0)
    SymsInDef = (SymsInDef << 2) + DecodeExtraLength();
  return(SymsInDef + 14);
}


void delta_transform(uint8_t * buffer, uint32_t len) {
  uint8_t * char_ptr = buffer;
  if (out_buffers_sent == 0) {
    if (stride > 4) {
      char_ptr = buffer + 1;
      do {
        *char_ptr += *(char_ptr - 1);
      } while (++char_ptr < buffer + stride);
    }
    char_ptr = buffer + stride;
    len -= stride;
  }
  if (stride == 1) {
    while (len-- != 0) {
      *char_ptr += *(char_ptr - 1);
      char_ptr++;
    }
  }
  else if (stride == 2) {
    while (len-- != 0) {
      if ((delta_format & 4) == 0) {
        *char_ptr += *(char_ptr - 2);
        char_ptr++;
      }
      else {
        char_ptr++;
        if (((char_ptr - buffer) & 1) == 0) {
          if ((delta_format & 8) == 0) {
            uint32_t value = (*(char_ptr - 4) << 8) + *(char_ptr - 3) + (*(char_ptr - 2) << 8) + *(char_ptr - 1) - 0x80;
            *(char_ptr - 2) = (value >> 8) & 0xFF;
            *(char_ptr - 1) = value & 0xFF;
          }
          else {
            uint32_t value = (*(char_ptr - 3) << 8) + *(char_ptr - 4) + (*(char_ptr - 1) << 8) + *(char_ptr - 2) - 0x80;
            *(char_ptr - 1) = (value >> 8) & 0xFF;
            *(char_ptr - 2) = value & 0xFF;
          }
        }
      }
    }
  }
  else if (stride == 3) {
    while (len-- != 0) {
      *char_ptr += *(char_ptr - 3);
      char_ptr++;
    }
  }
  else if (stride == 4) {
    while (len-- != 0) {
      char_ptr++;
      if ((delta_format & 4) == 0) {
        *(char_ptr - 1) += *(char_ptr - 5);
      }
      else if ((delta_format & 0x10) != 0) {
        if (((char_ptr - buffer) & 1) == 0) {
          if ((delta_format & 8) == 0) {
            uint32_t value = (*(char_ptr - 6) << 8) + *(char_ptr - 5) + (*(char_ptr - 2) << 8) + *(char_ptr - 1) - 0x80;
            *(char_ptr - 2) = (value >> 8) & 0xFF;
            *(char_ptr - 1) = value & 0xFF;
          }
          else {
            uint32_t value = (*(char_ptr - 5) << 8) + *(char_ptr - 6) + (*(char_ptr - 1) << 8) + *(char_ptr - 2) - 0x80;
            *(char_ptr - 1) = (value >> 8) & 0xFF;
            *(char_ptr - 2) = value & 0xFF;
          }
        }
      }
      else {
        if (((char_ptr - buffer) & 3) == 0) {
          if ((delta_format & 8) == 0) {
            uint32_t value = (*(char_ptr - 8) << 24) + (*(char_ptr - 7) << 16) + (*(char_ptr - 6) << 8) + *(char_ptr - 5)
                + (*(char_ptr - 4) << 24) + (*(char_ptr - 3) << 16) + (*(char_ptr - 2) << 8) + *(char_ptr - 1) - 0x808080;
            *(char_ptr - 4) = value >> 24;
            *(char_ptr - 3) = (value >> 16) & 0xFF;
            *(char_ptr - 2) = (value >> 8) & 0xFF;
            *(char_ptr - 1) = value & 0xFF;
          }
          else {
            uint32_t value = (*(char_ptr - 5) << 24) + (*(char_ptr - 6) << 16) + (*(char_ptr - 7) << 8) + *(char_ptr - 8)
                + (*(char_ptr - 1) << 24) + (*(char_ptr - 2) << 16) + (*(char_ptr - 3) << 8) + *(char_ptr - 4) - 0x808080;
            *(char_ptr - 1) = value >> 24;
            *(char_ptr - 2) = (value >> 16) & 0xFF;
            *(char_ptr - 3) = (value >> 8) & 0xFF;
            *(char_ptr - 4) = value & 0xFF;
          }
        }
      }
    }
  }
  else {
    while (len-- != 0) {
      *char_ptr += *(char_ptr - stride);
      char_ptr++;
    }
  }
}


uint8_t create_extended_UTF8_symbol(uint32_t base_symbol, uint32_t * string_index_ptr) {
  if (base_symbol < START_UTF8_3BYTE_SYMBOLS) {
    symbol_strings[(*string_index_ptr)++] = (uint8_t)(base_symbol >> 6) + 0xC0;
    symbol_strings[(*string_index_ptr)++] = (uint8_t)(base_symbol & 0x3F) + 0x80;
    if (base_symbol < 0x250)
      return(0x80);
    else if (base_symbol < 0x370)
      return(0x81);
    else if (base_symbol < 0x400)
      return(0x82);
    else if (base_symbol < 0x530)
      return(0x83);
    else if (base_symbol < 0x590)
      return(0x84);
    else if (base_symbol < 0x600)
      return(0x85);
    else if (base_symbol < 0x700)
      return(0x86);
    else
      return(0x87);
  }
  else if (base_symbol < START_UTF8_4BYTE_SYMBOLS) {
    symbol_strings[(*string_index_ptr)++] = (uint8_t)(base_symbol >> 12) + 0xE0;
    symbol_strings[(*string_index_ptr)++] = (uint8_t)((base_symbol >> 6) & 0x3F) + 0x80;
    symbol_strings[(*string_index_ptr)++] = (uint8_t)(base_symbol & 0x3F) + 0x80;
    if (base_symbol < 0x1000)
      return(0x88);
    else if (base_symbol < 0x2000)
      return(0x89);
    else if (base_symbol < 0x3000)
      return(0x8A);
    else if (base_symbol < 0x3040)
      return(0x8B);
    else if (base_symbol < 0x30A0)
      return(0x8C);
    else if (base_symbol < 0x3100)
      return(0x8D);
    else if (base_symbol < 0x3200)
      return(0x8E);
    else if (base_symbol < 0xA000)
      return(0x8F);
    else
      return(0x8E);
  }
  else {
    symbol_strings[(*string_index_ptr)++] = (uint8_t)(base_symbol >> 18) + 0xF0;
    symbol_strings[(*string_index_ptr)++] = (uint8_t)((base_symbol >> 12) & 0x3F) + 0x80;
    symbol_strings[(*string_index_ptr)++] = (uint8_t)((base_symbol >> 6) & 0x3F) + 0x80;
    symbol_strings[(*string_index_ptr)++] = (uint8_t)(base_symbol & 0x3F) + 0x80;
    return(0x90);
  }
}


uint8_t get_first_char(uint32_t index) {
  uint32_t UTF8_chars = (symbol_strings[index] << 8) + symbol_strings[index + 1];
  if (UTF8_chars < 0xE000) {
    if (UTF8_chars < 0xC990)
      return(0x80);
    else if (UTF8_chars < 0xCDB0)
      return(0x81);
    else if (UTF8_chars < 0xD000)
      return(0x82);
    else if (UTF8_chars < 0xD4B0)
      return(0x83);
    else if (UTF8_chars < 0xD690)
      return(0x84);
    else if (UTF8_chars < 0xD800)
      return(0x85);
    else if (UTF8_chars < 0xDC00)
      return(0x86);
    else
      return(0x87);
  }
  else if (UTF8_chars < 0xE100)
    return(0x88);
  else if (UTF8_chars < 0xE200)
    return(0x89);
  else if (UTF8_chars < 0xE300)
    return(0x8A);
  else if (UTF8_chars < 0xE381)
    return(0x8B);
  else {
    UTF8_chars = (UTF8_chars << 8) + symbol_strings[index + 2];
    if (UTF8_chars < 0xE382A0)
      return(0x8C);
    else if (UTF8_chars < 0xE38400)
      return(0x8D);
    else if (UTF8_chars < 0xE38800)
      return(0x8E);
    else if (UTF8_chars < 0xEA0000)
      return(0x8F);
    else if (UTF8_chars < 0xF00000)
      return(0x8E);
    else
      return(0x90);
  }
}


struct sym_data * decode_new(uint32_t * string_index_ptr) {
  uint8_t SID_symbol, sym_type, first_char;
  uint8_t put_in_mtf = 0;
  uint32_t symbols_in_definition, end_string_index;
  struct sym_data * sym_data_ptr;
  struct sym_data2 temp_sym_data;

  temp_sym_data.string_index = *string_index_ptr;
  end_string_index = *string_index_ptr;
  SID_symbol = DecodeSID(NOT_CAP);
  if (SID_symbol == 0) {
    temp_sym_data.bytes.repeats = DecodeINST(NOT_CAP, SID_symbol);
    if (temp_sym_data.bytes.repeats < MAX_INSTANCES_FOR_REMOVE - 1)
      temp_sym_data.code_length = queue_miss_code_length[++temp_sym_data.bytes.repeats];
    else if (temp_sym_data.bytes.repeats >= MAX_INSTANCES_FOR_REMOVE) {
      temp_sym_data.code_length = max_regular_code_length + MAX_INSTANCES_FOR_REMOVE - temp_sym_data.bytes.repeats;
      temp_sym_data.bytes.repeats = temp_sym_data.code_length + MAX_INSTANCES_FOR_REMOVE - 1;
    }
    else {
      temp_sym_data.bytes.repeats = 0;
      temp_sym_data.code_length = max_code_length + 1;
    }
    uint32_t base_symbol = DecodeBaseSymbol(num_base_symbols);
    if ((UTF8_compliant == 0) || (base_symbol < START_UTF8_2BYTE_SYMBOLS)) {
      if ((base_symbol & 1) != 0) {
        if (symbol_lengths[base_symbol] != 0) {
          base_symbol--;
          DoubleRangeDown();
        }
        else if (symbol_lengths[base_symbol - 1] != 0)
          DoubleRangeDown();
      }
      else if (symbol_lengths[base_symbol] != 0) {
        base_symbol++;
        DoubleRange();
      }
      else if (symbol_lengths[base_symbol + 1] != 0)
        DoubleRange();
    }
    temp_sym_data.bytes.type = 0;

    if (UTF8_compliant != 0) {
      if (base_symbol < START_UTF8_2BYTE_SYMBOLS) {
        symbol_strings[end_string_index++] = prior_end = (uint8_t)base_symbol;
        temp_sym_data.string_length = 1;
      }
      else {
        prior_end = create_extended_UTF8_symbol(base_symbol, &end_string_index);
        temp_sym_data.string_length = end_string_index - temp_sym_data.string_index;
      }
      if (symbol_lengths[prior_end] == 0) {
        symbol_lengths[prior_end] = temp_sym_data.code_length;
        uint8_t j1 = 0x90;
        do {
          InitFirstCharBin(j1, prior_end, temp_sym_data.code_length, cap_symbol_defined, cap_lock_symbol_defined);
        } while (j1-- != 0);
        j1 = 0x90;
        do {
          if (symbol_lengths[j1] != 0)
            InitTrailingCharBin(prior_end, j1, symbol_lengths[j1]);
        } while (j1-- != 0);
      }
    }
    else {
      symbol_strings[end_string_index++] = prior_end = (uint8_t)base_symbol;
      temp_sym_data.string_length = 1;
      symbol_lengths[prior_end] = temp_sym_data.code_length;
      uint8_t j1 = 0xFF;
      do {
        InitFirstCharBinBinary(j1, prior_end, temp_sym_data.code_length);
      } while (j1-- != 0);
      InitTrailingCharBinary(prior_end, symbol_lengths);
    }
    temp_sym_data.starts = temp_sym_data.bytes.ends = prior_end;

    if (find_first_symbol != 0) {
      find_first_symbol = 0;
      sum_nbob[prior_end] = bin_data[prior_end][max_code_length].nbob = 1;
    }
    if (temp_sym_data.bytes.repeats == 0) {
      sym_data_ptr = dadd_single_dictionary_symbol(temp_sym_data.starts);
      sym_data_ptr->bytes.type = 0;
      sym_data_ptr->string_index = temp_sym_data.string_index;
      sym_data_ptr->string_length = temp_sym_data.string_length;
      *string_index_ptr = end_string_index;
      return(sym_data_ptr);
    }

    temp_sym_data.bytes.remaining = temp_sym_data.bytes.repeats;
    if (temp_sym_data.bytes.repeats < MAX_INSTANCES_FOR_REMOVE) {
      uint16_t context = temp_sym_data.bytes.repeats;
      uint16_t context2 = 240 + temp_sym_data.code_length;
      if ((use_mtf != 0) && (DecodeERG(context, context2) != 0)) {
        temp_sym_data.bytes.type = 2;
        if ((temp_sym_data.bytes.repeats == 1) || (DecodeGoMtf(context, 2) != 0))
          put_in_mtf = 1;
      }
    }
    else {
      uint16_t context = temp_sym_data.bytes.repeats;
      uint16_t context2 = 240;
      if ((temp_sym_data.code_length >= 11) && (use_mtf != 0) && (DecodeERG(context, context2) != 0)) {
        temp_sym_data.bytes.type = 2;
        if (DecodeGoMtf(context, 2) != 0)
          put_in_mtf = 1;
      }
    }
    if (put_in_mtf == 0) {
      sym_data_ptr = dadd_dictionary_symbol(temp_sym_data.code_length, prior_end);
      sym_data_ptr->string_index = temp_sym_data.string_index;
      sym_data_ptr->string_length = temp_sym_data.string_length;
      sym_data_ptr->four_bytes = temp_sym_data.four_bytes;
    }
    else
      sym_data_ptr = (struct sym_data *)dadd_symbol_to_queue((struct sym_data *)&temp_sym_data,
          temp_sym_data.code_length, temp_sym_data.starts);
  }
  else {
    symbols_in_definition = SID_symbol + 1;
    if (symbols_in_definition == 16)
      symbols_in_definition = get_extra_length();

    do { // Build the symbol string from the next symbols_in_definition symbols
      if ((sym_type = DecodeSymType1(1)) == 0) {
        if (UTF8_compliant != 0)
          first_char = DecodeFirstChar(0, prior_end);
        else
          first_char = DecodeFirstCharBinary(prior_end);
        uint16_t bin_num = DecodeBin(sum_nbob[first_char]);
        uint8_t code_length = lookup_bits[first_char][bin_num];
        uint32_t index = get_dictionary_symbol(bin_num, code_length, first_char);
        sym_data_ptr = &bin_data[first_char][code_length].symbol_data[index];
        prior_end = sym_data_ptr->bytes.ends;
        uint8_t * symbol_string_ptr = &symbol_strings[sym_data_ptr->string_index];
        uint8_t * end_symbol_string_ptr = symbol_string_ptr + sym_data_ptr->string_length;
        do {
          symbol_strings[end_string_index++] = *symbol_string_ptr++;
        } while (symbol_string_ptr != end_symbol_string_ptr);
        if (sym_data_ptr->bytes.remaining < MAX_INSTANCES_FOR_REMOVE) {
          if (--sym_data_ptr->bytes.remaining == 0)
            dremove_dictionary_symbol(&bin_data[first_char][code_length], index);
          else if ((sym_data_ptr->bytes.type & 2) != 0) {
            if ((sym_data_ptr->bytes.remaining == 1) && ((sym_data_ptr->bytes.type & 8) == 0)) {
              sym_data_ptr = (struct sym_data *)dadd_symbol_to_queue(sym_data_ptr, code_length, first_char);
              dremove_dictionary_symbol(&bin_data[first_char][code_length], index);
            }
            else {
              uint16_t context = 6 * sym_data_ptr->bytes.repeats;
              if (DecodeGoMtf(context, 0) != 0) {
                sym_data_ptr = (struct sym_data *)dadd_symbol_to_queue(sym_data_ptr, code_length, first_char);
                dremove_dictionary_symbol(&bin_data[first_char][code_length], index);
              }
            }
          }

        }
        else if ((sym_data_ptr->bytes.type & 2) != 0) {
          uint16_t context = 6 * sym_data_ptr->bytes.remaining;
          if (DecodeGoMtf(context, 0) != 0) {
            sym_data_ptr = (struct sym_data *)dadd_symbol_to_queue(sym_data_ptr, code_length, first_char);
            dremove_dictionary_symbol(&bin_data[first_char][code_length], index);
          }
        }
      }
      else if (sym_type == 1)
        sym_data_ptr = decode_new(&end_string_index);
      else {
        sym_data_ptr = dupdate_queue(DecodeMtfPos(queue_size));
        prior_end = sym_data_ptr->bytes.ends;
        uint8_t * symbol_string_ptr = &symbol_strings[sym_data_ptr->string_index];
        uint8_t * end_symbol_string_ptr = symbol_string_ptr + sym_data_ptr->string_length;
        do {
          symbol_strings[end_string_index++] = *symbol_string_ptr++;
        } while (symbol_string_ptr != end_symbol_string_ptr);
      }
    } while (--symbols_in_definition != 0);
    if ((symbol_strings[*string_index_ptr] < 0x80) || (UTF8_compliant == 0))
      temp_sym_data.starts = symbol_strings[*string_index_ptr];
    else
      temp_sym_data.starts = get_first_char(temp_sym_data.string_index);
    temp_sym_data.bytes.ends = prior_end;
    temp_sym_data.string_length = end_string_index - temp_sym_data.string_index;

    temp_sym_data.bytes.repeats = DecodeINST(NOT_CAP, SID_symbol);
    if (temp_sym_data.bytes.repeats <= MAX_INSTANCES_FOR_REMOVE - 2) {
      temp_sym_data.code_length = queue_miss_code_length[++temp_sym_data.bytes.repeats];
      uint16_t context = temp_sym_data.bytes.repeats;
      uint16_t context2 = 240 + temp_sym_data.code_length;
      if ((use_mtf != 0) && (DecodeERG(context, context2) != 0)) {
        temp_sym_data.bytes.type = 2;
        if ((temp_sym_data.bytes.repeats == 1) || (DecodeGoMtf(context, 2) != 0))
          put_in_mtf = 1;
      }
      else
        temp_sym_data.bytes.type = 0;
    }
    else {
      temp_sym_data.code_length = max_regular_code_length + MAX_INSTANCES_FOR_REMOVE - 1 - temp_sym_data.bytes.repeats;
      temp_sym_data.bytes.repeats = temp_sym_data.code_length + MAX_INSTANCES_FOR_REMOVE - 1;
      uint16_t context = temp_sym_data.bytes.repeats;
      uint16_t context2 = 240;
      if ((temp_sym_data.code_length >= 11) && (use_mtf != 0) && (DecodeERG(context, context2) != 0)) {
        temp_sym_data.bytes.type = 2;
        if (DecodeGoMtf(context, 2) != 0)
          put_in_mtf = 1;
      }
      else
        temp_sym_data.bytes.type = 0;
    }
    temp_sym_data.bytes.remaining = temp_sym_data.bytes.repeats;
    if (put_in_mtf == 0) {
      sym_data_ptr = dadd_dictionary_symbol(temp_sym_data.code_length, temp_sym_data.starts);
      sym_data_ptr->string_index = temp_sym_data.string_index;
      sym_data_ptr->string_length = temp_sym_data.string_length;
      sym_data_ptr->four_bytes = temp_sym_data.four_bytes;
    }
    else
      sym_data_ptr = (struct sym_data *)dadd_symbol_to_queue((struct sym_data *)&temp_sym_data,
          temp_sym_data.code_length, temp_sym_data.starts);
  }
  *string_index_ptr = end_string_index;
  return(sym_data_ptr);
}


struct sym_data * decode_new_cap_encoded(uint32_t * string_index_ptr) {
  uint8_t SID_symbol, sym_type, first_char, saved_prior_is_cap;
  uint8_t tag_type = 0;
  uint8_t put_in_mtf = 0;
  uint32_t symbols_in_definition, end_string_index;
  struct sym_data * sym_data_ptr;
  struct sym_data2 temp_sym_data;

  temp_sym_data.string_index = *string_index_ptr;
  end_string_index = *string_index_ptr;
  SID_symbol = DecodeSID(prior_is_cap);
  if (SID_symbol == 0) {
    temp_sym_data.bytes.repeats = DecodeINST(prior_is_cap, SID_symbol);
    if (temp_sym_data.bytes.repeats < MAX_INSTANCES_FOR_REMOVE - 1) {
      temp_sym_data.code_length = queue_miss_code_length[++temp_sym_data.bytes.repeats];
    }
    else if (temp_sym_data.bytes.repeats >= MAX_INSTANCES_FOR_REMOVE) {
      temp_sym_data.code_length = max_regular_code_length + MAX_INSTANCES_FOR_REMOVE - temp_sym_data.bytes.repeats;
      temp_sym_data.bytes.repeats = temp_sym_data.code_length + MAX_INSTANCES_FOR_REMOVE - 1;
    }
    else {
      temp_sym_data.bytes.repeats = 0;
      temp_sym_data.code_length = max_code_length + 1;
    }
    uint32_t base_symbol = DecodeBaseSymbolCap(num_base_symbols);
    if (base_symbol > 0x42)
      base_symbol += 24;
    else if (base_symbol > 0x40)
      base_symbol += 1;
    temp_sym_data.bytes.type = 0;
    uint8_t saved_pic = prior_is_cap;
    prior_is_cap = 0;

    if ((UTF8_compliant == 0) || (base_symbol < START_UTF8_2BYTE_SYMBOLS)) {
      if ((base_symbol & 1) != 0) {
        if (symbol_lengths[base_symbol] != 0) {
          base_symbol--;
          DoubleRangeDown();
        }
        else if (symbol_lengths[base_symbol - 1] != 0)
          DoubleRangeDown();
      }
      else if (symbol_lengths[base_symbol] != 0) {
        base_symbol++;
        DoubleRange();
      }
      else if (symbol_lengths[base_symbol + 1] != 0)
        DoubleRange();

      symbol_lengths[base_symbol] = temp_sym_data.code_length;
      InitBaseSymbolCap(base_symbol, temp_sym_data.code_length, &cap_symbol_defined, &cap_lock_symbol_defined,
          symbol_lengths);
      symbol_strings[end_string_index++] = temp_sym_data.starts = temp_sym_data.bytes.ends = base_symbol;
      temp_sym_data.string_length = 1;

      if (base_symbol == 'C') {
        prior_is_cap = 1;
        if (max_code_length >= 14)
          temp_sym_data.bytes.type = 4;
      }
      else if (base_symbol == 'B') {
        prior_is_cap = 1;
        temp_sym_data.bytes.ends = 'C';
        if (max_code_length >= 14)
          temp_sym_data.bytes.type = 4;
      }
      else {
        if (base_symbol == ' ') {
          if (max_code_length >= 14)
            temp_sym_data.bytes.type = 4;
        }
        else if ((base_symbol >= 0x61) && (base_symbol <= 0x7A))
          temp_sym_data.bytes.type = 1;
      }
      prior_end = temp_sym_data.bytes.ends;
    }
    else {
      base_symbol = create_extended_UTF8_symbol(base_symbol, &end_string_index);
      prior_end = temp_sym_data.starts = temp_sym_data.bytes.ends = base_symbol;
        if (symbol_lengths[prior_end] == 0) {
        symbol_lengths[prior_end] = temp_sym_data.code_length;
        uint8_t j1 = 0x90;
        do {
          InitFirstCharBin(j1, prior_end, temp_sym_data.code_length, cap_symbol_defined, cap_lock_symbol_defined);
        } while (--j1 != 'Z');
        j1 = 'A' - 1;
        do {
          InitFirstCharBin(j1, prior_end, temp_sym_data.code_length, cap_symbol_defined, cap_lock_symbol_defined);
        } while (j1-- != 0);
        j1 = 0x90;
        do {
          if (symbol_lengths[j1] != 0)
            InitTrailingCharBin(prior_end, j1, symbol_lengths[j1]);
        } while (j1-- != 0);
      }
      temp_sym_data.string_length = end_string_index - temp_sym_data.string_index;
    }

    if (find_first_symbol != 0) {
      find_first_symbol = 0;
      sum_nbob[base_symbol] = bin_data[base_symbol][max_code_length].nbob = 1;
    }
    if (temp_sym_data.bytes.repeats == 0) {
      sym_data_ptr = dadd_single_dictionary_symbol(temp_sym_data.starts);
      sym_data_ptr->bytes.type = temp_sym_data.bytes.type;
      sym_data_ptr->string_index = temp_sym_data.string_index;
      sym_data_ptr->string_length = temp_sym_data.string_length;
      prior_type = temp_sym_data.bytes.type;
      *string_index_ptr = end_string_index;
      return(sym_data_ptr);
    }

    temp_sym_data.bytes.remaining = temp_sym_data.bytes.repeats;
    if (temp_sym_data.bytes.repeats < MAX_INSTANCES_FOR_REMOVE) {
      uint16_t context = 6 * temp_sym_data.bytes.repeats + saved_pic + (temp_sym_data.bytes.type & 1);
      uint16_t context2 = 240 + (4 * temp_sym_data.code_length) + (((temp_sym_data.bytes.type >> 4) == 2) << 1)
          + (temp_sym_data.bytes.type & 1);
      if ((use_mtf != 0) && (DecodeERG(context, context2) != 0)) {
        temp_sym_data.bytes.type |= 2;
        if ((temp_sym_data.bytes.repeats == 1) || (DecodeGoMtf(context, 2) != 0))
          put_in_mtf = 1;
      }
    }
    else {
      uint16_t context = 6 * temp_sym_data.bytes.repeats + saved_pic + (temp_sym_data.bytes.type & 1);
      uint16_t context2 = 240 + (temp_sym_data.bytes.type & 1);
      if ((temp_sym_data.code_length >= 11) && (use_mtf != 0) && (DecodeERG(context, context2) != 0)) {
        temp_sym_data.bytes.type |= 2;
        if (DecodeGoMtf(context, 2) != 0)
          put_in_mtf = 1;
      }
    }
    if (put_in_mtf == 0) {
      sym_data_ptr = dadd_dictionary_symbol(temp_sym_data.code_length, (uint8_t)base_symbol);
      sym_data_ptr->string_index = temp_sym_data.string_index;
      sym_data_ptr->string_length = temp_sym_data.string_length;
      sym_data_ptr->four_bytes = temp_sym_data.four_bytes;
    }
    else
      sym_data_ptr = (struct sym_data *)dadd_symbol_to_queue_cap_encoded((struct sym_data *)&temp_sym_data,
          temp_sym_data.code_length, temp_sym_data.starts);
  }
  else {
    symbols_in_definition = SID_symbol + 1;
    if (symbols_in_definition == 16)
      symbols_in_definition = get_extra_length();
    saved_prior_is_cap = prior_is_cap;

    do { // Build the symbol string from the next symbols_in_definition symbols
      if (prior_is_cap == 0) {
        uint8_t context = 5 + 8 * (prior_type >> 4) + 2 * (prior_type & 7);
        if ((sym_type = DecodeSymType3(1, context, prior_end)) == 0) {
          if (prior_end != 0xA)
            first_char = DecodeFirstChar(prior_type >> 4, prior_end);
          else
            first_char = 0x20;
          uint16_t bin_num = DecodeBin(sum_nbob[first_char]);
          uint8_t code_length = lookup_bits[first_char][bin_num];
          uint32_t index = get_dictionary_symbol(bin_num, code_length, first_char);
          sym_data_ptr = &bin_data[first_char][code_length].symbol_data[index];
          prior_is_cap = ((prior_end = sym_data_ptr->bytes.ends) == 'C');
          prior_type = sym_data_ptr->bytes.type;
          uint8_t * symbol_string_ptr = &symbol_strings[sym_data_ptr->string_index];
          uint8_t * end_symbol_string_ptr = symbol_string_ptr + sym_data_ptr->string_length;
          do {
            symbol_strings[end_string_index++] = *symbol_string_ptr++;
          } while (symbol_string_ptr != end_symbol_string_ptr);
          if (sym_data_ptr->bytes.remaining < MAX_INSTANCES_FOR_REMOVE) {
            if (--sym_data_ptr->bytes.remaining == 0)
              dremove_dictionary_symbol(&bin_data[first_char][code_length], index);
            else if ((sym_data_ptr->bytes.type & 2) != 0) {
              if ((sym_data_ptr->bytes.remaining == 1) && ((sym_data_ptr->bytes.type & 8) == 0)) {
                sym_data_ptr = (struct sym_data *)dadd_symbol_to_queue_cap_encoded(sym_data_ptr, code_length, first_char);
                dremove_dictionary_symbol(&bin_data[first_char][code_length], index);
              }
              else {
                uint16_t context = 6 * sym_data_ptr->bytes.repeats + (sym_data_ptr->bytes.type & 1)
                    + 3 * ((sym_data_ptr->bytes.type >> 4) == 2);
                if (DecodeGoMtf(context, 0) != 0) {
                  sym_data_ptr = (struct sym_data *)dadd_symbol_to_queue_cap_encoded(sym_data_ptr, code_length, first_char);
                  dremove_dictionary_symbol(&bin_data[first_char][code_length], index);
                }
              }
            }
          }
          else if ((sym_data_ptr->bytes.type & 2) != 0) {
            uint16_t context = 6 * sym_data_ptr->bytes.remaining + (sym_data_ptr->bytes.type & 1)
                + 3 * ((sym_data_ptr->bytes.type >> 4) == 2);
            if (DecodeGoMtf(context, 0) != 0) {
              sym_data_ptr = (struct sym_data *)dadd_symbol_to_queue_cap_encoded(sym_data_ptr, code_length, first_char);
              dremove_dictionary_symbol(&bin_data[first_char][code_length], index);
            }
          }
        }
        else if (sym_type == 1)
          sym_data_ptr = decode_new_cap_encoded(&end_string_index);
        else {
          uint8_t mtf_first;
          if (prior_end != 0xA)
            mtf_first = DecodeMtfFirst((prior_type & 0x30) == 0x20);
          else
            mtf_first = 1;
          if (mtf_first == 0)
            sym_data_ptr = dupdate_other_queue(DecodeMtfPosOther(queue_size_other));
          else if (mtf_first == 1)
            sym_data_ptr = dupdate_space_queue(DecodeMtfPosSpace(queue_size_space));
          else
            sym_data_ptr = dupdate_az_queue(DecodeMtfPosAz(queue_size_az));
          prior_is_cap = ((prior_end = sym_data_ptr->bytes.ends) == 'C');
          prior_type = sym_data_ptr->bytes.type;
          uint8_t * symbol_string_ptr = &symbol_strings[sym_data_ptr->string_index];
          uint8_t * end_symbol_string_ptr = symbol_string_ptr + sym_data_ptr->string_length;
          do {
            symbol_strings[end_string_index++] = *symbol_string_ptr++;
          } while (symbol_string_ptr != end_symbol_string_ptr);
        }
      }
      else { // prior_is_cap
        uint8_t context = 0x30 + (prior_type & 3);
        if ((sym_type = DecodeSymType2(3, context)) == 0) {
          first_char = DecodeFirstChar(0, 'C');
          uint16_t bin_num = DecodeBin(sum_nbob[first_char]);
          uint8_t code_length = lookup_bits[first_char][bin_num];
          uint32_t index = get_dictionary_symbol(bin_num, code_length, first_char);
          sym_data_ptr = &bin_data[first_char][code_length].symbol_data[index];
          prior_is_cap = ((prior_end = sym_data_ptr->bytes.ends) == 'C');
          prior_type = sym_data_ptr->bytes.type;
          uint8_t * symbol_string_ptr = &symbol_strings[sym_data_ptr->string_index];
          uint8_t * end_symbol_string_ptr = symbol_string_ptr + sym_data_ptr->string_length;
          do {
            symbol_strings[end_string_index++] = *symbol_string_ptr++;
          } while (symbol_string_ptr != end_symbol_string_ptr);
          if (sym_data_ptr->bytes.remaining < MAX_INSTANCES_FOR_REMOVE) {
            if (--sym_data_ptr->bytes.remaining == 0)
              dremove_dictionary_symbol(&bin_data[first_char][code_length], index);
            else if ((sym_data_ptr->bytes.type & 2) != 0) {
              if ((sym_data_ptr->bytes.remaining == 1) && ((sym_data_ptr->bytes.type & 8) == 0)) {
                sym_data_ptr = (struct sym_data *)dadd_symbol_to_queue_cap_encoded(sym_data_ptr, code_length, first_char);
                dremove_dictionary_symbol(&bin_data[first_char][code_length], index);
              }
              else {
                uint16_t context = 6 * sym_data_ptr->bytes.repeats + 2 + 3 * ((sym_data_ptr->bytes.type >> 4) == 2);
                if (DecodeGoMtf(context, 0) != 0) {
                  sym_data_ptr = (struct sym_data *)dadd_symbol_to_queue_cap_encoded(sym_data_ptr, code_length, first_char);
                  dremove_dictionary_symbol(&bin_data[first_char][code_length], index);
                }
              }
            }
          }
          else if ((sym_data_ptr->bytes.type & 2) != 0) {
            uint16_t context = 6 * sym_data_ptr->bytes.remaining + 2 + 3 * ((sym_data_ptr->bytes.type >> 4) == 2);
            if (DecodeGoMtf(context, 0) != 0) {
              sym_data_ptr = (struct sym_data *)dadd_symbol_to_queue_cap_encoded(sym_data_ptr, code_length, first_char);
              dremove_dictionary_symbol(&bin_data[first_char][code_length], index);
            }
          }
        }
        else if (sym_type == 1)
          sym_data_ptr = decode_new_cap_encoded(&end_string_index);
        else {
          sym_data_ptr = dupdate_az_queue(DecodeMtfPosAz(queue_size_az));
          prior_is_cap = ((prior_end = sym_data_ptr->bytes.ends) == 'C');
          prior_type = sym_data_ptr->bytes.type;
          uint8_t * symbol_string_ptr = &symbol_strings[sym_data_ptr->string_index];
          uint8_t * end_symbol_string_ptr = symbol_string_ptr + sym_data_ptr->string_length;
          do {
            symbol_strings[end_string_index++] = *symbol_string_ptr++;
          } while (symbol_string_ptr != end_symbol_string_ptr);
        }
      }
    } while (--symbols_in_definition != 0);

    temp_sym_data.bytes.ends = prior_end;
    temp_sym_data.string_length = end_string_index - temp_sym_data.string_index;
    if ((symbol_strings[temp_sym_data.string_index] < 0x80) || (UTF8_compliant == 0)) {
      temp_sym_data.starts = symbol_strings[temp_sym_data.string_index];
      temp_sym_data.bytes.type = ((symbol_strings[temp_sym_data.string_index] >= 'a')
          && (symbol_strings[temp_sym_data.string_index] <= 'z'));
    }
    else {
      temp_sym_data.starts = get_first_char(temp_sym_data.string_index);
      temp_sym_data.bytes.type = 0;
    }

    temp_sym_data.bytes.repeats = DecodeINST(saved_prior_is_cap, SID_symbol);
    if (temp_sym_data.bytes.repeats <= MAX_INSTANCES_FOR_REMOVE - 2) {
      temp_sym_data.code_length = queue_miss_code_length[++temp_sym_data.bytes.repeats];
      if ((prior_type & 4) != 0) {
        temp_sym_data.bytes.type |= 4;
        temp_sym_data.bytes.type |= prior_type & 0x30;
      }
      else if (max_code_length >= 14) {
        uint8_t * symbol_string_ptr = &symbol_strings[end_string_index - 2];
        do {
          if (*symbol_string_ptr == ' ') {
            temp_sym_data.bytes.type |= 4;
            temp_sym_data.bytes.type = (temp_sym_data.bytes.type & 0xF) + 0x10;
            break;
          }
        } while (symbol_string_ptr-- != &symbol_strings[temp_sym_data.string_index]);
      }
      uint16_t context = 6 * temp_sym_data.bytes.repeats + saved_prior_is_cap + (temp_sym_data.bytes.type & 1)
          + 3 * ((temp_sym_data.bytes.type >> 4) == 2);
      uint16_t context2 = 240 + (4 * temp_sym_data.code_length) + (((temp_sym_data.bytes.type >> 4) == 2) << 1)
          + (temp_sym_data.bytes.type & 1);
      if ((use_mtf != 0) && (DecodeERG(context, context2) != 0)) {
        temp_sym_data.bytes.type |= 2;
        if ((temp_sym_data.bytes.repeats == 1) || (DecodeGoMtf(context, 2) != 0))
          put_in_mtf = 1;
      }
    }
    else {
      temp_sym_data.code_length = max_regular_code_length + MAX_INSTANCES_FOR_REMOVE - 1 - temp_sym_data.bytes.repeats;
      temp_sym_data.bytes.repeats = temp_sym_data.code_length + MAX_INSTANCES_FOR_REMOVE - 1;
      if ((prior_type & 4) != 0) {
        temp_sym_data.bytes.type |= 4;
        if ((prior_type & 0x10) != 0) {
          tag_type = 1 + DecodeWordTag(prior_end);
          temp_sym_data.bytes.type |= (4 - tag_type) << 4;
        }
        else
          temp_sym_data.bytes.type |= prior_type & 0x30;
      }
      else if (max_code_length >= 14) {
        uint8_t * symbol_string_ptr = &symbol_strings[end_string_index - 2];
        do {
          if (*symbol_string_ptr == ' ') {
            temp_sym_data.bytes.type |= 4;
            if (temp_sym_data.bytes.repeats >= MAX_INSTANCES_FOR_REMOVE) {
              tag_type = 1 + DecodeWordTag(prior_end);
              temp_sym_data.bytes.type = (temp_sym_data.bytes.type & 0xF) + ((4 - tag_type) << 4);
            }
            else
              temp_sym_data.bytes.type = (temp_sym_data.bytes.type & 0xF) + 0x10;
            break;
          }
        } while (symbol_string_ptr-- != &symbol_strings[temp_sym_data.string_index]);
      }
      uint16_t context = 6 * temp_sym_data.bytes.repeats + saved_prior_is_cap + (temp_sym_data.bytes.type & 1)
          + 3 * ((temp_sym_data.bytes.type >> 4) == 2);
      uint16_t context2 = 240 + (((temp_sym_data.bytes.type >> 4) == 2) << 1) + (temp_sym_data.bytes.type & 1);
      if ((temp_sym_data.code_length >= 11) && (use_mtf != 0) && (DecodeERG(context, context2) != 0)) {
        temp_sym_data.bytes.type |= 2;
        if (DecodeGoMtf(context, 2) != 0)
          put_in_mtf = 1;
      }
    }
    temp_sym_data.bytes.remaining = temp_sym_data.bytes.repeats;
    if (put_in_mtf == 0) {
      sym_data_ptr = dadd_dictionary_symbol(temp_sym_data.code_length, temp_sym_data.starts);
      sym_data_ptr->string_index = temp_sym_data.string_index;
      sym_data_ptr->string_length = temp_sym_data.string_length;
      sym_data_ptr->four_bytes = temp_sym_data.four_bytes;
    }
    else
      sym_data_ptr = (struct sym_data *)dadd_symbol_to_queue_cap_encoded((struct sym_data *)&temp_sym_data,
          temp_sym_data.code_length, temp_sym_data.starts);
  }
  prior_type = temp_sym_data.bytes.type;
  *string_index_ptr = end_string_index;
  return(sym_data_ptr);
}


void transpose2(uint8_t * buffer, uint32_t len) {
  uint8_t temp_buf[0x30000];
  uint8_t *char_ptr, *char2_ptr;
  uint32_t block1_len = len - (len >> 1);
  memcpy(temp_buf, buffer + block1_len, len - block1_len);
  char2_ptr = buffer + 2 * block1_len;
  char_ptr = buffer + block1_len;
  while (char_ptr != buffer) {
    char2_ptr -= 2;
    *char2_ptr = *--char_ptr;
  }
  char2_ptr = buffer + 1;
  char_ptr = temp_buf;
  while (char2_ptr < buffer + len) {
    *char2_ptr = *char_ptr++;
    char2_ptr += 2;
  }
  return;
}


void transpose4(uint8_t * buffer, uint32_t len) {
  uint8_t temp_buf[0x30000];
  uint8_t *char_ptr, *char2_ptr;
  uint32_t block1_len = (len + 3) >> 2;
  memcpy(temp_buf, buffer + block1_len, len - block1_len);
  char2_ptr = buffer + 4 * block1_len;
  char_ptr = buffer + block1_len;
  while (char_ptr != buffer) {
    char2_ptr -= 4;
    *char2_ptr = *--char_ptr;
  }
  char2_ptr = buffer + 1;
  char_ptr = temp_buf;
  while (char2_ptr < buffer + len) {
    *char2_ptr = *char_ptr++;
    char2_ptr += 4;
  }
  char2_ptr = buffer + 2;
  while (char2_ptr < buffer + len) {
    *char2_ptr = *char_ptr++;
    char2_ptr += 4;
  }
  char2_ptr = buffer + 3;
  while (char2_ptr < buffer + len) {
    *char2_ptr = *char_ptr++;
    char2_ptr += 4;
  }
  return;
}


void write_output_buffer() {
  uint32_t chars_to_write = out_char_ptr - start_char_ptr;
  if (fd != 0) {
    fflush(fd);
    fwrite(start_char_ptr, 1, chars_to_write, fd);
    if ((out_buffers_sent & 1) == 0)
      out_char_ptr = out_char1;
    else
      out_char_ptr = out_char0;
  }
  outbuf_index += chars_to_write;
  start_char_ptr = out_char_ptr;
  end_outbuf = out_char_ptr + CHARS_TO_WRITE;
#ifdef PRINTON
  if ((out_buffers_sent & 0x7F) == 0)
    fprintf(stderr, "%u\r", (unsigned int)outbuf_index);
#endif
  out_buffers_sent++;
  return;
}


void write_output_buffer_delta() {
  uint32_t chars_to_write = out_char_ptr - start_char_ptr;
  uint32_t len = out_char_ptr - start_char_ptr;
  if (stride == 4) {
    transpose4(start_char_ptr, len);
    len = out_char_ptr - start_char_ptr;
  }
  else if (stride == 2) {
    transpose2(start_char_ptr, len);
    len = out_char_ptr - start_char_ptr;
  }
  delta_transform(start_char_ptr, len);
  if (fd != 0) {
    fflush(fd);
    fwrite(start_char_ptr, 1, chars_to_write, fd);
    if ((out_buffers_sent & 1) == 0) {
      uint8_t k;
      for (k = 1 ; k <= stride ; k++)
        out_char1[100 - k] = *(out_char_ptr - k);
      out_char_ptr = out_char1 + 100;
    }
    else {
      uint8_t k;
      for (k = 1 ; k <= stride ; k++)
        out_char0[100 - k] = *(out_char_ptr - k);
      out_char_ptr = out_char0 + 100;
    }
  }
  outbuf_index += chars_to_write;
  start_char_ptr = out_char_ptr;
  end_outbuf = out_char_ptr + CHARS_TO_WRITE;
#ifdef PRINTON
  if ((out_buffers_sent & 0x7F) == 0)
    fprintf(stderr, "%u\r", (unsigned int)outbuf_index);
#endif
  out_buffers_sent++;
  return;
}


#define write_string(len) { \
  while (out_char_ptr + len >= end_outbuf) { \
    uint32_t temp_len = end_outbuf - out_char_ptr; \
    len -= temp_len; \
    memcpy(out_char_ptr, symbol_string_ptr, temp_len); \
    out_char_ptr += temp_len; \
    symbol_string_ptr += temp_len; \
    write_output_buffer(); \
  } \
  memcpy(out_char_ptr, symbol_string_ptr, len); \
  out_char_ptr += len; \
}


#define write_string_cap_encoded(len) { \
  while (out_char_ptr + len >= end_outbuf) { \
    uint32_t temp_len = end_outbuf - out_char_ptr; \
    len -= temp_len; \
    while (temp_len-- != 0) { \
      if (write_cap_on == 0) { \
        if (skip_space_on == 0) { \
          if ((*symbol_string_ptr & 0xFE) == 0x42) { \
            write_cap_on = 1; \
            if (*symbol_string_ptr++ == 'B') \
              write_cap_lock_on = 1; \
          } \
          else { \
            *out_char_ptr++ = *symbol_string_ptr; \
            if (*symbol_string_ptr++ == 0xA) \
              skip_space_on = 1; \
          } \
        } \
        else { \
          symbol_string_ptr++; \
          skip_space_on = 0; \
        } \
      } \
      else { \
        if (write_cap_lock_on != 0) { \
          if ((*symbol_string_ptr >= 'a') && (*symbol_string_ptr <= 'z')) \
            *out_char_ptr++ = *symbol_string_ptr++ - 0x20; \
          else { \
            write_cap_lock_on = 0; \
            write_cap_on = 0; \
            if (*symbol_string_ptr == 'C') \
              symbol_string_ptr++; \
            else { \
              *out_char_ptr++ = *symbol_string_ptr; \
              if (*symbol_string_ptr++ == 0xA) \
                skip_space_on = 1; \
            } \
          } \
        } \
        else { \
          write_cap_on = 0; \
          *out_char_ptr++ = *symbol_string_ptr++ - 0x20; \
        } \
      } \
    } \
    write_output_buffer(); \
  } \
  while (len-- != 0) { \
    if (write_cap_on == 0) { \
      if (skip_space_on == 0) { \
        if ((*symbol_string_ptr & 0xFE) == 0x42) { \
          write_cap_on = 1; \
          if (*symbol_string_ptr++ == 'B') \
            write_cap_lock_on = 1; \
        } \
        else { \
          *out_char_ptr++ = *symbol_string_ptr; \
          if (*symbol_string_ptr++ == 0xA) \
            skip_space_on = 1; \
        } \
      } \
      else { \
        symbol_string_ptr++; \
        skip_space_on = 0; \
      } \
    } \
    else { \
      if (write_cap_lock_on != 0) { \
        if ((*symbol_string_ptr >= 'a') && (*symbol_string_ptr <= 'z')) \
          *out_char_ptr++ = *symbol_string_ptr++ - 0x20; \
        else { \
          write_cap_lock_on = 0; \
          write_cap_on = 0; \
          if (*symbol_string_ptr == 'C') { \
            symbol_string_ptr++; \
          } \
          else { \
            *out_char_ptr++ = *symbol_string_ptr; \
            if (*symbol_string_ptr++ == 0xA) \
              skip_space_on = 1; \
          } \
        } \
      } \
      else { \
        write_cap_on = 0; \
        *out_char_ptr++ = *symbol_string_ptr++ - 0x20; \
      } \
    } \
  } \
}


#define write_string_delta(len) { \
  while (out_char_ptr + len >= end_outbuf) { \
    uint32_t temp_len = end_outbuf - out_char_ptr; \
    len -= temp_len; \
    memcpy(out_char_ptr, symbol_string_ptr, temp_len); \
    out_char_ptr += temp_len; \
    symbol_string_ptr += temp_len; \
    write_output_buffer_delta(); \
  } \
  memcpy(out_char_ptr, symbol_string_ptr, len); \
  out_char_ptr += len; \
}


void write_single_threaded_output() {
  uint8_t * symbol_string_ptr;
  uint32_t * read_ptr = (uint32_t *)symbol_buffer;
  if (cap_encoded != 0) {
    while (read_ptr != (uint32_t *)symbol_buffer_write_ptr) {
      symbol_string_ptr = &symbol_strings[*read_ptr++];
      uint32_t length = *read_ptr++;
      write_string_cap_encoded(length);
    }
  }
  else if (stride == 0) {
    while (read_ptr != (uint32_t *)symbol_buffer_write_ptr) {
      symbol_string_ptr = &symbol_strings[*read_ptr++];
      uint32_t length = *read_ptr++;
      write_string(length);
    }
  }
  else {
    while (read_ptr != (uint32_t *)symbol_buffer_write_ptr) {
      symbol_string_ptr = &symbol_strings[*read_ptr++];
      uint32_t length = *read_ptr++;
      write_string_delta(length);
    }
  }
  return;
}


void write_symbol_buffer(uint8_t * buffer_number_ptr) {
  if (two_threads == 0) {
    write_single_threaded_output();
    symbol_buffer_write_ptr = symbol_buffer;
  }
  else {
    if (*buffer_number_ptr != 0)
      symbol_buffer_write_ptr = symbol_buffer;
    symbol_buffer_end_write_ptr = symbol_buffer_write_ptr + 0x400;
    atomic_store_explicit(&symbol_buffer_owner[*buffer_number_ptr], 1, memory_order_release);
    *buffer_number_ptr ^= 1;
    while (atomic_load_explicit(&symbol_buffer_owner[*buffer_number_ptr], memory_order_acquire) != 0) ;
  }
  return;
}


void *write_output_thread(void * outbuf) {
  uint8_t write_cap_on, write_cap_lock_on, skip_space_on, next_buffer, *symbol_string_ptr;
  uint32_t *buffer_ptr, *buffer_end_ptr;

  next_buffer = write_cap_on = write_cap_lock_on = skip_space_on = 0;
  if (fd != 0)
    out_char_ptr = out_char0 + 100;
  else
    out_char_ptr = outbuf;
  start_char_ptr = out_char_ptr;
  end_outbuf = out_char_ptr + CHARS_TO_WRITE;
  buffer_ptr = (uint32_t *)&symbol_buffer[0];
  buffer_end_ptr = buffer_ptr + 0x800;

  if (cap_encoded != 0) {
    while ((atomic_load_explicit(&done_parsing, memory_order_acquire) == 0)
        || (atomic_load_explicit(&symbol_buffer_owner[next_buffer], memory_order_acquire) != 0)) {
      if (atomic_load_explicit(&symbol_buffer_owner[next_buffer], memory_order_acquire) != 0) {
        do {
          symbol_string_ptr = &symbol_strings[*buffer_ptr++];
          uint32_t length = *buffer_ptr++;
          write_string_cap_encoded(length);
        } while (buffer_ptr != buffer_end_ptr);
        atomic_store_explicit(&symbol_buffer_owner[next_buffer], 0, memory_order_release);
        next_buffer ^= 1;
        buffer_ptr = (uint32_t *)symbol_buffer + ((buffer_ptr - (uint32_t *)symbol_buffer) & 0xFFF);
        buffer_end_ptr = buffer_ptr + 0x800;
      }
    }
    while (*buffer_ptr != MAX_U32_VALUE) {
      symbol_string_ptr = &symbol_strings[*buffer_ptr++];
      uint32_t length = *buffer_ptr++;
      write_string_cap_encoded(length);
    }
  }
  else if (stride == 0) {
    while ((atomic_load_explicit(&done_parsing, memory_order_acquire) == 0)
        || (atomic_load_explicit(&symbol_buffer_owner[next_buffer], memory_order_acquire) != 0)) {
      if (atomic_load_explicit(&symbol_buffer_owner[next_buffer], memory_order_acquire) != 0) {
        do {
          symbol_string_ptr = &symbol_strings[*buffer_ptr++];
          uint32_t length = *buffer_ptr++;
          write_string(length);
        } while (buffer_ptr != buffer_end_ptr);
        atomic_store_explicit(&symbol_buffer_owner[next_buffer], 0, memory_order_release);
        next_buffer ^= 1;
        buffer_ptr = (uint32_t *)symbol_buffer + ((buffer_ptr - (uint32_t *)symbol_buffer) & 0xFFF);
        buffer_end_ptr = buffer_ptr + 0x800;
      }
    }
    while (*buffer_ptr != MAX_U32_VALUE) {
      symbol_string_ptr = &symbol_strings[*buffer_ptr++];
      uint32_t length = *buffer_ptr++;
      write_string(length);
    }
  }
  else {
    while ((atomic_load_explicit(&done_parsing, memory_order_acquire) == 0)
        || (atomic_load_explicit(&symbol_buffer_owner[next_buffer], memory_order_acquire) != 0)) {
      if (atomic_load_explicit(&symbol_buffer_owner[next_buffer], memory_order_acquire) != 0) {
        do {
          symbol_string_ptr = &symbol_strings[*buffer_ptr++];
          uint32_t length = *buffer_ptr++;
          write_string_delta(length);
        } while (buffer_ptr != buffer_end_ptr);
        atomic_store_explicit(&symbol_buffer_owner[next_buffer], 0, memory_order_release);
        next_buffer ^= 1;
        buffer_ptr = (uint32_t *)symbol_buffer + ((buffer_ptr - (uint32_t *)symbol_buffer) & 0xFFF);
        buffer_end_ptr = buffer_ptr + 0x800;
      }
    }
    while (*buffer_ptr != MAX_U32_VALUE) {
      symbol_string_ptr = &symbol_strings[*buffer_ptr++];
      uint32_t length = *buffer_ptr++;
      write_string_delta(length);
    }
  }
  uint32_t chars_to_write = out_char_ptr - start_char_ptr;
  if (stride != 0) {
    if (stride == 4)
      transpose4(start_char_ptr, chars_to_write);
    else if (stride == 2)
      transpose2(start_char_ptr, chars_to_write);
    delta_transform(start_char_ptr, chars_to_write);
  }
  if (fd != 0)
    fwrite(start_char_ptr, 1, chars_to_write, fd);
  outbuf_index += chars_to_write;
  return(0);
}


uint8_t * GLZAdecode(size_t in_size, uint8_t * inbuf, size_t * outsize_ptr, uint8_t * outbuf, FILE * fd_out,
    struct param_data * params) {
  uint8_t sym_type, next_write_buffer, i, j;
  uint32_t new_string_index;
  struct sym_data * sym_data_ptr;
  pthread_t output_thread;

  fd = fd_out;
  stride = outbuf_index = out_buffers_sent = next_write_buffer = two_threads = 0;
  dictionary_size = (uint32_t)(pow(2.0, 10.0 + 0.08 * (double)inbuf[0]));
  cap_encoded = inbuf[1] >> 7;
  UTF8_compliant = (inbuf[1] >> 6) & 1;
  use_mtf = (inbuf[1] >> 5) & 1;
  max_code_length = (inbuf[1] & 0x1F) + 1;
  queue_miss_code_length[1] = max_code_length;
  min_code_length = (inbuf[2] & 0x1F) + 1;
  max_regular_code_length = max_code_length - (inbuf[3] & 0x1F);
  i = 2;
  do {
    queue_miss_code_length[i] = queue_miss_code_length[i - 1] - ((inbuf[2] >> (i + 4)) & 1);
  } while (++i != 4);
  do {
    queue_miss_code_length[i] = queue_miss_code_length[i - 1] - ((inbuf[3] >> (i + 1)) & 1);
  } while (++i != 7);
  do {
    queue_miss_code_length[i] = queue_miss_code_length[i - 1] - ((inbuf[4] >> (i - 7)) & 1);
  } while (++i != 15);
  if (UTF8_compliant != 0) {
    if (in_size < 6) {
      *outsize_ptr = 0;
      return(outbuf);
    }
    WriteInCharNum(6);
    num_base_symbols = 1 << inbuf[5];
    i = 0x90;
  }
  else {
    num_base_symbols = 0x100;
    delta_format = (inbuf[2] & 0x20) >> 5;
    if (delta_format != 0) {
      if (in_size < 6) {
        *outsize_ptr = 0;
        return(outbuf);
      }
      WriteInCharNum(6);
      delta_format = inbuf[5];
      if ((delta_format & 0x80) == 0)
        stride = (delta_format & 0x3) + 1;
      else
        stride = delta_format & 0x7F;
    }
    else {
      if (in_size < 5) {
        *outsize_ptr = 0;
        return(outbuf);
      }
      WriteInCharNum(5);
    }
    i = 0xFF;
  }

  if ((0 == (symbol_strings = (uint8_t *)malloc(dictionary_size)))
      || (0 == (queue_data = (struct sym_data2 *)malloc(0x100 * sizeof(struct sym_data2))))) {
    fprintf(stderr, "ERROR - memory allocation failed\n");
    return(0);
  }

  do {
    for (j = max_code_length + 1 ; j >= min_code_length ; j--) {
      bin_data[i][j].nsob = bin_data[i][j].nbob = bin_data[i][j].fbob = 0;
      bin_data[i][j].sym_list_size = 4;
      if (0 == (bin_data[i][j].symbol_data = (struct sym_data *)malloc(sizeof(struct sym_data) * 4))) {
        fprintf(stderr, "ERROR - memory allocation failed\n");
        return(0);
      }
    }
    sum_nbob[i] = 0;
    symbol_lengths[i] = 0;
    bin_code_length[i] = max_code_length;
  } while (i-- != 0);

  uint8_t * lookup_bits_ptr = &lookup_bits[0][0] + 0x100000;
  while (lookup_bits_ptr-- != &lookup_bits[0][0])
    *lookup_bits_ptr = max_code_length;

  symbol_buffer_write_ptr = symbol_buffer;
  symbol_buffer_end_write_ptr = symbol_buffer_write_ptr + 0x400;
  find_first_symbol = 1;
  queue_offset = queue_size = queue_size_az = queue_size_space = queue_size_other = prior_is_cap = 0;
  new_string_index = 0;
  if (UTF8_compliant != 0)
    i = 0x90;
  InitDecoder(max_code_length, i, MAX_INSTANCES_FOR_REMOVE + 1 + max_regular_code_length - min_code_length,
      cap_encoded, UTF8_compliant, use_mtf, inbuf);

  if ((params != 0) && (params->two_threads != 0)) {
    two_threads = 1;
    atomic_init(&done_parsing, 0);
    atomic_init(&symbol_buffer_owner[0], 0);
    atomic_init(&symbol_buffer_owner[1], 0);
    pthread_create(&output_thread, NULL, write_output_thread, (void *)outbuf);
  }
  else {
    write_cap_on = write_cap_lock_on = skip_space_on = 0;
    if (fd != 0)
      out_char_ptr = out_char0 + 100;
    else
      out_char_ptr = outbuf;
    start_char_ptr = out_char_ptr;
    end_outbuf = out_char_ptr + CHARS_TO_WRITE;
  }

  sym_data_ptr = bin_data[0][min_code_length].symbol_data;
  sym_data_ptr->bytes.type = 0;
  prior_end = 0;
  prior_type = 0;
  cap_symbol_defined = cap_lock_symbol_defined = 0;

  if (use_mtf != 0) {
    uint16_t i;
    for (i = 0 ; i < 0x100 ; i++)
      queue_data_free_list[i] = i;
  }

  // main decoding loop

  if (cap_encoded != 0) {
    sym_data_ptr = decode_new_cap_encoded(&new_string_index);
    *(uint64_t *)symbol_buffer_write_ptr++ = *(uint64_t *)&sym_data_ptr->string_index;
    while (1) {
      if (symbol_buffer_write_ptr == symbol_buffer_end_write_ptr)
        write_symbol_buffer(&next_write_buffer);
      if (prior_is_cap == 0) {
        if ((sym_type = DecodeSymType3(0, 4 + 8 * (prior_type >> 4)
            + 2 * (prior_type & 7), prior_end)) == 0) { // dictionary symbol
          uint8_t first_char = ' ';
          if (prior_end != 0xA)
            first_char = DecodeFirstChar(prior_type >> 4, prior_end);
          uint16_t bin_num = DecodeBin(sum_nbob[first_char]);
          uint8_t code_length = lookup_bits[first_char][bin_num];
          if (bin_data[first_char][code_length].nsob == 0)
            break; // EOF
          uint32_t index = get_dictionary_symbol(bin_num, code_length, first_char);
          sym_data_ptr = &bin_data[first_char][code_length].symbol_data[index];
          *(uint64_t *)symbol_buffer_write_ptr++ = *(uint64_t *)&sym_data_ptr->string_index;
          prior_is_cap = ((prior_end = sym_data_ptr->bytes.ends) == 'C');
          prior_type = sym_data_ptr->bytes.type;
          if (sym_data_ptr->bytes.remaining < MAX_INSTANCES_FOR_REMOVE) {
            if (--sym_data_ptr->bytes.remaining == 0)
              dremove_dictionary_symbol(&bin_data[first_char][code_length], index);
            else if ((sym_data_ptr->bytes.type & 2) != 0) {
              if ((sym_data_ptr->bytes.remaining == 1) && ((sym_data_ptr->bytes.type & 8) == 0)) {
                (void)dadd_symbol_to_queue_cap_encoded(sym_data_ptr, code_length, first_char);
                dremove_dictionary_symbol(&bin_data[first_char][code_length], index);
              }
              else {
                uint16_t context = 6 * sym_data_ptr->bytes.repeats + (sym_data_ptr->bytes.type & 1)
                    + 3 * ((sym_data_ptr->bytes.type >> 4) == 2);
                if (DecodeGoMtf(context, 0) != 0) {
                  (void)dadd_symbol_to_queue_cap_encoded(sym_data_ptr, code_length, first_char);
                  dremove_dictionary_symbol(&bin_data[first_char][code_length], index);
                }
              }
            }
          }
          else if ((sym_data_ptr->bytes.type & 2) != 0) {
            uint16_t context = 6 * sym_data_ptr->bytes.remaining + (sym_data_ptr->bytes.type & 1)
                + 3 * ((sym_data_ptr->bytes.type >> 4) == 2);
            if (DecodeGoMtf(context, 0) != 0) {
              (void)dadd_symbol_to_queue_cap_encoded(sym_data_ptr, code_length, first_char);
              dremove_dictionary_symbol(&bin_data[first_char][code_length], index);
            }
          }
        }
        else if (sym_type == 1) {
          sym_data_ptr = decode_new_cap_encoded(&new_string_index);
          *(uint64_t *)symbol_buffer_write_ptr++ = *(uint64_t *)&sym_data_ptr->string_index;
        }
        else {
          uint8_t mtf_first;
          if (prior_end != 0xA)
            mtf_first = DecodeMtfFirst((prior_type & 0x30) == 0x20);
          else
            mtf_first = 1;
          if (mtf_first == 0)
            sym_data_ptr = dupdate_other_queue(DecodeMtfPosOther(queue_size_other));
          else if (mtf_first == 1)
            sym_data_ptr = dupdate_space_queue(DecodeMtfPosSpace(queue_size_space));
          else
            sym_data_ptr = dupdate_az_queue(DecodeMtfPosAz(queue_size_az));
          prior_is_cap = ((prior_end = sym_data_ptr->bytes.ends) == 'C');
          prior_type = sym_data_ptr->bytes.type;
          *(uint64_t *)symbol_buffer_write_ptr++ = *(uint64_t *)&sym_data_ptr->string_index;
        }
      }
      else { // prior_is_cap
        if ((sym_type = DecodeSymType2(2, 0x2C + (prior_type & 3))) == 0) { // dictionary symbol
          uint8_t first_char = DecodeFirstChar(0, 'C');
          uint16_t bin_num = DecodeBin(sum_nbob[first_char]);
          uint8_t code_length = lookup_bits[first_char][bin_num];
          uint32_t index = get_dictionary_symbol(bin_num, code_length, first_char);
          sym_data_ptr = &bin_data[first_char][code_length].symbol_data[index];
          *(uint64_t *)symbol_buffer_write_ptr++ = *(uint64_t *)&sym_data_ptr->string_index;
          prior_is_cap = ((prior_end = sym_data_ptr->bytes.ends) == 'C');
          prior_type = sym_data_ptr->bytes.type;
          if (sym_data_ptr->bytes.remaining < MAX_INSTANCES_FOR_REMOVE) {
            if (--sym_data_ptr->bytes.remaining == 0)
              dremove_dictionary_symbol(&bin_data[first_char][code_length], index);
            else if ((sym_data_ptr->bytes.type & 2) != 0) {
              if ((sym_data_ptr->bytes.remaining == 1) && ((sym_data_ptr->bytes.type & 8) == 0)) {
                (void)dadd_symbol_to_queue_cap_encoded(sym_data_ptr, code_length, first_char);
                dremove_dictionary_symbol(&bin_data[first_char][code_length], index);
              }
              else {
                uint16_t context = 6 * sym_data_ptr->bytes.repeats + 1 + (sym_data_ptr->bytes.type & 1)
                     + 3 * ((sym_data_ptr->bytes.type >> 4) == 2);
                if (DecodeGoMtf(context, 0) != 0) {
                  (void)dadd_symbol_to_queue_cap_encoded(sym_data_ptr, code_length, first_char);
                  dremove_dictionary_symbol(&bin_data[first_char][code_length], index);
                }
              }
            }
          }
          else if ((sym_data_ptr->bytes.type & 2) != 0) {
            uint16_t context = 6 * sym_data_ptr->bytes.remaining + 1 + (sym_data_ptr->bytes.type & 1)
                + 3 * ((sym_data_ptr->bytes.type >> 4) == 2);
            if (DecodeGoMtf(context, 0) != 0) {
              (void)dadd_symbol_to_queue_cap_encoded(sym_data_ptr, code_length, first_char);
              dremove_dictionary_symbol(&bin_data[first_char][code_length], index);
            }
          }
        }
        else if (sym_type == 1) {
          sym_data_ptr = decode_new_cap_encoded(&new_string_index);
          *(uint64_t *)symbol_buffer_write_ptr++ = *(uint64_t *)&sym_data_ptr->string_index;
        }
        else {
          sym_data_ptr = dupdate_az_queue(DecodeMtfPosAz(queue_size_az));
          prior_is_cap = ((prior_end = sym_data_ptr->bytes.ends) == 'C');
          prior_type = sym_data_ptr->bytes.type;
          *(uint64_t *)symbol_buffer_write_ptr++ = *(uint64_t *)&sym_data_ptr->string_index;
        }
      }
    }
  }
  else { // not cap encoded
    sym_data_ptr = decode_new(&new_string_index);
    *(uint64_t *)symbol_buffer_write_ptr++ = *(uint64_t *)&sym_data_ptr->string_index;
    while (1) {
      if (symbol_buffer_write_ptr == symbol_buffer_end_write_ptr)
        write_symbol_buffer(&next_write_buffer);
      if ((sym_type = DecodeSymType1(0)) == 0) { // dictionary symbol
        uint8_t first_char;
        if (UTF8_compliant != 0)
          first_char = DecodeFirstChar(0, prior_end);
        else
          first_char = DecodeFirstCharBinary(prior_end);
        uint16_t bin_num = DecodeBin(sum_nbob[first_char]);
        uint8_t code_length = lookup_bits[first_char][bin_num];
        if (bin_data[first_char][code_length].nsob == 0)
          break; // EOF
        uint32_t index = get_dictionary_symbol(bin_num, code_length, first_char);
        sym_data_ptr = &bin_data[first_char][code_length].symbol_data[index];
        *(uint64_t *)symbol_buffer_write_ptr++ = *(uint64_t *)&sym_data_ptr->string_index;
        prior_end = sym_data_ptr->bytes.ends;
        if (sym_data_ptr->bytes.remaining < MAX_INSTANCES_FOR_REMOVE) {
          if (--sym_data_ptr->bytes.remaining == 0)
            dremove_dictionary_symbol(&bin_data[first_char][code_length], index);
          else if ((sym_data_ptr->bytes.type & 2) != 0) {
            if ((sym_data_ptr->bytes.remaining == 1) && ((sym_data_ptr->bytes.type & 8) == 0)) {
              (void)dadd_symbol_to_queue(sym_data_ptr, code_length, first_char);
              dremove_dictionary_symbol(&bin_data[first_char][code_length], index);
            }
            else {
              uint16_t context = 6 * sym_data_ptr->bytes.repeats;
              if (DecodeGoMtf(context, 0) != 0) {
                (void)dadd_symbol_to_queue(sym_data_ptr, code_length, first_char);
                dremove_dictionary_symbol(&bin_data[first_char][code_length], index);
              }
            }
          }

        }
        else if ((sym_data_ptr->bytes.type & 2) != 0) {
          uint16_t context = 6 * sym_data_ptr->bytes.remaining;
          if (DecodeGoMtf(context, 0) != 0) {
            (void)dadd_symbol_to_queue(sym_data_ptr, code_length, first_char);
            dremove_dictionary_symbol(&bin_data[first_char][code_length], index);
          }
        }
      }
      else if (sym_type == 1) {
        sym_data_ptr = decode_new(&new_string_index);
        *(uint64_t *)symbol_buffer_write_ptr++ = *(uint64_t *)&sym_data_ptr->string_index;
      }
      else {
        sym_data_ptr = dupdate_queue(DecodeMtfPos(queue_size));
        prior_end = sym_data_ptr->bytes.ends;
        *(uint64_t *)symbol_buffer_write_ptr++ = *(uint64_t *)&sym_data_ptr->string_index;
      }
    }
  }

  *symbol_buffer_write_ptr = MAX_U64_VALUE;
  if (two_threads != 0)
    atomic_store_explicit(&done_parsing, 1, memory_order_release);
  i = 0xFF;
  if (UTF8_compliant != 0)
    i = 0x90;
  do {
    for (j = max_code_length + 1 ; j >= min_code_length ; j--)
      free(bin_data[i][j].symbol_data);
  } while (i--);
  if (two_threads != 0)
    pthread_join(output_thread, NULL);
  else {
    write_single_threaded_output();
    uint32_t chars_to_write = out_char_ptr - start_char_ptr;
    if (stride != 0) {
      if (stride == 4)
        transpose4(start_char_ptr, chars_to_write);
      else if (stride == 2)
        transpose2(start_char_ptr, chars_to_write);
      delta_transform(start_char_ptr, chars_to_write);
    }
    if (fd != 0)
      fwrite(start_char_ptr, 1, chars_to_write, fd);
    outbuf_index += chars_to_write;
  }
  free(symbol_strings);
  free(queue_data);
  *outsize_ptr = outbuf_index;
  return(outbuf);
}

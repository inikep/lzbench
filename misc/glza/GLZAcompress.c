/***********************************************************************

Copyright 2014-2026 Kennon Conrad

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

// GLZAcompress.c
//   Iteratively does the following until there are no rules worth generating:
//     1. Counts the symbol occurences and calculates the log base 2 of each symbol's probability of occuring
//     2. Builds (in portions) the suffix tree and searches the nodes for the "most compressible" symbol strings
//     3. Invalidates less desireable strings that overlap with better ones
//     4. Replaces each occurence of the best strings with a rule symbol and adds the rule number followed by the rule
//        right hand side to the end of the data

#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "GLZA.h"

const uint32_t MAX_WRITE_SIZE = 0x200000;
const uint32_t MAX_PRIOR_MATCHES = 20;
const uint32_t MAX_MATCH_LENGTH = 8000;
const uint32_t BASE_NODES_CHILD_ARRAY_SIZE = 16;
const uint32_t NUM_PRECALCULATED_LOG2_X = 0x4000;
const uint32_t NUM_PRECALCULATED_X_LOG2_X = 0x1000000;
const uint32_t NUM_PRECALCULATED_NFSMR_LOGS = 0x400;
const uint32_t NUM_PRECALCULATED_SYMBOL_COSTS = 2000;
const uint32_t MAX_SCORES = 30000;
const uint32_t MAX_SCORES_FAST = 0x7FFF;
const float BIG_FLOAT = 1000000000.0;

uint32_t num_file_symbols, num_terminals;
uint32_t *start_symbol_ptr, *symbol_counts, *next_match_ptr[8], num_starts[0x100], num_ends[0x100], o1c[0x100][0x100];
int32_t *base_nodes_child_node_num;
int16_t *score_map;
uint8_t cap_encoded, fast_mode;
atomic_uint_least16_t rank_scores_write_index, rank_scores_read_index;
atomic_uint_least16_t substitute_data_write_index, substitute_data_read_index;
atomic_uintptr_t max_symbol_ptr, scan_symbol_ptr;
double log_file_symbols, num_file_symbols_p1_x_log_file_symbols_p1, new_rule_cost, *x_log2_x;
double order_ratio, log2_x[0x4000], nfs_profit[0x400];
float min_score;

struct node {
  uint32_t symbol;
  uint32_t last_match_index;
  int32_t sibling_node_num[2];
  int32_t child_node_num;
  uint32_t num_extra_symbols;
  uint32_t instances;
};

struct match_node {
  uint32_t symbol;
  uint32_t num_symbols;
  uint32_t score_number;
  struct match_node *child_ptr;
  uint32_t sibling_node_num[16];
  struct match_node *miss_ptr;
  struct match_node *hit_ptr;
};

struct tree_thread_data {
  uint32_t *start_cycle_symbol_ptr;
  uint32_t min_symbol;
  uint32_t max_symbol;
  uint32_t nodes_limit;
  uint32_t first_node_num;
  int32_t *base_nodes_child_node_num;
};

struct word_tree_thread_data {
  uint32_t first_node_num;
  int32_t start_positions[256];
  atomic_uint_least8_t write_index;
  atomic_uint_least8_t read_index;
};

struct node_score_data {
  float score;
  uint32_t last_match_index;
  uint32_t last_match_index2;
  uint16_t num_symbols;
};

struct rank_scores_thread_data {
  uint16_t *candidates_index;
  uint16_t max_scores;
  uint16_t num_candidates;
  uint32_t num_file_symbols;
  struct node_score_data rank_scores_buffer[0x10000];
  struct node_score_data candidates[0x8000];
  uint16_t *candidates_position;
};

struct substitute_thread_data {
  uint32_t * in_symbol_ptr;
  uint32_t * out_symbol_ptr;
  uint32_t * symbol_counts;
  uint32_t * substitute_data;
};

struct score_data {
  struct node *node_ptr;
  double string_entropy;
  double string_profit;
  float string_entropy_f;
  uint16_t num_symbols;
  uint8_t next_sibling;
};

struct overlap_check {
  uint32_t *start_symbol_ptr;
  uint32_t *stop_matches_symbol_ptr;
  uint32_t *stop_symbol_ptr;
  uint32_t **next_match_ptr_ptr;
  uint32_t num_overlaps;
  uint8_t *candidate_bad;
  struct match_node *match_nodes;
  uint32_t second[150000];
  int32_t next[150000];
};

struct find_substitutions_thread_data {
  uint32_t *start_symbol_ptr;
  uint32_t *stop_symbol_ptr;
  uint32_t extra_match_symbols;
  uint32_t data[0x800000];
  struct match_node *match_nodes;
  atomic_uchar done;
  atomic_uint_least32_t write_index;
  atomic_uint_least32_t read_index;
};

struct symbol_ends_data {
  uint8_t start;
  uint8_t end;
} *symbol_ends;

struct node * nodes;
struct node_score_data * candidates;
struct match_node ** child_ptr_array;

uint8_t get_UTF8_context(uint32_t symbol) {
  if (symbol < 0x80)
    return((uint8_t)symbol);
  else if (symbol < 0x250)
    return(0x80);
  else if (symbol < 0x370)
    return(0x81);
  else if (symbol < 0x400)
    return(0x82);
  else if (symbol < 0x530)
    return(0x83);
  else if (symbol < 0x590)
    return(0x84);
  else if (symbol < 0x600)
    return(0x85);
  else if (symbol < 0x700)
    return(0x86);
  else if (symbol < 0x800)
    return(0x87);
  else if (symbol < 0x1000)
    return(0x88);
  else if (symbol < 0x2000)
    return(0x89);
  else if (symbol < 0x3000)
    return(0x8A);
  else if (symbol < 0x3040)
    return(0x8B);
  else if (symbol < 0x30A0)
    return(0x8C);
  else if (symbol < 0x3100)
    return(0x8D);
  else if (symbol < 0x3200)
    return(0x8E);
  else if (symbol < 0xA000)
    return(0x8F);
  else if (symbol < 0x10000)
    return(0x8E);
  else
    return(0x90);
}


void init_match_node(struct match_node *match_node_ptr, uint32_t symbol, uint32_t match_num_symbols,
    uint32_t match_score_number) {
  match_node_ptr->symbol = symbol;
  match_node_ptr->num_symbols = match_num_symbols;
  match_node_ptr->score_number = match_score_number;
  match_node_ptr->child_ptr = 0;
  uint64_t * sibling_nodes_ptr = (uint64_t *)&match_node_ptr->sibling_node_num[0];
  *sibling_nodes_ptr = 0;
  *(sibling_nodes_ptr + 1) = 0;
  *(sibling_nodes_ptr + 2) = 0;
  *(sibling_nodes_ptr + 3) = 0;
  *(sibling_nodes_ptr + 4) = 0;
  *(sibling_nodes_ptr + 5) = 0;
  *(sibling_nodes_ptr + 6) = 0;
  *(sibling_nodes_ptr + 7) = 0;
  match_node_ptr->miss_ptr = 0;
  match_node_ptr->hit_ptr = 0;
  return;
}


uint8_t move_to_match_sibling(struct match_node *match_nodes, struct match_node **match_node_ptr_ptr,
    uint32_t symbol, uint8_t * sibling_number) {
  uint32_t shifted_symbol = symbol;
  *sibling_number = (uint8_t)(shifted_symbol & 0xF);
  while (symbol != (*match_node_ptr_ptr)->symbol) {
    if ((*match_node_ptr_ptr)->sibling_node_num[*sibling_number] == 0)
      return(0);
    *match_node_ptr_ptr = &match_nodes[(*match_node_ptr_ptr)->sibling_node_num[*sibling_number]];
    shifted_symbol >>= 4;
    *sibling_number = (uint8_t)(shifted_symbol & 0xF);
  }
  return(1);
}


void move_to_existing_match_sibling(struct match_node *match_nodes, struct match_node **match_node_ptr_ptr,
    uint32_t symbol) {
  uint8_t sibling_number;
  uint32_t shifted_symbol = symbol;
  while (symbol != (*match_node_ptr_ptr)->symbol) {
    sibling_number = (uint8_t)(shifted_symbol & 0xF);
    *match_node_ptr_ptr = &match_nodes[(*match_node_ptr_ptr)->sibling_node_num[sibling_number]];
    shifted_symbol >>= 4;
  }
  return;
}


uint8_t move_to_search_sibling(struct match_node *match_nodes, uint32_t symbol, struct match_node **search_node_ptr_ptr) {
  uint32_t shifted_symbol = symbol;
  uint8_t sibling_nibble = (uint8_t)(shifted_symbol & 0xF);
  while (symbol != (*search_node_ptr_ptr)->symbol) {
    if ((*search_node_ptr_ptr)->sibling_node_num[sibling_nibble] == 0)
      return(0);
    *search_node_ptr_ptr = &match_nodes[(*search_node_ptr_ptr)->sibling_node_num[sibling_nibble]];
    shifted_symbol >>= 4;
    sibling_nibble = (uint8_t)(shifted_symbol & 0xF);
  }
  return(1);
}


struct match_node * move_to_base_match_child_with_make(struct match_node *match_nodes, uint32_t symbol,
    uint32_t score_number, uint32_t *num_match_nodes_ptr, struct match_node ** child_ptr_ptr) {
  struct match_node * match_node_ptr;
  if (*child_ptr_ptr == 0) {
    *child_ptr_ptr = &match_nodes[(*num_match_nodes_ptr)++];
    match_node_ptr = *child_ptr_ptr;
    init_match_node(match_node_ptr, symbol, 2, score_number);
  } else {
    match_node_ptr = *child_ptr_ptr;
    uint8_t sibling_number;
    if (move_to_match_sibling(match_nodes, &match_node_ptr, symbol, &sibling_number) == 0) {
      match_node_ptr->sibling_node_num[sibling_number] = *num_match_nodes_ptr;
      match_node_ptr = &match_nodes[(*num_match_nodes_ptr)++];
      init_match_node(match_node_ptr, symbol, 2, score_number);
    }
  }
  return(match_node_ptr);
}


void move_to_match_child_with_make(struct match_node *match_nodes, struct match_node **match_node_ptr_ptr,
    uint32_t symbol, uint32_t score_number, uint32_t best_score_num_symbols, uint32_t *num_match_nodes_ptr) {
  if ((*match_node_ptr_ptr)->child_ptr == 0) {
    (*match_node_ptr_ptr)->child_ptr = &match_nodes[(*num_match_nodes_ptr)++];
    *match_node_ptr_ptr = (*match_node_ptr_ptr)->child_ptr;
    init_match_node(*match_node_ptr_ptr, symbol, best_score_num_symbols, score_number);
  } else {
    (*match_node_ptr_ptr) = (*match_node_ptr_ptr)->child_ptr;
    uint8_t sibling_number;
    if (move_to_match_sibling(match_nodes, match_node_ptr_ptr, symbol, &sibling_number) == 0) {
      (*match_node_ptr_ptr)->sibling_node_num[sibling_number] = *num_match_nodes_ptr;
      *match_node_ptr_ptr = &match_nodes[(*num_match_nodes_ptr)++];
      init_match_node(*match_node_ptr_ptr, symbol, best_score_num_symbols, score_number);
    }
  }
  return;
}


void write_siblings_miss_ptr(struct match_node *match_nodes, struct match_node *node_ptr, struct match_node *miss_ptr) {
  uint8_t sibling_nibble;
  node_ptr->miss_ptr = miss_ptr;
  for (sibling_nibble = 0 ; sibling_nibble < 16 ; sibling_nibble++) {
    uint32_t sibling_node_number = node_ptr->sibling_node_num[sibling_nibble];
    if (sibling_node_number != 0)
      write_siblings_miss_ptr(match_nodes, &match_nodes[sibling_node_number], miss_ptr);
  }
  return;
}


struct node * create_suffix_node(uint32_t suffix_symbol, uint32_t symbol_index,
      uint32_t * next_node_num_ptr) {
  struct node * node_ptr = &nodes[(*next_node_num_ptr)++];
  node_ptr->symbol = suffix_symbol;
  node_ptr->last_match_index = symbol_index;
  node_ptr->sibling_node_num[0] = 0;
  node_ptr->sibling_node_num[1] = 0;
  node_ptr->child_node_num = 0;
  node_ptr->num_extra_symbols = 0;
  node_ptr->instances = 1;
  return(node_ptr);
}


struct node * split_node_for_overlap(struct node * node_ptr, uint32_t split_index, uint32_t in_symbol_index,
    uint32_t * next_node_num_ptr) {
  uint32_t non_overlap_length = split_index - node_ptr->last_match_index;
  struct node * new_node_ptr = &nodes[*next_node_num_ptr];
  new_node_ptr->symbol = *(start_symbol_ptr + split_index);
  new_node_ptr->last_match_index = split_index;
  new_node_ptr->sibling_node_num[0] = 0;
  new_node_ptr->sibling_node_num[1] = 0;
  new_node_ptr->child_node_num = node_ptr->child_node_num;
  new_node_ptr->num_extra_symbols = node_ptr->num_extra_symbols - non_overlap_length;
  new_node_ptr->instances = node_ptr->instances;
  node_ptr->last_match_index = in_symbol_index;
  node_ptr->child_node_num = (*next_node_num_ptr)++;
  node_ptr->num_extra_symbols = non_overlap_length - 1;
  node_ptr->instances++;
  return(new_node_ptr);
}


void add_word_suffix(uint32_t *in_symbol_ptr, uint32_t *next_node_num_ptr) {
  uint32_t search_symbol = *in_symbol_ptr;
  int32_t * base_node_child_num_ptr;
  if (search_symbol < 0x80)
    base_node_child_num_ptr = &base_nodes_child_node_num[search_symbol];
  else
    base_node_child_num_ptr = &base_nodes_child_node_num[0x80 + (search_symbol & 0xF)];
  if (*base_node_child_num_ptr <= 0) {
    if (*base_node_child_num_ptr == 0) { // first occurence of the symbol, so create a child
      *base_node_child_num_ptr = in_symbol_ptr - start_symbol_ptr - 0x80000000;
      return;
    }
    uint32_t symbol_index = *base_node_child_num_ptr + 0x80000000;
    *base_node_child_num_ptr = *next_node_num_ptr;
    (void)create_suffix_node(*(start_symbol_ptr + symbol_index), symbol_index, next_node_num_ptr);
  }
  struct node * node_ptr = &nodes[*base_node_child_num_ptr];
  if (search_symbol != node_ptr->symbol) {  // follow siblings until match found or end of siblings found
    uint32_t shifted_search_symbol = search_symbol >> 4;
    do {
      int32_t * sibling_node_num_ptr = &node_ptr->sibling_node_num[shifted_search_symbol & 1];
      if (*sibling_node_num_ptr == 0) { // no match so add sibling
        *sibling_node_num_ptr = *next_node_num_ptr;
        (void)create_suffix_node(search_symbol, in_symbol_ptr - start_symbol_ptr, next_node_num_ptr);
        return;
      }
      node_ptr = &nodes[*sibling_node_num_ptr];
      shifted_search_symbol = shifted_search_symbol >> 1;
    } while (search_symbol != node_ptr->symbol);
  }

  // found a matching sibling
  uint32_t * first_symbol_ptr = in_symbol_ptr - 1;
  while (node_ptr->child_node_num != 0) {
    // matching sibling with child so check length of match
    uint32_t num_extra_symbols = node_ptr->num_extra_symbols;
    if (num_extra_symbols != 0) {
      uint32_t * node_symbol_ptr = start_symbol_ptr + node_ptr->last_match_index;
      uint32_t length = 1;
      do {
        if (*(node_symbol_ptr + length) != *(in_symbol_ptr + length)) { // insert node in branch
          struct node * new_node_ptr = &nodes[*next_node_num_ptr];
          new_node_ptr->last_match_index = node_ptr->last_match_index + length;
          new_node_ptr->symbol = *(node_symbol_ptr + length);
          new_node_ptr->sibling_node_num[0] = 0;
          new_node_ptr->sibling_node_num[1] = 0;
          new_node_ptr->child_node_num = node_ptr->child_node_num;
          new_node_ptr->num_extra_symbols = num_extra_symbols - length;
          new_node_ptr->instances = node_ptr->instances;
          node_ptr->num_extra_symbols = length - 1;
          node_ptr->child_node_num = (*next_node_num_ptr)++;
          node_ptr->instances++;
          new_node_ptr->sibling_node_num[(*(in_symbol_ptr + length)) & 1] = *next_node_num_ptr;
          (void)create_suffix_node(*(in_symbol_ptr + length), in_symbol_ptr + length - start_symbol_ptr,
              next_node_num_ptr);
          return;
        }
      } while (length++ != num_extra_symbols);
    }
    node_ptr->instances++;
    in_symbol_ptr += num_extra_symbols + 1;
    if (*(in_symbol_ptr - 1) == 0x20)
      return;
    search_symbol = *in_symbol_ptr;
    node_ptr = &nodes[node_ptr->child_node_num];
    if (search_symbol != node_ptr->symbol) { // follow siblings until match found or end of siblings found
      uint32_t shifted_search_symbol = search_symbol;
      do {
        int32_t * prior_node_num_ptr = &node_ptr->sibling_node_num[shifted_search_symbol & 1];
        if (*prior_node_num_ptr == 0) {
          *prior_node_num_ptr = *next_node_num_ptr;
          (void)create_suffix_node(search_symbol, in_symbol_ptr - start_symbol_ptr, next_node_num_ptr);
          return;
        }
        node_ptr = &nodes[*prior_node_num_ptr];
        shifted_search_symbol >>= 1;
      } while (search_symbol != node_ptr->symbol);
    }
  }

  // Matching node without child - extend branch, add child for previous instance, add child sibling
  node_ptr->instances = 2;
  node_ptr->child_node_num = *next_node_num_ptr;
  uint32_t * node_symbol_ptr = start_symbol_ptr + node_ptr->last_match_index;
  uint32_t * max_symbol_ptr = first_symbol_ptr + MAX_MATCH_LENGTH - 1;
  if ((*(node_symbol_ptr + 1) == *(in_symbol_ptr + 1)) && (*in_symbol_ptr != 0x20)
      && (in_symbol_ptr < max_symbol_ptr)) {
    uint32_t length = 2;
    while ((*(node_symbol_ptr + length) == *(in_symbol_ptr + length)) && (*(in_symbol_ptr + length - 1) != 0x20)
        && (in_symbol_ptr + length <= max_symbol_ptr))
      length++;
    node_ptr->num_extra_symbols = length - 1;
    node_ptr = create_suffix_node(*(node_symbol_ptr + length), node_symbol_ptr + length - start_symbol_ptr,
        next_node_num_ptr);
    node_ptr->sibling_node_num[*(in_symbol_ptr + length) & 1] = *next_node_num_ptr;
    (void)create_suffix_node(*(in_symbol_ptr + length), in_symbol_ptr + length - start_symbol_ptr,
        next_node_num_ptr);
    return;
  }
  node_ptr = create_suffix_node(*(node_symbol_ptr + 1), node_symbol_ptr + 1 - start_symbol_ptr,
      next_node_num_ptr);
  node_ptr->sibling_node_num[*(in_symbol_ptr + 1) & 1] = *next_node_num_ptr;
  (void)create_suffix_node(*(in_symbol_ptr + 1), in_symbol_ptr + 1 - start_symbol_ptr, next_node_num_ptr);
  return;
}


void add_suffix(uint32_t first_symbol, uint32_t *in_symbol_ptr, uint32_t *next_node_num_ptr) {
  struct node * node_ptr;
  uint32_t start_index = in_symbol_ptr - start_symbol_ptr - 1;
  uint32_t node_start_index = start_index + 1;
  uint32_t search_symbol = *in_symbol_ptr;
  int32_t * base_node_child_num_ptr
      = &base_nodes_child_node_num[first_symbol * BASE_NODES_CHILD_ARRAY_SIZE + (search_symbol & 0xF)];

  if (*base_node_child_num_ptr == 0) { // first occurence of the symbol, so create a child
    *base_node_child_num_ptr = node_start_index - 0x80000000;
    return;
  }
  if (*base_node_child_num_ptr < 0) {
    uint32_t symbol_index = *base_node_child_num_ptr + 0x80000000;
    uint32_t symbol = *(start_symbol_ptr + symbol_index);
    *base_node_child_num_ptr = *next_node_num_ptr;
    node_ptr = create_suffix_node(symbol, symbol_index, next_node_num_ptr);
    if (search_symbol != symbol) {
      node_ptr->sibling_node_num[(search_symbol >> 4) & 1] = node_start_index - 0x80000000;
      return;
    }
    // Matching node without child - extend branch, add child for previous instance, add child sibling
    uint32_t * node_symbol_ptr = start_symbol_ptr + node_ptr->last_match_index;
    if (*(node_symbol_ptr + 1) == *(in_symbol_ptr + 1)) {
      uint32_t length = 2;
      while ((*(node_symbol_ptr + length) == *(in_symbol_ptr + length)) && (length < MAX_MATCH_LENGTH - 1))
        length++;
      node_ptr->num_extra_symbols = length - 1;
      if (node_ptr->last_match_index + length <= start_index) {
        node_ptr->last_match_index = node_start_index;
        node_ptr->instances = 2;
      } else if (node_ptr->last_match_index < start_index)
        node_ptr = split_node_for_overlap(node_ptr, start_index, node_start_index, next_node_num_ptr);
      node_ptr->child_node_num = *next_node_num_ptr;
      node_ptr = create_suffix_node(*(node_symbol_ptr + length), node_symbol_ptr + length - start_symbol_ptr,
          next_node_num_ptr);
      node_ptr->sibling_node_num[*(in_symbol_ptr + length) & 1] = node_start_index + length - 0x80000000;
    } else {
      if (node_ptr->last_match_index < start_index) {
        node_ptr->last_match_index = node_start_index;
        node_ptr->instances = 2;
      }
      node_ptr->child_node_num = *next_node_num_ptr;
      node_ptr = create_suffix_node(*(node_symbol_ptr + 1), node_symbol_ptr + 1 - start_symbol_ptr,
          next_node_num_ptr);
      node_ptr->sibling_node_num[*(in_symbol_ptr + 1) & 1] = start_index + 2 - 0x80000000;
    }
    return;
  }

  node_ptr = &nodes[*base_node_child_num_ptr];
  if (search_symbol != node_ptr->symbol) {  // follow siblings until match found or end of siblings found
    uint32_t shifted_search_symbol = search_symbol >> 4;
    do {
      int32_t * sibling_node_num_ptr = &node_ptr->sibling_node_num[shifted_search_symbol & 1];
      if (*sibling_node_num_ptr <= 0) {
        if (*sibling_node_num_ptr == 0) { // no sibling so add sibling
          *sibling_node_num_ptr = node_start_index - 0x80000000;
          return;
        }
        // turn the sibling into a node
        uint32_t symbol_index = *sibling_node_num_ptr + 0x80000000;
        *sibling_node_num_ptr = *next_node_num_ptr;
        node_ptr = create_suffix_node(*(start_symbol_ptr + symbol_index), symbol_index, next_node_num_ptr);
        if (search_symbol != node_ptr->symbol) {
          node_ptr->sibling_node_num[(shifted_search_symbol >> 1) & 1] = node_start_index - 0x80000000;
          return;
        }
        // Matching node without child - extend branch, add child for previous instance, add child sibling
        uint32_t * node_symbol_ptr = start_symbol_ptr + node_ptr->last_match_index;
        if (*(node_symbol_ptr + 1) == *(in_symbol_ptr + 1)) {
          uint32_t length = 2;
          while ((*(node_symbol_ptr + length) == *(in_symbol_ptr + length)) && (length < MAX_MATCH_LENGTH - 1))
            length++;
          node_ptr->num_extra_symbols = length - 1;
          if (node_ptr->last_match_index + length <= start_index) {
            node_ptr->last_match_index = node_start_index;
            node_ptr->instances = 2;
          } else if (node_ptr->last_match_index < start_index)
            node_ptr = split_node_for_overlap(node_ptr, start_index, node_start_index, next_node_num_ptr);
          node_ptr->child_node_num = *next_node_num_ptr;
          node_ptr = create_suffix_node(*(node_symbol_ptr + length), node_symbol_ptr + length - start_symbol_ptr,
              next_node_num_ptr);
          node_ptr->sibling_node_num[*(in_symbol_ptr + length) & 1] = node_start_index + length - 0x80000000;
        } else {
          if (node_ptr->last_match_index < start_index) {
            node_ptr->last_match_index = node_start_index;
            node_ptr->instances = 2;
          }
          node_ptr->child_node_num = *next_node_num_ptr;
          node_ptr = create_suffix_node(*(node_symbol_ptr + 1), node_symbol_ptr + 1 - start_symbol_ptr,
              next_node_num_ptr);
          node_ptr->sibling_node_num[*(in_symbol_ptr + 1) & 1] = start_index + 2 - 0x80000000;
        }
        return;
      }
      node_ptr = &nodes[*sibling_node_num_ptr];
      shifted_search_symbol = shifted_search_symbol >> 1;
    } while (search_symbol != node_ptr->symbol);
  }

  // found a matching sibling
  while (node_ptr->child_node_num != 0) {
    // matching sibling with child so check length of match
    uint32_t num_extra_symbols = node_ptr->num_extra_symbols;
    if (num_extra_symbols != 0) {
      uint32_t * node_symbol_ptr = start_symbol_ptr + node_ptr->last_match_index;
      uint32_t length = 1;
      do {
        if (*(node_symbol_ptr + length) != *(in_symbol_ptr + length)) { // insert node in branch
          struct node * new_node_ptr = &nodes[*next_node_num_ptr];
          uint32_t new_node_lmi = node_ptr->last_match_index + length;
          new_node_ptr->last_match_index = new_node_lmi;
          new_node_ptr->symbol = *(node_symbol_ptr + length);
          new_node_ptr->sibling_node_num[0] = 0;
          new_node_ptr->sibling_node_num[1] = 0;
          new_node_ptr->child_node_num = node_ptr->child_node_num;
          new_node_ptr->num_extra_symbols = num_extra_symbols - length;
          new_node_ptr->instances = node_ptr->instances;
          node_ptr->num_extra_symbols = length - 1;
          node_ptr->child_node_num = (*next_node_num_ptr)++;
          new_node_ptr->sibling_node_num[(*(in_symbol_ptr + length)) & 1] = node_start_index + length - 0x80000000;
          if (new_node_lmi <= start_index) {
            node_ptr->last_match_index = node_start_index;
            node_ptr->instances++;
          } else if (node_ptr->last_match_index < start_index)
            (void)split_node_for_overlap(node_ptr, start_index, node_start_index, next_node_num_ptr);
          return;
        }
      } while (length++ != num_extra_symbols);
    }
    if (node_ptr->last_match_index + num_extra_symbols < start_index) {
      node_ptr->last_match_index = node_start_index;
      node_ptr->instances++;
    } else if (node_ptr->last_match_index < start_index)
      node_ptr = split_node_for_overlap(node_ptr, start_index, node_start_index, next_node_num_ptr);

    in_symbol_ptr += num_extra_symbols + 1;
    node_start_index += num_extra_symbols + 1;
    search_symbol = *in_symbol_ptr;
    node_ptr = &nodes[node_ptr->child_node_num];
    if (search_symbol != node_ptr->symbol) { // follow siblings until match found or end of siblings found
      uint32_t shifted_search_symbol = search_symbol;
      do {
        int32_t * prior_node_num_ptr = &node_ptr->sibling_node_num[shifted_search_symbol & 1];
        if (*prior_node_num_ptr == 0) {
          *prior_node_num_ptr = node_start_index - 0x80000000;
          return;
        }
        if (*prior_node_num_ptr < 0) { // turn the sibling into a node
          uint32_t symbol_index = *prior_node_num_ptr + 0x80000000;
          *prior_node_num_ptr = *next_node_num_ptr;
          node_ptr = create_suffix_node(*(start_symbol_ptr + symbol_index), symbol_index, next_node_num_ptr);
          if (search_symbol == node_ptr->symbol)
            break;
          node_ptr->sibling_node_num[(shifted_search_symbol >> 1) & 1] = node_start_index - 0x80000000;
          return;
        }
        node_ptr = &nodes[*prior_node_num_ptr];
        shifted_search_symbol >>= 1;
      } while (search_symbol != node_ptr->symbol);
    }
  }

  // Matching node without child - extend branch, add child for previous instance, add child sibling
  uint32_t * node_symbol_ptr = start_symbol_ptr + node_ptr->last_match_index;
  if (*(node_symbol_ptr + 1) == *(in_symbol_ptr + 1)) {
    int32_t max_length = start_index + MAX_MATCH_LENGTH - 1 - node_start_index;
    if (max_length > 0) {
      int32_t length = 2;
      while ((*(node_symbol_ptr + length) == *(in_symbol_ptr + length)) && (length <= max_length))
        length++;
      node_ptr->num_extra_symbols = length - 1;
      if (node_ptr->last_match_index + length <= start_index) {
        node_ptr->last_match_index = node_start_index;
        node_ptr->instances = 2;
      } else if (node_ptr->last_match_index < start_index)
        node_ptr = split_node_for_overlap(node_ptr, start_index, node_start_index, next_node_num_ptr);
      node_ptr->child_node_num = *next_node_num_ptr;
      node_ptr = create_suffix_node(*(node_symbol_ptr + length), node_symbol_ptr + length - start_symbol_ptr,
          next_node_num_ptr);
      node_ptr->sibling_node_num[*(in_symbol_ptr + length) & 1] = node_start_index + length - 0x80000000;
      return;
    }
  }
  if (node_ptr->last_match_index < start_index) {
    node_ptr->last_match_index = node_start_index;
    node_ptr->instances = 2;
  }
  node_ptr->child_node_num = *next_node_num_ptr;
  node_ptr = create_suffix_node(*(node_symbol_ptr + 1), node_symbol_ptr + 1 - start_symbol_ptr,
      next_node_num_ptr);
  node_ptr->sibling_node_num[*(in_symbol_ptr + 1) & 1] = node_start_index + 1 - 0x80000000;
  return;
}


void *build_tree_thread(void *arg) {
  struct tree_thread_data * thread_data_ptr = (struct tree_thread_data *)arg;
  uint32_t * in_symbol_ptr = thread_data_ptr->start_cycle_symbol_ptr;
  uint32_t min_symbol = thread_data_ptr->min_symbol;
  uint32_t max_symbol = thread_data_ptr->max_symbol;
  uint32_t next_node_num = thread_data_ptr->first_node_num;
  uint32_t node_num_limit = thread_data_ptr->nodes_limit - 10;
  int32_t * base_nodes_child_node_num = thread_data_ptr->base_nodes_child_node_num;

  memset(base_nodes_child_node_num + min_symbol * BASE_NODES_CHILD_ARRAY_SIZE, 0,
      4 * (max_symbol - min_symbol + 1) * BASE_NODES_CHILD_ARRAY_SIZE);
  while ((uint32_t *)atomic_load_explicit(&max_symbol_ptr, memory_order_relaxed) != in_symbol_ptr) {
    uint32_t * local_scan_symbol_ptr = (uint32_t *)atomic_load_explicit(&scan_symbol_ptr, memory_order_relaxed);
    if (in_symbol_ptr == local_scan_symbol_ptr)
      sched_yield();
    else {
      do {
        uint32_t symbol = *in_symbol_ptr++;
        if ((symbol >= min_symbol) && (symbol <= max_symbol)) {
          add_suffix(symbol, in_symbol_ptr, &next_node_num);
          if (next_node_num >= node_num_limit)
            return(0);
        }
      } while (in_symbol_ptr != local_scan_symbol_ptr);
    }
  }
  return(0);
}


void *word_build_tree_thread(void *arg) {
  struct word_tree_thread_data * thread_data_ptr = (struct word_tree_thread_data *)arg;
  uint32_t next_node_num = thread_data_ptr->first_node_num;
  uint8_t local_write_index;
  uint8_t local_read_index = 0;

  while (1) {
    while ((local_write_index = (uint8_t)atomic_load_explicit(&thread_data_ptr->write_index, memory_order_acquire))
        == local_read_index)
      sched_yield();
    do {
      if (thread_data_ptr->start_positions[local_read_index] < 0)
        return(0);
      add_word_suffix(start_symbol_ptr + thread_data_ptr->start_positions[local_read_index], &next_node_num);
      atomic_store_explicit(&thread_data_ptr->read_index, ++local_read_index, memory_order_relaxed);
    } while (local_read_index != local_write_index);
  }
}


void *rank_scores_thread(void *arg) {
  struct rank_scores_thread_data * thread_data_ptr = (struct rank_scores_thread_data *)arg;
  struct node_score_data * rank_scores_buffer = &thread_data_ptr->rank_scores_buffer[0];
  struct node_score_data * candidates = &thread_data_ptr->candidates[0];
  uint16_t score_index, node_score_num_symbols, num_candidates, node_ptrs_num, local_write_index;
  uint16_t max_scores = thread_data_ptr->max_scores;
  uint16_t * candidates_index = thread_data_ptr->candidates_index;
  float score;

  while ((local_write_index = atomic_load_explicit(&rank_scores_write_index, memory_order_acquire)) == 0); // wait
  if (rank_scores_buffer[0].last_match_index == 0) {
    thread_data_ptr->num_candidates = 0;
    return(0);
  }
  candidates_index[0] = 0;
  candidates[0].score = rank_scores_buffer[0].score;
  candidates[0].num_symbols = rank_scores_buffer[0].num_symbols;
  if (rank_scores_buffer[0].last_match_index < rank_scores_buffer[0].last_match_index2) {
    candidates[0].last_match_index = rank_scores_buffer[0].last_match_index;
    candidates[0].last_match_index2 = rank_scores_buffer[0].last_match_index2;
  } else {
    candidates[0].last_match_index = rank_scores_buffer[0].last_match_index2;
    candidates[0].last_match_index2 = rank_scores_buffer[0].last_match_index;
  }
  num_candidates = 1;
  node_ptrs_num = 1;

  while (1) {
    while ((local_write_index == node_ptrs_num)
        && ((local_write_index = atomic_load_explicit(&rank_scores_write_index, memory_order_acquire))
          == node_ptrs_num)); // wait
    if (rank_scores_buffer[node_ptrs_num].last_match_index == 0)
      break;
    score = rank_scores_buffer[node_ptrs_num].score;
    if (score > min_score) {
      // find the rank of the score
      uint16_t new_score_rank = 0;
      uint16_t max_rank = num_candidates;
      do {
        uint16_t temp_rank = (new_score_rank + max_rank) >> 1;
        if (score > candidates[candidates_index[temp_rank]].score)
          max_rank = temp_rank;
        else
          new_score_rank = temp_rank + 1;
      } while (new_score_rank != max_rank);

      // check for overlaps with candidates with better scores
      uint16_t num_symbols = rank_scores_buffer[node_ptrs_num].num_symbols;
      int32_t new_score_lmi, new_score_lmi2;
      if (rank_scores_buffer[node_ptrs_num].last_match_index < rank_scores_buffer[node_ptrs_num].last_match_index2) {
        new_score_lmi = rank_scores_buffer[node_ptrs_num].last_match_index;
        new_score_lmi2 = rank_scores_buffer[node_ptrs_num].last_match_index2;
      } else {
        new_score_lmi = rank_scores_buffer[node_ptrs_num].last_match_index2;
        new_score_lmi2 = rank_scores_buffer[node_ptrs_num].last_match_index;
      }
      int32_t new_score_pmi = new_score_lmi - num_symbols;
      int32_t new_score_pmi2 = new_score_lmi2 - num_symbols;
      uint16_t rank = 0;
      while (rank < new_score_rank) {
        score_index = candidates_index[rank];
        node_score_num_symbols = candidates[score_index].num_symbols;
        int32_t slmi2 = candidates[score_index].last_match_index2;
        if (slmi2 <= new_score_pmi)
          rank++;
        else {
          int32_t slmi1 = candidates[score_index].last_match_index;
          if (new_score_lmi2 + node_score_num_symbols <= slmi1)
            rank++;
          else if (new_score_lmi + node_score_num_symbols <= slmi2) { //score1 before newscore2
            if (slmi1 <= new_score_pmi) { // score1 after newscore1
              if ((slmi2 <= new_score_pmi2) || (new_score_lmi2 + node_score_num_symbols <= slmi2))
                rank++;
              else
                goto rank_scores_thread_node_done;
            } else if ((new_score_lmi + node_score_num_symbols <= slmi1) // score1 before newscore1
                && ((slmi2 <= new_score_pmi2)  // score2 after newscore2
                  || ((new_score_lmi2 + node_score_num_symbols <= slmi2) && (slmi1 <= new_score_pmi2))))
              rank++;
            else
              goto rank_scores_thread_node_done;
          } else
            goto rank_scores_thread_node_done;
        }
      }
      // no better candidate overlaps so node will be put on the list
      // look for overlaps with lower scoring candidates that should be removed (only looks for one)
      if (rank != num_candidates) {
        do {
          score_index = candidates_index[rank];
          node_score_num_symbols = candidates[score_index].num_symbols;
          int32_t slmi2 = candidates[score_index].last_match_index2;
          if (slmi2 > new_score_pmi) {
            int32_t slmi1 = candidates[score_index].last_match_index;
            if (new_score_lmi2 + node_score_num_symbols > slmi1) {
              if ((slmi2 > new_score_pmi2) && (new_score_lmi2 + node_score_num_symbols > slmi2))
                goto rank_scores_thread_move_down;
              if (slmi1 > new_score_pmi) {
                if ((new_score_lmi + node_score_num_symbols > slmi1) || (slmi1 > new_score_pmi2))
                  goto rank_scores_thread_move_down;
              } else if (new_score_lmi + node_score_num_symbols > slmi2)
                goto rank_scores_thread_move_down;
            }
          }

        } while (++rank != num_candidates);
      }

      if (num_candidates != max_scores) { // increment the list length if not at limit
        candidates_index[num_candidates] = num_candidates;
        num_candidates++;
      } else // otherwise throw away the lowest score instead of moving it
        rank--;

rank_scores_thread_move_down:
      // move the lower scoring nodes down one location
      score_index = candidates_index[rank];
//    memmove(&candidates_index[new_score_rank + 1], &candidates_index[new_score_rank], 2 * (rank - new_score_rank));
//    (can fail due to DF corruption during rep movsq)
      uint16_t * score_ptr = &candidates_index[new_score_rank];
      uint16_t * candidate_ptr = &candidates_index[rank];
      if (candidate_ptr >= score_ptr + 8) {
        uint64_t first_four = *(uint64_t *)&candidates_index[new_score_rank];
        uint64_t next_four = *(uint64_t *)&candidates_index[new_score_rank + 4];
        do {
          *candidate_ptr = *(candidate_ptr - 1);
          *(candidate_ptr - 1) = *(candidate_ptr - 2);
          *(candidate_ptr - 2) = *(candidate_ptr - 3);
          *(candidate_ptr - 3) = *(candidate_ptr - 4);
          *(candidate_ptr - 4) = *(candidate_ptr - 5);
          *(candidate_ptr - 5) = *(candidate_ptr - 6);
          *(candidate_ptr - 6) = *(candidate_ptr - 7);
          *(candidate_ptr - 7) = *(candidate_ptr - 8);
        } while ((candidate_ptr -= 8) >= score_ptr + 8);
        *(uint64_t *)&candidates_index[new_score_rank + 1] = first_four;
        *(uint64_t *)&candidates_index[new_score_rank + 5] = next_four;
      } else if (candidate_ptr >= score_ptr + 4)  {
        uint64_t first_four = *(uint64_t *)&candidates_index[new_score_rank];
        *(uint64_t *)(candidate_ptr - 3) = *(uint64_t *)(candidate_ptr - 4);
        *(uint64_t *)&candidates_index[new_score_rank + 1] = first_four;
      } else if (candidate_ptr >= score_ptr + 2)  {
        uint16_t first = candidates_index[new_score_rank];
        *(uint32_t *)(candidate_ptr - 1) = *(uint32_t *)(candidate_ptr - 2);
        candidates_index[new_score_rank + 1] = first;
      } else if (candidate_ptr > score_ptr) {
        *candidate_ptr = *(candidate_ptr - 1);
      }
      candidates_index[new_score_rank] = score_index;

      // save the new score
      candidates[score_index].score = score;
      candidates[score_index].num_symbols = num_symbols;
      candidates[score_index].last_match_index = new_score_lmi;
      candidates[score_index].last_match_index2 = new_score_lmi2;
      if (num_candidates == max_scores)
        min_score = candidates[candidates_index[max_scores - 1]].score;
    }
rank_scores_thread_node_done:
    atomic_store_explicit(&rank_scores_read_index, ++node_ptrs_num, memory_order_relaxed);
  }
  thread_data_ptr->num_candidates = num_candidates;
  return(0);
}


void *rank_scores_thread_fast(void *arg) {
  struct rank_scores_thread_data * thread_data_ptr = (struct rank_scores_thread_data *)arg;
  struct node_score_data * rank_scores_buffer = &thread_data_ptr->rank_scores_buffer[0];
  struct node_score_data * candidates = &thread_data_ptr->candidates[0];
  uint32_t new_score_lmi, slmi, i;
  uint16_t max_scores = thread_data_ptr->max_scores;
  uint16_t * candidates_index = thread_data_ptr->candidates_index;
  uint16_t * candidates_position = thread_data_ptr->candidates_position;
  uint16_t num_symbols, score_index, num_found_overlaps, prior_score;
  uint16_t position, next_position, min_position, max_position, first_unused_position, section, min_section, max_section;
  uint16_t new_score_rank, max_new_score_rank, num_candidates, candidate_index;
  uint16_t found_overlaps[MAX_SCORES_FAST];
  uint16_t local_write_index = 0;
  uint16_t node_ptrs_num = 0;
  uint8_t candidates_index_starts[0x80];
  float score;

  memset(candidates_index_starts, 0 , 0x80);
  memset(score_map, 0, 2 * thread_data_ptr->num_file_symbols);
  for (i = 0 ; i < max_scores ; i++)
    candidates_index[i] = i;
  num_candidates = 0;

  while (1) {
    while ((local_write_index == node_ptrs_num)
        && ((local_write_index = atomic_load_explicit(&rank_scores_write_index, memory_order_acquire))
          == node_ptrs_num)); // wait
    if (rank_scores_buffer[node_ptrs_num].last_match_index == 0)
      break;
    score = rank_scores_buffer[node_ptrs_num].score;
    if (score > min_score) {
      // find the rank of the score
      max_section = num_candidates >> 8;
      min_section = 0;
      while (min_section != max_section) {
        section = (min_section + max_section) >> 1;
        if (score > candidates[candidates_index[0x100 * section
            + (uint16_t)((uint8_t)(candidates_index_starts[section] - 1))]].score)
          max_section = section;
        else
          min_section = section + 1;
      }
      section = max_section;
      if (num_candidates > 0x100 * section + 0xFF)
        max_new_score_rank = 0x100 * section + 0xFF;
      else
        max_new_score_rank = num_candidates;
      new_score_rank = 0x100 * section;
      while (max_new_score_rank != new_score_rank) {
        uint16_t temp_rank = (max_new_score_rank + new_score_rank) >> 1;
        if (score > candidates[candidates_index[0x100 * section
            + (uint16_t)((uint8_t)(temp_rank + candidates_index_starts[section]))]].score)
          max_new_score_rank = temp_rank;
        else
          new_score_rank = temp_rank + 1;
      }

      // make overlap list and check for overlaps with candidates with better scores
      num_symbols = rank_scores_buffer[node_ptrs_num].num_symbols;
      new_score_lmi = rank_scores_buffer[node_ptrs_num].last_match_index;
      num_found_overlaps = 0;
      prior_score = 0;
      for (i = new_score_lmi - num_symbols + 1 ; i <= new_score_lmi ; i++) {
        if ((score_map[i] != 0) && (score_map[i] != prior_score)) {
          prior_score = score_map[i];
          uint8_t duplicate = 0;
          uint16_t j;
          for (j = 0 ; j < num_found_overlaps ; j++)
            if (found_overlaps[j] == score_map[i] - 1)
              duplicate = 1;
          if (duplicate == 0) {
            section = candidates_position[score_map[i] - 1] >> 8;
            if (new_score_rank
                > 0x100 * section + (uint8_t)(candidates_position[score_map[i] - 1] - candidates_index_starts[section]))
              goto rank_scores_thread_fast_node_done;
            found_overlaps[num_found_overlaps++] = score_map[i] - 1;
          }
        }
      }

      if (num_found_overlaps != 0) { // sort overlap list
        uint16_t j, k;
        for (j = 0 ; j < num_found_overlaps - 1 ; j++) {
          for (k = j + 1 ; k < num_found_overlaps ; k++) {
            section = candidates_position[found_overlaps[j]] >> 8;
            uint16_t rank_j = 0x100 * section
              + (uint8_t)(candidates_position[found_overlaps[j]] - candidates_index_starts[section]);
            section = candidates_position[found_overlaps[k]] >> 8;
            uint16_t rank_k = 0x100 * section
              + (uint8_t)(candidates_position[found_overlaps[k]] - candidates_index_starts[section]);
            if (rank_k < rank_j) {
              uint16_t temp_found_overlap = found_overlaps[j];
              found_overlaps[j] = found_overlaps[k];
              found_overlaps[k] = temp_found_overlap;
            }
          }
        }

        score_index = found_overlaps[0];
        first_unused_position = candidates_position[score_index];
        slmi = candidates[score_index].last_match_index;
        for (i = slmi - candidates[score_index].num_symbols + 1 ; i <= slmi ; i++)
          score_map[i] = 0;
        section = first_unused_position >> 8;

        while (--num_found_overlaps != 0) { // remove lower scoring overlaps
          score_index = found_overlaps[num_found_overlaps];
          position = candidates_position[score_index];
          slmi = candidates[score_index].last_match_index;
          for (i = slmi - candidates[score_index].num_symbols + 1 ; i <= slmi ; i++)
            score_map[i] = 0;
          section = position >> 8;
          max_section = --num_candidates >> 8;
          if (section != max_section) {
            max_position = 0x100 * section + (uint8_t)(candidates_index_starts[section] - 1);
            while (position != max_position) {
              next_position = (position & 0xFF00) + (uint8_t)(position + 1);
              candidates_index[position] = candidates_index[next_position];
              candidates_position[candidates_index[position]] = position;
              position = next_position;
            }
            section++;
            next_position = 0x100 * section + candidates_index_starts[section];
            candidates_index[position] = candidates_index[next_position];
            candidates_position[candidates_index[position]] = position;
            while (section != max_section) {
              candidates_index_starts[section++]++;
              position = next_position;
              next_position = 0x100 * section + candidates_index_starts[section];
              candidates_index[position] = candidates_index[next_position];
              candidates_position[candidates_index[position]] = position;
            }
            position = next_position;
          }
          max_position = (num_candidates & 0xFF00) + (uint8_t)(num_candidates + candidates_index_starts[section]);
          while (position != max_position) {
            next_position = (position & 0xFF00) + (uint8_t)(position + 1);
            candidates_index[position] = candidates_index[next_position];
            candidates_position[candidates_index[position]] = position;
            position = next_position;
          }
          candidates_index[position] = score_index;
        }
        section = first_unused_position >> 8;
      } else if (num_candidates != max_scores) { // increment the list length if not at limit
        section = num_candidates >> 8;
        first_unused_position = 0x100 * section + ((uint8_t)(num_candidates + candidates_index_starts[section]));
        num_candidates++;
      } else { // otherwise remove the lowest score
        section = (num_candidates - 1) >> 8;
        first_unused_position = 0x100 * section + ((uint8_t)(num_candidates - 1 + candidates_index_starts[section]));
        candidate_index = candidates_index[first_unused_position];
        for (i = candidates[candidate_index].last_match_index - candidates[candidate_index].num_symbols + 1 ;
            i <= candidates[candidate_index].last_match_index ; i++)
          score_map[i] = 0;
      }

      // move the lower scoring nodes down one location
      position = first_unused_position;
      score_index = candidates_index[position];  // save the index - use later to hold new score
      min_section = new_score_rank >> 8;
      if (section != min_section) {
        min_position = 0x100 * section + candidates_index_starts[section];
        while (position != min_position) {
          next_position = (position & 0xFF00) + (uint8_t)(position - 1);
          candidates_index[position] = candidates_index[next_position];
          candidates_position[candidates_index[position]] = position;
          position = next_position;
        }
        section--;
        next_position = 0x100 * section + (uint8_t)(candidates_index_starts[section] - 1);
        candidates_index[position] = candidates_index[next_position];
        candidates_position[candidates_index[position]] = position;
        position = next_position;
        while (section != min_section) {
          --candidates_index_starts[section--];
          next_position = 0x100 * section + (uint8_t)(candidates_index_starts[section] - 1);
          candidates_index[position] = candidates_index[next_position];
          candidates_position[candidates_index[position]] = position;
          position = next_position;
        }
      }
      min_position = (0xFF00 & new_score_rank) + (uint8_t)(new_score_rank + candidates_index_starts[section]);
      while (position != min_position) {
        next_position = (position & 0xFF00) + (uint8_t)(position - 1);
        candidates_index[position] = candidates_index[next_position];
        candidates_position[candidates_index[position]] = position;
        position = next_position;
      }

      // save the new score
      candidates_index[position] = score_index;
      candidates_position[score_index] = position;
      candidates[score_index].score = score;
      candidates[score_index].num_symbols = num_symbols;
      candidates[score_index].last_match_index = new_score_lmi;
      for (i = new_score_lmi - num_symbols + 1 ; i <= new_score_lmi ; i++)
        score_map[i] = score_index + 1;
      if (num_candidates == max_scores) {
        section = (max_scores - 1) >> 8;
        position = 0x100 * section + (uint16_t)((uint8_t)(max_scores - 1 + candidates_index_starts[section]));
        min_score = candidates[candidates_index[position]].score;
      }
    }
rank_scores_thread_fast_node_done:
    atomic_store_explicit(&rank_scores_read_index, ++node_ptrs_num, memory_order_relaxed);
  }
  thread_data_ptr->num_candidates = num_candidates;
  if (num_candidates != 0) {
    max_section = (num_candidates - 1) >> 8;
    section = 1;
    uint16_t temp[0x100];
    while (section < max_section) {
      if (candidates_index_starts[section] != 0) {
        memcpy(&temp[0], &candidates_index[0x100 * section], 0x200);
        memcpy(&candidates_index[0x100 * section], &temp[candidates_index_starts[section]],
            0x200 - 2 * candidates_index_starts[section]);
        memcpy(&candidates_index[0x100 * section] + (0x100 - candidates_index_starts[section]),
            &temp[0], 2 * candidates_index_starts[section]);
      }
      section++;
    }
  }
  return(0);
}


void *rank_word_scores_thread(void *arg) {
  struct rank_scores_thread_data * thread_data_ptr = (struct rank_scores_thread_data *)arg;
  struct node_score_data * rank_scores_buffer = &thread_data_ptr->rank_scores_buffer[0];
  struct node_score_data * candidates = &thread_data_ptr->candidates[0];
  uint16_t max_scores = thread_data_ptr->max_scores;
  uint16_t * candidates_index = thread_data_ptr->candidates_index;
  uint16_t score_index;
  uint16_t local_write_index = 0;
  uint16_t node_ptrs_num = 0;
  uint16_t num_candidates = 0;

  while (1) {
    while ((local_write_index == node_ptrs_num)
        && ((local_write_index = atomic_load_explicit(&rank_scores_write_index, memory_order_acquire))
          == node_ptrs_num)); // wait
    if (rank_scores_buffer[node_ptrs_num].last_match_index == 0)
      break;
    float score = rank_scores_buffer[node_ptrs_num].score;
    if (score > min_score) {
      // find the position in the score list this node would go in
      uint16_t rank, new_score_rank, candidate_search_size;
      new_score_rank = num_candidates;
      candidate_search_size = num_candidates + 1;
      do {
        candidate_search_size = (candidate_search_size + 1) >> 1;
        if (candidate_search_size > new_score_rank)
          candidate_search_size = new_score_rank;
        if (score > candidates[candidates_index[new_score_rank - candidate_search_size]].score)
          new_score_rank -= candidate_search_size;
      } while (candidate_search_size > 1);

      if (num_candidates != max_scores) { // increment the list length if not at limit
        candidates_index[num_candidates] = num_candidates;
        num_candidates++;
      }
      score_index = candidates_index[num_candidates - 1];
      memmove(&candidates_index[new_score_rank + 1], &candidates_index[new_score_rank],
          2 * (num_candidates - 1 - new_score_rank));
      candidates_index[new_score_rank] = score_index;
      candidates[score_index].score = score;
      candidates[score_index].num_symbols = rank_scores_buffer[node_ptrs_num].num_symbols;
      candidates[score_index].last_match_index = rank_scores_buffer[node_ptrs_num].last_match_index;
      if (num_candidates == max_scores)
        min_score = candidates[candidates_index[max_scores - 1]].score;
    }
    atomic_store_explicit(&rank_scores_read_index, ++node_ptrs_num, memory_order_relaxed);
  }
  thread_data_ptr->num_candidates = num_candidates;
  return(0);
}


void score_base_node_tree(struct node *node_ptr, struct score_data *node_data, double profit_ratio_power,
    double *symbol_entropy, struct node_score_data *rank_scores_buffer, uint16_t *node_ptrs_num_ptr, uint32_t prior_symbol) {
  uint32_t instances, node_instances;
  uint16_t num_symbols = 2;
  uint16_t level = 0;
  uint16_t node_ptrs_num = *node_ptrs_num_ptr;
  double repeats, profit_per_substitution2, bits_saved, bits_saved2, string_profit, string_entropy2;
  double string_entropy = symbol_entropy[prior_symbol];
  double first_symbol_entropy = string_entropy;

  if (symbol_counts[prior_symbol] < NUM_PRECALCULATED_X_LOG2_X)
    string_profit = -x_log2_x[symbol_counts[prior_symbol]] - new_rule_cost;
  else
    string_profit = -(double)symbol_counts[prior_symbol] * log2((double)symbol_counts[prior_symbol]) - new_rule_cost;
  if ((node_ptr->instances == symbol_counts[prior_symbol]) &&  (prior_symbol >= num_terminals))
    string_profit += new_rule_cost;

  while (1) {
    node_instances = node_ptr->instances;
    if (node_instances >= 2)  {
      if ((node_ptr->sibling_node_num[0] > 0) || (node_ptr->sibling_node_num[1] > 0)) {
        node_data[level].string_entropy = string_entropy;
        node_data[level].string_profit = string_profit;
        node_data[level].node_ptr = node_ptr;
        node_data[level].num_symbols = num_symbols;
        node_data[level++].next_sibling = (node_ptr->sibling_node_num[0] <= 0);
      }
      uint32_t num_extra_symbols = node_ptr->num_extra_symbols;
      repeats = (double)(node_instances - 1);
      if (node_instances <= NUM_PRECALCULATED_X_LOG2_X)
        bits_saved = x_log2_x[node_instances - 1];
      else
        bits_saved = repeats * log2(repeats);
      uint32_t * symbol_ptr = start_symbol_ptr + node_ptr->last_match_index - num_symbols + 1;
      do {
        instances = symbol_counts[*symbol_ptr];
        if (instances - node_instances + 1 < NUM_PRECALCULATED_X_LOG2_X)
          bits_saved += x_log2_x[instances - node_instances + 1];
        else
          bits_saved += (double)(instances - node_instances + 1) * log2((double)(instances - node_instances + 1));
      } while (++symbol_ptr < start_symbol_ptr + node_ptr->last_match_index);

      uint32_t * end_symbol_ptr = start_symbol_ptr + node_ptr->last_match_index + num_extra_symbols;
      while (symbol_ptr <= end_symbol_ptr) {
        instances = symbol_counts[*symbol_ptr];
        if (instances < NUM_PRECALCULATED_X_LOG2_X) {
          string_profit -= x_log2_x[instances];
          bits_saved += x_log2_x[instances - node_instances + 1];
        } else {
          string_profit -= (double)instances * log2((double)instances);
          bits_saved += (double)(instances - node_instances + 1) * log2((double)(instances - node_instances + 1));
        }
        if ((node_instances == instances) && (*symbol_ptr >= num_terminals))
          string_profit += new_rule_cost;
        string_entropy += symbol_entropy[*symbol_ptr++];
      }
      bits_saved += (double)((node_instances - 1) * (num_symbols + num_extra_symbols - 1)) * nfs_profit[1];
      bits_saved += string_profit;

      // calculate score
      if (bits_saved > 0.0) {
        double score = (profit_ratio_power + 1.0) * log2(bits_saved) - profit_ratio_power * log2(repeats * string_entropy);
        if (order_ratio == 0.0)
          score += 40.0;
        else if ((score > 2.0 * min_score - 98.0) || (score > min_score - 40.5)) {
          string_entropy2 = first_symbol_entropy;
          symbol_ptr = start_symbol_ptr + node_ptr->last_match_index - num_symbols + 2;
          while (symbol_ptr <= end_symbol_ptr) {
            string_entropy2 += log2(((double)num_ends[symbol_ends[*(symbol_ptr - 1)].end] - 0.9 * repeats)
                * (double)num_starts[symbol_ends[*symbol_ptr].start]
              / (((double)o1c[symbol_ends[*(symbol_ptr - 1)].end][symbol_ends[*symbol_ptr].start] - 0.9 * repeats)
                * (double)symbol_counts[*symbol_ptr]));
            symbol_ptr++;
          }
          if (node_instances <= NUM_PRECALCULATED_LOG2_X)
            profit_per_substitution2 = string_entropy2 + log2_x[node_instances - 1] - log_file_symbols;
          else
            profit_per_substitution2 = string_entropy2 + log2(repeats) - log_file_symbols;
          bits_saved2 = repeats * profit_per_substitution2 - new_rule_cost;
          if (bits_saved2 > 0.0) {
            score = score * ((float)1.0 - (float)order_ratio)
              + (float)(order_ratio * (log2(bits_saved2) + profit_ratio_power * log2(profit_per_substitution2 / string_entropy2)));
            score += 40.0;
          }
        }
        if (score > min_score) {
          struct node * child_ptr = &nodes[node_ptr->child_node_num];
          if ((node_ptrs_num & 0xFFF) == 0)
            while ((uint16_t)(node_ptrs_num - atomic_load_explicit(&rank_scores_read_index, memory_order_acquire))
                >= 0xF000); // wait
          rank_scores_buffer[node_ptrs_num].score = score;
          rank_scores_buffer[node_ptrs_num].num_symbols = num_symbols + num_extra_symbols;
          rank_scores_buffer[node_ptrs_num].last_match_index = child_ptr->last_match_index - 1;
          rank_scores_buffer[node_ptrs_num].last_match_index2 = node_ptr->last_match_index + num_extra_symbols;
          if (rank_scores_buffer[node_ptrs_num].last_match_index == rank_scores_buffer[node_ptrs_num].last_match_index2) {
            int32_t * sibling_node_num_ptr = &child_ptr->sibling_node_num[0];
            if (*sibling_node_num_ptr > 0)
              rank_scores_buffer[node_ptrs_num].last_match_index = nodes[*sibling_node_num_ptr].last_match_index - 1;
            else if (*sibling_node_num_ptr != 0)
              rank_scores_buffer[node_ptrs_num].last_match_index = *sibling_node_num_ptr + 0x7FFFFFFF;
            else if (*(sibling_node_num_ptr + 1) > 0)
              rank_scores_buffer[node_ptrs_num].last_match_index = nodes[*(sibling_node_num_ptr + 1)].last_match_index - 1;
            else if (*(sibling_node_num_ptr + 1) != 0)
              rank_scores_buffer[node_ptrs_num].last_match_index = *(sibling_node_num_ptr + 1) + 0x7FFFFFFF;
          }
          atomic_store_explicit(&rank_scores_write_index, ++node_ptrs_num, memory_order_release);
        }
      }
      num_symbols += num_extra_symbols + 1;
      node_ptr = &nodes[node_ptr->child_node_num]; // move to child
    } else {
      int32_t sib_node_num = node_ptr->sibling_node_num[0];
      struct node * tnp = &nodes[sib_node_num];
      if ((sib_node_num > 0)
          && ((tnp->instances > 1) || (tnp->sibling_node_num[0] > 0) || (tnp->sibling_node_num[1] > 0))) {
        tnp = &nodes[node_ptr->sibling_node_num[1]];
        if ((node_ptr->sibling_node_num[1] > 0)
             &&((tnp->instances > 1) || (tnp->sibling_node_num[0] > 0) || (tnp->sibling_node_num[1] > 0))) {
          node_data[level].node_ptr = node_ptr;
          node_data[level].num_symbols = num_symbols;
          node_data[level].string_entropy = string_entropy;
          node_data[level].string_profit = string_profit;
          node_data[level++].next_sibling = 1;
        }
        node_ptr = &nodes[sib_node_num]; // move to sibling 0
      } else {
        sib_node_num = node_ptr->sibling_node_num[1];
        tnp = &nodes[sib_node_num];
        if ((sib_node_num > 0)
            && ((tnp->instances > 1) || (tnp->sibling_node_num[0] > 0) || (tnp->sibling_node_num[1] > 0)))
          node_ptr = &nodes[sib_node_num]; // move to sibling 1 - prior symbol unchanged (okay)
        else {
          if (level == 0) {
            *node_ptrs_num_ptr = node_ptrs_num;
            return;
          }
          string_entropy = node_data[--level].string_entropy; // pop stack
          string_profit = node_data[level].string_profit;
          num_symbols = node_data[level].num_symbols;
          node_ptr = node_data[level].node_ptr;
          if (node_data[level].next_sibling == 0) {
            if (node_ptr->sibling_node_num[1] > 0)
              node_data[level++].next_sibling = 1; // put sibling 1 on stack
            node_ptr = &nodes[node_ptr->sibling_node_num[0]]; // move to sibling 0
          } else
            node_ptr = &nodes[node_ptr->sibling_node_num[1]]; // move to sibling 1
        }
      }
    }
  }
}


void score_base_node_tree_fast(struct node *node_ptr, struct score_data *node_data, float string_entropy,
    float production_cost, float profit_ratio_power, float log2_num_symbols_plus_substitution_cost,
    float *new_symbol_cost, float *symbol_entropy, struct node_score_data *rank_scores_buffer,
    uint16_t *node_ptrs_num_ptr) {
  uint16_t num_symbols = 2;
  uint16_t level = 0;
  uint16_t node_ptrs_num = *node_ptrs_num_ptr;
  float profit_per_substitution, bits_saved;

  while (1) {
    uint32_t node_instances = node_ptr->instances;
    if (node_instances >= 2)  {
      node_data[level].string_entropy_f = string_entropy;
      uint32_t symbol = node_ptr->symbol;
      string_entropy += symbol_entropy[symbol];
      uint32_t num_extra_symbols = 0;
      float repeats = (float)(node_instances - 1);
      while (num_extra_symbols != node_ptr->num_extra_symbols) {
        symbol = *(start_symbol_ptr + node_ptr->last_match_index + ++num_extra_symbols);
        string_entropy += symbol_entropy[symbol];
      }

      // calculate score
      if (node_instances < NUM_PRECALCULATED_SYMBOL_COSTS)
        profit_per_substitution = string_entropy - new_symbol_cost[node_instances];
      else
        profit_per_substitution = string_entropy - (log2_num_symbols_plus_substitution_cost - log2f(repeats));
      if (profit_per_substitution >= 0.0) {
        bits_saved = repeats * profit_per_substitution - production_cost;
        if (bits_saved > min_score) {
          float profit_ratio = profit_per_substitution / string_entropy;
          float score = log2f(bits_saved) + profit_ratio_power * log2f(profit_ratio);
          score += 2.125;
          if (score > min_score) {
            uint32_t new_score_lmi = node_ptr->last_match_index + num_extra_symbols;
            if ((node_ptrs_num & 0xFFF) == 0)
              while ((uint16_t)(node_ptrs_num - atomic_load_explicit(&rank_scores_read_index, memory_order_acquire))
                  >= 0xF000); // wait
            rank_scores_buffer[node_ptrs_num].score = score;
            rank_scores_buffer[node_ptrs_num].last_match_index = new_score_lmi;
            rank_scores_buffer[node_ptrs_num].num_symbols = num_symbols + num_extra_symbols;
            atomic_store_explicit(&rank_scores_write_index, ++node_ptrs_num, memory_order_release);
          }
        }
      }
      if ((node_ptr->sibling_node_num[0] > 0) || (node_ptr->sibling_node_num[1] > 0)) {
        node_data[level].node_ptr = node_ptr;
        node_data[level].num_symbols = num_symbols;
        node_data[level++].next_sibling = (node_ptr->sibling_node_num[0] <= 0);
      }
      num_symbols += num_extra_symbols + 1;
      node_ptr = &nodes[node_ptr->child_node_num];
    } else {
      int32_t sib_node_num = node_ptr->sibling_node_num[0];
      struct node * tnp = &nodes[sib_node_num];
      if ((sib_node_num > 0)
          && ((tnp->instances > 1) || (tnp->sibling_node_num[0] > 0) || (tnp->sibling_node_num[1] > 0))) {
        tnp = &nodes[node_ptr->sibling_node_num[1]];
        if ((node_ptr->sibling_node_num[1] > 0)
             &&((tnp->instances > 1) || (tnp->sibling_node_num[0] > 0) || (tnp->sibling_node_num[1] > 0))) {
          node_data[level].node_ptr = node_ptr;
          node_data[level].num_symbols = num_symbols;
          node_data[level].string_entropy_f = string_entropy;
          node_data[level++].next_sibling = 1;
        }
        node_ptr = &nodes[sib_node_num];
      } else {
        sib_node_num = node_ptr->sibling_node_num[1];
        tnp = &nodes[sib_node_num];
        if ((sib_node_num > 0)
            && ((tnp->instances > 1) || (tnp->sibling_node_num[0] > 0) || (tnp->sibling_node_num[1] > 0)))
          node_ptr = &nodes[sib_node_num];
        else {
          if (level == 0) {
            *node_ptrs_num_ptr = node_ptrs_num;
            return;
          }
          string_entropy = node_data[--level].string_entropy_f;
          num_symbols = node_data[level].num_symbols;
          node_ptr = node_data[level].node_ptr;
          if (node_data[level].next_sibling == 0) {
            if (node_ptr->sibling_node_num[1] > 0)
              node_data[level++].next_sibling = 1;
            node_ptr = &nodes[node_ptr->sibling_node_num[0]];
          } else
            node_ptr = &nodes[node_ptr->sibling_node_num[1]];
        }
      }
    }
  }
}


void score_base_node_tree_cap(struct node *node_ptr, struct score_data *node_data, double profit_ratio_power,
    double *symbol_entropy, struct node_score_data *rank_scores_buffer, uint16_t *node_ptrs_num_ptr, uint32_t prior_symbol) {
  uint32_t instances, node_instances;
  uint16_t num_symbols = 2;
  uint16_t level = 0;
  uint16_t node_ptrs_num = *node_ptrs_num_ptr;
  double repeats, profit_per_substitution2, bits_saved, bits_saved2, string_profit, string_entropy2;
  double string_entropy = symbol_entropy[prior_symbol];
  double first_symbol_entropy = string_entropy;

  if (symbol_counts[prior_symbol] < NUM_PRECALCULATED_X_LOG2_X)
    string_profit = -x_log2_x[symbol_counts[prior_symbol]] - new_rule_cost;
  else
    string_profit = -(double)symbol_counts[prior_symbol] * log2((double)symbol_counts[prior_symbol]) - new_rule_cost;
  if ((node_ptr->instances == symbol_counts[prior_symbol]) && (prior_symbol >= num_terminals))
    string_profit += new_rule_cost;

  while (1) {
    node_instances = node_ptr->instances;
    if (node_instances >= 2)  {
      double score, short_score;
      int8_t send_score = -1;
      if ((node_ptr->sibling_node_num[0] > 0) || (node_ptr->sibling_node_num[1] > 0)){
        node_data[level].string_entropy = string_entropy;
        node_data[level].string_profit = string_profit;
        node_data[level].node_ptr = node_ptr;
        node_data[level].num_symbols = num_symbols;
        node_data[level++].next_sibling = (node_ptr->sibling_node_num[0] <= 0);
      }
      uint32_t num_extra_symbols = node_ptr->num_extra_symbols;
      repeats = (double)(node_instances - 1);
      if (node_instances <= NUM_PRECALCULATED_X_LOG2_X)
        bits_saved = x_log2_x[node_instances - 1];
      else
        bits_saved = repeats * log2(repeats);

      uint32_t * symbol_ptr = start_symbol_ptr + node_ptr->last_match_index - num_symbols + 1;
      do {
        instances = symbol_counts[*symbol_ptr];
        if (instances - node_instances + 1 < NUM_PRECALCULATED_X_LOG2_X)
          bits_saved += x_log2_x[instances - node_instances + 1];
        else
          bits_saved += (double)(instances - node_instances + 1) * log2((double)(instances - node_instances + 1));
      } while (++symbol_ptr < start_symbol_ptr + node_ptr->last_match_index);

      if (num_extra_symbols == 0) {
        instances = symbol_counts[node_ptr->symbol];
        if (instances < NUM_PRECALCULATED_X_LOG2_X) {
          string_profit -= x_log2_x[instances];
          bits_saved += x_log2_x[instances - node_instances + 1];
        } else {
          string_profit -= (double)instances * log2((double)instances);
          bits_saved += (double)(instances - node_instances + 1) * log2((double)(instances - node_instances + 1));
        }
        if ((node_instances == instances) && (*symbol_ptr >= num_terminals))
          string_profit += new_rule_cost;
        if ((num_symbols - 1) * (node_instances - 1) < 0x400)
          bits_saved += nfs_profit[(num_symbols - 1) * (node_instances - 1)];
        else
          bits_saved += num_file_symbols_p1_x_log_file_symbols_p1
            - (double)(num_file_symbols + 1 - (num_symbols - 1) * (node_instances - 1))
            * log2((double)(num_file_symbols + 1 - (num_symbols - 1) * (node_instances - 1)));
        bits_saved += string_profit;
        string_entropy += symbol_entropy[*symbol_ptr];

        // calculate score
        if (bits_saved > 0.0) {
          score = (profit_ratio_power + 1.0) * log2(bits_saved) - profit_ratio_power * log2(repeats * string_entropy);
          double penalty;
          if (*symbol_ptr == 0x20) {
            if (*(symbol_ptr + 1) != 0x20) {
              score -= 2.0;
              penalty = 2.0;
            } else {
              score -= 1.0;
              penalty = 1.0;
            }
          } else if ((*symbol_ptr & 0xF2) != 0x42) {
            score -= 1.0;
            penalty = 1.0;
          } else {
            penalty = 0.0;
          }
          if (order_ratio == 0.0) {
            score += 40.0;
            if (score > min_score)
              send_score = 0;
          } else if ((score > 2.0 * min_score - 98.0) || (score > min_score - 40.5)) {
            string_entropy2 = first_symbol_entropy;
            symbol_ptr = start_symbol_ptr + node_ptr->last_match_index - num_symbols + 2;
            do {
              string_entropy2 += log2(((double)num_ends[symbol_ends[*(symbol_ptr - 1)].end] - 0.9 * repeats)
                  * (double)num_starts[symbol_ends[*symbol_ptr].start]
                / (((double)o1c[symbol_ends[*(symbol_ptr - 1)].end][symbol_ends[*symbol_ptr].start] - 0.9 * repeats)
                  * (double)symbol_counts[*symbol_ptr]));
            } while (symbol_ptr++ < start_symbol_ptr + node_ptr->last_match_index);
            if (node_instances <= NUM_PRECALCULATED_LOG2_X)
              profit_per_substitution2 = string_entropy2 + log2_x[node_instances - 1] - log_file_symbols;
            else
              profit_per_substitution2 = string_entropy2 + log2(repeats) - log_file_symbols;
            bits_saved2 = repeats * profit_per_substitution2 - new_rule_cost;
            if (bits_saved2 > 0.0) {
              score = score * (1.0 - order_ratio) + (order_ratio
                * (log2(bits_saved2) + profit_ratio_power * log2(profit_per_substitution2 / string_entropy2) - penalty));
              score += 40.0;
              if (score > min_score)
                send_score = 0;
            }
          }
        }
      } else {
        uint32_t * end_symbol_ptr = start_symbol_ptr + node_ptr->last_match_index + num_extra_symbols;
        while (symbol_ptr < end_symbol_ptr) {
          instances = symbol_counts[*symbol_ptr];
          if (instances < NUM_PRECALCULATED_X_LOG2_X) {
            string_profit -= x_log2_x[instances];
            bits_saved += x_log2_x[instances - node_instances + 1];
          } else {
            string_profit -= (double)instances * log2((double)instances);
            bits_saved += (double)(instances - node_instances + 1) * log2((double)(instances - node_instances + 1));
          }
          if ((node_instances == instances) && (*symbol_ptr >= num_terminals))
            string_profit += new_rule_cost;
          string_entropy += symbol_entropy[*symbol_ptr++];
        }
        string_entropy2 = first_symbol_entropy;
        short_score = min_score;
        if ((*symbol_ptr == 0x20) && (*(symbol_ptr + 1) != 0x20)) {
          double temp_bits_saved;
          if ((node_instances - 1) * (num_symbols + num_extra_symbols - 2) < 0x400)
            temp_bits_saved = nfs_profit[(node_instances - 1) * (num_symbols + num_extra_symbols - 2)];
          else
            temp_bits_saved = num_file_symbols_p1_x_log_file_symbols_p1
              - ((double)(num_file_symbols + 1 - (node_instances - 1) * (num_symbols + num_extra_symbols - 2))
                * log2((double)(num_file_symbols + 1 - (node_instances - 1) * (num_symbols + num_extra_symbols - 2))));
          temp_bits_saved += bits_saved + string_profit;

          // calculate score
          if (temp_bits_saved > 0.0) {
            short_score = (profit_ratio_power + 1.0) * log2(temp_bits_saved)
              - profit_ratio_power * log2(repeats * string_entropy) - 1.0;
            if (order_ratio == 0.0) {
              short_score += 40.0;
              if (short_score > min_score)
                send_score = 1;
            } else if ((short_score > 2.0 * min_score - 98.0) || (short_score > min_score - 40.5)) {
              symbol_ptr = start_symbol_ptr + node_ptr->last_match_index - num_symbols + 2;
              while (symbol_ptr < end_symbol_ptr) {
                string_entropy2 += log2(((double)num_ends[symbol_ends[*(symbol_ptr - 1)].end] - 0.9 * repeats)
                    * (double)num_starts[symbol_ends[*symbol_ptr].start]
                  / (((double)o1c[symbol_ends[*(symbol_ptr - 1)].end][symbol_ends[*symbol_ptr].start] - 0.9 * repeats)
                    * (double)symbol_counts[*symbol_ptr]));
                symbol_ptr++;
              }
              if (node_instances <= NUM_PRECALCULATED_LOG2_X)
                profit_per_substitution2 = string_entropy2 + log2_x[node_instances - 1] - log_file_symbols;
              else
                profit_per_substitution2 = string_entropy2 + log2(repeats) - log_file_symbols;
              bits_saved2 = repeats * profit_per_substitution2 - new_rule_cost;
              if (bits_saved2 > 0.0) {
                short_score = short_score * (1.0 - order_ratio) + (order_ratio
                  * (log2(bits_saved2) + profit_ratio_power * log2(profit_per_substitution2 / string_entropy2) - 1.0));
                short_score += 40.0;
                if (short_score > min_score)
                  send_score = 1;
              }
            }
          }
        }

        instances = symbol_counts[*symbol_ptr];
        if (instances < NUM_PRECALCULATED_X_LOG2_X) {
          string_profit -= x_log2_x[instances];
          bits_saved += x_log2_x[instances - node_instances + 1];
        } else {
          string_profit -= (double)instances * log2((double)instances);
          bits_saved += (double)(instances - node_instances + 1) * log2((double)(instances - node_instances + 1));
        }
        if ((node_instances == instances) && (*symbol_ptr >= num_terminals))
          string_profit += new_rule_cost;
        string_entropy += symbol_entropy[*symbol_ptr];
        if ((node_instances - 1) * (num_symbols + num_extra_symbols - 1) < 0x400)
          bits_saved += nfs_profit[(node_instances - 1) * (num_symbols + num_extra_symbols - 1)];
        else
          bits_saved += num_file_symbols_p1_x_log_file_symbols_p1
            - (double)(num_file_symbols + 1 - (node_instances - 1) * (num_symbols + num_extra_symbols - 1))
            * log2((double)(num_file_symbols + 1 - (node_instances - 1) * (num_symbols + num_extra_symbols - 1)));
        bits_saved += string_profit;

        // calculate score
        if (bits_saved > 0.0) {
          score = (profit_ratio_power + 1.0) * log2(bits_saved) - profit_ratio_power * log2(repeats * string_entropy);
          double penalty;
          if (*symbol_ptr == 0x20) {
            if (*(symbol_ptr + 1) != 0x20) {
              score -= 2.0;
              penalty = 2.0;
            } else {
              score -= 1.0;
              penalty = 1.0;
            }
          } else if ((*symbol_ptr & 0xF2) != 0x42) {
            score -= 1.0;
            penalty = 1.0;
          } else {
            penalty = 0.0;
          }
          if (order_ratio == 0.0) {
            score += 40.0;
            if ((score > min_score) && (score > short_score))
              send_score = 0;
          } else if ((score > 2.0 * min_score - 98.0) || (score > min_score - 40.5)) {
            if (string_entropy2 == first_symbol_entropy) {
              symbol_ptr = start_symbol_ptr + node_ptr->last_match_index - num_symbols + 2;
              while (symbol_ptr < end_symbol_ptr) {
                string_entropy2 += log2(((double)num_ends[symbol_ends[*(symbol_ptr - 1)].end] - 0.9 * repeats)
                    * (double)num_starts[symbol_ends[*symbol_ptr].start]
                  / (((double)o1c[symbol_ends[*(symbol_ptr - 1)].end][symbol_ends[*symbol_ptr].start] - 0.9 * repeats)
                    * (double)symbol_counts[*symbol_ptr]));
                symbol_ptr++;
              }
            }
            string_entropy2 += log2(((double)num_ends[symbol_ends[*(symbol_ptr - 1)].end] - 0.9 * repeats)
                * (double)num_starts[symbol_ends[*symbol_ptr].start]
              / (((double)o1c[symbol_ends[*(symbol_ptr - 1)].end][symbol_ends[*symbol_ptr].start] - 0.9 * repeats)
                * (double)symbol_counts[*symbol_ptr]));
            if (node_instances <= NUM_PRECALCULATED_LOG2_X)
              profit_per_substitution2 = string_entropy2 + log2_x[node_instances - 1] - log_file_symbols;
            else
              profit_per_substitution2 = string_entropy2 + log2(repeats) - log_file_symbols;
            bits_saved2 = repeats * profit_per_substitution2 - new_rule_cost;
            if (bits_saved2 > 0.0) {
              score = score * (1.0 - order_ratio) + (order_ratio
                * (log2(bits_saved2) + profit_ratio_power * log2(profit_per_substitution2 / string_entropy2) - penalty));
              score += 40.0;
              if ((score > min_score) && (score > short_score))
                send_score = 0;
            }
          }
        }
      }
      if (send_score >= 0) {
        struct node * child_ptr = &nodes[node_ptr->child_node_num];
        if ((node_ptrs_num & 0xFFF) == 0)
          while ((uint16_t)(node_ptrs_num - atomic_load_explicit(&rank_scores_read_index, memory_order_acquire))
              >= 0xF000); // wait
        rank_scores_buffer[node_ptrs_num].score = score;
        rank_scores_buffer[node_ptrs_num].num_symbols = num_symbols + num_extra_symbols - send_score;
        rank_scores_buffer[node_ptrs_num].last_match_index = child_ptr->last_match_index - 1 - send_score;
        rank_scores_buffer[node_ptrs_num].last_match_index2 = node_ptr->last_match_index + num_extra_symbols - send_score;
        if (rank_scores_buffer[node_ptrs_num].last_match_index == rank_scores_buffer[node_ptrs_num].last_match_index2) {
          int32_t * sibling_node_num_ptr = &child_ptr->sibling_node_num[0];
          if (*sibling_node_num_ptr > 0)
            rank_scores_buffer[node_ptrs_num].last_match_index = nodes[*sibling_node_num_ptr].last_match_index - 1 - send_score;
          else if (*sibling_node_num_ptr != 0)
            rank_scores_buffer[node_ptrs_num].last_match_index = *sibling_node_num_ptr + 0x7FFFFFFF - send_score;
          else if (*(sibling_node_num_ptr + 1) > 0)
            rank_scores_buffer[node_ptrs_num].last_match_index = nodes[*(sibling_node_num_ptr + 1)].last_match_index - 1 - send_score;
          else if (*(sibling_node_num_ptr + 1) != 0)
            rank_scores_buffer[node_ptrs_num].last_match_index = *(sibling_node_num_ptr + 1) + 0x7FFFFFFF - send_score;
        }
        atomic_store_explicit(&rank_scores_write_index, ++node_ptrs_num, memory_order_release);
      }
      num_symbols += num_extra_symbols + 1;
      node_ptr = &nodes[node_ptr->child_node_num]; // move to child
    } else {
      int32_t sib_node_num = node_ptr->sibling_node_num[0];
      struct node * tnp = &nodes[sib_node_num];
      if ((sib_node_num > 0)
          && ((tnp->instances > 1) || (tnp->sibling_node_num[0] > 0) || (tnp->sibling_node_num[1] > 0))) {
        tnp = &nodes[node_ptr->sibling_node_num[1]];
        if ((node_ptr->sibling_node_num[1] > 0)
             &&((tnp->instances > 1) || (tnp->sibling_node_num[0] > 0) || (tnp->sibling_node_num[1] > 0))) {
          node_data[level].node_ptr = node_ptr;
          node_data[level].num_symbols = num_symbols;
          node_data[level].string_entropy = string_entropy;
          node_data[level].string_profit = string_profit;
          node_data[level++].next_sibling = 1;
        }
        node_ptr = &nodes[sib_node_num]; // move to sibling 0
      } else {
        sib_node_num = node_ptr->sibling_node_num[1];
        tnp = &nodes[sib_node_num];
        if ((sib_node_num > 0)
            && ((tnp->instances > 1) || (tnp->sibling_node_num[0] > 0) || (tnp->sibling_node_num[1] > 0)))
          node_ptr = &nodes[sib_node_num]; // move to sibling 1 - prior symbol unchanged (okay)
        else {
          if (level == 0) {
            *node_ptrs_num_ptr = node_ptrs_num;
            return;
          }
          string_entropy = node_data[--level].string_entropy; // pop stack
          string_profit = node_data[level].string_profit;
          num_symbols = node_data[level].num_symbols;
          node_ptr = node_data[level].node_ptr;
          if (node_data[level].next_sibling == 0) {
            if (node_ptr->sibling_node_num[1] > 0)
              node_data[level++].next_sibling = 1; // put sibling 1 on stack
            node_ptr = &nodes[node_ptr->sibling_node_num[0]]; // move to sibling 0
          } else
            node_ptr = &nodes[node_ptr->sibling_node_num[1]]; // move to sibling 1
        }
      }
    }
  }
}


void score_base_node_tree_cap_fast(struct node *node_ptr, struct score_data *node_data, float string_entropy,
    float production_cost, float profit_ratio_power, float log2_num_symbols_plus_substitution_cost,
    float *new_symbol_cost, float *symbol_entropy, struct node_score_data *rank_scores_buffer,
    uint16_t *node_ptrs_num_ptr) {
  uint16_t num_symbols = 2;
  uint16_t level = 0;
  uint16_t node_ptrs_num = *node_ptrs_num_ptr;
  float profit_per_substitution, bits_saved;

  while (1) {
    uint32_t node_instances = node_ptr->instances;
    if (node_instances >= 2)  {
      float score;
      float repeats = (float)(node_instances - 1);
      node_data[level].string_entropy_f = string_entropy;
      uint32_t symbol = node_ptr->symbol;
      int8_t send_score = -1;
      uint32_t num_extra_symbols = node_ptr->num_extra_symbols;
      if (num_extra_symbols == 0) {
        string_entropy += symbol_entropy[symbol];
        // calculate score
        if (node_instances < NUM_PRECALCULATED_SYMBOL_COSTS)
          profit_per_substitution = string_entropy - new_symbol_cost[node_instances];
        else
          profit_per_substitution = string_entropy - (log2_num_symbols_plus_substitution_cost - log2f(repeats));
        bits_saved = repeats * profit_per_substitution - production_cost;
        if (bits_saved > min_score) {
          float profit_ratio = profit_per_substitution / string_entropy;
          score = log2f(bits_saved) + profit_ratio_power * log2f(profit_ratio);
          if (symbol == 0x20)
            score -= 0.25;
          else if ((symbol & 0xF2) != 0x42)
            score += 1.125;
          else
            score += 2.125;
          if (score > min_score)
            send_score = 0;
        }
      } else {
        uint32_t * symbol_ptr = start_symbol_ptr + node_ptr->last_match_index;
        uint32_t * end_symbol_ptr = symbol_ptr + num_extra_symbols;
        string_entropy += symbol_entropy[*symbol_ptr++];
        while (symbol_ptr < end_symbol_ptr)
          string_entropy += symbol_entropy[*symbol_ptr++];

        if ((*symbol_ptr == 0x20) && (*(symbol_ptr - 1) != 0x20)) {
          // calculate score
          if (node_instances < NUM_PRECALCULATED_SYMBOL_COSTS)
            profit_per_substitution = string_entropy - new_symbol_cost[node_instances];
          else
            profit_per_substitution = string_entropy - (log2_num_symbols_plus_substitution_cost - log2f(repeats));
          bits_saved = repeats * profit_per_substitution - production_cost;
          if (bits_saved > min_score) {
            float profit_ratio = profit_per_substitution / (string_entropy + symbol_entropy[0x20]);
            score = log2f(bits_saved) + profit_ratio_power * log2f(profit_ratio) + 1.125;
            if (score > min_score)
              send_score = 1;
          }
        }

        string_entropy += symbol_entropy[*symbol_ptr];
        // calculate score
        if (send_score < 0) {
          if (node_instances < NUM_PRECALCULATED_SYMBOL_COSTS)
            profit_per_substitution = string_entropy - new_symbol_cost[node_instances];
          else
            profit_per_substitution = string_entropy - (log2_num_symbols_plus_substitution_cost - log2f(repeats));
          bits_saved = repeats * profit_per_substitution - production_cost;
          if (bits_saved > min_score) {
            float profit_ratio = profit_per_substitution / string_entropy;
            score = log2f(bits_saved) + profit_ratio_power * log2f(profit_ratio);
            if (*symbol_ptr == 0x20)
              score -= 0.25;
            else if (((*symbol_ptr) & 0xF2) != 0x42)
              score += 1.125;
            else
              score += 2.125;
            if (score > min_score)
              send_score = 0;
          }
        }
      }
      if (send_score >= 0) {
        uint32_t new_score_lmi = node_ptr->last_match_index + num_extra_symbols;
        if ((node_ptrs_num & 0xFFF) == 0)
          while ((uint16_t)(node_ptrs_num - atomic_load_explicit(&rank_scores_read_index, memory_order_acquire))
              >= 0xF000); // wait
        rank_scores_buffer[node_ptrs_num].score = score;
        rank_scores_buffer[node_ptrs_num].last_match_index = new_score_lmi - send_score;
        rank_scores_buffer[node_ptrs_num].num_symbols = num_symbols + num_extra_symbols - send_score;
        atomic_store_explicit(&rank_scores_write_index, ++node_ptrs_num, memory_order_release);
      }
      if ((node_ptr->sibling_node_num[0] > 0) || (node_ptr->sibling_node_num[1] > 0)) {
        node_data[level].node_ptr = node_ptr;
        node_data[level].num_symbols = num_symbols;
        node_data[level++].next_sibling = (node_ptr->sibling_node_num[0] <= 0);
      }
      num_symbols += num_extra_symbols + 1;
      node_ptr = &nodes[node_ptr->child_node_num];
    } else {
      int32_t sib_node_num = node_ptr->sibling_node_num[0];
      struct node * tnp = &nodes[sib_node_num];
      if ((sib_node_num > 0)
          && ((tnp->instances > 1) || (tnp->sibling_node_num[0] > 0) || (tnp->sibling_node_num[1] > 0))) {
        tnp = &nodes[node_ptr->sibling_node_num[1]];
        if ((node_ptr->sibling_node_num[1] > 0)
             &&((tnp->instances > 1) || (tnp->sibling_node_num[0] > 0) || (tnp->sibling_node_num[1] > 0))) {
          node_data[level].node_ptr = node_ptr;
          node_data[level].num_symbols = num_symbols;
          node_data[level].string_entropy_f = string_entropy;
          node_data[level++].next_sibling = 1;
        }
        node_ptr = &nodes[sib_node_num];
      } else {
        sib_node_num = node_ptr->sibling_node_num[1];
        tnp = &nodes[sib_node_num];
        if ((sib_node_num > 0)
            && ((tnp->instances > 1) || (tnp->sibling_node_num[0] > 0) || (tnp->sibling_node_num[1] > 0)))
          node_ptr = &nodes[sib_node_num];
        else {
          if (level == 0) {
            *node_ptrs_num_ptr = node_ptrs_num;
            return;
          }
          string_entropy = node_data[--level].string_entropy_f;
          num_symbols = node_data[level].num_symbols;
          node_ptr = node_data[level].node_ptr;
          if (node_data[level].next_sibling == 0) {
            if (node_ptr->sibling_node_num[1] > 0)
              node_data[level++].next_sibling = 1;
            node_ptr = &nodes[node_ptr->sibling_node_num[0]];
          } else
            node_ptr = &nodes[node_ptr->sibling_node_num[1]];
        }
      }
    }
  }
}


void score_base_node_tree_words(struct node* node_ptr, struct score_data *node_data, float production_cost,
    float log2_num_symbols_plus_substitution_cost, float *new_symbol_cost, float *symbol_entropy,
    struct node_score_data *rank_scores_buffer, uint16_t *node_ptrs_num_ptr) {
  int32_t sib_node_num;
  uint16_t num_symbols = 2;
  uint16_t level = 0;
  uint16_t node_ptrs_num = *node_ptrs_num_ptr;
  float string_entropy = symbol_entropy[0x20];

  while (1) {
    uint32_t node_instances = node_ptr->instances;
    node_data[level].string_entropy = string_entropy;
    if (node_instances >= 2) {
      uint32_t num_extra_symbols = 0;
      while (num_extra_symbols != node_ptr->num_extra_symbols)
        string_entropy += symbol_entropy[*(start_symbol_ptr + node_ptr->last_match_index + num_extra_symbols++)];
      if (*(start_symbol_ptr + node_ptr->last_match_index + num_extra_symbols) == 0x20) {
        // calculate score
        uint32_t last_symbol = *(start_symbol_ptr + node_ptr->last_match_index + num_extra_symbols - 1);
        if (((last_symbol >= (uint32_t)'a') && (last_symbol <= (uint32_t)'z'))
            || ((last_symbol >= (uint32_t)'0') && (last_symbol <= (uint32_t)'9')) || (last_symbol >= 0x80)) {
          float repeats = (float)(node_instances - 1);
          float profit_per_substitution;
          if (node_instances < NUM_PRECALCULATED_SYMBOL_COSTS)
            profit_per_substitution = string_entropy - new_symbol_cost[node_instances];
          else
            profit_per_substitution = string_entropy - (log2_num_symbols_plus_substitution_cost - log2f(repeats));
          if (profit_per_substitution >= 0.0) {
            float score = repeats * profit_per_substitution - production_cost;
            if (score > min_score) {
              if ((node_ptrs_num & 0xFFF) == 0)
                while ((uint16_t)(node_ptrs_num - atomic_load_explicit(&rank_scores_read_index, memory_order_acquire))
                    >= 0xF000); // wait
              rank_scores_buffer[node_ptrs_num].score = score;
              rank_scores_buffer[node_ptrs_num].last_match_index = node_ptr->last_match_index + num_extra_symbols - 1;
              rank_scores_buffer[node_ptrs_num].num_symbols = num_symbols + num_extra_symbols - 1;
              atomic_store_explicit(&rank_scores_write_index, ++node_ptrs_num, memory_order_release);
            }
          }
        }
        goto score_siblings;
      }
      string_entropy += symbol_entropy[*(start_symbol_ptr + node_ptr->last_match_index + num_extra_symbols)];
      if ((node_ptr->sibling_node_num[0] > 0) || (node_ptr->sibling_node_num[1] > 0)) {
        node_data[level].node_ptr = node_ptr;
        node_data[level].num_symbols = num_symbols;
        node_data[level++].next_sibling = (node_ptr->sibling_node_num[0] <= 0);
      }
      num_symbols += num_extra_symbols + 1;
      node_ptr = &nodes[node_ptr->child_node_num];
    } else {
score_siblings:
      sib_node_num = node_ptr->sibling_node_num[0];
      struct node * tnp = &nodes[sib_node_num];
      if ((sib_node_num > 0)
          && ((tnp->instances > 1) || (tnp->sibling_node_num[0] > 0) || (tnp->sibling_node_num[1] > 0))) {
        tnp = &nodes[node_ptr->sibling_node_num[1]];
        if ((node_ptr->sibling_node_num[1] > 0) &&
            ((tnp->instances > 1) || (tnp->sibling_node_num[0] > 0) || (tnp->sibling_node_num[1] > 0))) {
          node_data[level].node_ptr = node_ptr;
          node_data[level].num_symbols = num_symbols;
          node_data[level++].next_sibling = 1;
        }
        node_ptr = &nodes[sib_node_num];
      } else {
        sib_node_num = node_ptr->sibling_node_num[1];
        if (sib_node_num > 0)
          node_ptr = &nodes[sib_node_num];
        else {
          if (level == 0) {
            *node_ptrs_num_ptr = node_ptrs_num;
            return;
          }
          string_entropy = node_data[--level].string_entropy;
          num_symbols = node_data[level].num_symbols;
          node_ptr = node_data[level].node_ptr;
          if (node_data[level].next_sibling == 0) {
            if (node_ptr->sibling_node_num[1] > 0)
              node_data[level++].next_sibling = 1;
            node_ptr = &nodes[node_ptr->sibling_node_num[0]];
          } else
            node_ptr = &nodes[node_ptr->sibling_node_num[1]];
        }
      }
    }
  }
}


void score_symbol_tree(uint32_t min_symbol, uint32_t max_symbol, struct node_score_data *rank_scores_buffer,
    struct score_data *node_data, uint16_t *node_ptrs_num_ptr, double profit_ratio_power, double *symbol_entropy,
    uint32_t * symbol_counts) {
  int32_t * next_base_node_child_num_ptr;
  int32_t * base_node_child_num_ptr = &base_nodes_child_node_num[min_symbol * BASE_NODES_CHILD_ARRAY_SIZE];
  uint32_t symbol = min_symbol;
  while (symbol <= max_symbol) {
    if (symbol_counts[symbol] > 1) {
      next_base_node_child_num_ptr = base_node_child_num_ptr + BASE_NODES_CHILD_ARRAY_SIZE;
      do {
        if (*base_node_child_num_ptr > 0) {
          if (cap_encoded != 0)
            score_base_node_tree_cap(&nodes[*base_node_child_num_ptr], node_data, profit_ratio_power, symbol_entropy,
                rank_scores_buffer, node_ptrs_num_ptr, symbol);
          else
            score_base_node_tree(&nodes[*base_node_child_num_ptr], node_data, profit_ratio_power, symbol_entropy,
                rank_scores_buffer, node_ptrs_num_ptr, symbol);
        }
        base_node_child_num_ptr++;
      } while (base_node_child_num_ptr != next_base_node_child_num_ptr);
    } else
      base_node_child_num_ptr += 16;
    symbol++;
  }
  return;
}


void score_symbol_tree_fast(uint32_t min_symbol, uint32_t max_symbol, struct node_score_data *rank_scores_buffer,
    struct score_data *node_data, uint16_t *node_ptrs_num_ptr, float production_cost, double profit_ratio_power,
    float log2_num_symbols_plus_substitution_cost, float *new_symbol_cost, float *symbol_entropy,
    uint32_t * symbol_counts) {
  int32_t * next_base_node_child_num_ptr;
  int32_t * base_node_child_num_ptr = &base_nodes_child_node_num[min_symbol * BASE_NODES_CHILD_ARRAY_SIZE];
  uint32_t symbol = min_symbol;
  while (symbol <= max_symbol) {
    if (symbol_counts[symbol] > 1) {
      next_base_node_child_num_ptr = base_node_child_num_ptr + BASE_NODES_CHILD_ARRAY_SIZE;
      do {
        if (*base_node_child_num_ptr > 0) {
          if (cap_encoded != 0)
            score_base_node_tree_cap_fast(&nodes[*base_node_child_num_ptr], node_data, symbol_entropy[symbol],
                production_cost, (float)profit_ratio_power, log2_num_symbols_plus_substitution_cost, new_symbol_cost,
                symbol_entropy, rank_scores_buffer, node_ptrs_num_ptr);
          else
            score_base_node_tree_fast(&nodes[*base_node_child_num_ptr], node_data, symbol_entropy[symbol],
                production_cost, (float)profit_ratio_power, log2_num_symbols_plus_substitution_cost, new_symbol_cost,
                symbol_entropy, rank_scores_buffer, node_ptrs_num_ptr);
        }
        base_node_child_num_ptr++;
      } while (base_node_child_num_ptr != next_base_node_child_num_ptr);
    } else
      base_node_child_num_ptr += 16;
    symbol++;
  }
  return;
}


void score_symbol_tree_words(struct node_score_data *rank_scores_buffer, struct score_data *node_data,
    uint16_t *node_ptrs_num_ptr, float production_cost, float log2_num_symbols_plus_substitution_cost,
    float *new_symbol_cost, float *symbol_entropy) {
  int32_t * base_node_child_num_ptr = &base_nodes_child_node_num[0];
  int32_t * base_node_child_num_end_ptr = &base_nodes_child_node_num[0x90];
  do {
    if (*base_node_child_num_ptr > 0)
      score_base_node_tree_words(&nodes[*base_node_child_num_ptr], node_data, production_cost,
          log2_num_symbols_plus_substitution_cost, new_symbol_cost, symbol_entropy, rank_scores_buffer,
          node_ptrs_num_ptr);
  } while (++base_node_child_num_ptr <= base_node_child_num_end_ptr);
  return;
}


void *overlap_check_thread(void *arg) {
  struct overlap_check * thread_data_ptr = (struct overlap_check *)arg;
  struct match_node * match_nodes = thread_data_ptr->match_nodes;
  struct match_node *match_node_ptr;
  uint32_t * in_symbol_ptr = thread_data_ptr->start_symbol_ptr;
  uint32_t * end_symbol_ptr = thread_data_ptr->stop_symbol_ptr;
  uint8_t * candidate_bad = thread_data_ptr->candidate_bad;
  uint32_t num_overlaps = thread_data_ptr->num_overlaps;
  uint32_t symbol, prior_match_score_number[MAX_PRIOR_MATCHES], *prior_match_end_ptr[MAX_PRIOR_MATCHES];
  uint32_t num_prior_matches = 0;

  for (symbol = 0 ; symbol < num_overlaps ; symbol++)
    thread_data_ptr->next[symbol] = -1;

thread_overlap_check_loop_no_match:
  symbol = *in_symbol_ptr++;
  if (in_symbol_ptr >= end_symbol_ptr)
    return(0);
  if (((int32_t)symbol < 0) || (child_ptr_array[symbol] == 0))
    goto thread_overlap_check_loop_no_match;
  match_node_ptr = child_ptr_array[symbol];
thread_overlap_check_loop_match:
  symbol = *in_symbol_ptr++;
  if (symbol != match_node_ptr->symbol) {
    uint32_t shifted_symbol = symbol;
    do {
      if (match_node_ptr->sibling_node_num[shifted_symbol & 0xF] != 0) {
        match_node_ptr = &match_nodes[match_node_ptr->sibling_node_num[shifted_symbol & 0xF]];
        shifted_symbol >>= 4;
      } else {
        if (match_node_ptr->miss_ptr == 0) {
          if (((int32_t)symbol < 0) || (child_ptr_array[symbol] == 0))
            goto thread_overlap_check_loop_no_match;
          match_node_ptr = child_ptr_array[symbol];
          goto thread_overlap_check_loop_match;
        } else {
          match_node_ptr = match_node_ptr->miss_ptr;
          shifted_symbol = symbol;
        }
      }
    } while (symbol != match_node_ptr->symbol);
  }
  if (match_node_ptr->child_ptr != 0) {
    match_node_ptr = match_node_ptr->child_ptr;
    goto thread_overlap_check_loop_match;
  }

  // no child, so found a match - check for overlaps
  uint32_t node_score_number = match_node_ptr->score_number;
  if ((in_symbol_ptr - match_node_ptr->num_symbols < thread_data_ptr->stop_matches_symbol_ptr)
      && (candidate_bad[node_score_number] == 0)) {
    **thread_data_ptr->next_match_ptr_ptr = node_score_number;
    (*thread_data_ptr->next_match_ptr_ptr)++;
    **thread_data_ptr->next_match_ptr_ptr = in_symbol_ptr - start_symbol_ptr - match_node_ptr->num_symbols;
    (*thread_data_ptr->next_match_ptr_ptr)++;
  }
  if ((num_prior_matches != 0) && (in_symbol_ptr - match_node_ptr->num_symbols
      <= prior_match_end_ptr[num_prior_matches - 1])) {
    if (num_prior_matches == 1) {
      if (prior_match_score_number[0] != node_score_number) {
        if (fast_mode == 0) {
          if (prior_match_score_number[0] > node_score_number)
            candidate_bad[prior_match_score_number[0]] = 1;
          else
            candidate_bad[node_score_number] = 1;
        } else {
          uint32_t low_score, high_score;
          if (node_score_number < prior_match_score_number[0]) {
            low_score = node_score_number;
            high_score = prior_match_score_number[0];
          } else {
            low_score = prior_match_score_number[0];
            high_score = node_score_number;
          }
          int32_t * next_overlap_num_ptr = &thread_data_ptr->next[low_score];
          while ((*next_overlap_num_ptr != -1) && (thread_data_ptr->second[*next_overlap_num_ptr] < high_score))
            next_overlap_num_ptr = &thread_data_ptr->next[*next_overlap_num_ptr];
          if ((*next_overlap_num_ptr == -1) || (thread_data_ptr->second[*next_overlap_num_ptr] != high_score)) {
            if (num_overlaps < 150000) {
              thread_data_ptr->second[num_overlaps] = high_score;
              thread_data_ptr->next[num_overlaps] = *next_overlap_num_ptr;
              *next_overlap_num_ptr = num_overlaps++;
            } else
              candidate_bad[high_score] = 1;
          }
        }
        prior_match_end_ptr[1] = in_symbol_ptr - 1;
        prior_match_score_number[1] = node_score_number;
        num_prior_matches = 2;
      }
    } else {
      uint32_t prior_match_number = 0;
      uint8_t found_same_score_prior_match = 0;
      do {
        if (in_symbol_ptr - match_node_ptr->num_symbols > prior_match_end_ptr[prior_match_number]) {
          num_prior_matches--;
          uint32_t i;
          for (i = prior_match_number ; i < num_prior_matches ; i++) {
            prior_match_end_ptr[i] = prior_match_end_ptr[i + 1];
            prior_match_score_number[i] = prior_match_score_number[i + 1];
          }
        } else { // overlapping candidates - invalidate the lower score
          if (prior_match_score_number[prior_match_number] == node_score_number)
            found_same_score_prior_match = 1;
          else {
            if (fast_mode == 0) {
              if (prior_match_score_number[prior_match_number] > node_score_number)
               candidate_bad[prior_match_score_number[prior_match_number]] = 1;
              else
                candidate_bad[node_score_number] = 1;
            } else {
              uint32_t low_score, high_score;
              if (node_score_number < prior_match_score_number[prior_match_number]) {
                low_score = node_score_number;
                high_score = prior_match_score_number[prior_match_number];
              } else {
                low_score = prior_match_score_number[prior_match_number];
                high_score = node_score_number;
              }
              int32_t * next_overlap_num_ptr = &thread_data_ptr->next[low_score];
              while ((*next_overlap_num_ptr != -1) && (thread_data_ptr->second[*next_overlap_num_ptr] < high_score))
                next_overlap_num_ptr = &thread_data_ptr->next[*next_overlap_num_ptr];
              if ((*next_overlap_num_ptr == -1) || (thread_data_ptr->second[*next_overlap_num_ptr] != high_score)) {
                if (num_overlaps < 150000) {
                  thread_data_ptr->second[num_overlaps] = high_score;
                  thread_data_ptr->next[num_overlaps] = *next_overlap_num_ptr;
                  *next_overlap_num_ptr = num_overlaps++;
                } else
                  candidate_bad[high_score] = 1;
              }
            }
          }
          prior_match_number++;
        }
      } while (prior_match_number < num_prior_matches);
      if (found_same_score_prior_match == 0) {
        prior_match_end_ptr[num_prior_matches] = in_symbol_ptr - 1;
        prior_match_score_number[num_prior_matches++] = node_score_number;
      }
    }
  } else {
    num_prior_matches = 1;
    prior_match_end_ptr[0] = in_symbol_ptr - 1;
    prior_match_score_number[0] = node_score_number;
  }
  match_node_ptr = match_node_ptr->hit_ptr;

  if (match_node_ptr == 0) {
    if (child_ptr_array[symbol] == 0)
      goto thread_overlap_check_loop_no_match;
    match_node_ptr = child_ptr_array[symbol];
    goto thread_overlap_check_loop_match;
  }
  match_node_ptr = match_node_ptr->child_ptr;
  goto thread_overlap_check_loop_match;
}


void *overlap_check_no_defs_thread(void *arg) {
  struct overlap_check * thread_data_ptr = (struct overlap_check *)arg;
  struct match_node * match_nodes = thread_data_ptr->match_nodes;
  struct match_node *match_node_ptr;
  uint32_t * in_symbol_ptr = thread_data_ptr->start_symbol_ptr;
  uint32_t * end_symbol_ptr = thread_data_ptr->stop_symbol_ptr;
  uint8_t * candidate_bad = thread_data_ptr->candidate_bad;
  uint32_t num_overlaps = thread_data_ptr->num_overlaps;
  uint32_t symbol, prior_match_score_number[MAX_PRIOR_MATCHES], *prior_match_end_ptr[MAX_PRIOR_MATCHES];
  uint32_t num_prior_matches = 0;

  for (symbol = 0 ; symbol < num_overlaps ; symbol++)
    thread_data_ptr->next[symbol] = -1;

thread_overlap_check_no_defs_loop_no_match:
  symbol = *in_symbol_ptr++;
  if (in_symbol_ptr >= end_symbol_ptr)
    return(0);
  if (child_ptr_array[symbol] == 0)
    goto thread_overlap_check_no_defs_loop_no_match;
  match_node_ptr = child_ptr_array[symbol];
thread_overlap_check_no_defs_loop_match:
  symbol = *in_symbol_ptr++;
  if (symbol != match_node_ptr->symbol) {
    uint32_t shifted_symbol = symbol;
    do {
      if (match_node_ptr->sibling_node_num[shifted_symbol & 0xF] != 0) {
        match_node_ptr = &match_nodes[match_node_ptr->sibling_node_num[shifted_symbol & 0xF]];
        shifted_symbol >>= 4;
      } else {
        if (match_node_ptr->miss_ptr == 0) {
          if (child_ptr_array[symbol] == 0)
            goto thread_overlap_check_no_defs_loop_no_match;
          if (in_symbol_ptr > end_symbol_ptr)
            return(0);
          match_node_ptr = child_ptr_array[symbol];
          goto thread_overlap_check_no_defs_loop_match;
        } else {
          match_node_ptr = match_node_ptr->miss_ptr;
          shifted_symbol = symbol;
        }
      }
    } while (symbol != match_node_ptr->symbol);
  }
  if (match_node_ptr->child_ptr != 0) {
    if (in_symbol_ptr > end_symbol_ptr)
      if (in_symbol_ptr - match_node_ptr->num_symbols >= end_symbol_ptr)
        return(0);
    match_node_ptr = match_node_ptr->child_ptr;
    goto thread_overlap_check_no_defs_loop_match;
  }

  // no child, so found a match - check for overlaps
  uint32_t node_score_number = match_node_ptr->score_number;
  if ((in_symbol_ptr - match_node_ptr->num_symbols < thread_data_ptr->stop_matches_symbol_ptr)
      && (candidate_bad[node_score_number] == 0)) {
    **thread_data_ptr->next_match_ptr_ptr = node_score_number;
    (*thread_data_ptr->next_match_ptr_ptr)++;
    **thread_data_ptr->next_match_ptr_ptr = in_symbol_ptr - start_symbol_ptr - match_node_ptr->num_symbols;
    (*thread_data_ptr->next_match_ptr_ptr)++;
  }
  if ((num_prior_matches != 0) && (in_symbol_ptr - match_node_ptr->num_symbols
      <= prior_match_end_ptr[num_prior_matches - 1])) {
    if (num_prior_matches == 1) {
      if (prior_match_score_number[0] != node_score_number) {
        if (fast_mode == 0) {
          if (prior_match_score_number[0] > node_score_number)
            candidate_bad[prior_match_score_number[0]] = 1;
          else
            candidate_bad[node_score_number] = 1;
        } else {
          uint32_t low_score, high_score;
          if (node_score_number < prior_match_score_number[0]) {
            low_score = node_score_number;
            high_score = prior_match_score_number[0];
          } else {
            low_score = prior_match_score_number[0];
            high_score = node_score_number;
          }
          int32_t * next_overlap_num_ptr = &thread_data_ptr->next[low_score];
          while ((*next_overlap_num_ptr != -1) && (thread_data_ptr->second[*next_overlap_num_ptr] < high_score))
            next_overlap_num_ptr = &thread_data_ptr->next[*next_overlap_num_ptr];
          if ((*next_overlap_num_ptr == -1) || (thread_data_ptr->second[*next_overlap_num_ptr] != high_score)) {
            if (num_overlaps < 150000) {
              thread_data_ptr->second[num_overlaps] = high_score;
              thread_data_ptr->next[num_overlaps] = *next_overlap_num_ptr;
              *next_overlap_num_ptr = num_overlaps++;
            } else
              candidate_bad[high_score] = 1;
          }
        }
        prior_match_end_ptr[1] = in_symbol_ptr - 1;
        prior_match_score_number[1] = node_score_number;
        num_prior_matches = 2;
      }
    } else {
      uint32_t prior_match_number = 0;
      uint8_t found_same_score_prior_match = 0;
      do {
        if (in_symbol_ptr - match_node_ptr->num_symbols > prior_match_end_ptr[prior_match_number]) {
          num_prior_matches--;
          uint32_t i;
          for (i = prior_match_number ; i < num_prior_matches ; i++) {
            prior_match_end_ptr[i] = prior_match_end_ptr[i + 1];
            prior_match_score_number[i] = prior_match_score_number[i + 1];
          }
        } else { // overlapping candidates - invalidate the lower score
          if (prior_match_score_number[prior_match_number] == node_score_number)
            found_same_score_prior_match = 1;
          else {
            if (fast_mode == 0) {
              if (prior_match_score_number[prior_match_number] > node_score_number)
                candidate_bad[prior_match_score_number[prior_match_number]] = 1;
              else
               candidate_bad[node_score_number] = 1;
            } else {
              uint32_t low_score, high_score;
              if (node_score_number < prior_match_score_number[prior_match_number]) {
                low_score = node_score_number;
                high_score = prior_match_score_number[prior_match_number];
              } else {
                low_score = prior_match_score_number[prior_match_number];
                high_score = node_score_number;
              }
              int32_t * next_overlap_num_ptr = &thread_data_ptr->next[low_score];
              while ((*next_overlap_num_ptr != -1) && (thread_data_ptr->second[*next_overlap_num_ptr] < high_score))
                next_overlap_num_ptr = &thread_data_ptr->next[*next_overlap_num_ptr];
              if ((*next_overlap_num_ptr == -1) || (thread_data_ptr->second[*next_overlap_num_ptr] != high_score)) {
                if (num_overlaps < 150000) {
                  thread_data_ptr->second[num_overlaps] = high_score;
                  thread_data_ptr->next[num_overlaps] = *next_overlap_num_ptr;
                  *next_overlap_num_ptr = num_overlaps++;
                } else
                  candidate_bad[high_score] = 1;
              }
            }
          }
          prior_match_number++;
        }
      } while (prior_match_number < num_prior_matches);
      if (found_same_score_prior_match == 0) {
        prior_match_end_ptr[num_prior_matches] = in_symbol_ptr - 1;
        prior_match_score_number[num_prior_matches++] = node_score_number;
      }
    }
  } else {
    num_prior_matches = 1;
    prior_match_end_ptr[0] = in_symbol_ptr - 1;
    prior_match_score_number[0] = node_score_number;
  }
  match_node_ptr = match_node_ptr->hit_ptr;
  if (match_node_ptr == 0) {
    if (child_ptr_array[symbol] == 0)
      goto thread_overlap_check_no_defs_loop_no_match;
    match_node_ptr = child_ptr_array[symbol];
    goto thread_overlap_check_no_defs_loop_match;
  }
  if ((in_symbol_ptr <= end_symbol_ptr) || (in_symbol_ptr - match_node_ptr->num_symbols < end_symbol_ptr)) {
    match_node_ptr = match_node_ptr->child_ptr;
    goto thread_overlap_check_no_defs_loop_match;
  }
  return(0);
}


void *find_substitutions_thread(void *arg) {
  struct find_substitutions_thread_data * thread_data_ptr = (struct find_substitutions_thread_data *)arg;
  struct match_node * match_nodes = thread_data_ptr->match_nodes;
  struct match_node *match_node_ptr;
  uint32_t * in_symbol_ptr = thread_data_ptr->start_symbol_ptr;
  uint32_t * previous_in_symbol_ptr = in_symbol_ptr;
  uint32_t * end_symbol_ptr = thread_data_ptr->stop_symbol_ptr;
  uint32_t symbol;
  uint32_t substitute_index = 0;
  uint32_t local_read_index = 0;

  thread_data_ptr->extra_match_symbols = 0;
thread_symbol_substitution_loop_top:
  symbol = *in_symbol_ptr++;
  if (symbol == 0x20) {
    match_node_ptr = child_ptr_array[0];
    symbol = *in_symbol_ptr++;
    if ((int32_t)symbol < 0) {
      if (in_symbol_ptr < end_symbol_ptr)
        goto thread_symbol_substitution_loop_top;
      goto thread_symbol_substitution_loop_end;
    } else {
thread_symbol_substitution_loop_match_search:
      if (symbol != match_node_ptr->symbol) {
        uint32_t sibling_nibble = symbol;
        do {
          if (match_node_ptr->sibling_node_num[sibling_nibble & 0xF] != 0) {
            match_node_ptr = &match_nodes[match_node_ptr->sibling_node_num[sibling_nibble & 0xF]];
            sibling_nibble = sibling_nibble >> 4;
          } else { // no match, so use miss node and output missed symbols
            if (match_node_ptr->miss_ptr == 0) {
              if (symbol == 0x20) {
                if (in_symbol_ptr > end_symbol_ptr)
                  goto thread_symbol_substitution_loop_end;
                if ((int32_t)*in_symbol_ptr >= 0) {
                  match_node_ptr = child_ptr_array[0];
                  symbol = *in_symbol_ptr++;
                  goto thread_symbol_substitution_loop_match_search;
                }
                if (++in_symbol_ptr < end_symbol_ptr)
                  goto thread_symbol_substitution_loop_top;
                goto thread_symbol_substitution_loop_end;
              }
              if (in_symbol_ptr < end_symbol_ptr)
                goto thread_symbol_substitution_loop_top;
              goto thread_symbol_substitution_loop_end;
            }
            if ((in_symbol_ptr > end_symbol_ptr)
                && (in_symbol_ptr - match_node_ptr->miss_ptr->num_symbols >= end_symbol_ptr))
              goto thread_symbol_substitution_loop_end;
            match_node_ptr = match_node_ptr->miss_ptr;
            sibling_nibble = symbol;
          }
        } while (symbol != match_node_ptr->symbol);
      }
      if (match_node_ptr->child_ptr != 0) {
        symbol = *in_symbol_ptr++;
        if ((int32_t)symbol >= 0) {
          match_node_ptr = match_node_ptr->child_ptr;
          goto thread_symbol_substitution_loop_match_search;
        } else
          goto thread_symbol_substitution_loop_match_no_match;
      }
      // found a match
      while ((((substitute_index - local_read_index) & 0x7FFFFC) == 0x7FFFFC)
          && (((substitute_index - (local_read_index = atomic_load_explicit(&thread_data_ptr->read_index,
            memory_order_acquire))) & 0x7FFFFC) == 0x7FFFFC))
        sched_yield();
      if (in_symbol_ptr - previous_in_symbol_ptr - match_node_ptr->num_symbols != 0) {
        thread_data_ptr->data[substitute_index] = in_symbol_ptr - previous_in_symbol_ptr - match_node_ptr->num_symbols;
        substitute_index = (substitute_index + 1) & 0x7FFFFF;
      }
      thread_data_ptr->data[substitute_index] = 0x80000000 + match_node_ptr->num_symbols;
      substitute_index = (substitute_index + 1) & 0x7FFFFF;
      thread_data_ptr->data[substitute_index] = match_node_ptr->score_number;
      substitute_index = (substitute_index + 1) & 0x7FFFFF;
      atomic_store_explicit(&thread_data_ptr->write_index, substitute_index, memory_order_release);
      previous_in_symbol_ptr = in_symbol_ptr;
      if (in_symbol_ptr < end_symbol_ptr)
        goto thread_symbol_substitution_loop_top;
      thread_data_ptr->extra_match_symbols = in_symbol_ptr - end_symbol_ptr;
      goto thread_symbol_substitution_loop_end2;
    }
thread_symbol_substitution_loop_match_no_match:
    if (in_symbol_ptr < end_symbol_ptr)
      goto thread_symbol_substitution_loop_top;
    goto thread_symbol_substitution_loop_end;
  }
  if (in_symbol_ptr < end_symbol_ptr)
    goto thread_symbol_substitution_loop_top;

thread_symbol_substitution_loop_end:
  while ((((substitute_index - local_read_index) & 0x7FFFFF) == 0x7FFFFF)
     && (((substitute_index - (local_read_index = atomic_load_explicit(&thread_data_ptr->read_index,
          memory_order_acquire))) & 0x7FFFFF) == 0x7FFFFF))
    sched_yield();
  thread_data_ptr->data[substitute_index] = end_symbol_ptr - previous_in_symbol_ptr;
  substitute_index = (substitute_index + 1) & 0x7FFFFF;
  atomic_store_explicit(&thread_data_ptr->write_index, substitute_index, memory_order_release);
thread_symbol_substitution_loop_end2:
  atomic_store_explicit(&thread_data_ptr->done, 1, memory_order_relaxed);
  return(0);
}


void *substitute_thread(void *arg) {
  struct substitute_thread_data * thread_data_ptr = (struct substitute_thread_data *)arg;
  uint32_t data;
  uint16_t local_write_index;
  uint16_t substitute_data_index = 0;

  thread_data_ptr->out_symbol_ptr = thread_data_ptr->in_symbol_ptr;
  while (1) {
    while ((local_write_index = atomic_load_explicit(&substitute_data_write_index, memory_order_relaxed))
        == substitute_data_index); // wait
    do {
      if ((int32_t)(data = thread_data_ptr->substitute_data[substitute_data_index++]) >= 0) {
        memmove(thread_data_ptr->out_symbol_ptr, thread_data_ptr->in_symbol_ptr, data * 4);
        thread_data_ptr->in_symbol_ptr += data;
        thread_data_ptr->out_symbol_ptr += data;
      } else if (data != 0xFFFFFFFF) {
        thread_data_ptr->in_symbol_ptr += (size_t)(data + 0x80000000);
        uint32_t symbol = thread_data_ptr->substitute_data[substitute_data_index++];
        *thread_data_ptr->out_symbol_ptr++ = symbol;
        thread_data_ptr->symbol_counts[symbol]++;
      } else
        return(0);
      atomic_store_explicit(&substitute_data_read_index, substitute_data_index, memory_order_relaxed);
    } while (local_write_index != substitute_data_index);
  }
}


uint8_t GLZAcompress(size_t in_size, size_t * outsize_ptr, uint8_t ** iobuf, struct param_data * params) {
  const uint8_t INSERT_SYMBOL_CHAR = 0xFE;
  const uint8_t DEFINE_SYMBOL_CHAR = 0xFF;
  uint64_t available_RAM, max_memory_usage;
  uint32_t UTF8_value, max_UTF8_value, max_rules, i, j, prior_cycle_symbols;
  uint32_t num_terminals_used, num_rules, num_prior_matches, num_match_nodes;
  uint32_t symbol, next_new_symbol_number, initial_max_scores, max_scores, first_define_index;
  uint32_t node_score_number, suffix_node_number, next_node_num, node_num_limit, max_match_length;
  uint32_t best_score_num_symbols, num_overlaps, max_x_log2_x, prior_match_score_number[MAX_PRIOR_MATCHES];
  uint32_t *in_symbol_ptr, *previous_in_symbol_ptr, *out_symbol_ptr, *stop_symbol_ptr, *end_symbol_ptr;
  uint32_t *search_match_ptr, *start_cycle_symbol_ptr, *end_cycle_symbol_ptr, *node_string_start_ptr, *block_ptr;
  uint32_t *match_string_start_ptr, *match_strings, *new_symbol_number, *substitute_data;
  uint32_t *prior_match_end_ptr[MAX_PRIOR_MATCHES], *stop_matches_symbol_ptr[8], new_rule_number[0x8000];
  int32_t *base_node_child_num_ptr;
  uint16_t scan_cycle, num_candidates, candidate_num, node_ptrs_num, *candidates_index, *candidates_position;
  uint8_t UTF8_compliant, format, create_words, scan_mode;
  uint8_t fast_sections, fast_section, section_repeats, max_terminal;
  uint8_t *in_char_ptr, *end_char_ptr, *candidate_bad, *free_RAM_ptr, *end_RAM_ptr;
  size_t block_size;
  double d_num_file_symbols, order_0_entropy, profit_ratio_power, *symbol_entropy;
  float log2_num_symbols_plus_substitution_cost, production_cost, *symbol_entropy_f;
  float prior_min_score, new_min_score, cycle_start_ratio, cycle_end_ratio, fast_min_score;
  float new_symbol_cost[NUM_PRECALCULATED_SYMBOL_COSTS];
  float section_scores[23];
  struct tree_thread_data tree_thread_data[13];
  struct word_tree_thread_data word_tree_thread_data[4];
  struct rank_scores_thread_data *rank_scores_data_ptr;
  struct score_data *node_data;
  struct find_substitutions_thread_data *find_substitutions_thread_data;
  struct substitute_thread_data substitute_thread_data;
  struct match_node *match_node_ptr;
  struct overlap_check *overlap_check_data;
  pthread_t build_tree_threads[7], word_build_tree_threads[4], rank_scores_thread1, substitute_thread1;
  pthread_t overlap_check_threads[7], find_substitutions_threads[7];


  if (sizeof(uint32_t *) >= 8)
    max_memory_usage = 0x800000000;
  else
    max_memory_usage = 0x70000000;

  if (params != 0) {
    if (params->user_set_profit_ratio_power != 0)
      profit_ratio_power = params->profit_ratio_power;
    create_words = params->create_words;
    if (in_size < 1000)
      fast_mode = 0;
    else
      fast_mode = params->fast_mode;
    order_ratio = (double)params->order;
  } else {
    create_words = 1;
    fast_mode = 0;
    order_ratio = 0.0;
  }
  max_rules = 0xA00000;
  if (max_rules > (in_size >> 4) + 0x110000)
    max_rules = (in_size >> 4) + 0x110000;

  if ((0 == (symbol_counts = (uint32_t *)malloc(4 * max_rules)))
      || (0 == (symbol_ends = (struct symbol_ends_data *)malloc(max_rules * sizeof(struct symbol_ends_data))))
      || (0 == (rank_scores_data_ptr = (struct rank_scores_thread_data *)malloc(sizeof(struct rank_scores_thread_data))))
      || ((fast_mode != 0) && (0 == (score_map = (int16_t *)malloc(2 * in_size))))) {
    fprintf(stderr, "ERROR - memory allocation failed\n");
    return(0);
  }

  if (fast_mode == 0)
    max_scores = MAX_SCORES;
  else
    max_scores = MAX_SCORES_FAST;
  candidates = &rank_scores_data_ptr->candidates[0];
  memset(num_starts, 0, 0x400);
  memset(num_ends, 0, 0x400);
  memset(o1c, 0, 0x40000);

  if ((params != 0) && (params->user_set_RAM_size != 0)) {
    available_RAM = (uint64_t)(params->RAM_usage * (float)0x100000);
    if (available_RAM > max_memory_usage)
      available_RAM = max_memory_usage;
    if (0 == (start_symbol_ptr = (uint32_t *)malloc(available_RAM))) {
      fprintf(stderr, "ERROR - Insufficient RAM to compress - unable to allocate %zu bytes\n", (size_t)available_RAM);
      return(0);
    }
    if (available_RAM < (41 * (uint64_t)in_size) / 10) {
      fprintf(stderr, "ERROR - Insufficient RAM to compress - program requires at least %.2lf MB\n",
          (float)((41 * (uint64_t)in_size) / 10) / (float)0x100000 + 0.005);
      return(0);
    }
  } else {
    available_RAM = (uint64_t)in_size * 250 + 40000000;
    if (available_RAM > max_memory_usage)
      available_RAM = max_memory_usage;
    if (available_RAM > 0x80000000 + 6 * (uint64_t)in_size)
      available_RAM = 0x80000000 + 6 * (uint64_t)in_size;
    do {
      start_symbol_ptr = (uint32_t *)malloc(available_RAM);
      if (start_symbol_ptr != 0)
        break;
      available_RAM = (available_RAM / 10) * 9;
    } while (available_RAM > 1500000000);
    if ((start_symbol_ptr == 0) || (available_RAM < (uint64_t)in_size * 9 / 2)) {
      fprintf(stderr, "ERROR - Insufficient RAM to compress - unable to allocate %zu bytes\n",
          (size_t)((available_RAM * 10) / 9));
      return(0);
    }
  }
  end_RAM_ptr = (uint8_t *)start_symbol_ptr + available_RAM;

  // parse the file to determine UTF8_compliant
  in_symbol_ptr = start_symbol_ptr;
  num_rules = 0;
  UTF8_compliant = 0;
  format = **iobuf;
  cap_encoded = (format == 1);
  max_UTF8_value = 0x7F;
  in_char_ptr = *iobuf + 1;
  end_char_ptr = *iobuf + in_size;
  if (format < 2) {
    do {
      uint8_t this_char = *in_char_ptr++;
      if (this_char < 0x80)
        *in_symbol_ptr++ = (uint32_t)this_char;
      else if ((this_char < 0xC0) || (this_char >= 0xF2) || ((*in_char_ptr & 0xC0) != 0x80)) break;
      else {
        UTF8_value = 0x40 * (uint32_t)(this_char & 0x1F) + (*in_char_ptr++ & 0x3F);
        if (this_char >= 0xE0) {
          if ((*in_char_ptr & 0xC0) != 0x80) break;
          UTF8_value = 0x40 * UTF8_value + (uint32_t)(*in_char_ptr++ & 0x3F);
          if (this_char >= 0xF0) {
            if ((*in_char_ptr & 0xC0) != 0x80) break;
            UTF8_value = 0x40 * (UTF8_value & 0x7FFF) + (uint32_t)(*in_char_ptr++ & 0x3F);
          }
        }
        *in_symbol_ptr++ = UTF8_value;
        if (UTF8_value > max_UTF8_value)
          max_UTF8_value = UTF8_value;
      }
    } while (in_char_ptr < end_char_ptr);
    if (in_char_ptr == end_char_ptr)
      UTF8_compliant = 1;
  }

#ifdef PRINTON
  fprintf(stderr, "cap encoded: %u, UTF8 compliant %u\n", (unsigned int)cap_encoded, (unsigned int)UTF8_compliant);
#endif

  // create the initial grammar and count symbols
  in_char_ptr = *iobuf + 1;
  if (UTF8_compliant != 0) {
    num_terminals = max_UTF8_value + 1;
    max_terminal = 0x7F;
    memset(symbol_counts, 0, 4 * num_terminals);
    num_file_symbols = in_symbol_ptr - start_symbol_ptr;
    end_symbol_ptr = in_symbol_ptr;
    in_symbol_ptr = start_symbol_ptr;
    while (in_symbol_ptr != end_symbol_ptr)
      symbol_counts[*in_symbol_ptr++]++;
#ifdef PRINTON
    fprintf(stderr, "%u symbols, maximum UTF-8 value 0x%x\n",
        (unsigned int)num_file_symbols, (unsigned int)max_UTF8_value);
#endif
    if ((params == 0) || (params->user_set_profit_ratio_power == 0)) {
      if (fast_mode == 0)
        profit_ratio_power = 2.0;
      else
        profit_ratio_power = 1.0;
    }
    for (i = 0 ; i < num_terminals ; i++)
      symbol_ends[i].start = symbol_ends[i].end = get_UTF8_context(i);
  } else {
    num_terminals = 0x100;
    max_terminal = 0xFF;
    memset(symbol_counts, 0, 0x400);
    in_symbol_ptr = start_symbol_ptr;
    while (in_char_ptr != end_char_ptr) {
      *in_symbol_ptr = (uint32_t)*in_char_ptr++;
      symbol_counts[*in_symbol_ptr++]++;
    }
    num_file_symbols = in_symbol_ptr - start_symbol_ptr;
    end_symbol_ptr = in_symbol_ptr;
#ifdef PRINTON
    fprintf(stderr, "%u symbols\n", (unsigned int)num_file_symbols);
#endif
    if ((params == 0) || (params->user_set_profit_ratio_power == 0)) {
      if ((fast_mode == 0) && (cap_encoded != 0))
        profit_ratio_power = 2.0;
      else if ((format & 0xFE) == 0)
        profit_ratio_power = 1.0;
      else
        profit_ratio_power = 0.0;
    }
    for (i = 0 ; i < num_terminals ; i++)
      symbol_ends[i].start = symbol_ends[i].end = i;
  }
  free(*iobuf);
  if (available_RAM
      < 4 * (uint64_t)in_size + 4 * BASE_NODES_CHILD_ARRAY_SIZE * num_terminals + 0x10 * MAX_SCORES_FAST) {
    fprintf(stderr, "ERROR - Insufficient RAM to compress - unable to allocate %zu bytes\n",
        (size_t)(4 * (uint64_t)in_size + 4 * BASE_NODES_CHILD_ARRAY_SIZE * num_terminals + 0x10 * MAX_SCORES_FAST));
    return(0);
  }
  if (params != 0) {
    if (params->max_rules + num_terminals < max_rules)
      max_rules = params->max_rules + num_terminals;
  }

  in_symbol_ptr = start_symbol_ptr;
  uint8_t sym1, sym2;
  sym2 = *in_symbol_ptr++;
  while (in_symbol_ptr != end_symbol_ptr) {
    sym1 = sym2;
    sym2 = symbol_ends[*in_symbol_ptr++].end;
    o1c[sym1][sym2]++;
    num_ends[sym1]++;
    num_starts[sym2]++;
  }
  max_x_log2_x = 0;
  for (i = 0 ; i < 0x100 ; i++)
    if (num_ends[i] > max_x_log2_x)
      max_x_log2_x = num_ends[i];
  max_x_log2_x += 2;
  if (max_x_log2_x > NUM_PRECALCULATED_X_LOG2_X)
    max_x_log2_x = NUM_PRECALCULATED_X_LOG2_X;

  first_define_index = in_symbol_ptr - start_symbol_ptr;
  *end_symbol_ptr = 0xFFFFFFFE;
  size_t min_RAM = end_symbol_ptr - start_symbol_ptr + 2 * MAX_MATCH_LENGTH * sizeof(struct node);
  if (min_RAM > available_RAM) {
    fprintf(stderr, "ERROR - Insufficient RAM to compress - program requires at least %.2lf MB\n",
        (float)min_RAM / (float)0x100000 + 0.005);
    return(0);
  }
  if ((0 == (new_symbol_number = (uint32_t *)malloc(4 * max_scores)))
      || (0 == (node_data = (struct score_data *)malloc(max_scores * sizeof(struct score_data))))
      || (0 == (candidates_index = (uint16_t *)malloc(2 * max_scores)))
      || (0 == (candidate_bad = (uint8_t *)malloc(max_scores)))
      || ((fast_mode == 0) && (0 == (x_log2_x = (double *)malloc(8 * max_x_log2_x))))) {
    fprintf(stderr, "ERROR - memory allocation failed\n");
    return(0);
  }

  num_terminals_used = 0;
  for (i = 0 ; i < num_terminals ; i++)
    if (symbol_counts[i] != 0)
      num_terminals_used++;
  for (i = 1 ; i < NUM_PRECALCULATED_LOG2_X ; i++)
    log2_x[i] = log2((double)i);
  rank_scores_data_ptr->candidates_index = candidates_index;
  if (fast_mode == 0) {
    for (i = 1 ; i < max_x_log2_x ; i++)
      x_log2_x[i] = (double)i * log2((double)i);
    initial_max_scores = (uint32_t)(500.0 + 0.075 * sqrt((double)num_file_symbols));
    fast_sections = 1;
  } else {
    if (0 == (candidates_position = (uint16_t *)malloc(2 * max_scores))) {
      fprintf(stderr, "ERROR - memory allocation failed\n");
      return(0);
    }
    rank_scores_data_ptr->candidates_position = candidates_position;
    fast_sections = 23;
    fast_section = 0;
    fast_min_score = 4.0;
    section_repeats = 0;
    for (i = 0 ; i < 23 ; i++)
      section_scores[i] = BIG_FLOAT;
    initial_max_scores = (uint32_t)(100.0 + 22.0 * pow((double)num_file_symbols, 0.3333));
  }
  memset(candidate_bad, 0, max_scores);
  min_score = 10.0;
  prior_min_score = BIG_FLOAT;
  cycle_start_ratio = 0.0;
  cycle_end_ratio = 1.0;
  prior_cycle_symbols = num_file_symbols;
  scan_cycle = 0;
  scan_mode = ((cap_encoded == 0) && ((UTF8_compliant == 0) || (fast_mode == 0))) || (create_words == 0);

  do {
top_main_loop:
    next_new_symbol_number = num_terminals + num_rules;
    d_num_file_symbols = (double)num_file_symbols;
    log_file_symbols = log2(d_num_file_symbols);
    free_RAM_ptr = (char *)(((size_t)end_symbol_ptr + 8) & ~7);
    symbol_entropy = (double *)free_RAM_ptr;
    symbol_entropy_f = (float *)free_RAM_ptr;
    free_RAM_ptr += (1 + ((scan_mode != 0) & (fast_mode == 0))) * sizeof(float) * (size_t)next_new_symbol_number;
    if ((scan_mode != 0) && (fast_mode == 0)) {
      num_file_symbols_p1_x_log_file_symbols_p1 = (double)(num_file_symbols + 1) * log2((double)(num_file_symbols + 1));
      for (i = 1 ; i < NUM_PRECALCULATED_NFSMR_LOGS ; i++)
        nfs_profit[i] = num_file_symbols_p1_x_log_file_symbols_p1
          - (double)(num_file_symbols - i + 1) * log2((double)(num_file_symbols - i + 1));
      if (num_rules != 0)
        new_rule_cost = num_file_symbols_p1_x_log_file_symbols_p1 + (double)num_rules * log2((double)num_rules)
          - (double)(num_rules + 1) * log2((double)(num_rules + 1)) - d_num_file_symbols * log2(d_num_file_symbols) + 1.0;
      else
        new_rule_cost =  num_file_symbols_p1_x_log_file_symbols_p1 - d_num_file_symbols * log2(d_num_file_symbols) + 1.0;
    } else {
      log2_num_symbols_plus_substitution_cost = (float)log_file_symbols + 1.4;
      for (i = 2 ; i < NUM_PRECALCULATED_SYMBOL_COSTS ; i++)
        new_symbol_cost[i] = log2_num_symbols_plus_substitution_cost - (float)log2_x[i - 1]; // -1 for repeats only
      if (scan_mode == 0)
        production_cost = log2f((float)d_num_file_symbols / (float)num_terminals_used) + 1.2;
      else
        production_cost = log2f((float)d_num_file_symbols / (float)(num_rules + 1)) + 1.2;
    }

    if (fast_mode == 0) {
      order_0_entropy = 0.0;
      i = 0;
      do {
        if (symbol_counts[i] != 0) {
          if (symbol_counts[i] < NUM_PRECALCULATED_LOG2_X)
            symbol_entropy[i] = log_file_symbols - log2_x[symbol_counts[i]];
          else
            symbol_entropy[i] = log_file_symbols - log2((double)symbol_counts[i]);
          order_0_entropy += symbol_entropy[i] * (double)symbol_counts[i];
        }
      } while (++i < next_new_symbol_number);
      if (scan_mode == 0) {
        i = 0;
        do {
          if (symbol_counts[i] != 0)
            symbol_entropy_f[i] = (float)symbol_entropy[i];
        } while (++i < next_new_symbol_number);
      }
      if (num_rules != 0)
        order_0_entropy += (double)(num_rules + 1) * (log_file_symbols - log2((double)num_rules));
#ifdef PRINTON
      fprintf(stderr, "%u: grammar size: %u, %u rules, %.4f bits/sym, o0e %u bytes\n",
          (unsigned int)++scan_cycle, (unsigned int)num_file_symbols + 1, (unsigned int)num_rules,
          (float)(order_0_entropy / d_num_file_symbols), (unsigned int)(order_0_entropy * 0.125));
#endif
    } else {
#ifdef PRINTON
      fprintf(stderr, "PASS %u: grammar size %u, %u production rules\r",
          (unsigned int)++scan_cycle, (unsigned int)num_file_symbols + 1, (unsigned int)num_rules + 1);
#endif
    }

    // Set the memory adddress for the suffix tree nodes
    base_nodes_child_node_num = (int32_t *)free_RAM_ptr;
    nodes = (struct node *)((size_t)free_RAM_ptr + sizeof(int32_t) * (size_t)next_new_symbol_number * BASE_NODES_CHILD_ARRAY_SIZE);
    node_num_limit = (uint32_t)(((uint8_t *)start_symbol_ptr + available_RAM - (uint8_t *)nodes) / sizeof(struct node));

    if (scan_mode == 0) {
      scan_mode = 1;

      // build the words suffix tree
      base_node_child_num_ptr = &base_nodes_child_node_num[0];
      while (base_node_child_num_ptr <= base_nodes_child_node_num + 0x90)
        *base_node_child_num_ptr++ = 0;

      uint8_t local_write_index[4];
      for (i = 0 ; i < 4 ; i++) {
        local_write_index[i] = 0;
        word_tree_thread_data[i].first_node_num = 1 + i * (node_num_limit >> 3);
        word_tree_thread_data[i].write_index = 0;
        word_tree_thread_data[i].read_index = 0;
        pthread_create(&word_build_tree_threads[i], NULL, word_build_tree_thread, (void *)&word_tree_thread_data[i]);
      }
      find_substitutions_thread_data = (struct find_substitutions_thread_data *)(nodes + (node_num_limit >> 1));

      in_symbol_ptr = start_symbol_ptr;
      uint8_t word_start[0x80];
      for (i = 0 ; i < 0x80 ; i++)
        word_start[i] = 0;
      for (i = 'a' ; i <= 'z' ; i++)
        word_start[i] = 1;
      for (i = '0' ; i <= '9' ; i++)
        word_start[i] = 1;
      word_start['$'] = 1;
      while (1) {
        symbol = *in_symbol_ptr++;
        if (symbol == 0x20) {
          if (((*in_symbol_ptr >= 0x80) && (UTF8_compliant != 0))
              || ((*in_symbol_ptr < 0x80) && (word_start[*in_symbol_ptr] != 0))) {
            uint8_t thread_num = *in_symbol_ptr & 3;
            if ((local_write_index[thread_num] & 0x7F) == 0)
              while ((uint8_t)(local_write_index[thread_num]
                  - atomic_load_explicit(&word_tree_thread_data[thread_num].read_index, memory_order_acquire))
                >= 0x80); // wait
            word_tree_thread_data[thread_num].start_positions[local_write_index[thread_num]++]
                = in_symbol_ptr - start_symbol_ptr;
            atomic_store_explicit(&word_tree_thread_data[thread_num].write_index, local_write_index[thread_num],
                memory_order_release);
          }
        } else if (symbol == 0xFFFFFFFE) {
          in_symbol_ptr--;
          break; // exit loop on EOF
        }
      }
      for (i = 0 ; i < 4 ; i++) {
        word_tree_thread_data[i].start_positions[local_write_index[i]++] = -1;
        atomic_store_explicit(&word_tree_thread_data[i].write_index, local_write_index[i], memory_order_release);
      }

      if (fast_mode != 0) {
        i = 0;
        do {
          if (symbol_counts[i] != 0) {
            if (symbol_counts[i] < NUM_PRECALCULATED_LOG2_X)
              symbol_entropy_f[i] = (float)(log_file_symbols - log2_x[symbol_counts[i]]);
            else
              symbol_entropy_f[i] = (float)log_file_symbols - log2f((float)symbol_counts[i]);
          }
        } while (++i < next_new_symbol_number);
      }

      for (i = 0 ; i < 4 ; i++)
        pthread_join(word_build_tree_threads[i], NULL);

      node_ptrs_num = 0;
      rank_scores_write_index = 0;
      rank_scores_read_index = 0;
      rank_scores_data_ptr->max_scores = (uint16_t)max_scores;
      pthread_create(&rank_scores_thread1, NULL, rank_word_scores_thread, (void *)rank_scores_data_ptr);
      score_symbol_tree_words(rank_scores_data_ptr->rank_scores_buffer, node_data, &node_ptrs_num,
          production_cost, log2_num_symbols_plus_substitution_cost, new_symbol_cost, symbol_entropy_f);
      while (node_ptrs_num != atomic_load_explicit(&rank_scores_read_index, memory_order_acquire)); // wait
      rank_scores_data_ptr->rank_scores_buffer[node_ptrs_num].last_match_index = 0;
      atomic_store_explicit(&rank_scores_write_index, node_ptrs_num + 1, memory_order_release);
      pthread_join(rank_scores_thread1, NULL);

      prior_cycle_symbols = in_symbol_ptr - start_symbol_ptr;
      float min_score;
      if (fast_mode == 0)
        min_score = (float)(1000.0 + 400.0 * (log2(order_0_entropy + 3000000.0) - log2(3000000.0)));
      else
        min_score = (float)(5.0 + (log2(d_num_file_symbols + 5000000.0) - log2(5000000.0)));
      num_candidates = rank_scores_data_ptr->num_candidates;
      if (next_new_symbol_number + num_candidates > max_rules)
        num_candidates = max_rules - next_new_symbol_number;
      if ((num_candidates != 0) && (candidates[candidates_index[0]].score >= min_score)) {
        free_RAM_ptr = (char *)(((size_t)end_symbol_ptr + 8) & ~7);
        substitute_data = (uint32_t *)free_RAM_ptr;
        free_RAM_ptr += 0x40000 * sizeof(uint32_t);
        substitute_thread_data.symbol_counts = symbol_counts;
        substitute_thread_data.substitute_data = substitute_data;
        child_ptr_array = (struct match_node **)free_RAM_ptr;
        struct match_node * match_nodes = (struct match_node *)(free_RAM_ptr + sizeof(struct match_node *));
        num_match_nodes = 1;
        max_match_length = 0;
        candidate_num = 0;
        while (candidate_num < num_candidates) {
          if (candidates[candidates_index[candidate_num]].num_symbols > max_match_length)
            max_match_length = candidates[candidates_index[candidate_num]].num_symbols;
          if (candidates[candidates_index[candidate_num]].score < min_score)
            num_candidates = candidate_num;
          num_match_nodes += candidates[candidates_index[candidate_num]].num_symbols - 1;
          if ((size_t)match_nodes + num_match_nodes * sizeof(struct match_node) + 4 * max_match_length
              >= (size_t)end_RAM_ptr)
            num_candidates = candidate_num - 1;
          candidate_num++;
        }

        match_strings = (uint32_t *)((size_t)match_nodes + (size_t)num_match_nodes * sizeof(struct match_node));
        candidate_num = 0;
        // save the match strings so they can be added to the end of the data after symbol substitution is done
        while (candidate_num < num_candidates) {
          match_string_start_ptr = &match_strings[candidate_num * max_match_length];
          node_string_start_ptr = start_symbol_ptr + candidates[candidates_index[candidate_num]].last_match_index
              - candidates[candidates_index[candidate_num]].num_symbols + 1;
          for (j = 0 ; j < candidates[candidates_index[candidate_num]].num_symbols ; j++)
            *(match_string_start_ptr + j) = *(node_string_start_ptr + j);
          candidate_num++;
        }
        overlap_check_data = (struct overlap_check *)(((size_t)&match_strings[num_candidates * max_match_length] + 7) & ~7);
        for (i = 1 ; i < 8 ; i++)
          overlap_check_data[i].candidate_bad = &candidate_bad[0];

        uint16_t num_candidates_processed = 0;
        do {
          next_new_symbol_number = num_terminals + num_rules;
          // build a prefix tree of the match strings, defer shorter overlapping strings
          num_match_nodes = 0;
          candidate_num = 0;
          while (candidate_num < num_candidates) {
            if (candidate_bad[candidate_num] == 0) {
              uint32_t *best_score_last_match_ptr, *best_score_match_ptr;
              best_score_match_ptr = match_strings + (candidate_num * max_match_length);
              best_score_last_match_ptr = best_score_match_ptr + candidates[candidates_index[candidate_num]].num_symbols - 1;
              best_score_match_ptr++;
              if (num_match_nodes == 0) {
                child_ptr_array[0] = match_nodes;
                init_match_node(match_nodes, *best_score_match_ptr, 2, candidate_num);
                num_match_nodes = 1;
              }
              match_node_ptr = match_nodes;
              while (best_score_match_ptr <= best_score_last_match_ptr) {
                symbol = *best_score_match_ptr;
                if (match_node_ptr->child_ptr == 0) {
                  match_node_ptr->child_ptr = &match_nodes[num_match_nodes++];
                  match_node_ptr = match_node_ptr->child_ptr;
                  init_match_node(match_node_ptr, symbol, 0, candidate_num);
                } else {
                  match_node_ptr = match_node_ptr->child_ptr;
                  uint8_t sibling_number;
                  if (move_to_match_sibling(match_nodes, &match_node_ptr, symbol, &sibling_number) != 0) {
                    if (match_node_ptr->child_ptr == 0)
                      candidate_bad[match_node_ptr->score_number] = 1;
                  } else {
                    match_node_ptr->sibling_node_num[sibling_number] = num_match_nodes;
                    match_node_ptr = &match_nodes[num_match_nodes++];
                    init_match_node(match_node_ptr, symbol, 0, candidate_num);
                  }
                }
                best_score_match_ptr++;
              }
              if (match_node_ptr->child_ptr != 0)
                candidate_bad[candidate_num] = 1;
            }
            candidate_num++;
          }

          // Redo the tree build with just this subcycle's candidates
          child_ptr_array[0] = 0;
          num_match_nodes = 0;
          j = next_new_symbol_number;
          candidate_num = 0;
          while (candidate_num < num_candidates) {
            if (candidate_bad[candidate_num] == 0) {
              uint32_t *best_score_last_match_ptr, *best_score_match_ptr;
              best_score_match_ptr = match_strings + (candidate_num * max_match_length);
              best_score_last_match_ptr = best_score_match_ptr + candidates[candidates_index[candidate_num]].num_symbols - 1;
              best_score_match_ptr++;
              symbol = *best_score_match_ptr++;
              best_score_num_symbols = 2;
              match_node_ptr = move_to_base_match_child_with_make(match_nodes, symbol, j, &num_match_nodes,
                  &child_ptr_array[0]);
              while (best_score_match_ptr <= best_score_last_match_ptr) {
                symbol = *best_score_match_ptr++;
                best_score_num_symbols++;
                move_to_match_child_with_make(match_nodes, &match_node_ptr, symbol, j, best_score_num_symbols,
                    &num_match_nodes);
              }
              symbol_counts[j++] = 0;
            }
            candidate_num++;
          }

          // scan the data following the prefix tree and substitute new symbols on end matches (child is 0)
          if (num_file_symbols >= 1000000) {
            stop_symbol_ptr = start_symbol_ptr + 64 * (num_file_symbols >> 9);
            find_substitutions_thread_data[0].start_symbol_ptr = stop_symbol_ptr;
            block_ptr = stop_symbol_ptr + 68 * (num_file_symbols >> 9);
            find_substitutions_thread_data[0].stop_symbol_ptr = block_ptr;
            find_substitutions_thread_data[1].start_symbol_ptr = block_ptr;
            block_ptr += 72 * (num_file_symbols >> 9);
            find_substitutions_thread_data[1].stop_symbol_ptr = block_ptr;
            find_substitutions_thread_data[2].start_symbol_ptr = block_ptr;
            block_ptr += 75 * (num_file_symbols >> 9);
            find_substitutions_thread_data[2].stop_symbol_ptr = block_ptr;
            find_substitutions_thread_data[3].start_symbol_ptr = block_ptr;
            block_ptr += 77 * (num_file_symbols >> 9);
            find_substitutions_thread_data[3].stop_symbol_ptr = block_ptr;
            find_substitutions_thread_data[4].start_symbol_ptr = block_ptr;
            block_ptr += 78 * (num_file_symbols >> 9);
            find_substitutions_thread_data[4].stop_symbol_ptr = block_ptr;
            find_substitutions_thread_data[5].start_symbol_ptr = block_ptr;
            find_substitutions_thread_data[5].stop_symbol_ptr = end_symbol_ptr;
            for (i = 0 ; i < 6 ; i++) {
              find_substitutions_thread_data[i].match_nodes = match_nodes;
              find_substitutions_thread_data[i].done = 0;
              find_substitutions_thread_data[i].read_index = 0;
              find_substitutions_thread_data[i].write_index = 0;
              pthread_create(&find_substitutions_threads[i], NULL, find_substitutions_thread,
                  (void *)&find_substitutions_thread_data[i]);
            }
          } else
            stop_symbol_ptr = end_symbol_ptr;

          uint32_t extra_match_symbols = 0;
          uint16_t substitute_index = 0;
          in_symbol_ptr = start_symbol_ptr;
          previous_in_symbol_ptr = start_symbol_ptr;
          out_symbol_ptr = start_symbol_ptr;

          substitute_thread_data.in_symbol_ptr = start_symbol_ptr;
          substitute_data_write_index = 0;
          substitute_data_read_index = 0;
          pthread_create(&substitute_thread1, NULL, substitute_thread, (void *)&substitute_thread_data);

wmain_symbol_substitution_loop_top:
          if (*in_symbol_ptr++ == 0x20) {
            symbol = *in_symbol_ptr++;
            if ((int32_t)symbol < 0) {
              if (in_symbol_ptr < stop_symbol_ptr)
                goto wmain_symbol_substitution_loop_top;
              goto wmain_symbol_substitution_loop_end;
            } else {
              match_node_ptr = child_ptr_array[0];
wmain_symbol_substitution_loop_match_search:
              if (symbol != match_node_ptr->symbol) {
                uint32_t sibling_nibble = symbol;
                do {
                  if (match_node_ptr->sibling_node_num[sibling_nibble & 0xF] != 0) {
                    match_node_ptr = &match_nodes[match_node_ptr->sibling_node_num[sibling_nibble & 0xF]];
                    sibling_nibble = sibling_nibble >> 4;
                  } else { // no match, so output missed symbols
                    if (symbol == 0x20) {
                      if (in_symbol_ptr > stop_symbol_ptr)
                        goto wmain_symbol_substitution_loop_end;
                      symbol = *in_symbol_ptr++;
                      if ((int32_t)symbol >= 0) {
                        match_node_ptr = child_ptr_array[0];
                        goto wmain_symbol_substitution_loop_match_search;
                      }
                      if (in_symbol_ptr < stop_symbol_ptr)
                        goto wmain_symbol_substitution_loop_top;
                      goto wmain_symbol_substitution_loop_end;
                    }
                    if (in_symbol_ptr < stop_symbol_ptr)
                      goto wmain_symbol_substitution_loop_top;
                    goto wmain_symbol_substitution_loop_end;
                  }
                } while (symbol != match_node_ptr->symbol);
              }
              if (match_node_ptr->child_ptr != 0) {
                symbol = *in_symbol_ptr++;
                if ((int32_t)symbol >= 0) {
                  match_node_ptr = match_node_ptr->child_ptr;
                  goto wmain_symbol_substitution_loop_match_search;
                }
                if (in_symbol_ptr < stop_symbol_ptr)
                  goto wmain_symbol_substitution_loop_top;
                goto wmain_symbol_substitution_loop_end;
              }
              // found a match
              if (((substitute_index + 2) & 0x7FFC) == 0)
                while ((uint16_t)(substitute_index - atomic_load_explicit(&substitute_data_read_index,
                    memory_order_acquire)) >= 0x7FFE); // wait
              if (in_symbol_ptr - previous_in_symbol_ptr - match_node_ptr->num_symbols != 0)
                substitute_data[substitute_index++] = in_symbol_ptr - previous_in_symbol_ptr - match_node_ptr->num_symbols;
              substitute_data[substitute_index++] = 0x80000000 + match_node_ptr->num_symbols;
              substitute_data[substitute_index++] = match_node_ptr->score_number;
              atomic_store_explicit(&substitute_data_write_index, substitute_index, memory_order_release);
              previous_in_symbol_ptr = in_symbol_ptr;
              if (in_symbol_ptr < stop_symbol_ptr)
                goto wmain_symbol_substitution_loop_top;
              extra_match_symbols = in_symbol_ptr - stop_symbol_ptr;
              goto wmain_symbol_substitution_loop_end2;
            }
          }
          if (in_symbol_ptr < stop_symbol_ptr)
            goto wmain_symbol_substitution_loop_top;

wmain_symbol_substitution_loop_end:
          if ((substitute_index & 0x7FFF) == 0)
            while ((uint16_t)(substitute_index - atomic_load_explicit(&substitute_data_read_index,
                memory_order_acquire)) >= 0x8000); // wait
          substitute_data[substitute_index++] = stop_symbol_ptr - previous_in_symbol_ptr;
          atomic_store_explicit(&substitute_data_write_index, substitute_index, memory_order_release);
wmain_symbol_substitution_loop_end2:
          if ((substitute_index & 0x7FFF) == 0)
            while (substitute_index != (uint16_t)atomic_load_explicit(&substitute_data_read_index,
                memory_order_acquire)); // wait
          substitute_data[substitute_index++] = 0xFFFFFFFF;
          atomic_store_explicit(&substitute_data_write_index, substitute_index, memory_order_release);
          pthread_join(substitute_thread1, NULL);
          in_symbol_ptr = substitute_thread_data.in_symbol_ptr;
          out_symbol_ptr = substitute_thread_data.out_symbol_ptr;

          if (num_file_symbols >= 1000000) {
            for (i = 0 ; i < 6 ; i++) {
              uint32_t local_substitutions_write_index;
              uint32_t substitutions_index = 0;
              if (extra_match_symbols != 0) {
                while ((local_substitutions_write_index
                    = atomic_load_explicit(&find_substitutions_thread_data[i].write_index,
                        memory_order_acquire)) == 0); // wait
                if (find_substitutions_thread_data[i].data[0] > extra_match_symbols)
                  find_substitutions_thread_data[i].data[0] -= extra_match_symbols;
                else
                  substitutions_index = 1;
                extra_match_symbols = 0;
              }

              while ((atomic_load_explicit(&find_substitutions_thread_data[i].done, memory_order_acquire) == 0)
                  || (substitutions_index != atomic_load_explicit(&find_substitutions_thread_data[i].write_index,
                    memory_order_acquire))) {
                local_substitutions_write_index
                    = atomic_load_explicit(&find_substitutions_thread_data[i].write_index, memory_order_acquire);
                if (substitutions_index != local_substitutions_write_index) {
                  do {
                    uint32_t data = find_substitutions_thread_data[i].data[substitutions_index];
                    if ((int32_t)data < 0) {
                      in_symbol_ptr += (size_t)(data + 0x80000000);
                      substitutions_index = (substitutions_index + 1) & 0x7FFFFF;
                      uint32_t symbol = find_substitutions_thread_data[i].data[substitutions_index];
                      *out_symbol_ptr++ = symbol;
                      symbol_counts[symbol]++;
                      substitutions_index = (substitutions_index + 1) & 0x7FFFFF;
                      atomic_store_explicit(&find_substitutions_thread_data[i].read_index, substitutions_index,
                          memory_order_release);
                    } else {
                      memmove(out_symbol_ptr, in_symbol_ptr, data * 4);
                      in_symbol_ptr += data;
                      out_symbol_ptr += data;
                      substitutions_index = (substitutions_index + 1) & 0x7FFFFF;
                    }
                  } while (substitutions_index != local_substitutions_write_index);
                }
              }
              atomic_store_explicit(&find_substitutions_thread_data[i].read_index, substitutions_index,
                  memory_order_release);
              pthread_join(find_substitutions_threads[i], NULL);
              extra_match_symbols += find_substitutions_thread_data[i].extra_match_symbols;
            }
          }

          if (num_rules == 0)
            first_define_index = out_symbol_ptr - start_symbol_ptr;
          else {
            if (out_symbol_ptr < start_symbol_ptr + first_define_index)
              first_define_index = out_symbol_ptr - start_symbol_ptr;
            if (*(start_symbol_ptr + first_define_index) != 0x80000001)
              while (*(start_symbol_ptr + --first_define_index) != 0x80000001); // decrement index until found
          }

          // Add new production rules and update symbol counts
          for (i = 0 ; i < num_candidates ; i++) {
            if (candidate_bad[i] == 0) {
              num_candidates_processed++;
              candidate_bad[i] = 2;
              uint32_t *match_string_ptr, *match_string_end_ptr;
              *out_symbol_ptr++ = num_rules + 0x80000001;
              match_string_ptr = match_strings + max_match_length * i;
              match_string_end_ptr = match_string_ptr + candidates[candidates_index[i]].num_symbols;
              uint32_t num_repeats = symbol_counts[num_terminals + num_rules] - 1;
              uint32_t sym1, sym2;
              sym1 = *match_string_ptr;
              symbol_ends[num_terminals + num_rules].start = symbol_ends[sym1].start;
              symbol_counts[sym1] -= num_repeats;
              *out_symbol_ptr++ = *match_string_ptr++;
              while (match_string_ptr != match_string_end_ptr) {
                sym2 = *match_string_ptr;
                symbol_counts[sym2] -= num_repeats;
                o1c[symbol_ends[sym1].end][symbol_ends[sym2].start] -= num_repeats;
                num_ends[symbol_ends[sym1].end] -= num_repeats;
                num_starts[symbol_ends[sym2].start] -= num_repeats;
                sym1 = sym2;
                *out_symbol_ptr++ = *match_string_ptr++;
              }
              symbol_ends[num_terminals + num_rules++].end = symbol_ends[sym1].end;
            } else if (candidate_bad[i] == 1)
              candidate_bad[i] = 0;
          }
          end_symbol_ptr = out_symbol_ptr;
          *end_symbol_ptr = 0xFFFFFFFE;
          num_file_symbols = end_symbol_ptr - start_symbol_ptr;
#ifdef PRINTON
          if (fast_mode == 0)
            fprintf(stderr, "Replaced %u of %u words\n", num_candidates_processed, num_candidates);
#endif
        } while (num_candidates_processed != num_candidates);  // should go to end here if hit maximum dictionary size
        memset(candidate_bad, 0, num_candidates);
      }
      goto top_main_loop;
    }

    if (scan_mode == 1) {
      scan_mode = 2;
      max_scores = initial_max_scores;
      uint32_t max_run_length[0x100];
      uint32_t run_length = 0;
      uint32_t prior_symbol = 0xFFFFFFFE;
      memset(max_run_length, 0, 0x400);
      in_symbol_ptr = start_symbol_ptr;
      do {
        symbol = *in_symbol_ptr++;
        if (symbol == prior_symbol)
          run_length++;
        else {
          if (run_length != 0) {
            if (run_length > max_run_length[prior_symbol])
              max_run_length[prior_symbol] = run_length;
            run_length = 0;
          }
          prior_symbol = symbol <= max_terminal ? symbol : 0xFFFFFFFE;
        }
      } while (in_symbol_ptr != end_symbol_ptr);
      in_symbol_ptr = start_symbol_ptr;
      if ((run_length != 0) && (run_length > max_run_length[prior_symbol]))
        max_run_length[prior_symbol] = run_length;

      uint8_t found_run = 0;
      for (i = 0 ; i <= max_terminal ; i++) {
        if ((max_run_length[i] >= 63) && (next_new_symbol_number < max_rules)) {
          max_run_length[i] = 1 << (uint32_t)log2(sqrt((double)max_run_length[i] + 1.5));
          symbol_counts[next_new_symbol_number] = 0;
          new_symbol_number[i] = next_new_symbol_number++;
          found_run = 1;
        } else
          max_run_length[i] = 0;
      }

      if (found_run != 0) {
#ifdef PRINTON
        if (fast_mode == 0)
          fprintf(stderr, "Deduplicating runs\n");
#endif
        run_length = 0;
        out_symbol_ptr = start_symbol_ptr;
        prior_symbol = *in_symbol_ptr;
        while (in_symbol_ptr++ != end_symbol_ptr) {
          symbol = *in_symbol_ptr;
          if ((symbol == prior_symbol) && (symbol <= max_terminal)) {
            if (++run_length == max_run_length[symbol] - 1) {
              prior_symbol = new_symbol_number[symbol];
              run_length = 0;
              symbol_counts[new_symbol_number[symbol]]++;
            }
          } else {
            *out_symbol_ptr++ = prior_symbol;
            while (run_length != 0) {
              *out_symbol_ptr++ = prior_symbol;
              run_length--;
            }
            prior_symbol = symbol;
          }
        }

        if (num_rules == 0)
          first_define_index = out_symbol_ptr - start_symbol_ptr;
        else {
          if (out_symbol_ptr < start_symbol_ptr + first_define_index)
            first_define_index = out_symbol_ptr - start_symbol_ptr;
          if (*(start_symbol_ptr + first_define_index) != 0x80000001) // decrement index until found
            while (*(start_symbol_ptr + --first_define_index) != 0x80000001);
        }

        // Add the new symbol definitions to the end of the data
        for (i = 0 ; i <= max_terminal ; i++) {
          if (max_run_length[i] != 0) {
            *out_symbol_ptr++ = 0x80000001 + num_rules;
            uint32_t j = 0;
            while (j++ != max_run_length[i])
              *out_symbol_ptr++ = i;
            symbol_counts[i] -= max_run_length[i] * (symbol_counts[new_symbol_number[i]] - 1);
            o1c[i][i] -= (max_run_length[i] - 1) * (symbol_counts[new_symbol_number[i]] - 1);
            num_ends[i] -= (max_run_length[i] - 1) * (symbol_counts[new_symbol_number[i]] - 1);
            num_starts[i] -= (max_run_length[i] - 1) * (symbol_counts[new_symbol_number[i]] - 1);
            symbol_ends[num_terminals + num_rules].start = i;
            symbol_ends[num_terminals + num_rules++].end = i;
          }
        }
        end_symbol_ptr = out_symbol_ptr;
        *end_symbol_ptr = 0xFFFFFFFE;
        num_file_symbols = end_symbol_ptr - start_symbol_ptr;
        if (fast_mode == 0)
          min_score = 10.0;
        else
          min_score = 40.0;
        prior_min_score = BIG_FLOAT;
        goto top_main_loop;
      }
    }

    if (fast_mode == 0) {
      if (cycle_start_ratio == 0.0) {
        if (cycle_end_ratio < 1.0) {
          if (cycle_end_ratio > 0.5)
            cycle_start_ratio = 1.0 - 0.99 * cycle_end_ratio;
          else
            cycle_start_ratio = cycle_end_ratio;
        }
      } else if ((cycle_end_ratio >= 0.99) || (prior_cycle_symbols >= num_file_symbols)
          || (1.5 * (1.0 - cycle_end_ratio) <= cycle_end_ratio - cycle_start_ratio))
        cycle_start_ratio = 0.0;
      else if ((uint32_t)((1.0 - cycle_end_ratio) * (float)num_file_symbols) >= prior_cycle_symbols)
        cycle_start_ratio = cycle_end_ratio;
      else
        cycle_start_ratio = 1.0 - 0.97 * (cycle_end_ratio - cycle_start_ratio);
    } else
      cycle_start_ratio = (float)fast_section / (float)fast_sections;
    start_cycle_symbol_ptr = start_symbol_ptr + (uint32_t)(cycle_start_ratio * (float)num_file_symbols);
    in_symbol_ptr = start_cycle_symbol_ptr;

    // setup to build the suffix tree
    uint32_t sum_symbols, symbols_limit, main_max_symbol, main_nodes_limit;
    uint32_t symbols_div_100 = (num_file_symbols - num_rules) / 100;
    uint32_t nodes_div_100 = node_num_limit / 100;
    i = 1;
    if (fast_mode == 0) {
      sum_symbols = symbol_counts[0];
      symbols_limit = symbols_div_100 * 5;
      while (sum_symbols < symbols_limit)
        sum_symbols += symbol_counts[i++];
      main_max_symbol = i - 1;
      tree_thread_data[0].min_symbol = i;
      if (i < next_new_symbol_number - 1)
        sum_symbols += symbol_counts[i++];
      symbols_limit = symbols_div_100 * 11;
      while (sum_symbols < symbols_limit)
        sum_symbols += symbol_counts[i++];
      tree_thread_data[0].max_symbol = i - 1;
      tree_thread_data[1].min_symbol = i;
      if (i < next_new_symbol_number - 1)
        sum_symbols += symbol_counts[i++];
      symbols_limit = symbols_div_100 * 17;
      while (sum_symbols < symbols_limit)
        sum_symbols += symbol_counts[i++];
      tree_thread_data[1].max_symbol = i - 1;
      tree_thread_data[2].min_symbol = i;
      if (i < next_new_symbol_number - 1)
        sum_symbols += symbol_counts[i++];
      symbols_limit = symbols_div_100 * 24;
      while (sum_symbols < symbols_limit)
        sum_symbols += symbol_counts[i++];
      tree_thread_data[2].max_symbol = i - 1;
      tree_thread_data[3].min_symbol = i;
      if (i < next_new_symbol_number - 1)
        sum_symbols += symbol_counts[i++];
      symbols_limit = symbols_div_100 * 32;
      while (sum_symbols < symbols_limit)
        sum_symbols += symbol_counts[i++];
      tree_thread_data[3].max_symbol = i - 1;
      tree_thread_data[4].min_symbol = i;
      if (i < next_new_symbol_number - 1)
        sum_symbols += symbol_counts[i++];
      symbols_limit = symbols_div_100 * 42;
      while (sum_symbols < symbols_limit)
        sum_symbols += symbol_counts[i++];
      tree_thread_data[4].max_symbol = i - 1;
      tree_thread_data[5].min_symbol = i;
      if (i < next_new_symbol_number - 1)
        sum_symbols += symbol_counts[i++];
      symbols_limit = symbols_div_100 * 52;
      while (sum_symbols < symbols_limit)
        sum_symbols += symbol_counts[i++];
      tree_thread_data[5].max_symbol = i - 1;
      tree_thread_data[6].min_symbol = i;
      if (i < next_new_symbol_number - 1)
        sum_symbols += symbol_counts[i++];
      symbols_limit = symbols_div_100 * 61;
      while (sum_symbols < symbols_limit)
        sum_symbols += symbol_counts[i++];
      tree_thread_data[6].max_symbol = i - 1;
      tree_thread_data[7].min_symbol = i;
      if (i < next_new_symbol_number - 1)
        sum_symbols += symbol_counts[i++];
      symbols_limit = symbols_div_100 * 69;
      while (sum_symbols < symbols_limit)
        sum_symbols += symbol_counts[i++];
      tree_thread_data[7].max_symbol = i - 1;
      tree_thread_data[8].min_symbol = i;
      if (i < next_new_symbol_number - 1)
        sum_symbols += symbol_counts[i++];
      symbols_limit = symbols_div_100 * 77;
      while (sum_symbols < symbols_limit)
        sum_symbols += symbol_counts[i++];
      tree_thread_data[8].max_symbol = i - 1;
      tree_thread_data[9].min_symbol = i;
      if (i < next_new_symbol_number - 1)
        sum_symbols += symbol_counts[i++];
      symbols_limit = symbols_div_100 * 86;
      while (sum_symbols < symbols_limit)
        sum_symbols += symbol_counts[i++];
      tree_thread_data[9].max_symbol = i - 1;
      tree_thread_data[10].min_symbol = i;
      if (i < next_new_symbol_number - 1)
        sum_symbols += symbol_counts[i++];
      symbols_limit = symbols_div_100 * 93;
      while (sum_symbols < symbols_limit)
        sum_symbols += symbol_counts[i++];
      tree_thread_data[10].max_symbol = i - 1;
      tree_thread_data[11].min_symbol = i;
      tree_thread_data[11].max_symbol = next_new_symbol_number - 1;

      next_node_num = 1;
      main_nodes_limit = nodes_div_100 * 18 - 10;
      tree_thread_data[6].first_node_num = 1;
      tree_thread_data[6].nodes_limit = nodes_div_100 * 16;
      tree_thread_data[0].first_node_num = nodes_div_100 * 18;
      tree_thread_data[7].first_node_num = nodes_div_100 * 16;
      tree_thread_data[0].nodes_limit = nodes_div_100 * 31;
      tree_thread_data[7].nodes_limit = nodes_div_100 * 31;
      tree_thread_data[1].first_node_num = nodes_div_100 * 31;
      tree_thread_data[8].first_node_num = nodes_div_100 * 31;
      tree_thread_data[1].nodes_limit = nodes_div_100 * 43;
      tree_thread_data[8].nodes_limit = nodes_div_100 * 43;
      tree_thread_data[2].first_node_num = nodes_div_100 * 43;
      tree_thread_data[9].first_node_num = nodes_div_100 * 43;
      tree_thread_data[2].nodes_limit = nodes_div_100 * 56;
      tree_thread_data[9].nodes_limit = nodes_div_100 * 56;
      tree_thread_data[3].first_node_num = nodes_div_100 * 56;
      tree_thread_data[10].first_node_num = nodes_div_100 * 56;
      tree_thread_data[3].nodes_limit = nodes_div_100 * 70;
      tree_thread_data[10].nodes_limit = nodes_div_100 * 70;
      tree_thread_data[4].first_node_num = nodes_div_100 * 70;
      tree_thread_data[11].first_node_num = nodes_div_100 * 70;
      tree_thread_data[4].nodes_limit = nodes_div_100 * 85;
      tree_thread_data[11].nodes_limit = nodes_div_100 * 85;
      tree_thread_data[5].first_node_num = nodes_div_100 * 85;
      tree_thread_data[5].nodes_limit = node_num_limit;
      for (i = 0 ; i < 12 ; i++) {
        tree_thread_data[i].start_cycle_symbol_ptr = start_cycle_symbol_ptr;
        tree_thread_data[i].base_nodes_child_node_num = base_nodes_child_node_num;
      }
      scan_symbol_ptr = (uintptr_t)in_symbol_ptr;
      max_symbol_ptr = 0;
      for (i = 0 ; i < 6 ; i++)
        pthread_create(&build_tree_threads[i], NULL, build_tree_thread, (void *)&tree_thread_data[i]);
    } else {
      sum_symbols = symbol_counts[0];
      symbols_limit = symbols_div_100 * 6;
      while (sum_symbols < symbols_limit)
        sum_symbols += symbol_counts[i++];
      main_max_symbol = i - 1;
      tree_thread_data[0].min_symbol = i;
      if (i < next_new_symbol_number - 1)
        sum_symbols += symbol_counts[i++];
      symbols_limit = symbols_div_100 * 12;
      while (sum_symbols < symbols_limit)
        sum_symbols += symbol_counts[i++];
      tree_thread_data[0].max_symbol = i - 1;
      tree_thread_data[1].min_symbol = i;
      if (i < next_new_symbol_number - 1)
        sum_symbols += symbol_counts[i++];
      symbols_limit = symbols_div_100 * 19;
      while (sum_symbols < symbols_limit)
        sum_symbols += symbol_counts[i++];
      tree_thread_data[1].max_symbol = i - 1;
      tree_thread_data[2].min_symbol = i;
      if (i < next_new_symbol_number - 1)
        sum_symbols += symbol_counts[i++];
      symbols_limit = symbols_div_100 * 26;
      while (sum_symbols < symbols_limit)
        sum_symbols += symbol_counts[i++];
      tree_thread_data[2].max_symbol = i - 1;
      tree_thread_data[3].min_symbol = i;
      if (i < next_new_symbol_number - 1)
        sum_symbols += symbol_counts[i++];
      symbols_limit = symbols_div_100 * 34;
      while (sum_symbols < symbols_limit)
        sum_symbols += symbol_counts[i++];
      tree_thread_data[3].max_symbol = i - 1;
      tree_thread_data[4].min_symbol = i;
      if (i < next_new_symbol_number - 1)
        sum_symbols += symbol_counts[i++];
      symbols_limit = symbols_div_100 * 43;
      while (sum_symbols < symbols_limit)
        sum_symbols += symbol_counts[i++];
      tree_thread_data[4].max_symbol = i - 1;
      tree_thread_data[5].min_symbol = i;
      if (i < next_new_symbol_number - 1)
        sum_symbols += symbol_counts[i++];
      symbols_limit = symbols_div_100 * 54;
      while (sum_symbols < symbols_limit)
        sum_symbols += symbol_counts[i++];
      tree_thread_data[5].max_symbol = i - 1;
      tree_thread_data[6].min_symbol = i;
      if (i < next_new_symbol_number - 1)
        sum_symbols += symbol_counts[i++];
      symbols_limit = symbols_div_100 * 67;
      while (sum_symbols < symbols_limit)
        sum_symbols += symbol_counts[i++];
      tree_thread_data[6].max_symbol = i - 1;
      tree_thread_data[7].min_symbol = i;
      if (i < next_new_symbol_number - 1)
        sum_symbols += symbol_counts[i++];
      symbols_limit = symbols_div_100 * 73;
      while (sum_symbols < symbols_limit)
        sum_symbols += symbol_counts[i++];
      tree_thread_data[7].max_symbol = i - 1;
      tree_thread_data[8].min_symbol = i;
      if (i < next_new_symbol_number - 1)
        sum_symbols += symbol_counts[i++];
      symbols_limit = symbols_div_100 * 79;
      while (sum_symbols < symbols_limit)
        sum_symbols += symbol_counts[i++];
      tree_thread_data[8].max_symbol = i - 1;
      tree_thread_data[9].min_symbol = i;
      if (i < next_new_symbol_number - 1)
        sum_symbols += symbol_counts[i++];
      symbols_limit = symbols_div_100 * 85;
      while (sum_symbols < symbols_limit)
        sum_symbols += symbol_counts[i++];
      tree_thread_data[9].max_symbol = i - 1;
      tree_thread_data[10].min_symbol = i;
      if (i < next_new_symbol_number - 1)
        sum_symbols += symbol_counts[i++];
      symbols_limit = symbols_div_100 * 90;
      while (sum_symbols < symbols_limit)
        sum_symbols += symbol_counts[i++];
      tree_thread_data[10].max_symbol = i - 1;
      tree_thread_data[11].min_symbol = i;
      if (i < next_new_symbol_number - 1)
        sum_symbols += symbol_counts[i++];
      symbols_limit = symbols_div_100 * 95;
      while (sum_symbols < symbols_limit)
        sum_symbols += symbol_counts[i++];
      tree_thread_data[11].max_symbol = i - 1;
      tree_thread_data[12].min_symbol = i;
      tree_thread_data[12].max_symbol = next_new_symbol_number - 1;

      next_node_num = 1;
      tree_thread_data[7].first_node_num = 1;
      main_nodes_limit = nodes_div_100 * 6 - 10;
      tree_thread_data[0].first_node_num = nodes_div_100 * 6;
      tree_thread_data[0].nodes_limit = nodes_div_100 * 12;
      tree_thread_data[7].nodes_limit = nodes_div_100 * 12;
      tree_thread_data[1].first_node_num = tree_thread_data[8].first_node_num = nodes_div_100 * 12;
      tree_thread_data[1].nodes_limit = tree_thread_data[8].nodes_limit = nodes_div_100 * 22;
      tree_thread_data[2].first_node_num = tree_thread_data[9].first_node_num = nodes_div_100 * 22;
      tree_thread_data[2].nodes_limit = tree_thread_data[9].nodes_limit = nodes_div_100 * 34;
      tree_thread_data[3].first_node_num = tree_thread_data[10].first_node_num = nodes_div_100 * 34;
      tree_thread_data[3].nodes_limit = tree_thread_data[10].nodes_limit = nodes_div_100 * 48;
      tree_thread_data[4].first_node_num = tree_thread_data[11].first_node_num = nodes_div_100 * 48;
      tree_thread_data[4].nodes_limit = tree_thread_data[11].nodes_limit = nodes_div_100 * 64;
      tree_thread_data[5].first_node_num = tree_thread_data[12].first_node_num = nodes_div_100 * 64;
      tree_thread_data[5].nodes_limit = tree_thread_data[12].nodes_limit = nodes_div_100 * 81;
      tree_thread_data[6].first_node_num = nodes_div_100 * 81;
      tree_thread_data[6].nodes_limit = node_num_limit;
      for (i = 0 ; i < 13 ; i++) {
        tree_thread_data[i].start_cycle_symbol_ptr = start_cycle_symbol_ptr;
        tree_thread_data[i].base_nodes_child_node_num = base_nodes_child_node_num;
      }
      if (fast_section == fast_sections - 1)
        end_cycle_symbol_ptr = end_symbol_ptr;
      else
        end_cycle_symbol_ptr = start_symbol_ptr
            + (uint32_t)((float)num_file_symbols * (float)(fast_section + 1) / (float)fast_sections);
      atomic_store_explicit(&scan_symbol_ptr, (uintptr_t)end_cycle_symbol_ptr, memory_order_relaxed);
      atomic_store_explicit(&max_symbol_ptr, (uintptr_t)end_cycle_symbol_ptr, memory_order_release);
      for (i = 0 ; i < 7 ; i++)
        pthread_create(&build_tree_threads[i], NULL, build_tree_thread, (void *)&tree_thread_data[i]);
    }
    memset(base_nodes_child_node_num, 0, 4 * (main_max_symbol + 1) * BASE_NODES_CHILD_ARRAY_SIZE);

    if (fast_mode == 0) {
      do {
        symbol = *in_symbol_ptr++;
        if (symbol <= main_max_symbol) {
          atomic_store_explicit(&scan_symbol_ptr, (uintptr_t)in_symbol_ptr, memory_order_relaxed);
          if ((int32_t)*in_symbol_ptr >= 0) {
            add_suffix(symbol, in_symbol_ptr, &next_node_num);
            if (next_node_num >= main_nodes_limit)
              goto done_building_tree_tree;
          }
        }
      } while (symbol != 0xFFFFFFFE);
      in_symbol_ptr--;
done_building_tree_tree:
      atomic_store_explicit(&scan_symbol_ptr, (uintptr_t)in_symbol_ptr, memory_order_relaxed);
      atomic_store_explicit(&max_symbol_ptr, (uintptr_t)in_symbol_ptr, memory_order_release);
      node_ptrs_num = 0;
      atomic_store_explicit(&rank_scores_write_index, 0, memory_order_relaxed);
      atomic_store_explicit(&rank_scores_read_index, 0, memory_order_relaxed);
#ifdef PRINTON
      fprintf(stderr, ".");
#endif
      rank_scores_data_ptr->max_scores = (uint16_t)max_scores;

      pthread_create(&rank_scores_thread1, NULL, rank_scores_thread, (void *)rank_scores_data_ptr);
      score_symbol_tree(0, main_max_symbol, rank_scores_data_ptr->rank_scores_buffer, node_data, &node_ptrs_num,
          profit_ratio_power, symbol_entropy, symbol_counts);
      for (i = 0 ; i < 12 ; i++) {
#ifdef PRINTON
        fprintf(stderr, ".");
#endif
        if (i < 6) {
          pthread_join(build_tree_threads[i], NULL);
          pthread_create(&build_tree_threads[i], NULL, build_tree_thread, (void *)&tree_thread_data[i + 6]);
        } else
          pthread_join(build_tree_threads[i - 6], NULL);
#ifdef PRINTON
        fprintf(stderr, ".");
#endif
        score_symbol_tree(tree_thread_data[i].min_symbol, tree_thread_data[i].max_symbol,
            rank_scores_data_ptr->rank_scores_buffer, node_data, &node_ptrs_num, profit_ratio_power,
            symbol_entropy, symbol_counts);
      }

      if ((node_ptrs_num & 0xFFF) == 0)
        while ((uint16_t)(node_ptrs_num - atomic_load_explicit(&rank_scores_read_index, memory_order_acquire))
            >= 0xF000); // wait
      rank_scores_data_ptr->rank_scores_buffer[node_ptrs_num].last_match_index = 0;
      atomic_store_explicit(&rank_scores_write_index, node_ptrs_num + 1, memory_order_release);
      pthread_join(rank_scores_thread1, NULL);
#ifdef PRINTON
      fprintf(stderr, "\rStart %.4f", cycle_start_ratio);
#endif
      cycle_end_ratio = (float)(in_symbol_ptr - start_symbol_ptr) / (float)num_file_symbols;
    } else {
      do {
        symbol = *in_symbol_ptr++;
        if (symbol <= main_max_symbol) {
          if ((int32_t)*in_symbol_ptr >= 0) {
            add_suffix(symbol, in_symbol_ptr, &next_node_num);
            if (next_node_num >= main_nodes_limit)
              break;
          }
        }
      } while (in_symbol_ptr != end_cycle_symbol_ptr);
      node_ptrs_num = 0;
      atomic_store_explicit(&rank_scores_write_index, 0, memory_order_relaxed);
      atomic_store_explicit(&rank_scores_read_index, 0, memory_order_relaxed);
      i = 0;
      do {
        if (symbol_counts[i] != 0) {
          if (symbol_counts[i] < NUM_PRECALCULATED_LOG2_X)
            symbol_entropy_f[i] = (float)(log_file_symbols - log2_x[symbol_counts[i]]);
          else
            symbol_entropy_f[i] = (float)log_file_symbols - log2f((float)symbol_counts[i]);
        }
      } while (++i < next_new_symbol_number);
      rank_scores_data_ptr->max_scores = (uint16_t)max_scores;
      rank_scores_data_ptr->num_file_symbols = num_file_symbols;
      pthread_join(build_tree_threads[0], NULL);
      pthread_create(&rank_scores_thread1, NULL, rank_scores_thread_fast, (void *)rank_scores_data_ptr);
      score_symbol_tree_fast(0, tree_thread_data[0].max_symbol, rank_scores_data_ptr->rank_scores_buffer, node_data,
          &node_ptrs_num, production_cost, profit_ratio_power, log2_num_symbols_plus_substitution_cost, new_symbol_cost,
          symbol_entropy_f, symbol_counts);
      for (i = 1 ; i <= 12 ; i++) {
        if (i <= 6) {
          pthread_join(build_tree_threads[i], NULL);
          pthread_create(&build_tree_threads[i - 1], NULL, build_tree_thread, (void *)&tree_thread_data[i + 6]);
        } else
          pthread_join(build_tree_threads[i - 7], NULL);
        score_symbol_tree_fast(tree_thread_data[i].min_symbol, tree_thread_data[i].max_symbol,
            rank_scores_data_ptr->rank_scores_buffer, node_data, &node_ptrs_num, production_cost, profit_ratio_power,
            log2_num_symbols_plus_substitution_cost, new_symbol_cost, symbol_entropy_f, symbol_counts);
      }
      if ((node_ptrs_num & 0xFFF) == 0)
        while ((uint16_t)(node_ptrs_num - atomic_load_explicit(&rank_scores_read_index, memory_order_acquire))
            >= 0xF000); // wait
      rank_scores_data_ptr->rank_scores_buffer[node_ptrs_num].last_match_index = 0;
      atomic_store_explicit(&rank_scores_write_index, node_ptrs_num + 1, memory_order_release);
      pthread_join(rank_scores_thread1, NULL);
    }
    num_candidates = rank_scores_data_ptr->num_candidates;
    prior_cycle_symbols = in_symbol_ptr - start_cycle_symbol_ptr;

    if (num_candidates == 0) {
#ifdef PRINTON
      if (fast_mode == 0)
        fprintf(stderr, "\r");
#endif
      if (scan_mode == 3) {
        if (min_score > 0.0) {
          num_candidates = 1;
          prior_min_score = min_score;
          min_score = 0.0;
        } else if (fast_sections != 1) {
          if (fast_sections == 23) {
            fast_sections = 9;
            fast_section = 0;
            min_score = 8.0;
          } else {
            fast_sections = (fast_sections + 1) >> 1;
            fast_section >>= 1;
            min_score = 4.0;
          }
          prior_min_score = BIG_FLOAT;
          fast_min_score = 1.0;

          num_candidates = 1;
          scan_mode = 2;
        }
      } else {
        scan_mode = 3;
        num_candidates = 1;
        prior_min_score = min_score;
        min_score = 0.25 * min_score;
      }
    } else {
      free_RAM_ptr = (char *)(((size_t)end_symbol_ptr + 8) & ~7);
      struct node_score_data * tmp_candidates = (struct node_score_data *)free_RAM_ptr;
        memcpy(&tmp_candidates[0], &candidates[0], MAX_SCORES_FAST * sizeof(struct node_score_data));
        for (candidate_num = 0 ; candidate_num < MAX_SCORES_FAST ; candidate_num++)
          memcpy(&candidates[candidate_num], &tmp_candidates[candidates_index[candidate_num]],
              sizeof(struct node_score_data));

      if (fast_mode == 0) {
#ifdef PRINTON
        fprintf(stderr, " score[0-%hu] = %.5f-%.5f\n", (unsigned short int)num_candidates - 1,
            candidates[0].score, candidates[num_candidates - 1].score);
#endif
        if (candidates[num_candidates - 1].score < 0.1 * candidates[0].score - 1.0) {
          candidate_num = 1;
          while ((candidate_num + 0x100 < num_candidates)
              && (candidates[candidate_num + 0x100].score >= 0.1 * candidates[0].score - 1.0))
            candidate_num += 0x100;
          while (candidate_num < num_candidates) {
            if (candidates[candidate_num].score < 0.1 * candidates[0].score - 1.0) {
              num_candidates = candidate_num;
              break;
            }
            candidate_num++;
          }
        }
      } else if (fast_sections != 1) {
        section_scores[fast_section] = candidates[num_candidates - 1].score;
        uint8_t old_fast_section = fast_section;
        if (++fast_section == fast_sections)
          fast_section = 0;
        if (candidates[num_candidates - 1].score < fast_min_score) {
          if (fast_sections == 23) {
            fast_sections = 9;
            fast_section = 0;
            min_score = 8.0;
          } else {
            fast_sections = (fast_sections + 1) >> 1;
            fast_section >>= 1;
            min_score = 4.0;
          }
          section_repeats = 0;
          for (i = 0 ; i < fast_sections ; i++)
            section_scores[i] = BIG_FLOAT;
          prior_min_score = BIG_FLOAT;
          fast_min_score = 1.0;
          scan_mode = 2;
        } else {
          i = fast_section + 1;
          if (i == fast_sections)
            i = 0;
          while (i != old_fast_section) {
            if (section_scores[i] > section_scores[fast_section])
              fast_section = i;
            if (++i == fast_sections)
              i = 0;
          }
          if ((section_repeats < 2) && (section_scores[old_fast_section] > section_scores[fast_section])) {
            fast_section = old_fast_section;
            section_repeats++;
          } else
            section_repeats = 0;
        }
      }

      if (next_new_symbol_number + num_candidates > max_rules)
        num_candidates = max_rules - next_new_symbol_number;

      // build a prefix tree of the match strings
      child_ptr_array = (struct match_node **)free_RAM_ptr;
      struct match_node * match_nodes = (struct match_node *)(free_RAM_ptr
          + sizeof(struct match_node *) * next_new_symbol_number);
      num_match_nodes = 0;
      max_match_length = 0;
      candidate_num = 0;
      while (candidate_num < num_candidates) {
        if (free_RAM_ptr + (next_new_symbol_number + num_match_nodes) * sizeof(struct match_node)
            + max_match_length * sizeof(uint32_t) >= end_RAM_ptr) {
          num_candidates = candidate_num - 1;
          break;
        }
        uint32_t *best_score_last_match_ptr, *best_score_match_ptr;
        if (candidates[candidate_num].num_symbols > max_match_length)
          max_match_length = candidates[candidate_num].num_symbols;
        best_score_last_match_ptr = start_symbol_ptr + candidates[candidate_num].last_match_index;
        best_score_match_ptr = best_score_last_match_ptr - candidates[candidate_num].num_symbols + 1;
        if (num_match_nodes == 0) {
          init_match_node(match_nodes, *best_score_match_ptr, 0, candidate_num);
          num_match_nodes = 1;
        }
        match_node_ptr = match_nodes;
        while (best_score_match_ptr <= best_score_last_match_ptr) {
          symbol = *best_score_match_ptr;
          if (match_node_ptr->child_ptr == 0) {
            match_node_ptr->child_ptr = &match_nodes[num_match_nodes++];
            match_node_ptr = match_node_ptr->child_ptr;
            init_match_node(match_node_ptr, symbol, 0, candidate_num);
          } else {
            match_node_ptr = match_node_ptr->child_ptr;
            uint8_t sibling_number;
            if (move_to_match_sibling(match_nodes, &match_node_ptr, symbol, &sibling_number) != 0) {
              if (match_node_ptr->child_ptr == 0) {
                candidate_bad[candidate_num] = 1;
                break;
              }
            } else {
              match_node_ptr->sibling_node_num[sibling_number] = num_match_nodes;
              match_node_ptr = &match_nodes[num_match_nodes++];
              init_match_node(match_node_ptr, symbol, 0, candidate_num);
            }
          }
          best_score_match_ptr++;
        }
        if (match_node_ptr->child_ptr != 0)
          candidate_bad[candidate_num] = 1;
        candidate_num++;
      }

      // for each candidate, search substrings for matches with other candidates, if found invalidate lower score
      candidate_num = 0;
      while (candidate_num < num_candidates) {
        uint32_t *best_score_last_match_ptr, *best_score_match_ptr;
        best_score_last_match_ptr = start_symbol_ptr + candidates[candidate_num].last_match_index;
        best_score_match_ptr = best_score_last_match_ptr - candidates[candidate_num].num_symbols + 1;
        // read the first symbol
        symbol = *best_score_match_ptr++;
        match_node_ptr = &match_nodes[1];
        move_to_existing_match_sibling(match_nodes, &match_node_ptr, symbol);
        while (best_score_match_ptr <= best_score_last_match_ptr) {
          // starting with the second symbol, look for suffixes that are in the prefix tree
          search_match_ptr = best_score_match_ptr;
          struct match_node * search_node_ptr = match_nodes;
          while (1) { // follow the tree until find child = 0 or sibling = 0
            if (search_node_ptr->child_ptr == 0) { // found a scored string that is a substring of this string
              if (fast_mode == 0) {
                if (search_node_ptr->score_number > candidate_num)
                  candidate_bad[search_node_ptr->score_number] = 1;
                else if (search_node_ptr->score_number != candidate_num)
                  candidate_bad[candidate_num] = 1;
                break;
              } else {
                if (search_node_ptr->score_number > candidate_num) {
                  if (candidate_bad[candidate_num] == 0)
                    candidate_bad[search_node_ptr->score_number] = 1;
                } else if (search_node_ptr->score_number != candidate_num) {
                  if (candidate_bad[search_node_ptr->score_number] == 0)
                    candidate_bad[candidate_num] = 1;
                }
                break;
              }
            }
            search_node_ptr = search_node_ptr->child_ptr;
            symbol = *search_match_ptr;
            if (move_to_search_sibling(match_nodes, symbol, &search_node_ptr) == 0)
              break;
            match_node_ptr->miss_ptr = search_node_ptr;
            search_match_ptr++;
          }
          symbol = *best_score_match_ptr++;
        }
        candidate_num++;
      }

      // Redo the tree build and miss values with just the valid score symbols
      match_node_ptr = match_nodes + next_new_symbol_number;
      num_match_nodes = 0;
      j = next_new_symbol_number;
      while (j-- != 0)
        child_ptr_array[j] = 0;
      candidate_num = 0;
      while (candidate_num < num_candidates) {
        if (candidate_bad[candidate_num] == 0) {
          uint32_t *best_score_last_match_ptr, *best_score_match_ptr;
          best_score_last_match_ptr = start_symbol_ptr + candidates[candidate_num].last_match_index;
          best_score_match_ptr = best_score_last_match_ptr - candidates[candidate_num].num_symbols + 1;
          struct match_node ** child_ptr_ptr = &child_ptr_array[*best_score_match_ptr++];
          symbol = *best_score_match_ptr++;
          best_score_num_symbols = 2;
          match_node_ptr = move_to_base_match_child_with_make(match_nodes, symbol, candidate_num, &num_match_nodes,
              child_ptr_ptr);
          while (best_score_match_ptr <= best_score_last_match_ptr)
            move_to_match_child_with_make(match_nodes, &match_node_ptr, *best_score_match_ptr++, candidate_num,
                ++best_score_num_symbols, &num_match_nodes);
        }
        candidate_num++;
      }

      // span nodes entering the longest (first) suffix match for each node
      candidate_num = 0;
      while (candidate_num < num_candidates) {
        if (candidate_bad[candidate_num] == 0) {
          uint32_t *best_score_last_match_ptr, *best_score_suffix_ptr;
          best_score_last_match_ptr = start_symbol_ptr + candidates[candidate_num].last_match_index;
          best_score_suffix_ptr = best_score_last_match_ptr - candidates[candidate_num].num_symbols + 1;
          suffix_node_number = child_ptr_array[*best_score_suffix_ptr++] - match_nodes;
          // starting at the node of the 2nd symbol in string, match strings with prefix tree until no match found,
          //   for each match node found, if suffix miss symbol is zero, set it to the tree symbol node
          while (best_score_suffix_ptr <= best_score_last_match_ptr) {
            // follow the suffix until the end (or break on no tree matches)
            symbol = *best_score_suffix_ptr++;
            uint32_t shifted_symbol = symbol;
            while (symbol != match_nodes[suffix_node_number].symbol) {
              suffix_node_number = match_nodes[suffix_node_number].sibling_node_num[shifted_symbol & 0xF];
              shifted_symbol >>= 4;
            }
            match_node_ptr = &match_nodes[suffix_node_number];
            uint32_t *best_score_match_ptr;
            best_score_match_ptr = best_score_suffix_ptr;
            if (child_ptr_array[symbol] != 0) {
              if ((match_node_ptr->child_ptr != 0) && (match_node_ptr->child_ptr->miss_ptr == 0))
                write_siblings_miss_ptr(match_nodes, match_node_ptr->child_ptr, child_ptr_array[symbol]);
              struct match_node * search_node_ptr = child_ptr_array[symbol];
              while (best_score_match_ptr <= best_score_last_match_ptr) {
                // follow the tree until end of match string or find child = 0 or sibling = 0
                symbol = *best_score_match_ptr++;
                match_node_ptr = match_node_ptr->child_ptr;
                move_to_existing_match_sibling(match_nodes, &match_node_ptr, symbol);
                if (move_to_search_sibling(match_nodes, symbol, &search_node_ptr) == 0)
                  break;
                if (match_node_ptr->child_ptr == 0) {
                  if (match_node_ptr->hit_ptr == 0)
                    match_node_ptr->hit_ptr = search_node_ptr;
                } else if (match_node_ptr->child_ptr->miss_ptr == 0)
                  write_siblings_miss_ptr(match_nodes, match_node_ptr->child_ptr, search_node_ptr->child_ptr);
                if (search_node_ptr->child_ptr == 0) // no child, so done with this suffix
                  break;
                search_node_ptr = search_node_ptr->child_ptr;
              }
            }
            suffix_node_number = match_nodes[suffix_node_number].child_ptr - match_nodes;
          }
        }
        candidate_num++;
      }

      // save the match strings so they can be added to the end of the data after symbol substitution is done
      match_strings = (uint32_t *)((size_t)match_nodes + (size_t)num_match_nodes * sizeof(struct match_node));
      overlap_check_data = (struct overlap_check *)(((size_t)&match_strings[num_candidates * max_match_length] + 7) & ~7);
      for (i = 1 ; i < 8 ; i++)
        overlap_check_data[i].candidate_bad = &candidate_bad[0];

      candidate_num = 0;
      while (candidate_num < num_candidates) {
        if (candidate_bad[candidate_num] == 0) {
          match_string_start_ptr = &match_strings[candidate_num * max_match_length];
          node_string_start_ptr = start_symbol_ptr + candidates[candidate_num].last_match_index
              - candidates[candidate_num].num_symbols + 1;
          for (j = 0 ; j < candidates[candidate_num].num_symbols ; j++)
            *(match_string_start_ptr + j) = *(node_string_start_ptr + j);
        }
        candidate_num++;
      }

      uint32_t *matches_start_ptr[8], *next_match_start_ptr[8];
      uint32_t *begin_matches = (uint32_t *)((size_t)overlap_check_data + 8 * sizeof(struct overlap_check));
      for (i = 0 ; i < 8 ; i++) {
        next_match_start_ptr[i] = matches_start_ptr[i] = begin_matches + i * (((uint32_t *)end_RAM_ptr - begin_matches) >> 3);
        next_match_ptr[i] = next_match_start_ptr[i];
      }

#ifdef PRINTON
      if (fast_mode == 0)
        fprintf(stderr, "Overlap search\r");
#endif
      block_size = num_file_symbols >> 3;
      block_ptr = start_symbol_ptr + block_size;
      stop_matches_symbol_ptr[0] = block_ptr;
      stop_symbol_ptr = block_ptr + MAX_MATCH_LENGTH;
      if (stop_symbol_ptr >= end_symbol_ptr) {
        stop_symbol_ptr = end_symbol_ptr;
        stop_matches_symbol_ptr[0] = end_symbol_ptr;
      } else {
        for (i = 1 ; i < 8 ; i++) {
          overlap_check_data[i].start_symbol_ptr = block_ptr;
          block_ptr += block_size;
          if (i < 7) {
            overlap_check_data[i].stop_matches_symbol_ptr = block_ptr;
            overlap_check_data[i].stop_symbol_ptr = block_ptr + MAX_MATCH_LENGTH;
            if (overlap_check_data[i].stop_symbol_ptr > end_symbol_ptr)
              overlap_check_data[i].stop_symbol_ptr = end_symbol_ptr;
          } else {
            overlap_check_data[7].stop_matches_symbol_ptr = end_symbol_ptr;
            overlap_check_data[7].stop_symbol_ptr = end_symbol_ptr;
          }
          stop_matches_symbol_ptr[i] = overlap_check_data[i].stop_matches_symbol_ptr;
          overlap_check_data[i].next_match_ptr_ptr = &next_match_ptr[i];
          overlap_check_data[i].num_overlaps = num_candidates;
          overlap_check_data[i].match_nodes = match_nodes;
          if (overlap_check_data[i].stop_symbol_ptr - start_symbol_ptr + MAX_MATCH_LENGTH < first_define_index)
            pthread_create(&overlap_check_threads[i - 1], NULL, overlap_check_no_defs_thread,
               (void *)&overlap_check_data[i]);
          else
            pthread_create(&overlap_check_threads[i - 1], NULL, overlap_check_thread, (void *)&overlap_check_data[i]);
        }
      }

      num_overlaps = num_candidates;
      for (j = 0 ; j < num_candidates ; j++)
        overlap_check_data[0].next[j] = -1;

      // scan the data, following prefix tree
      uint8_t found_same_score_prior_match;
      uint32_t prior_match_number;
      num_prior_matches = 0;
      in_symbol_ptr = start_symbol_ptr;

      if (stop_symbol_ptr - start_symbol_ptr + MAX_MATCH_LENGTH >= first_define_index) {
main_overlap_check_loop_no_match:
        symbol = *in_symbol_ptr++;
        if (in_symbol_ptr >= stop_symbol_ptr)
          goto main_overlap_check_loop_end;
        if (((int32_t)symbol < 0) || (child_ptr_array[symbol] == 0))
          goto main_overlap_check_loop_no_match;
        match_node_ptr = child_ptr_array[symbol];
main_overlap_check_loop_match:
        symbol = *in_symbol_ptr++;
        if (symbol != match_node_ptr->symbol) {
          uint32_t shifted_symbol = symbol;
          do {
            if (match_node_ptr->sibling_node_num[shifted_symbol & 0xF] != 0) {
              match_node_ptr = &match_nodes[match_node_ptr->sibling_node_num[shifted_symbol & 0xF]];
              shifted_symbol >>= 4;
            } else {
              if (match_node_ptr->miss_ptr == 0) {
                if (((int32_t)symbol < 0) || (child_ptr_array[symbol] == 0))
                  goto main_overlap_check_loop_no_match;
                match_node_ptr = child_ptr_array[symbol];
                goto main_overlap_check_loop_match;
              } else {
                match_node_ptr = match_node_ptr->miss_ptr;
                shifted_symbol = symbol;
              }
            }
          } while (symbol != match_node_ptr->symbol);
        }
        if (match_node_ptr->child_ptr != 0) {
          match_node_ptr = match_node_ptr->child_ptr;
          goto main_overlap_check_loop_match;
        }

        // no child, so found a match - check for overlaps
        node_score_number = match_node_ptr->score_number;
        if ((in_symbol_ptr - match_node_ptr->num_symbols < stop_matches_symbol_ptr[0])
            && (candidate_bad[node_score_number] == 0)) {
          *next_match_ptr[0] = node_score_number;
          next_match_ptr[0]++;
          *next_match_ptr[0] = in_symbol_ptr - start_symbol_ptr - match_node_ptr->num_symbols;
          next_match_ptr[0]++;
        }

        if ((num_prior_matches != 0) && (in_symbol_ptr - match_node_ptr->num_symbols
            <= prior_match_end_ptr[num_prior_matches - 1])) {
          if (num_prior_matches == 1) {
            if (prior_match_score_number[0] != node_score_number) {
              if (fast_mode == 0) {
                if (prior_match_score_number[0] > node_score_number)
                  candidate_bad[prior_match_score_number[0]] = 1;
                else
                  candidate_bad[node_score_number] = 1;
              } else {
                uint32_t low_score, high_score;
                if (node_score_number < prior_match_score_number[0]) {
                  low_score = node_score_number;
                  high_score = prior_match_score_number[0];
                } else {
                  low_score = prior_match_score_number[0];
                  high_score = node_score_number;
                }
                int32_t * next_overlap_num_ptr = &overlap_check_data[0].next[low_score];
                while ((*next_overlap_num_ptr != -1)
                    && (overlap_check_data[0].second[*next_overlap_num_ptr] < high_score))
                  next_overlap_num_ptr = &overlap_check_data[0].next[*next_overlap_num_ptr];
                if ((*next_overlap_num_ptr == -1)
                    || (overlap_check_data[0].second[*next_overlap_num_ptr] != high_score)) {
                  if (num_overlaps < 150000) {
                    overlap_check_data[0].second[num_overlaps] = high_score;
                    overlap_check_data[0].next[num_overlaps] = *next_overlap_num_ptr;
                    *next_overlap_num_ptr = num_overlaps++;
                  } else
                    candidate_bad[high_score] = 1;
                }
              }
              prior_match_end_ptr[1] = in_symbol_ptr - 1;
              prior_match_score_number[1] = node_score_number;
              num_prior_matches = 2;
            }
          } else {
            prior_match_number = 0;
            found_same_score_prior_match = 0;
            do {
              if (in_symbol_ptr - match_node_ptr->num_symbols > prior_match_end_ptr[prior_match_number]) {
                num_prior_matches--;
                for (j = prior_match_number ; j < num_prior_matches ; j++) {
                  prior_match_end_ptr[j] = prior_match_end_ptr[j + 1];
                  prior_match_score_number[j] = prior_match_score_number[j + 1];
                }
              } else { // overlapping symbol substitution strings, so invalidate the lower score
                if (prior_match_score_number[prior_match_number] == node_score_number)
                  found_same_score_prior_match = 1;
                else if (fast_mode == 0) {
                  if (prior_match_score_number[prior_match_number] > node_score_number)
                    candidate_bad[prior_match_score_number[prior_match_number]] = 1;
                  else
                    candidate_bad[node_score_number] = 1;
                } else {
                  uint32_t low_score, high_score;
                  if (node_score_number < prior_match_score_number[prior_match_number]) {
                    low_score = node_score_number;
                    high_score = prior_match_score_number[prior_match_number];
                  } else {
                    low_score = prior_match_score_number[prior_match_number];
                    high_score = node_score_number;
                  }
                  int32_t * next_overlap_num_ptr = &overlap_check_data[0].next[low_score];
                  while ((*next_overlap_num_ptr != -1)
                      && (overlap_check_data[0].second[*next_overlap_num_ptr] < high_score))
                    next_overlap_num_ptr = &overlap_check_data[0].next[*next_overlap_num_ptr];
                  if ((*next_overlap_num_ptr == -1)
                      || (overlap_check_data[0].second[*next_overlap_num_ptr] != high_score)) {
                    if (num_overlaps < 150000) {
                      overlap_check_data[0].second[num_overlaps] = high_score;
                      overlap_check_data[0].next[num_overlaps] = *next_overlap_num_ptr;
                      *next_overlap_num_ptr = num_overlaps++;
                    } else
                      candidate_bad[high_score] = 1;
                  }
                }
                prior_match_number++;
              }
            } while (prior_match_number < num_prior_matches);
            if (found_same_score_prior_match == 0) {
              prior_match_end_ptr[num_prior_matches] = in_symbol_ptr - 1;
              prior_match_score_number[num_prior_matches++] = node_score_number;
            }
          }
        } else {
          num_prior_matches = 1;
          prior_match_end_ptr[0] = in_symbol_ptr - 1;
          prior_match_score_number[0] = node_score_number;
        }
        match_node_ptr = match_node_ptr->hit_ptr;
        if (match_node_ptr == 0) {
          if (child_ptr_array[symbol] == 0)
            goto main_overlap_check_loop_no_match;
          match_node_ptr = child_ptr_array[symbol];
          goto main_overlap_check_loop_match;
        }
        match_node_ptr = match_node_ptr->child_ptr;
        goto main_overlap_check_loop_match;
      } else {
main_overlap_check_no_defs_loop_no_match:
        symbol = *in_symbol_ptr++;
        if (in_symbol_ptr >= stop_symbol_ptr)
          goto main_overlap_check_loop_end;
        if (child_ptr_array[symbol] == 0)
          goto main_overlap_check_no_defs_loop_no_match;
        match_node_ptr = child_ptr_array[symbol];
main_overlap_check_no_defs_loop_match:
        symbol = *in_symbol_ptr++;
        if (symbol != match_node_ptr->symbol) {
          uint32_t shifted_symbol = symbol;
          do {
            if (match_node_ptr->sibling_node_num[shifted_symbol & 0xF] != 0) {
              match_node_ptr = &match_nodes[match_node_ptr->sibling_node_num[shifted_symbol & 0xF]];
              shifted_symbol >>= 4;
            } else if (match_node_ptr->miss_ptr == 0) {
              if (child_ptr_array[symbol] == 0)
                goto main_overlap_check_no_defs_loop_no_match;
              if (in_symbol_ptr <= stop_symbol_ptr) {
                match_node_ptr = child_ptr_array[symbol];
                goto main_overlap_check_no_defs_loop_match;
              }
              goto main_overlap_check_loop_end;
            } else {
              match_node_ptr = match_node_ptr->miss_ptr;
              shifted_symbol = symbol;
            }
          } while (symbol != match_node_ptr->symbol);
        }
        if (match_node_ptr->child_ptr != 0) {
          if ((in_symbol_ptr > stop_symbol_ptr) && (in_symbol_ptr - match_node_ptr->num_symbols >= stop_symbol_ptr))
            goto main_overlap_check_loop_end;
          match_node_ptr = match_node_ptr->child_ptr;
          goto main_overlap_check_no_defs_loop_match;
        }

        // no child, so found a match - check for overlaps
        node_score_number = match_node_ptr->score_number;
        if ((in_symbol_ptr - match_node_ptr->num_symbols < stop_matches_symbol_ptr[0])
            && (candidate_bad[node_score_number] == 0)) {
          *next_match_ptr[0] = node_score_number;
          next_match_ptr[0]++;
          *next_match_ptr[0] = in_symbol_ptr - start_symbol_ptr - match_node_ptr->num_symbols;
          next_match_ptr[0]++;
        }

        if ((num_prior_matches != 0) && (in_symbol_ptr - match_node_ptr->num_symbols
            <= prior_match_end_ptr[num_prior_matches - 1])) {
          if (num_prior_matches == 1) {
            if (prior_match_score_number[0] != node_score_number) {
              if (fast_mode == 0) {
                if (prior_match_score_number[0] > node_score_number)
                  candidate_bad[prior_match_score_number[0]] = 1;
                else
                  candidate_bad[node_score_number] = 1;
              } else {
                uint32_t low_score, high_score;
                if (node_score_number < prior_match_score_number[0]) {
                  low_score = node_score_number;
                  high_score = prior_match_score_number[0];
                } else {
                  low_score = prior_match_score_number[0];
                  high_score = node_score_number;
                }
                int32_t * next_overlap_num_ptr = &overlap_check_data[0].next[low_score];
                while ((*next_overlap_num_ptr != -1)
                    && (overlap_check_data[0].second[*next_overlap_num_ptr] < high_score))
                  next_overlap_num_ptr = &overlap_check_data[0].next[*next_overlap_num_ptr];
                if ((*next_overlap_num_ptr == -1)
                    || (overlap_check_data[0].second[*next_overlap_num_ptr] != high_score)) {
                  if (num_overlaps < 150000) {
                    overlap_check_data[0].second[num_overlaps] = high_score;
                    overlap_check_data[0].next[num_overlaps] = *next_overlap_num_ptr;
                    *next_overlap_num_ptr = num_overlaps++;
                  } else
                    candidate_bad[high_score] = 1;
                }
              }
              prior_match_end_ptr[1] = in_symbol_ptr - 1;
              prior_match_score_number[1] = node_score_number;
              num_prior_matches = 2;
            }
          } else {
            prior_match_number = 0;
            found_same_score_prior_match = 0;
            do {
              if (in_symbol_ptr - match_node_ptr->num_symbols > prior_match_end_ptr[prior_match_number]) {
                num_prior_matches--;
                for (j = prior_match_number ; j < num_prior_matches ; j++) {
                  prior_match_end_ptr[j] = prior_match_end_ptr[j + 1];
                  prior_match_score_number[j] = prior_match_score_number[j + 1];
                }
              } else { // overlapping symbol substitution strings, so invalidate the lower score
                if (prior_match_score_number[prior_match_number] == node_score_number)
                  found_same_score_prior_match = 1;
                else if (fast_mode == 0) {
                  if (prior_match_score_number[prior_match_number] > node_score_number)
                    candidate_bad[prior_match_score_number[prior_match_number]] = 1;
                  else
                    candidate_bad[node_score_number] = 1;
                } else {
                  uint32_t low_score, high_score;
                  if (node_score_number < prior_match_score_number[prior_match_number]) {
                    low_score = node_score_number;
                    high_score = prior_match_score_number[prior_match_number];
                  } else {
                    low_score = prior_match_score_number[prior_match_number];
                    high_score = node_score_number;
                  }
                  int32_t * next_overlap_num_ptr = &overlap_check_data[0].next[low_score];
                  while ((*next_overlap_num_ptr != -1)
                      && (overlap_check_data[0].second[*next_overlap_num_ptr] < high_score))
                    next_overlap_num_ptr = &overlap_check_data[0].next[*next_overlap_num_ptr];
                  if ((*next_overlap_num_ptr == -1)
                      || (overlap_check_data[0].second[*next_overlap_num_ptr] != high_score)) {
                    if (num_overlaps < 150000) {
                      overlap_check_data[0].second[num_overlaps] = high_score;
                      overlap_check_data[0].next[num_overlaps] = *next_overlap_num_ptr;
                      *next_overlap_num_ptr = num_overlaps++;
                    } else
                      candidate_bad[high_score] = 1;
                  }
                }
                prior_match_number++;
              }
            } while (prior_match_number < num_prior_matches);
            if (found_same_score_prior_match == 0) {
              prior_match_end_ptr[num_prior_matches] = in_symbol_ptr - 1;
              prior_match_score_number[num_prior_matches++] = node_score_number;
            }
          }
        } else {
          num_prior_matches = 1;
          prior_match_end_ptr[0] = in_symbol_ptr - 1;
          prior_match_score_number[0] = node_score_number;
        }
        match_node_ptr = match_node_ptr->hit_ptr;
        if (match_node_ptr == 0) {
          if (child_ptr_array[symbol] == 0)
            goto main_overlap_check_no_defs_loop_no_match;
          match_node_ptr = child_ptr_array[symbol];
          goto main_overlap_check_no_defs_loop_match;
        }
        if ((in_symbol_ptr <= stop_symbol_ptr) || (in_symbol_ptr - match_node_ptr->num_symbols < stop_symbol_ptr)) {
          match_node_ptr = match_node_ptr->child_ptr;
          goto main_overlap_check_no_defs_loop_match;
        }
      }

main_overlap_check_loop_end:
      if (stop_symbol_ptr < end_symbol_ptr) {
        for (i = 1 ; i < 8 ; i++)
          pthread_join(overlap_check_threads[i - 1], NULL);
        if (fast_mode == 1) {
          for (candidate_num = 0 ; candidate_num < num_candidates - 1 ; candidate_num++) {
            if (candidate_bad[candidate_num] == 0) {
              for (j = 0 ; j < 8 ; j++) {
                int32_t next_overlap_num = overlap_check_data[j].next[candidate_num];
                while (next_overlap_num != -1) {
                  candidate_bad[overlap_check_data[j].second[next_overlap_num]] = -1;
                  next_overlap_num = overlap_check_data[j].next[next_overlap_num];
                }
              }
            }
          }
        }
      } else if (fast_mode == 1) {
        for (candidate_num = 0 ; candidate_num < num_candidates - 1 ; candidate_num++) {
          if (candidate_bad[candidate_num] == 0) {
            int32_t next_overlap_num = overlap_check_data[0].next[candidate_num];
            while (next_overlap_num != -1) {
              candidate_bad[overlap_check_data[0].second[next_overlap_num]] = -1;
              next_overlap_num = overlap_check_data[0].next[next_overlap_num];
            }
          }
        }
      }

      j = next_new_symbol_number;
      for (i = 0 ; i < num_candidates ; i++)
        if (candidate_bad[i] == 0) {
          symbol_counts[j] = 0;
          new_rule_number[i] = j++;
        }

      in_symbol_ptr = out_symbol_ptr = start_symbol_ptr;
      int32_t prior_match_end = -1;
      uint8_t max_i = 0;
      if (stop_symbol_ptr < end_symbol_ptr)
        max_i = 7;
      for (i = 0 ; i <= max_i ; i++) {
        for (j = 0 ; j < (next_match_ptr[i] - next_match_start_ptr[i]) >> 1 ; j++) {
          uint16_t candidate_num = *(next_match_start_ptr[i] + 2 * j);
          if (candidate_bad[candidate_num] == 0) {
            uint32_t start_index = *(next_match_start_ptr[i] + 2 * j + 1);
            if ((int32_t)start_index > prior_match_end) {
              prior_match_end = start_index + candidates[candidate_num].num_symbols - 1;
              uint32_t *start_ptr = start_symbol_ptr + start_index;
              while (in_symbol_ptr < start_ptr)
                *out_symbol_ptr++ = *in_symbol_ptr++;
              *out_symbol_ptr++ = new_rule_number[candidate_num];
              symbol_counts[new_rule_number[candidate_num]]++;
              in_symbol_ptr += candidates[candidate_num].num_symbols;
            }
          }
        }
      }
      while (in_symbol_ptr < end_symbol_ptr)
        *out_symbol_ptr++ = *in_symbol_ptr++;

      // Add new production rules and update symbol counts
      for (i = 0 ; i < num_candidates ; i++) {
        if (candidate_bad[i] == 0) {
          *out_symbol_ptr++ = num_rules + 0x80000001;
          uint32_t *match_string_ptr, *match_string_end_ptr;
          match_string_ptr = match_strings + max_match_length * i;
          match_string_end_ptr = match_string_ptr + candidates[i].num_symbols;
          uint32_t sym1 = *match_string_ptr++;
          *out_symbol_ptr++ = sym1;
          symbol_ends[num_terminals + num_rules].start = symbol_ends[sym1].start;
          uint32_t repeats = symbol_counts[num_terminals + num_rules] - 1;
          symbol_counts[sym1] -= repeats;
          while (match_string_ptr != match_string_end_ptr) {
            uint32_t sym2 = *match_string_ptr;
            symbol_counts[sym2] -= repeats;
            o1c[symbol_ends[sym1].end][symbol_ends[sym2].start] -= repeats;
            num_ends[symbol_ends[sym1].end] -= repeats;
            num_starts[symbol_ends[sym2].start] -= repeats;
            sym1 = sym2;
            *out_symbol_ptr++ = *match_string_ptr++;
          }
          symbol_ends[num_terminals + num_rules++].end = symbol_ends[sym1].end;
        } else
          candidate_bad[i] = 0;
      }

      if (num_rules == 0)
        first_define_index = out_symbol_ptr - start_symbol_ptr;
      else {
        if (out_symbol_ptr < start_symbol_ptr + first_define_index)
          first_define_index = out_symbol_ptr - start_symbol_ptr;
        if (*(start_symbol_ptr + first_define_index) != 0x80000001)
          while (*(start_symbol_ptr + --first_define_index) != 0x80000001); // decrement index until found
      }
      end_symbol_ptr = out_symbol_ptr;
      *end_symbol_ptr = 0xFFFFFFFE;
      num_file_symbols = end_symbol_ptr - start_symbol_ptr;

      if (fast_mode == 0) {
        if (scan_mode == 3) {
          if (rank_scores_data_ptr->num_candidates != 0) {
            if (rank_scores_data_ptr->num_candidates == (uint16_t)max_scores) {
              if (min_score < prior_min_score) {
                if (max_scores > 10000)
                  new_min_score = min_score + min_score - prior_min_score - 0.015;
                else {
                  new_min_score = min_score + min_score - prior_min_score - 0.1;
                  if (new_min_score < min_score - 1.0)
                    new_min_score = min_score - 1.0;
                  if (new_min_score < 0.0)
                    new_min_score = 0.0;
                }
                prior_min_score = min_score;
              } else {
                new_min_score = 0.5 * (prior_min_score + min_score) - 0.1;
                if (new_min_score >= prior_min_score)
                  new_min_score = prior_min_score - 0.05;
                if (new_min_score < min_score)
                  new_min_score = min_score - 0.09;
                prior_min_score = candidates[candidates_index[num_candidates - 1]].score;
              }
              min_score = new_min_score;
            } else if (min_score < prior_min_score) {
              new_min_score = min_score + min_score - prior_min_score - 0.15;
              prior_min_score = min_score;
              min_score = new_min_score;
            } else {
              new_min_score = min_score + min_score - prior_min_score - 0.15;
              if (new_min_score < prior_min_score)
                min_score = new_min_score;
              else
                min_score = prior_min_score - 0.03;
            }
            if (min_score < 0.0)
              min_score = 0.0;
          } else if (min_score > 0.0) {
            prior_min_score = min_score;
            min_score = 0.0;
            num_candidates = 1;
          }
        } else {
          scan_mode = 3;
          prior_min_score = min_score;
          min_score = 0.25 * min_score;
          if (min_score < 10.0)
            min_score = 10.0;
        }
      } else if (scan_mode == 3) {
        if (num_candidates == (uint16_t)max_scores) {
          if (min_score < prior_min_score) {
            if (prior_min_score != BIG_FLOAT) {
              if (scan_cycle > 50) {
                if (scan_cycle > 100) {
                  if (max_scores == MAX_SCORES_FAST)
                    new_min_score = 0.995 * min_score * (min_score / prior_min_score) - 0.002;
                  else
                    new_min_score = 0.998 * min_score * (min_score / prior_min_score) - 0.002;
                } else
                  new_min_score = 0.99 * min_score * (min_score / prior_min_score) - 0.002;
              } else
                new_min_score = 0.98 * min_score * (min_score / prior_min_score) - 0.002;
              prior_min_score = min_score;
              min_score = new_min_score;
            } else {
              prior_min_score = min_score;
              min_score *= 0.5;
            }
          } else
            min_score = 0.95 * prior_min_score - 0.002;
        } else if (min_score < prior_min_score) {
          if (prior_min_score != BIG_FLOAT) {
            new_min_score = 0.95 * min_score * (min_score / prior_min_score) - 0.002;
            prior_min_score = min_score;
            min_score = new_min_score;
          } else {
            prior_min_score = min_score;
            min_score *= 0.5;
          }
        } else
          min_score = 0.95 * prior_min_score - 0.002;
        if (min_score > 0.9 * section_scores[fast_section]) {
          if ((0.9 * section_scores[fast_section] < fast_min_score) && (min_score >= fast_min_score))
            min_score = fast_min_score;
          else
            min_score = 0.9 * section_scores[fast_section];
        } else if (min_score < 0.0)
          min_score = 0.0;
      } else
        scan_mode = 3;
    }

    if (fast_mode == 0) {
      uint32_t prior_max_scores = max_scores;
      max_scores = (max_scores
          + 2 * (num_terminals + num_rules - next_new_symbol_number + initial_max_scores)) / 3;
      if (max_scores > MAX_SCORES)
        max_scores = MAX_SCORES;
      if (max_scores > prior_max_scores)
        min_score -= (prior_min_score - min_score) * 25.0 * (double)(max_scores - prior_max_scores) / (double)prior_max_scores;
      if (min_score < 0.0)
        min_score = 0.0;
    } else {
      if ((prior_min_score <= fast_min_score) && (fast_sections == 1))
        break;
      max_scores = (20 * max_scores
          + 35 * (num_terminals + num_rules - next_new_symbol_number + initial_max_scores)) >> 6;
      if (max_scores > MAX_SCORES_FAST)
        max_scores = MAX_SCORES_FAST;
    }
    if (max_scores > 100 * num_candidates)
      max_scores = 100 * num_candidates;
  } while ((num_candidates != 0) && (num_terminals + num_rules < max_rules));

  if (fast_mode != 0) {
    free(score_map);
    free(candidates_position);
  }
  else
    free(x_log2_x);
  free(symbol_counts);
  free(symbol_ends);
  free(rank_scores_data_ptr);
  free(new_symbol_number);
  free(node_data);
  free(candidates_index);
  free(candidate_bad);

  if ((*iobuf = (uint8_t *)malloc(4 * num_file_symbols + 1)) == 0) {
    fprintf(stderr, "ERROR - Compressed output buffer memory allocation failed\n");
    return(0);
  }
  in_char_ptr = *iobuf;
  if (UTF8_compliant != 0) {
    *in_char_ptr++ = 5 | (cap_encoded << 1);
    uint8_t base_bits = 7;
    while ((max_UTF8_value >> base_bits) != 0)
      base_bits++;
    *in_char_ptr++ = base_bits;
  } else if (cap_encoded != 0)
    *in_char_ptr++ = 3;
  else
    *in_char_ptr++ = format;
  in_symbol_ptr = start_symbol_ptr;
  if (UTF8_compliant != 0) {
    while (in_symbol_ptr != end_symbol_ptr) {
      uint32_t symbol_value = *in_symbol_ptr++;
      if (symbol_value < 0x80)
        *in_char_ptr++ = (uint8_t)symbol_value;
      else if (symbol_value < num_terminals) {
        if (symbol_value < 0x800)
          *in_char_ptr++ = 0xC0 + (symbol_value >> 6);
        else if (symbol_value < 0x10000) {
          *in_char_ptr++ = 0xE0 + (symbol_value >> 12);
          *in_char_ptr++ = 0x80 + ((symbol_value >> 6) & 0x3F);
        } else {
          *in_char_ptr++ = 0xF0 + (symbol_value >> 18);
          *in_char_ptr++ = 0x80 + ((symbol_value >> 12) & 0x3F);
          *in_char_ptr++ = 0x80 + ((symbol_value >> 6) & 0x3F);
        }
        *in_char_ptr++ = 0x80 + (symbol_value & 0x3F);
      } else {
        if ((int32_t)symbol_value >= 0) {
          symbol_value -= num_terminals;
          *in_char_ptr++ = INSERT_SYMBOL_CHAR;
        } else {
          symbol_value--;
          *in_char_ptr++ = DEFINE_SYMBOL_CHAR;
        }
        *in_char_ptr++ = (uint8_t)((symbol_value >> 16) & 0xFF);
        *in_char_ptr++ = (uint8_t)((symbol_value >> 8) & 0xFF);
        *in_char_ptr++ = (uint8_t)(symbol_value & 0xFF);
      }
    }
  } else {
    while (in_symbol_ptr != end_symbol_ptr) {
      uint32_t symbol_value = *in_symbol_ptr++;
      if (symbol_value <= DEFINE_SYMBOL_CHAR) {
        *in_char_ptr++ = (uint8_t)symbol_value;
        if (symbol_value >= INSERT_SYMBOL_CHAR)
          *in_char_ptr++ = DEFINE_SYMBOL_CHAR;
      } else {
        if ((int32_t)symbol_value >= 0) {
          symbol_value -= 0x100;
          *in_char_ptr++ = INSERT_SYMBOL_CHAR;
        } else
          *in_char_ptr++ = DEFINE_SYMBOL_CHAR;
        *in_char_ptr++ = (uint8_t)((symbol_value >> 16) & 0xFF);
        *in_char_ptr++ = (uint8_t)((symbol_value >> 8) & 0xFF);
        *in_char_ptr++ = (uint8_t)(symbol_value & 0xFF);
      }
    }
  }

  in_size = in_char_ptr - *iobuf;
  if ((*iobuf = (uint8_t *)realloc(*iobuf, in_size)) == 0) {
    fprintf(stderr, "ERROR - Compressed output buffer memory reallocation failed\n");
    return(0);
  }
  *outsize_ptr = in_size;
  free(start_symbol_ptr);
#ifdef PRINTON
  if (fast_mode != 0)
    fprintf(stderr, "PASS %u: grammar size %u, %u production rules  \n",
        (unsigned int)scan_cycle, (unsigned int)num_file_symbols + 1, (unsigned int)num_rules);
#endif
  return(1);
}

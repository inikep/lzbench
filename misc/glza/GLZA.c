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

// GLZA.c

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "GLZA.h"
#include "GLZAcomp.h"
#include "GLZAdecode.h"


struct param_data params;

void print_usage() {
  fprintf(stderr, "ERROR - Invalid format\n");
  fprintf(stderr, " Use GLZA c|d [-c#] [-d0] [-l0] [-m#] [-o#] [-p#] [-r#] [-t1|2] [-v1|2] [-w0] [-x] [-C0|1] [-D#]\n");
  fprintf(stderr, "   <infile> <outfile>\n");
  fprintf(stderr, " where:\n");
  fprintf(stderr, "   -c#   sets the cost of a new grammar rule in bits\n");
  fprintf(stderr, "   -d0   disables delta transformation\n");
  fprintf(stderr, "   -l0   disables capital letter lock transformation\n");
  fprintf(stderr, "   -m0|1 overrides the program's decision on whether to use MTF queues\n");
  fprintf(stderr, "         -m0 disables MTF, -m1 enables MTF\n");
  fprintf(stderr, "   -o#   sets the dedupication candidate score order model.  0.0 is order 0 based, 1.0 is\n");
  fprintf(stderr, "         order 1 trailing char/leading char based.  Intermediate values are a blend.\n");
  fprintf(stderr, "   -p#   sets the profit power ratio.  0.0 is most compressive, larger values favor\n");
  fprintf(stderr, "         longer strings\n");
  fprintf(stderr, "   -r#   sets memory usage in millions of bytes\n");
  fprintf(stderr, "   -t#   sets the decoder multithreading option. -t1 = 1 thread, -t2 = 2 threads\n");
  fprintf(stderr, "   -v1|2 -v1 causes the dictionary to be printed to stdout, most frequent first\n");
  fprintf(stderr, "         -v2 causes the dictionary to be printed to stdout, in the order of creation\n");
  fprintf(stderr, "   -x    enables extreme compression mode\n");
  fprintf(stderr, "   -w0   disables the initial word deduplication cycle for files that appear to be text\n");
  fprintf(stderr, "   -C0|1 overrides the program's decision on whether to capital transform\n");
  fprintf(stderr, "         -C0 disables, -C1 enables\n");
  fprintf(stderr, "   -D#   sets an upper limit for the number of grammar rules created\n");
  return;
}


int main(int argc, char* argv[])
{
  uint8_t mode, user_set_order;
  uint8_t *inbuf, *outbuf = NULL;
  int32_t arg_num;
  size_t insize, outsize;
  struct timespec start_time, end_time;
  FILE *fd_in, *fd_out;

  clock_gettime(CLOCK_MONOTONIC, &start_time);
  user_set_order = 0;
  params.user_set_profit_ratio_power = 0;
  params.user_set_production_cost = 0;
  params.user_set_RAM_size = 0;
  params.cap_encoded = 0;
  params.cap_lock_disabled = 0;
  params.delta_disabled = 0;
  params.fast_mode = 1;
  params.print_dictionary = 0;
  params.max_rules = 0xA00000;
  params.use_mtf = 2;
  params.create_words = 1;
  params.two_threads = 1;
  params.order = 0.0;
  if (argc < 4) {
    print_usage();
    exit(EXIT_FAILURE);
  }

  mode = *(argv[1]) - 'c';
  if (mode > 1) {
    fprintf(stderr, "ERROR - mode must be c or d\n");
    exit(EXIT_FAILURE);
  }
  arg_num = 2;

  while (*argv[arg_num] ==  '-') {
    if (*(argv[arg_num] + 1) == 'C') {
      if (*(argv[arg_num++] + 2) == '1')
        params.cap_encoded = 1;
      else
        params.cap_encoded = 2;
    }
    else if (*(argv[arg_num] + 1) == 'D') {
      params.max_rules = (uint32_t)atoi(argv[arg_num++] + 2);
      if (params.max_rules > 0xC00000)
        params.max_rules = 0xC00000;
    }
    else if (*(argv[arg_num] + 1) == 'c') {
      params.production_cost = (double)atof(argv[arg_num++] + 2);
      params.user_set_production_cost = 1;
    }
    else if (*(argv[arg_num] + 1) == 'd') {
      if (*(argv[arg_num] + 2) == '0')
        params.delta_disabled = 1;
      arg_num++;
    }
    else if (*(argv[arg_num] + 1) == 'l') {
      if (*(argv[arg_num] + 2) == '0')
        params.cap_lock_disabled = 1;
      arg_num++;
    }
    else if (*(argv[arg_num] + 1) == 'm') {
      if (*(argv[arg_num] + 2) == '0')
        params.use_mtf = 0;
      else if (*(argv[arg_num] + 2) == '1')
        params.use_mtf = 1;
      arg_num++;
    }
    else if (*(argv[arg_num] + 1) == 'o') {
      params.order = (double)atof(argv[arg_num++] + 2);
      user_set_order = 1;
    }
    else if (*(argv[arg_num] + 1) == 'p') {
      params.profit_ratio_power = (double)atof(argv[arg_num++] + 2);
      params.user_set_profit_ratio_power = 1;
    }
    else if (*(argv[arg_num] + 1) == 'r') {
      params.user_set_RAM_size = 1;
      params.RAM_usage = (double)atof(argv[arg_num++] + 2);
      if (params.RAM_usage < 60.0) {
        fprintf(stderr,"ERROR: -r value must be >= 60.0 (MB)\n");
        exit(EXIT_FAILURE);
      }
    }
    else if (*(argv[arg_num] + 1) == 't') {
      if (*(argv[arg_num++] + 2) != '2')
        params.two_threads = 0;
    }
    else if (*(argv[arg_num] + 1) == 'v') {
      if (*(argv[arg_num] + 2) == '1')
        params.print_dictionary = 1;
      else if (*(argv[arg_num] + 2) == '2')
        params.print_dictionary = 2;
      arg_num++;
    }
    else if (*(argv[arg_num] + 1) == 'w') {
      if (*(argv[arg_num] + 2) == '0')
        params.create_words = 0;
      arg_num++;
    }
    else if (*(argv[arg_num] + 1) == 'x') {
      params.fast_mode = 0;
      arg_num++;
    }
    else {
      fprintf(stderr, "ERROR - Invalid format '-%c'.  Only -c<value>, -d<value>, -l0, -m<value>, -o<value>, -p<value>,\n",
          *(argv[arg_num] + 1));
      fprintf(stderr, "    -r<value>, -t<value>, -v<value>, -w<value> -x -C<value> and -D<value> allowed.\n");
      exit(EXIT_FAILURE);
    }
    if (argc < arg_num + 2) {
      print_usage();
      exit(EXIT_FAILURE);
    }
  }

  if (user_set_order != 0)
    params.fast_mode = 0;

  if (argc != arg_num + 2) {
    print_usage();
    exit(EXIT_FAILURE);
  }
  if ((fd_in = fopen(argv[arg_num], "rb")) == NULL) {
    fprintf(stderr, "ERROR - Unable to open input file '%s'\n", argv[arg_num]);
    exit(EXIT_FAILURE);
  }
  fseeko(fd_in, 0, SEEK_END);
  insize = (size_t)ftello(fd_in);
  if (insize > 0xFFFFFFF0) {
    fprintf(stderr, "ERROR - maximum file size is %u bytes\n", 0xFFFFFFF0);
    exit(EXIT_FAILURE);
  }
  rewind(fd_in);

  if ((inbuf = (uint8_t *)malloc(insize)) == 0) {
    fprintf(stderr, "ERROR - Input buffer memory allocation failed\n");
    exit(EXIT_FAILURE);
  }
  if (fread(inbuf, 1, insize, fd_in) != insize) {
    fprintf(stderr, "ERROR - Read infile failed\n");
    exit(EXIT_FAILURE);
  }
  fclose(fd_in);

  fd_out = 0;
  if ((fd_out = fopen(argv[++arg_num], "wb")) == NULL) {
    fprintf(stderr, "ERROR - Unable to open output file '%s'\n", argv[arg_num]);
    exit(EXIT_FAILURE);
  }

  if (mode == 0) {
    if (insize == 0)
      outsize = 0;
    else if (GLZAcomp(insize, inbuf, &outsize, 0, fd_out, &params) == 0)
      exit(EXIT_FAILURE);
    fprintf(stderr, "Compressed %lu bytes -> %lu bytes (%.4f bpB)",
        (long unsigned int)insize, (long unsigned int)outsize, 8.0 * (float)outsize / (float)insize);
  }
  else {
    if (insize == 0)
      outsize = 0;
    else {
      outbuf = GLZAdecode(insize, inbuf, &outsize, outbuf, fd_out, &params);
      free(inbuf);
    }
    fprintf(stderr, "Decompressed %lu bytes -> %lu bytes (%.4f bpB)",
        (long unsigned int)insize, (long unsigned int) outsize, 8.0 * (float)insize / (float)outsize);
  }
  fclose(fd_out);
  clock_gettime(CLOCK_MONOTONIC, &end_time);
  fprintf(stderr," in %.3lf seconds.\n", (float)(end_time.tv_sec * 1000000000L + end_time.tv_nsec
      - (start_time.tv_sec * 1000000000L + start_time.tv_nsec)) * 1e-9);
  return(EXIT_SUCCESS);
}

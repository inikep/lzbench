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

// GLZA.c


#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "GLZA.h"
#include "GLZAcomp.h"
#include "GLZAdecode.h"


void print_usage() {
  fprintf(stderr,"ERROR - Invalid format - Use GLZA c|d [-c#] [-p#] [-r#] <infile> <outfile>\n");
  fprintf(stderr," where -c# sets the grammar production cost in bits\n");
  fprintf(stderr,"       -p# sets the profit power ratio.  0.0 is most compressive, larger\n");
  fprintf(stderr,"           values favor longer strings\n");
  fprintf(stderr,"       -r# sets memory usage in millions of bytes\n");
  return;
}


int main(int argc, char* argv[])
{
  uint8_t mode, user_set_profit_ratio_power, user_set_production_cost, user_set_RAM_size;
  uint8_t *inbuf, *outbuf;
  int32_t arg_num;
  clock_t start_time;
  size_t insize, outsize, startsize;
  double production_cost, profit_ratio_power, RAM_usage;
  FILE *fd_in, *fd_out;


  params.user_set_profit_ratio_power = 0;
  params.user_set_production_cost = 0;
  params.user_set_RAM_size = 0;
  if (argc < 4) {
    print_usage();
    exit(EXIT_FAILURE);
  }

  mode = *(argv[1]) - 'c';
  if (mode > 1) {
    fprintf(stderr,"ERROR - mode must be c or d\n");
    exit(EXIT_FAILURE);
  }
  arg_num = 2;

  while (*argv[arg_num] ==  '-') {
    if (*(argv[arg_num]+1) == 'c') {
      params.production_cost = (double)atof(argv[arg_num++]+2);
      params.user_set_production_cost = 1;
    }
    else if (*(argv[arg_num]+1) == 'p') {
      params.profit_ratio_power = (double)atof(argv[arg_num++]+2);
      params.user_set_profit_ratio_power = 1;
    }
    else if (*(argv[arg_num]+1) == 'r') {
      params.user_set_RAM_size = 1;
      params.RAM_usage = (double)atof(argv[arg_num++]+2);
      if (params.RAM_usage < 60.0) {
        fprintf(stderr,"ERROR: -r value must be >= 60.0 (MB)\n");
        exit(EXIT_FAILURE);
      }
    }
    else {
      fprintf(stderr,"ERROR - Invalid '-' format.  Only -c<value>, -p<value> and -r<value> allowed\n");
      exit(EXIT_FAILURE);
    }
    if (argc < arg_num + 2) {
      print_usage();
      exit(EXIT_FAILURE);
    }
  }

  if (argc != arg_num + 2) {
    print_usage();
    exit(EXIT_FAILURE);
  }
  if ((fd_in = fopen(argv[arg_num],"rb"))==NULL) {
    fprintf(stderr,"ERROR - Unable to open input file '%s'\n",argv[arg_num]);
    exit(EXIT_FAILURE);
  }
  fseek(fd_in, 0, SEEK_END);
  startsize = insize = (size_t)ftell(fd_in);
  rewind(fd_in);

  if ((inbuf = (uint8_t *)malloc(insize)) == 0) {
    fprintf(stderr,"ERROR - Input buffer memory allocation failed\n");
    exit(EXIT_FAILURE);
  }
  if (fread(inbuf, 1, insize, fd_in) != insize) {
    fprintf(stderr,"ERROR - Read infile failed\n");
    exit(EXIT_FAILURE);
  }
  fclose(fd_in);

  fd_out = 0;
  if ((fd_out = fopen(argv[++arg_num],"wb"))==NULL) {
    fprintf(stderr,"ERROR - Unable to open output file '%s'\n",argv[arg_num]);
    exit(EXIT_FAILURE);
  }

  start_time = clock();
  if (mode == 0) {
    if (insize == 0)
      outsize = 0;
    else if (GLZAcomp(insize, inbuf, &outsize, 0, fd_out, &params) == 0)
      exit(EXIT_FAILURE);
    fprintf(stderr,"Compressed %lu bytes -> %lu bytes (%.4f bpB)",
        (long unsigned int)startsize,(long unsigned int)outsize,8.0*(float)outsize/(float)startsize);
  }
  else {
    if (insize == 0)
      outsize = 0;
    else {
      outbuf = GLZAdecode(insize, inbuf, &outsize, outbuf, fd_out);
      free(inbuf);
    }
    fprintf(stderr,"Decompressed %lu bytes -> %lu bytes (%.4f bpB)",
        (long unsigned int)startsize,(long unsigned int)outsize,8.0*(float)startsize/(float)outsize);
  }
  fprintf(stderr," in %.3f seconds.\n",(float)(clock()-start_time)/CLOCKS_PER_SEC);

  fclose(fd_out);
  return(EXIT_SUCCESS);
}
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
//
// Usage:
//   GLZAformat [-c#] [-d#] [-l#] <infilename> <outfilename>, where
//       -c0 disables capital encoding
//       -c1 forces text processing and capital encoding
//       -d0 disables delta encoding
//       -l0 disables capital lock encoding


#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "GLZAcomp.h"
#include "GLZAdecode.h"


void print_usage() {
  fprintf(stderr,"ERROR - Invalid format - Use GLZAformat [-c#] [-d#] [-l#] <infile> <outfile>\n");
  fprintf(stderr," where -c0 disables capital encoding\n");
  fprintf(stderr,"       -c1 forces capital encoding\n");
  fprintf(stderr,"       -d0 disables delta coding\n");
  fprintf(stderr,"       -l0 disables capital lock encoding\n");
  return;
}


int main(int argc, char* argv[])
{
  uint8_t mode;
  uint8_t *inbuf, *outbuf;
  int32_t arg_num;
  clock_t start_time;
  size_t insize, outsize, startsize;
  FILE *fd_in, *fd_out;


  arg_num = 1;
  if (argc != arg_num + 3) {
    fprintf(stderr,"ERROR - Command format is \"GLZA c|d <infile> <outfile>\"\n");
    exit(EXIT_FAILURE);
  }
  mode = *(argv[arg_num++]) - 'c';
  if (mode > 1) {
    fprintf(stderr,"ERROR - mode must be c or d\n");
    exit(EXIT_FAILURE);
  }
  if ((fd_in = fopen(argv[arg_num],"rb"))==NULL) {
    fprintf(stderr,"ERROR - Unable to open input file '%s'\n",argv[arg_num]);
    exit(EXIT_FAILURE);
  }
  fseek(fd_in, 0, SEEK_END);
  startsize = insize = (size_t)ftell(fd_in);
  rewind(fd_in);
  fprintf(stderr,"Reading %lu byte file\n",(long unsigned int)insize);

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
    else if (GLZAcomp(insize, inbuf, &outsize, 0, fd_out) == 0)
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
      if (outbuf == 0) exit(EXIT_FAILURE);
      free(outbuf);
    }
    fprintf(stderr,"Decompressed %lu bytes -> %lu bytes (%.4f bpB)",
        (long unsigned int)startsize,(long unsigned int)outsize,8.0*(float)startsize/(float)outsize);
  }
  fprintf(stderr," in %.3f seconds.\n",(float)(clock()-start_time)/CLOCKS_PER_SEC);

  fclose(fd_out);
  return(EXIT_SUCCESS);
}
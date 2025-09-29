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

// GLZAcomp.c
//


#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "GLZA.h"
#include "GLZAformat.h"
#include "GLZAcompress.h"
#include "GLZAencode.h"


uint8_t GLZAcomp(size_t insize, uint8_t * inbuf, size_t * outsize_ptr, uint8_t *outbuf, FILE * fd,
    struct param_data * params)
{
  uint8_t status;
  uint8_t * temp2buf;
  uint8_t * tempbuf;
  size_t tempsize;

  if (insize == 0) {
    *outsize_ptr = 0;
    return(1);
  }

  if (fd == 0) {
    tempbuf = (uint8_t *)malloc(insize);
    if (tempbuf == 0) return(0);
    memcpy(tempbuf, inbuf, insize);
    status = GLZAformat(insize, (uint8_t *)tempbuf, outsize_ptr, &temp2buf, params);
    free(tempbuf);
  }
  else {
    status = GLZAformat(insize, (uint8_t *)inbuf, outsize_ptr, &temp2buf, params);
    free(inbuf);
  }
  if (status == 0)
    return(0);
  tempsize = *outsize_ptr;
  status = GLZAcompress(tempsize, outsize_ptr, &temp2buf, params);
  if (status == 0)
    return(0);
  tempsize = *outsize_ptr;
  status = GLZAencode(tempsize, temp2buf, outsize_ptr, outbuf, (FILE *)fd, insize, params);
  free(temp2buf);
  return(status);
}

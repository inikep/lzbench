/*
Copyright (c) 2015-2016, Apple Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:  

1.  Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2.  Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer
    in the documentation and/or other materials provided with the distribution.

3.  Neither the name of the copyright holder(s) nor the names of any contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// LZVN decode API

#include "lzvn.h"
#include "lzvn_decode_base.h"

size_t lzvn_decode_scratch_size() { return sizeof(lzvn_decoder_state); }

size_t lzvn_decode_buffer_scratch(uint8_t *__restrict dst_buffer, size_t dst_size,
                           const uint8_t *__restrict src_buffer,
                           size_t src_size, void *__restrict scratch_buffer) {
  lzvn_decoder_state *s = (lzvn_decoder_state *)scratch_buffer;
  memset(s, 0x00, sizeof(*s));

  // Initialize state
  s->src = src_buffer;
  s->src_end = s->src + src_size;
  s->dst = dst_buffer;
  s->dst_begin = dst_buffer;
  s->dst_end = dst_buffer + dst_size;

  // Decode
  lzvn_decode(s);
  return (size_t)(s->dst - dst_buffer); // bytes written
}

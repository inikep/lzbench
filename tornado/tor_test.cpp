/*
(C) 2011, Dell Inc. Written by Przemyslaw Skibinski (inikep@gmail.com)

    LICENSE

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation; either version 3 of
    the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    General Public License for more details at
    Visit <http://www.gnu.org/copyleft/gpl.html>.

*/

#define FREEARC_STANDALONE_TORNADO

#include <stdint.h>
#include "tor_test.h" 
#include "Tornado.cpp"
#include "Common.cpp"


int compress_all_at_once = 0; 

struct Results 
{
	uint32_t inlen, outlen;
	uint32_t inpos;
	uint8_t *inbuf, *outbuf;
};

// Callback function called by compression routine to read/write data.
// Also it's called by the driver to init/shutdown its processing
int ReadWriteCallback (const char *what, void *buf, int size, void *r_)
{
  Results &r = *(Results*)r_;        // Accumulator for compression statistics

//  printf("what=%s size=%d\n", what, size);

  if (strequ(what,"init")) {
  
	  r.inpos = r.outlen = 0;
	  return FREEARC_OK;

  } else if (strequ(what,"read")) {
    if (r.inpos + size > r.inlen)
		size = r.inlen - r.inpos;

	memcpy(buf, r.inbuf+r.inpos, size);
	r.inpos += size;
    return size;

  } else if (strequ(what,"write") || strequ(what,"quasiwrite")) {
    if (strequ(what,"write")) {
		memcpy(r.outbuf+r.outlen, buf, size);
		r.outlen += size;
		return size;
	}

  } else if (strequ(what,"done")) {
    // Print final compression statistics
    return FREEARC_OK;

  } else {
    return FREEARC_ERRCODE_NOT_IMPLEMENTED;
  }
}


// #define CHUNK_SIZE 10240
// 	GetTime(start_ticks); 
// 	ReadWriteCallback("init", NULL, 0, &r);
// 	ArithCoder<EOB_CODE> ari(ReadWriteCallback, &r, CHUNK_SIZE, CHUNK_SIZE, 256);
// 	for (int i=0; i<size; i++)
// 	{
// 		if (i%CHUNK_SIZE == 0)
// 			ari.flush();
// 		ari.encode(inbuf[i]); 
// 	}
// 	outlen = r.outlen;
// 	Print_Time("ArithCoder", &ticksPerSecond, &start_ticks, size, outlen);
// 
// 
// 	GetTime(start_ticks); 
// 	ReadWriteCallback("init", NULL, 0, &r);
// 	HuffmanEncoder<EOB_CODE> huff(ReadWriteCallback, &r, CHUNK_SIZE, CHUNK_SIZE, 256);
// 	for (int i=0; i<size; i++)
// 	{
// 		if (i%CHUNK_SIZE == 0)
// 			huff.flush();
// 		huff.encode(inbuf[i]); 
// 	}
// 	outlen = r.outlen;
// 	Print_Time("HuffmanEncoder", &ticksPerSecond, &start_ticks, size, outlen);
// 
// 
// 	GetTime(start_ticks); 
// 	ReadWriteCallback("init", NULL, 0, &r);
// 	HuffmanEncoderOrder1<256, EOB_CODE> huff1(ReadWriteCallback, &r, CHUNK_SIZE, CHUNK_SIZE, 256);
// 	for (int i=0; i<size; i++)
// 	{
// 		if (i%CHUNK_SIZE == 0)
// 			huff1.flush();
// 		huff1.encode(inbuf[i-1], inbuf[i]); 
// 	}
// 	outlen = r.outlen;
// 	Print_Time("HuffmanEncoderO1", &ticksPerSecond, &start_ticks, size, outlen);



PackMethod second_Tornado_method[] =
//                 tables row  hashsize  matchfinder      buffer parser  hash3 shift update   auxhash fast_bytes
{ {  0, STORING,   false,   0,        0, NON_CACHING_MF,   1*mb,  0     ,   0,    0,  999,       0,    0,  128 }
, {  1, BYTECODER, false,   1,    16*kb, NON_CACHING_MF,   1*mb,  GREEDY,   0,    0,  999,       0,    0,  128 }
, {  2, BYTECODER, false,   1,   128*kb, NON_CACHING_MF,   2*mb,  GREEDY,   0,    0,  999,       0,    0,  128 }
, {  3, BYTECODER, false,   1,   128*kb, NON_CACHING_MF,   8*mb,  GREEDY,   0,    0,  999,       0,    0,  128 }
, {  4, BYTECODER, false,   1,     4*mb, NON_CACHING_MF,   8*mb,  GREEDY,   0,    0,  999,       0,    0,  128 }
, {  5, BITCODER,  false,   1,   128*kb, NON_CACHING_MF,   8*mb,  GREEDY,   0,    0,  999,       0,    0,  128 }
, {  6, BITCODER,  false,   1,     4*mb, NON_CACHING_MF,   8*mb,  GREEDY,   0,    0,  999,       0,    0,  128 }
, {  7, BITCODER,  false,   1,     4*mb, NON_CACHING_MF,  32*mb,  GREEDY,   0,    0,  999,       0,    0,  128 }
};

	
uint32_t tor_compress(uint8_t method, uint8_t* inbuf, uint8_t* outbuf, uint32_t size)
{
	PackMethod m;
	static Results r; 
	r.inbuf = inbuf;
	r.outbuf = outbuf;
	r.inlen = size;

	ReadWriteCallback ("init", NULL, 0, &r);
	if (method >= 20)
		m = second_Tornado_method[method-20];
	else
		m = std_Tornado_method[method];
			
	if (r.inlen >= 0)
		m.buffer = mymin (m.buffer, r.inlen+LOOKAHEAD*2);
	int result = tor_compress (m, ReadWriteCallback, &r, NULL, -1); 
	return r.outlen;
}

uint32_t tor_decompress(uint8_t* inbuf, uint8_t* outbuf, uint32_t size)
{
	static Results r; 
	r.inbuf = inbuf;
	r.outbuf = outbuf;
	r.inlen = size;

	ReadWriteCallback ("init", NULL, 0, &r);
	int result = tor_decompress(ReadWriteCallback, &r, NULL, -1); 
	ReadWriteCallback ("done", NULL, 0, &r); 
	return r.outlen;
}

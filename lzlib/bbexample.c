/* Buffer to buffer example - Test program for the library lzlib
   Copyright (C) 2010-2025 Antonio Diaz Diaz.

   This program is free software: you have unlimited permission
   to copy, distribute, and modify it.

   Usage: bbexample filename

   This program is an example of how buffer-to-buffer
   compression/decompression can be implemented using lzlib.
*/

#define _FILE_OFFSET_BITS 64

#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "lzlib.h"

#ifndef min
  #define min(x,y) ((x) <= (y) ? (x) : (y))
#endif


/* Return the address of a malloc'd buffer containing the file data and
   the file size in '*file_sizep'.
   In case of error, return 0 and do not modify '*file_sizep'.
*/
uint8_t * read_file( const char * const name, long * const file_sizep )
  {
  long buffer_size = 1 << 20, file_size;
  uint8_t * buffer, * tmp;
  FILE * const f = fopen( name, "rb" );
  if( !f )
    { fprintf( stderr, "bbexample: %s: Can't open input file: %s\n",
               name, strerror( errno ) ); return 0; }

  buffer = (uint8_t *)malloc( buffer_size );
  if( !buffer )
    { fputs( "bbexample: read_file: Not enough memory.\n", stderr );
      fclose( f ); return 0; }
  file_size = fread( buffer, 1, buffer_size, f );
  while( file_size >= buffer_size )
    {
    if( buffer_size >= LONG_MAX )
      {
      fprintf( stderr, "bbexample: %s: Input file is too large.\n", name );
      free( buffer ); fclose( f ); return 0;
      }
    buffer_size = (buffer_size <= LONG_MAX / 2) ? 2 * buffer_size : LONG_MAX;
    tmp = (uint8_t *)realloc( buffer, buffer_size );
    if( !tmp )
      { fputs( "bbexample: read_file: Not enough memory.\n", stderr );
        free( buffer ); fclose( f ); return 0; }
    buffer = tmp;
    file_size += fread( buffer + file_size, 1, buffer_size - file_size, f );
    }
  if( ferror( f ) || !feof( f ) )
    {
    fprintf( stderr, "bbexample: %s: Error reading file: %s\n",
             name, strerror( errno ) );
    free( buffer ); fclose( f ); return 0;
    }
  fclose( f );
  *file_sizep = file_size;
  return buffer;
  }


/* Compress 'insize' bytes from 'inbuf'.
   Return the address of a malloc'd buffer containing the compressed data,
   and the size of the data in '*outlenp'.
   In case of error, return 0 and do not modify '*outlenp'.
*/
uint8_t * bbcompressl( const uint8_t * const inbuf, const long insize,
                       const int level, long * const outlenp )
  {
  typedef struct Lzma_options
    {
    int dictionary_size;		/* 4 KiB .. 512 MiB */
    int match_len_limit;		/* 5 .. 273 */
    } Lzma_options;
  /* Mapping from gzip/bzip2 style 0..9 compression levels to the
     corresponding LZMA compression parameters. */
  const Lzma_options option_mapping[] =
    {
    {   65535,  16 },		/* -0 (65535,16 chooses fast encoder) */
    { 1 << 20,   5 },		/* -1 */
    { 3 << 19,   6 },		/* -2 */
    { 1 << 21,   8 },		/* -3 */
    { 3 << 20,  12 },		/* -4 */
    { 1 << 22,  20 },		/* -5 */
    { 1 << 23,  36 },		/* -6 */
    { 1 << 24,  68 },		/* -7 */
    { 3 << 23, 132 },		/* -8 */
    { 1 << 25, 273 } };		/* -9 */
  Lzma_options encoder_options;
  LZ_Encoder * encoder;
  uint8_t * outbuf;
  const long delta_size = insize / 4 + 64;	/* insize may be zero */
  long outsize = delta_size;			/* initial outsize */
  long inpos = 0;
  long outpos = 0;
  bool error = false;

  if( level < 0 || level > 9 ) return 0;
  encoder_options = option_mapping[level];

  if( encoder_options.dictionary_size > insize && level != 0 )
    encoder_options.dictionary_size = insize;		/* saves memory */
  if( encoder_options.dictionary_size < LZ_min_dictionary_size() )
    encoder_options.dictionary_size = LZ_min_dictionary_size();
  encoder = LZ_compress_open( encoder_options.dictionary_size,
                              encoder_options.match_len_limit, INT64_MAX );
  outbuf = (uint8_t *)malloc( outsize );
  if( !encoder || LZ_compress_errno( encoder ) != LZ_ok || !outbuf )
    { free( outbuf ); LZ_compress_close( encoder ); return 0; }

  while( true )
    {
    int ret = LZ_compress_write( encoder, inbuf + inpos,
                                 min( INT_MAX, insize - inpos ) );
    if( ret < 0 ) { error = true; break; }
    inpos += ret;
    if( inpos >= insize ) LZ_compress_finish( encoder );
    ret = LZ_compress_read( encoder, outbuf + outpos,
                            min( INT_MAX, outsize - outpos ) );
    if( ret < 0 ) { error = true; break; }
    outpos += ret;
    if( LZ_compress_finished( encoder ) == 1 ) break;
    if( outpos >= outsize )
      {
      uint8_t * tmp;
      if( outsize > LONG_MAX - delta_size ) { error = true; break; }
      outsize += delta_size;
      tmp = (uint8_t *)realloc( outbuf, outsize );
      if( !tmp ) { error = true; break; }
      outbuf = tmp;
      }
    }

  if( LZ_compress_close( encoder ) < 0 ) error = true;
  if( error ) { free( outbuf ); return 0; }
  *outlenp = outpos;
  return outbuf;
  }


/* Decompress 'insize' bytes from 'inbuf'.
   Return the address of a malloc'd buffer containing the decompressed
   data, and the size of the data in '*outlenp'.
   In case of error, return 0 and do not modify '*outlenp'.
*/
uint8_t * bbdecompressl( const uint8_t * const inbuf, const long insize,
                         long * const outlenp )
  {
  LZ_Decoder * const decoder = LZ_decompress_open();
  const long delta_size = insize;		/* insize must be > zero */
  long outsize = delta_size;			/* initial outsize */
  uint8_t * outbuf = (uint8_t *)malloc( outsize );
  long inpos = 0;
  long outpos = 0;
  bool error = false;
  if( !decoder || LZ_decompress_errno( decoder ) != LZ_ok || !outbuf )
    { free( outbuf ); LZ_decompress_close( decoder ); return 0; }

  while( true )
    {
    int ret = LZ_decompress_write( decoder, inbuf + inpos,
                                   min( INT_MAX, insize - inpos ) );
    if( ret < 0 ) { error = true; break; }
    inpos += ret;
    if( inpos >= insize ) LZ_decompress_finish( decoder );
    ret = LZ_decompress_read( decoder, outbuf + outpos,
                              min( INT_MAX, outsize - outpos ) );
    if( ret < 0 ) { error = true; break; }
    outpos += ret;
    if( LZ_decompress_finished( decoder ) == 1 ) break;
    if( outpos >= outsize )
      {
      uint8_t * tmp;
      if( outsize > LONG_MAX - delta_size ) { error = true; break; }
      outsize += delta_size;
      tmp = (uint8_t *)realloc( outbuf, outsize );
      if( !tmp ) { error = true; break; }
      outbuf = tmp;
      }
    }

  if( LZ_decompress_close( decoder ) < 0 ) error = true;
  if( error ) { free( outbuf ); return 0; }
  *outlenp = outpos;
  return outbuf;
  }


/* Test the whole file at all levels. */
int full_test( const uint8_t * const inbuf, const long insize )
  {
  int level;
  for( level = 0; level <= 9; ++level )
    {
    long midsize = 0, outsize = 0;
    uint8_t * outbuf;
    uint8_t * midbuf = bbcompressl( inbuf, insize, level, &midsize );
    if( !midbuf )
      { fputs( "bbexample: full_test: Not enough memory or compress error.\n",
               stderr ); return 1; }

    outbuf = bbdecompressl( midbuf, midsize, &outsize );
    free( midbuf );
    if( !outbuf )
      { fputs( "bbexample: full_test: Not enough memory or decompress error.\n",
               stderr ); return 1; }

    if( insize != outsize ||
        ( insize > 0 && memcmp( inbuf, outbuf, insize ) != 0 ) )
      { fputs( "bbexample: full_test: Decompressed data differs from original.\n",
               stderr ); free( outbuf ); return 1; }

    free( outbuf );
    }
  return 0;
  }


/* Compress 'insize' bytes from 'inbuf' to 'outbuf'.
   Return the size of the compressed data in '*outlenp'.
   In case of error, or if 'outsize' is too small, return false and do not
   modify '*outlenp'.
*/
bool bbcompress( const uint8_t * const inbuf, const int insize,
                 const int dictionary_size, const int match_len_limit,
                 uint8_t * const outbuf, const int outsize,
                 int * const outlenp )
  {
  int inpos = 0, outpos = 0;
  bool error = false;
  LZ_Encoder * const encoder =
    LZ_compress_open( dictionary_size, match_len_limit, INT64_MAX );
  if( !encoder || LZ_compress_errno( encoder ) != LZ_ok )
    { LZ_compress_close( encoder ); return false; }

  while( true )
    {
    int ret = LZ_compress_write( encoder, inbuf + inpos, insize - inpos );
    if( ret < 0 ) { error = true; break; }
    inpos += ret;
    if( inpos >= insize ) LZ_compress_finish( encoder );
    ret = LZ_compress_read( encoder, outbuf + outpos, outsize - outpos );
    if( ret < 0 ) { error = true; break; }
    outpos += ret;
    if( LZ_compress_finished( encoder ) == 1 ) break;
    if( outpos >= outsize ) { error = true; break; }
    }

  if( LZ_compress_close( encoder ) < 0 ) error = true;
  if( error ) return false;
  *outlenp = outpos;
  return true;
  }


/* Decompress 'insize' bytes from 'inbuf' to 'outbuf'.
   Return the size of the decompressed data in '*outlenp'.
   In case of error, or if 'outsize' is too small, return false and do not
   modify '*outlenp'.
*/
bool bbdecompress( const uint8_t * const inbuf, const int insize,
                   uint8_t * const outbuf, const int outsize,
                   int * const outlenp )
  {
  int inpos = 0, outpos = 0;
  bool error = false;
  LZ_Decoder * const decoder = LZ_decompress_open();
  if( !decoder || LZ_decompress_errno( decoder ) != LZ_ok )
    { LZ_decompress_close( decoder ); return false; }

  while( true )
    {
    int ret = LZ_decompress_write( decoder, inbuf + inpos, insize - inpos );
    if( ret < 0 ) { error = true; break; }
    inpos += ret;
    if( inpos >= insize ) LZ_decompress_finish( decoder );
    ret = LZ_decompress_read( decoder, outbuf + outpos, outsize - outpos );
    if( ret < 0 ) { error = true; break; }
    outpos += ret;
    if( LZ_decompress_finished( decoder ) == 1 ) break;
    if( outpos >= outsize ) { error = true; break; }
    }

  if( LZ_decompress_close( decoder ) < 0 ) error = true;
  if( error ) return false;
  *outlenp = outpos;
  return true;
  }


/* Test at most INT_MAX bytes from the file with buffers of fixed size. */
int fixed_test( const uint8_t * const inbuf, const int insize )
  {
  int dictionary_size = 65535;		/* fast encoder */
  int midsize = min( INT_MAX, ( insize / 8 ) * 9LL + 44 ), outsize = insize;
  uint8_t * midbuf = (uint8_t *)malloc( midsize );
  uint8_t * outbuf = (uint8_t *)malloc( outsize );
  if( !midbuf || !outbuf )
    { fputs( "bbexample: fixed_test: Not enough memory.\n", stderr );
      free( outbuf ); free( midbuf ); return 1; }

  for( ; dictionary_size <= 8 << 20; dictionary_size += 8323073 )
    {
    int midlen, outlen;
    if( !bbcompress( inbuf, insize, dictionary_size, 16, midbuf, midsize, &midlen ) )
      { fputs( "bbexample: fixed_test: Not enough memory or compress error.\n",
               stderr ); free( outbuf ); free( midbuf ); return 1; }

    if( !bbdecompress( midbuf, midlen, outbuf, outsize, &outlen ) )
      { fputs( "bbexample: fixed_test: Not enough memory or decompress error.\n",
               stderr ); free( outbuf ); free( midbuf ); return 1; }

    if( insize != outlen ||
        ( insize > 0 && memcmp( inbuf, outbuf, insize ) != 0 ) )
      { fputs( "bbexample: fixed_test: Decompressed data differs from original.\n",
               stderr ); free( outbuf ); free( midbuf ); return 1; }

    }
  free( outbuf );
  free( midbuf );
  return 0;
  }


int main( const int argc, const char * const argv[] )
  {
  int retval = 0, i;
  int open_failures = 0;
  const bool verbose = argc > 2;

  if( argc < 2 )
    {
    fputs( "Usage: bbexample filename\n", stderr );
    return 1;
    }

  for( i = 1; i < argc && retval == 0; ++i )
    {
    long insize;
    uint8_t * const inbuf = read_file( argv[i], &insize );
    if( !inbuf ) { ++open_failures; continue; }
    if( verbose ) fprintf( stderr, "  Testing file '%s'\n", argv[i] );

    retval = full_test( inbuf, insize );
    if( retval == 0 ) retval = fixed_test( inbuf, min( INT_MAX, insize ) );
    free( inbuf );
    }
  if( open_failures > 0 && verbose )
    fprintf( stderr, "bbexample: warning: %d %s failed to open.\n",
             open_failures, ( open_failures == 1 ) ? "file" : "files" );
  if( retval == 0 && open_failures ) retval = 1;
  return retval;
  }

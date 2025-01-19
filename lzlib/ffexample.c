/* File to file example - Test program for the library lzlib
   Copyright (C) 2010-2025 Antonio Diaz Diaz.

   This program is free software: you have unlimited permission
   to copy, distribute, and modify it.

   Try 'ffexample -h' for usage information.

   This program is an example of how file-to-file
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
#if defined __MSVCRT__ || defined __OS2__ || defined __DJGPP__
#include <fcntl.h>
#include <io.h>
#endif

#include "lzlib.h"

#ifndef min
  #define min(x,y) ((x) <= (y) ? (x) : (y))
#endif


static void show_help( void )
  {
  printf( "ffexample is an example program showing how file-to-file (de)compression can\n"
          "be implemented using lzlib. The content of infile is compressed,\n"
          "decompressed, or both, and then written to outfile.\n"
          "\nUsage: ffexample operation [infile [outfile]]\n" );
  printf( "\nOperation:\n"
          "  -h        display this help and exit\n"
          "  -c        compress infile to outfile\n"
          "  -d        decompress infile to outfile\n"
          "  -b        both (compress then decompress) infile to outfile\n"
          "  -m        compress (multimember) infile to outfile\n"
          "  -l        compress (1 member per line) infile to outfile\n"
          "  -r        decompress with resync if data error or leading garbage\n"
          "\nIf infile or outfile are omitted, or are specified as '-', standard input or\n"
          "standard output are used in their place respectively.\n"
          "\nReport bugs to lzip-bug@nongnu.org\n"
          "Lzlib home page: http://www.nongnu.org/lzip/lzlib.html\n" );
  }


int ffcompress( LZ_Encoder * const encoder,
                FILE * const infile, FILE * const outfile )
  {
  enum { buffer_size = 16384 };
  uint8_t buffer[buffer_size];
  while( true )
    {
    int len, ret;
    int size = min( buffer_size, LZ_compress_write_size( encoder ) );
    if( size > 0 )
      {
      len = fread( buffer, 1, size, infile );
      ret = LZ_compress_write( encoder, buffer, len );
      if( ret < 0 || ferror( infile ) ) break;
      if( feof( infile ) ) LZ_compress_finish( encoder );
      }
    ret = LZ_compress_read( encoder, buffer, buffer_size );
    if( ret < 0 ) break;
    len = fwrite( buffer, 1, ret, outfile );
    if( len < ret ) break;
    if( LZ_compress_finished( encoder ) == 1 ) return 0;
    }
  return 1;
  }


int ffdecompress( LZ_Decoder * const decoder,
                  FILE * const infile, FILE * const outfile )
  {
  enum { buffer_size = 16384 };
  uint8_t buffer[buffer_size];
  while( true )
    {
    int len, ret;
    int size = min( buffer_size, LZ_decompress_write_size( decoder ) );
    if( size > 0 )
      {
      len = fread( buffer, 1, size, infile );
      ret = LZ_decompress_write( decoder, buffer, len );
      if( ret < 0 || ferror( infile ) ) break;
      if( feof( infile ) ) LZ_decompress_finish( decoder );
      }
    ret = LZ_decompress_read( decoder, buffer, buffer_size );
    if( ret < 0 ) break;
    len = fwrite( buffer, 1, ret, outfile );
    if( len < ret ) break;
    if( LZ_decompress_finished( decoder ) == 1 ) return 0;
    }
  return 1;
  }


int ffboth( LZ_Encoder * const encoder, LZ_Decoder * const decoder,
            FILE * const infile, FILE * const outfile )
  {
  enum { buffer_size = 16384 };
  uint8_t buffer[buffer_size];
  while( true )
    {
    int len, ret;
    int size = min( buffer_size, LZ_compress_write_size( encoder ) );
    if( size > 0 )
      {
      len = fread( buffer, 1, size, infile );
      ret = LZ_compress_write( encoder, buffer, len );
      if( ret < 0 || ferror( infile ) ) break;
      if( feof( infile ) ) LZ_compress_finish( encoder );
      }
    size = min( buffer_size, LZ_decompress_write_size( decoder ) );
    if( size > 0 )
      {
      ret = LZ_compress_read( encoder, buffer, size );
      if( ret < 0 ) break;
      ret = LZ_decompress_write( decoder, buffer, ret );
      if( ret < 0 ) break;
      if( LZ_compress_finished( encoder ) == 1 )
        LZ_decompress_finish( decoder );
      }
    ret = LZ_decompress_read( decoder, buffer, buffer_size );
    if( ret < 0 ) break;
    len = fwrite( buffer, 1, ret, outfile );
    if( len < ret ) break;
    if( LZ_decompress_finished( decoder ) == 1 ) return 0;
    }
  return 1;
  }


int ffmmcompress( FILE * const infile, FILE * const outfile )
  {
  enum { buffer_size = 16384, member_size = 4096 };
  uint8_t buffer[buffer_size];
  bool done = false;
  LZ_Encoder * const encoder = LZ_compress_open( 65535, 16, member_size );
  if( !encoder || LZ_compress_errno( encoder ) != LZ_ok  )
    { fputs( "ffexample: Not enough memory.\n", stderr );
      LZ_compress_close( encoder ); return 1; }
  while( true )
    {
    int len, ret;
    int size = min( buffer_size, LZ_compress_write_size( encoder ) );
    if( size > 0 )
      {
      len = fread( buffer, 1, size, infile );
      ret = LZ_compress_write( encoder, buffer, len );
      if( ret < 0 || ferror( infile ) ) break;
      if( feof( infile ) ) LZ_compress_finish( encoder );
      }
    ret = LZ_compress_read( encoder, buffer, buffer_size );
    if( ret < 0 ) break;
    len = fwrite( buffer, 1, ret, outfile );
    if( len < ret ) break;
    if( LZ_compress_member_finished( encoder ) == 1 )
      {
      if( LZ_compress_finished( encoder ) == 1 ) { done = true; break; }
      if( LZ_compress_restart_member( encoder, member_size ) < 0 ) break;
      }
    }
  if( LZ_compress_close( encoder ) < 0 ) done = false;
  return done;
  }


/* Compress 'infile' to 'outfile' as a multimember stream with one member
   for each line of text terminated by a newline character or by EOF.
   Return 0 if success, 1 if error.
*/
int fflfcompress( LZ_Encoder * const encoder,
                  FILE * const infile, FILE * const outfile )
  {
  enum { buffer_size = 16384 };
  uint8_t buffer[buffer_size];
  while( true )
    {
    int len, ret;
    int size = min( buffer_size, LZ_compress_write_size( encoder ) );
    if( size > 0 )
      {
      for( len = 0; len < size; )
        {
        int ch = getc( infile );
        if( ch == EOF || ( buffer[len++] = ch ) == '\n' ) break;
        }
      /* avoid writing an empty member to outfile */
      if( len == 0 && LZ_compress_data_position( encoder ) == 0 ) return 0;
      ret = LZ_compress_write( encoder, buffer, len );
      if( ret < 0 || ferror( infile ) ) break;
      if( feof( infile ) || buffer[len-1] == '\n' )
        LZ_compress_finish( encoder );
      }
    ret = LZ_compress_read( encoder, buffer, buffer_size );
    if( ret < 0 ) break;
    len = fwrite( buffer, 1, ret, outfile );
    if( len < ret ) break;
    if( LZ_compress_member_finished( encoder ) == 1 )
      {
      if( feof( infile ) && LZ_compress_finished( encoder ) == 1 ) return 0;
      if( LZ_compress_restart_member( encoder, INT64_MAX ) < 0 ) break;
      }
    }
  return 1;
  }


/* Decompress 'infile' to 'outfile' with automatic resynchronization to
   next member in case of data error, including the automatic removal of
   leading garbage.
*/
int ffrsdecompress( LZ_Decoder * const decoder,
                    FILE * const infile, FILE * const outfile )
  {
  enum { buffer_size = 16384 };
  uint8_t buffer[buffer_size];
  while( true )
    {
    int len, ret;
    int size = min( buffer_size, LZ_decompress_write_size( decoder ) );
    if( size > 0 )
      {
      len = fread( buffer, 1, size, infile );
      ret = LZ_decompress_write( decoder, buffer, len );
      if( ret < 0 || ferror( infile ) ) break;
      if( feof( infile ) ) LZ_decompress_finish( decoder );
      }
    ret = LZ_decompress_read( decoder, buffer, buffer_size );
    if( ret < 0 )
      {
      if( LZ_decompress_errno( decoder ) == LZ_header_error ||
          LZ_decompress_errno( decoder ) == LZ_data_error )
        { LZ_decompress_sync_to_member( decoder ); continue; }
      break;
      }
    len = fwrite( buffer, 1, ret, outfile );
    if( len < ret ) break;
    if( LZ_decompress_finished( decoder ) == 1 ) return 0;
    }
  return 1;
  }


int main( const int argc, const char * const argv[] )
  {
#if defined __MSVCRT__ || defined __OS2__ || defined __DJGPP__
  setmode( STDIN_FILENO, O_BINARY );
  setmode( STDOUT_FILENO, O_BINARY );
#endif

  LZ_Encoder * const encoder = LZ_compress_open( 65535, 16, INT64_MAX );
  LZ_Decoder * const decoder = LZ_decompress_open();
  FILE * const infile = (argc >= 3 && strcmp( argv[2], "-" ) != 0) ?
                        fopen( argv[2], "rb" ) : stdin;
  FILE * const outfile = (argc >= 4 && strcmp( argv[3], "-" ) != 0) ?
                         fopen( argv[3], "wb" ) : stdout;
  int retval;

  if( argc < 2 || argc > 4 || strlen( argv[1] ) != 2 || argv[1][0] != '-' )
    { show_help(); return 1; }
  if( !encoder || LZ_compress_errno( encoder ) != LZ_ok ||
      !decoder || LZ_decompress_errno( decoder ) != LZ_ok )
    { fputs( "ffexample: Not enough memory.\n", stderr );
      LZ_compress_close( encoder ); LZ_decompress_close( decoder ); return 1; }
  if( !infile )
    { fprintf( stderr, "ffexample: %s: Can't open input file: %s\n",
               argv[2], strerror( errno ) ); return 1; }
  if( !outfile )
    { fprintf( stderr, "ffexample: %s: Can't open output file: %s\n",
               argv[3], strerror( errno ) ); return 1; }

  switch( argv[1][1] )
    {
    case 'c': retval = ffcompress( encoder, infile, outfile ); break;
    case 'd': retval = ffdecompress( decoder, infile, outfile ); break;
    case 'b': retval = ffboth( encoder, decoder, infile, outfile ); break;
    case 'm': retval = ffmmcompress( infile, outfile ); break;
    case 'l': retval = fflfcompress( encoder, infile, outfile ); break;
    case 'r': retval = ffrsdecompress( decoder, infile, outfile ); break;
    default: show_help(); return argv[1][1] != 'h';
    }

  if( LZ_decompress_close( decoder ) < 0 || LZ_compress_close( encoder ) < 0 ||
      fclose( outfile ) != 0 || fclose( infile ) != 0 ) retval = 1;
  return retval;
  }

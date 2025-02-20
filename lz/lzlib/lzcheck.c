/* Lzcheck - Test program for the library lzlib
   Copyright (C) 2009-2025 Antonio Diaz Diaz.

   This program is free software: you have unlimited permission
   to copy, distribute, and modify it.

   Usage: lzcheck [-m|-s] filename.txt...

   This program reads each text file specified and then compresses it,
   line by line, to test the flushing mechanism and the member
   restart/reset/sync functions.
*/

#define _FILE_OFFSET_BITS 64

#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

#include "lzlib.h"


const unsigned long long member_size = INT64_MAX;
enum { buffer_size = 32749 };			/* largest prime < 32768 */
uint8_t in_buffer[buffer_size];
uint8_t mid_buffer[buffer_size];
uint8_t out_buffer[buffer_size];


static void show_line( const uint8_t * const buffer, const int size )
  {
  int i;
  for( i = 0; i < size; ++i )
    fputc( isprint( buffer[i] ) ? buffer[i] : '.', stderr );
  fputc( '\n', stderr );
  }


static LZ_Encoder * xopen_encoder( const int dictionary_size )
  {
  const int match_len_limit = 16;
  LZ_Encoder * const encoder =
    LZ_compress_open( dictionary_size, match_len_limit, member_size );
  if( !encoder || LZ_compress_errno( encoder ) != LZ_ok )
    {
    const bool bad_arg =
      encoder && ( LZ_compress_errno( encoder ) == LZ_bad_argument );
    LZ_compress_close( encoder );
    if( bad_arg )
      {
      fputs( "lzcheck: internal error: Invalid argument to encoder.\n", stderr );
      exit( 3 );
      }
    fputs( "lzcheck: Not enough memory.\n", stderr );
    exit( 1 );
    }
  return encoder;
  }


static LZ_Decoder * xopen_decoder( void )
  {
  LZ_Decoder * const decoder = LZ_decompress_open();
  if( !decoder || LZ_decompress_errno( decoder ) != LZ_ok )
    {
    LZ_decompress_close( decoder );
    fputs( "lzcheck: Not enough memory.\n", stderr );
    exit( 1 );
    }
  return decoder;
  }


static void xclose_encoder( LZ_Encoder * const encoder, const bool finish )
  {
  if( finish )
    {
    unsigned long long size = 0;
    LZ_compress_finish( encoder );
    while( true )
      {
      const int rd = LZ_compress_read( encoder, mid_buffer, buffer_size );
      if( rd < 0 )
        {
        fprintf( stderr, "lzcheck: xclose: LZ_compress_read error: %s\n",
                 LZ_strerror( LZ_compress_errno( encoder ) ) );
        exit( 3 );
        }
      size += rd;
      if( LZ_compress_finished( encoder ) == 1 ) break;
      }
    if( size > 0 )
      {
      fprintf( stderr, "lzcheck: %lld bytes remain in encoder.\n", size );
      exit( 3 );
      }
    }
  if( LZ_compress_close( encoder ) < 0 ) exit( 1 );
  }


static void xclose_decoder( LZ_Decoder * const decoder, const bool finish )
  {
  if( finish )
    {
    unsigned long long size = 0;
    LZ_decompress_finish( decoder );
    while( true )
      {
      const int rd = LZ_decompress_read( decoder, out_buffer, buffer_size );
      if( rd < 0 )
        {
        fprintf( stderr, "lzcheck: xclose: LZ_decompress_read error: %s\n",
                 LZ_strerror( LZ_decompress_errno( decoder ) ) );
        exit( 3 );
        }
      size += rd;
      if( LZ_decompress_finished( decoder ) == 1 ) break;
      }
    if( size > 0 )
      {
      fprintf( stderr, "lzcheck: %lld bytes remain in decoder.\n", size );
      exit( 3 );
      }
    }
  if( LZ_decompress_close( decoder ) < 0 ) exit( 1 );
  }


/* Return the next (usually newline-terminated) chunk of data from file.
   The size returned in *sizep is always <= buffer_size.
   If sizep is a null pointer, rewind the file, reset state, and return.
   If file is at EOF, return an empty line.
*/
static const uint8_t * next_line( FILE * const file, int * const sizep )
  {
  static int l = 0;
  static int read_size = 0;
  int r;

  if( !sizep ) { rewind( file ); l = read_size = 0; return in_buffer; }
  if( l >= read_size )
    {
    l = 0; read_size = fread( in_buffer, 1, buffer_size, file );
    if( l >= read_size ) { *sizep = 0; return in_buffer; }  /* end of file */
    }

  for( r = l + 1; r < read_size && in_buffer[r-1] != '\n'; ++r );
  *sizep = r - l; l = r;
  return in_buffer + l - *sizep;
  }


static int check_sync_flush( FILE * const file, const int dictionary_size )
  {
  LZ_Encoder * const encoder = xopen_encoder( dictionary_size );
  LZ_Decoder * const decoder = xopen_decoder();
  int retval = 0;

  while( retval <= 1 )			/* test LZ_compress_sync_flush */
    {
    int in_size, mid_size, out_size;
    int line_size;
    const uint8_t * const line_buf = next_line( file, &line_size );
    if( line_size <= 0 ) break;			/* end of file */

    in_size = LZ_compress_write( encoder, line_buf, line_size );
    if( in_size < 0 )
      {
      fprintf( stderr, "lzcheck: LZ_compress_write error: %s\n",
               LZ_strerror( LZ_compress_errno( encoder ) ) );
      retval = 3; break;
      }
    if( in_size < line_size )
      {
      fprintf( stderr, "lzcheck: sync: LZ_compress_write only accepted %d "
               "of %d bytes\n", in_size, line_size );
      mid_size = LZ_compress_read( encoder, mid_buffer, buffer_size );
      const int wr =
        LZ_compress_write( encoder, line_buf + in_size, line_size - in_size );
      if( wr < 0 )
        {
        fprintf( stderr, "lzcheck: LZ_compress_write error: %s\n",
                 LZ_strerror( LZ_compress_errno( encoder ) ) );
        retval = 3; break;
        }
      if( wr + in_size != line_size )
        {
        fprintf( stderr, "lzcheck: sync: LZ_compress_write only accepted %d "
                 "of %d remaining bytes\n", wr, line_size - in_size );
        retval = 3; break;
        }
      in_size += wr;
      LZ_compress_sync_flush( encoder );
      const int rd = LZ_compress_read( encoder, mid_buffer + mid_size,
                                       buffer_size - mid_size );
      if( rd > 0 ) mid_size += rd;
      else if( rd < 0 ) mid_size = -1;
      }
    else
      {
      LZ_compress_sync_flush( encoder );
      if( line_buf[0] & 1 )	/* read all data at once or byte by byte */
        mid_size = LZ_compress_read( encoder, mid_buffer, buffer_size );
      else for( mid_size = 0; mid_size < buffer_size; )
        {
        const int rd = LZ_compress_read( encoder, mid_buffer + mid_size, 1 );
        if( rd > 0 ) mid_size += rd;
        else { if( rd < 0 ) { mid_size = -1; } break; }
        }
      }
    if( mid_size < 0 )
      {
      fprintf( stderr, "lzcheck: LZ_compress_read error: %s\n",
               LZ_strerror( LZ_compress_errno( encoder ) ) );
      retval = 3; break;
      }
    LZ_decompress_write( decoder, mid_buffer, mid_size );
    out_size = LZ_decompress_read( decoder, out_buffer, buffer_size );
    if( out_size < 0 )
      {
      fprintf( stderr, "lzcheck: LZ_decompress_read error: %s\n",
               LZ_strerror( LZ_decompress_errno( decoder ) ) );
      retval = 3; break;
      }

    if( out_size != in_size || memcmp( line_buf, out_buffer, out_size ) )
      {
      fprintf( stderr, "lzcheck: LZ_compress_sync_flush error: "
                       "in_size = %d, out_size = %d\n", in_size, out_size );
      show_line( line_buf, in_size );
      show_line( out_buffer, out_size );
      retval = 1;
      }
    }

  if( retval <= 1 )
    {
    int rd = 0;
    if( LZ_compress_finish( encoder ) < 0 ||
        ( rd = LZ_compress_read( encoder, mid_buffer, buffer_size ) ) < 0 )
      {
      fprintf( stderr, "lzcheck: Can't drain encoder: %s\n",
               LZ_strerror( LZ_compress_errno( encoder ) ) );
      retval = 3;
      }
    LZ_decompress_write( decoder, mid_buffer, rd );
    }

  xclose_decoder( decoder, retval == 0 );
  xclose_encoder( encoder, retval == 0 );
  return retval;
  }


/* Test member by member decompression without calling LZ_decompress_finish,
   inserting leading garbage before some members, and resetting the
   decompressor sometimes. Test that the increase in total_in_size when
   syncing to member is equal to the size of the leading garbage skipped.
*/
static int check_members( FILE * const file, const int dictionary_size )
  {
  LZ_Encoder * const encoder = xopen_encoder( dictionary_size );
  LZ_Decoder * const decoder = xopen_decoder();
  int retval = 0;

  while( retval <= 1 )			/* test LZ_compress_restart_member */
    {
    unsigned long long garbage_begin = 0;  /* avoid warning from gcc 3.3.6 */
    int leading_garbage, in_size, mid_size, out_size;
    int line_size;
    const uint8_t * const line_buf = next_line( file, &line_size );
    if( line_size <= 0 &&	/* end of file, write at least 1 member */
        LZ_decompress_total_in_size( decoder ) != 0 ) break;

    if( LZ_compress_finished( encoder ) == 1 )
      {
      if( LZ_compress_restart_member( encoder, member_size ) < 0 )
        {
        fprintf( stderr, "lzcheck: Can't restart member: %s\n",
                 LZ_strerror( LZ_compress_errno( encoder ) ) );
        retval = 3; break;
        }
      if( line_size >= 2 && line_buf[1] == 'h' )
        LZ_decompress_reset( decoder );
      }
    in_size = LZ_compress_write( encoder, line_buf, line_size );
    if( in_size < line_size )
      fprintf( stderr, "lzcheck: member: LZ_compress_write only accepted %d of %d bytes\n",
               in_size, line_size );
    LZ_compress_finish( encoder );
    if( line_size * 3 < buffer_size && line_buf[0] == 't' )
      { leading_garbage = line_size;
        memset( mid_buffer, in_buffer[0], leading_garbage );
        garbage_begin = LZ_decompress_total_in_size( decoder ); }
    else leading_garbage = 0;
    mid_size = LZ_compress_read( encoder, mid_buffer + leading_garbage,
                                 buffer_size - leading_garbage );
    if( mid_size < 0 )
      {
      fprintf( stderr, "lzcheck: member: LZ_compress_read error: %s\n",
               LZ_strerror( LZ_compress_errno( encoder ) ) );
      retval = 3; break;
      }
    LZ_decompress_write( decoder, mid_buffer, leading_garbage + mid_size );
    out_size = LZ_decompress_read( decoder, out_buffer, buffer_size );
    if( out_size < 0 )
      {
      if( leading_garbage &&
          ( LZ_decompress_errno( decoder ) == LZ_header_error ||
            LZ_decompress_errno( decoder ) == LZ_data_error ) )
        {
        LZ_decompress_sync_to_member( decoder );  /* skip leading garbage */
        const unsigned long long garbage_end =
          LZ_decompress_total_in_size( decoder );
        if( garbage_end - garbage_begin != (unsigned)leading_garbage )
          {
          fprintf( stderr, "lzcheck: member: LZ_decompress_sync_to_member error:\n"
                   "  garbage_begin = %llu garbage_end = %llu "
                   "difference = %llu expected = %d\n", garbage_begin,
                   garbage_end, garbage_end - garbage_begin, leading_garbage );
          retval = 3; break;
          }
        out_size = LZ_decompress_read( decoder, out_buffer, buffer_size );
        }
      if( out_size < 0 )
        {
        fprintf( stderr, "lzcheck: member: LZ_decompress_read error: %s\n",
                 LZ_strerror( LZ_decompress_errno( decoder ) ) );
        retval = 3; break;
        }
      }

    if( out_size != in_size || memcmp( line_buf, out_buffer, out_size ) )
      {
      fprintf( stderr, "lzcheck: LZ_compress_restart_member error: "
                       "in_size = %d, out_size = %d\n", in_size, out_size );
      show_line( line_buf, in_size );
      show_line( out_buffer, out_size );
      retval = 1;
      }
    }

  xclose_decoder( decoder, retval == 0 );
  xclose_encoder( encoder, retval == 0 );
  return retval;
  }


int main( const int argc, const char * const argv[] )
  {
  int retval = 0, i;
  int open_failures = 0;
  const char opt = ( argc > 2 &&
    ( strcmp( argv[1], "-m" ) == 0 || strcmp( argv[1], "-s" ) == 0 ) ) ?
    argv[1][1] : 0;
  const int first = opt ? 2 : 1;
  const bool verbose = opt != 0 || argc > first + 1;

  if( argc < 2 )
    {
    fputs( "Usage: lzcheck [-m|-s] filename.txt...\n", stderr );
    return 1;
    }

  for( i = first; i < argc && retval == 0; ++i )
    {
    struct stat st;
    if( stat( argv[i], &st ) != 0 || !S_ISREG( st.st_mode ) ) continue;
    FILE * file = fopen( argv[i], "rb" );
    if( !file )
      {
      fprintf( stderr, "lzcheck: %s: Can't open file for reading.\n", argv[i] );
      ++open_failures; continue;
      }
    if( verbose ) fprintf( stderr, "  Testing file '%s'\n", argv[i] );

    /* 65535,16 chooses fast encoder */
    if( opt != 'm' ) retval = check_sync_flush( file, 65535 );
    if( retval == 0 && opt != 'm' )
      { next_line( file, 0 ); retval = check_sync_flush( file, 1 << 20 ); }
    if( retval == 0 && opt != 's' )
      { next_line( file, 0 ); retval = check_members( file, 65535 ); }
    if( retval == 0 && opt != 's' )
      { next_line( file, 0 ); retval = check_members( file, 1 << 20 ); }
    fclose( file );
    }
  if( open_failures > 0 && verbose )
    fprintf( stderr, "lzcheck: warning: %d %s failed to open.\n",
             open_failures, ( open_failures == 1 ) ? "file" : "files" );
  if( retval == 0 && open_failures ) retval = 1;
  return retval;
  }

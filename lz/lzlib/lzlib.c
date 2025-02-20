/* Lzlib - Compression library for the lzip format
   Copyright (C) 2009-2025 Antonio Diaz Diaz.

   This library is free software. Redistribution and use in source and
   binary forms, with or without modification, are permitted provided
   that the following conditions are met:

   1. Redistributions of source code must retain the above copyright
   notice, this list of conditions, and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions, and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "lzlib.h"
#include "lzip.h"
#include "cbuffer.c"
#include "decoder.h"
#include "decoder.c"
#include "encoder_base.h"
#include "encoder_base.c"
#include "encoder.h"
#include "encoder.c"
#include "fast_encoder.h"
#include "fast_encoder.c"


struct LZ_Encoder
  {
  unsigned long long partial_in_size;
  unsigned long long partial_out_size;
  LZ_encoder_base * lz_encoder_base;		/* these 3 pointers make a */
  LZ_encoder * lz_encoder;			/* polymorphic encoder */
  FLZ_encoder * flz_encoder;
  LZ_Errno lz_errno;
  bool fatal;
  };

static void LZ_Encoder_init( LZ_Encoder * const e )
  {
  e->partial_in_size = 0;
  e->partial_out_size = 0;
  e->lz_encoder_base = 0;
  e->lz_encoder = 0;
  e->flz_encoder = 0;
  e->lz_errno = LZ_ok;
  e->fatal = false;
  }


struct LZ_Decoder
  {
  unsigned long long partial_in_size;
  unsigned long long partial_out_size;
  Range_decoder * rdec;
  LZ_decoder * lz_decoder;
  LZ_Errno lz_errno;
  Lzip_header member_header;		/* header of current member */
  bool fatal;
  bool first_header;			/* true until first header is read */
  bool seeking;
  };

static void LZ_Decoder_init( LZ_Decoder * const d )
  {
  int i;
  d->partial_in_size = 0;
  d->partial_out_size = 0;
  d->rdec = 0;
  d->lz_decoder = 0;
  d->lz_errno = LZ_ok;
  for( i = 0; i < Lh_size; ++i ) d->member_header[i] = 0;
  d->fatal = false;
  d->first_header = true;
  d->seeking = false;
  }


static bool check_encoder( LZ_Encoder * const e )
  {
  if( !e ) return false;
  if( !e->lz_encoder_base || ( !e->lz_encoder && !e->flz_encoder ) ||
      ( e->lz_encoder && e->flz_encoder ) )
    { e->lz_errno = LZ_bad_argument; return false; }
  return true;
  }


static bool check_decoder( LZ_Decoder * const d )
  {
  if( !d ) return false;
  if( !d->rdec )
    { d->lz_errno = LZ_bad_argument; return false; }
  return true;
  }


/* ------------------------- Misc Functions ------------------------- */

int LZ_api_version( void ) { return LZ_API_VERSION; }

const char * LZ_version( void ) { return LZ_version_string; }

const char * LZ_strerror( const LZ_Errno lz_errno )
  {
  switch( lz_errno )
    {
    case LZ_ok            : return "ok";
    case LZ_bad_argument  : return "Bad argument";
    case LZ_mem_error     : return "Not enough memory";
    case LZ_sequence_error: return "Sequence error";
    case LZ_header_error  : return "Header error";
    case LZ_unexpected_eof: return "Unexpected EOF";
    case LZ_data_error    : return "Data error";
    case LZ_library_error : return "Library error";
    }
  return "Invalid error code";
  }


int LZ_min_dictionary_bits( void ) { return min_dictionary_bits; }
int LZ_min_dictionary_size( void ) { return min_dictionary_size; }
int LZ_max_dictionary_bits( void ) { return max_dictionary_bits; }
int LZ_max_dictionary_size( void ) { return max_dictionary_size; }
int LZ_min_match_len_limit( void ) { return min_match_len_limit; }
int LZ_max_match_len_limit( void ) { return max_match_len; }


/* --------------------- Compression Functions --------------------- */

LZ_Encoder * LZ_compress_open( const int dictionary_size,
                               const int match_len_limit,
                               const unsigned long long member_size )
  {
  Lzip_header header;
  LZ_Encoder * const e = (LZ_Encoder *)malloc( sizeof (LZ_Encoder) );
  if( !e ) return 0;
  LZ_Encoder_init( e );
  if( !Lh_set_dictionary_size( header, dictionary_size ) ||
      match_len_limit < min_match_len_limit ||
      match_len_limit > max_match_len ||
      member_size < min_dictionary_size )
    e->lz_errno = LZ_bad_argument;
  else
    {
    if( dictionary_size == 65535 && match_len_limit == 16 )
      {
      e->flz_encoder = (FLZ_encoder *)malloc( sizeof (FLZ_encoder) );
      if( e->flz_encoder && FLZe_init( e->flz_encoder, member_size ) )
        { e->lz_encoder_base = &e->flz_encoder->eb; return e; }
      free( e->flz_encoder ); e->flz_encoder = 0;
      }
    else
      {
      e->lz_encoder = (LZ_encoder *)malloc( sizeof (LZ_encoder) );
      if( e->lz_encoder && LZe_init( e->lz_encoder, Lh_get_dictionary_size( header ),
                                     match_len_limit, member_size ) )
        { e->lz_encoder_base = &e->lz_encoder->eb; return e; }
      free( e->lz_encoder ); e->lz_encoder = 0;
      }
    e->lz_errno = LZ_mem_error;
    }
  e->fatal = true;
  return e;
  }


int LZ_compress_close( LZ_Encoder * const e )
  {
  if( !e ) return -1;
  if( e->lz_encoder_base )
    { LZeb_free( e->lz_encoder_base );
      free( e->lz_encoder ); free( e->flz_encoder ); }
  free( e );
  return 0;
  }


int LZ_compress_finish( LZ_Encoder * const e )
  {
  if( !check_encoder( e ) || e->fatal ) return -1;
  Mb_finish( &e->lz_encoder_base->mb );
  /* if (open --> write --> finish) use same dictionary size as lzip. */
  /* this does not save any memory. */
  if( Mb_data_position( &e->lz_encoder_base->mb ) == 0 &&
      Re_member_position( &e->lz_encoder_base->renc ) == Lh_size )
    {
    Mb_adjust_dictionary_size( &e->lz_encoder_base->mb );
    Lh_set_dictionary_size( e->lz_encoder_base->renc.header,
                            e->lz_encoder_base->mb.dictionary_size );
    e->lz_encoder_base->renc.cb.buffer[5] = e->lz_encoder_base->renc.header[5];
    }
  return 0;
  }


int LZ_compress_restart_member( LZ_Encoder * const e,
                                const unsigned long long member_size )
  {
  if( !check_encoder( e ) || e->fatal ) return -1;
  if( !LZeb_member_finished( e->lz_encoder_base ) )
    { e->lz_errno = LZ_sequence_error; return -1; }
  if( member_size < min_dictionary_size )
    { e->lz_errno = LZ_bad_argument; return -1; }

  e->partial_in_size += Mb_data_position( &e->lz_encoder_base->mb );
  e->partial_out_size += Re_member_position( &e->lz_encoder_base->renc );

  if( e->lz_encoder ) LZe_reset( e->lz_encoder, member_size );
  else FLZe_reset( e->flz_encoder, member_size );
  e->lz_errno = LZ_ok;
  return 0;
  }


int LZ_compress_sync_flush( LZ_Encoder * const e )
  {
  if( !check_encoder( e ) || e->fatal ) return -1;
  if( !e->lz_encoder_base->mb.at_stream_end )
    e->lz_encoder_base->mb.sync_flush_pending = true;
  return 0;
  }


int LZ_compress_read( LZ_Encoder * const e,
                      uint8_t * const buffer, const int size )
  {
  if( !check_encoder( e ) || e->fatal ) return -1;
  if( size < 0 ) return 0;

  { LZ_encoder_base * const eb = e->lz_encoder_base;
  int out_size = Re_read_data( &eb->renc, buffer, size );
  /* minimize number of calls to encode_member */
  if( out_size < size || size == 0 )
    {
    if( ( e->flz_encoder && !FLZe_encode_member( e->flz_encoder ) ) ||
        ( e->lz_encoder && !LZe_encode_member( e->lz_encoder ) ) )
      { e->lz_errno = LZ_library_error; e->fatal = true; return -1; }
    if( eb->mb.sync_flush_pending && Mb_available_bytes( &eb->mb ) <= 0 )
      LZeb_try_sync_flush( eb );
    out_size += Re_read_data( &eb->renc, buffer + out_size, size - out_size );
    }
  return out_size; }
  }


int LZ_compress_write( LZ_Encoder * const e,
                       const uint8_t * const buffer, const int size )
  {
  if( !check_encoder( e ) || e->fatal ) return -1;
  return Mb_write_data( &e->lz_encoder_base->mb, buffer, size );
  }


int LZ_compress_write_size( LZ_Encoder * const e )
  {
  if( !check_encoder( e ) || e->fatal ) return -1;
  return Mb_free_bytes( &e->lz_encoder_base->mb );
  }


LZ_Errno LZ_compress_errno( LZ_Encoder * const e )
  {
  if( !e ) return LZ_bad_argument;
  return e->lz_errno;
  }


int LZ_compress_finished( LZ_Encoder * const e )
  {
  if( !check_encoder( e ) ) return -1;
  return Mb_data_finished( &e->lz_encoder_base->mb ) &&
         LZeb_member_finished( e->lz_encoder_base );
  }


int LZ_compress_member_finished( LZ_Encoder * const e )
  {
  if( !check_encoder( e ) ) return -1;
  return LZeb_member_finished( e->lz_encoder_base );
  }


unsigned long long LZ_compress_data_position( LZ_Encoder * const e )
  {
  if( !check_encoder( e ) ) return 0;
  return Mb_data_position( &e->lz_encoder_base->mb );
  }


unsigned long long LZ_compress_member_position( LZ_Encoder * const e )
  {
  if( !check_encoder( e ) ) return 0;
  return Re_member_position( &e->lz_encoder_base->renc );
  }


unsigned long long LZ_compress_total_in_size( LZ_Encoder * const e )
  {
  if( !check_encoder( e ) ) return 0;
  return e->partial_in_size + Mb_data_position( &e->lz_encoder_base->mb );
  }


unsigned long long LZ_compress_total_out_size( LZ_Encoder * const e )
  {
  if( !check_encoder( e ) ) return 0;
  return e->partial_out_size + Re_member_position( &e->lz_encoder_base->renc );
  }


/* -------------------- Decompression Functions -------------------- */

LZ_Decoder * LZ_decompress_open( void )
  {
  LZ_Decoder * const d = (LZ_Decoder *)malloc( sizeof (LZ_Decoder) );
  if( !d ) return 0;
  LZ_Decoder_init( d );

  d->rdec = (Range_decoder *)malloc( sizeof (Range_decoder) );
  if( !d->rdec || !Rd_init( d->rdec ) )
    {
    if( d->rdec ) { Rd_free( d->rdec ); free( d->rdec ); d->rdec = 0; }
    d->lz_errno = LZ_mem_error; d->fatal = true;
    }
  return d;
  }


int LZ_decompress_close( LZ_Decoder * const d )
  {
  if( !d ) return -1;
  if( d->lz_decoder )
    { LZd_free( d->lz_decoder ); free( d->lz_decoder ); }
  if( d->rdec ) { Rd_free( d->rdec ); free( d->rdec ); }
  free( d );
  return 0;
  }


int LZ_decompress_finish( LZ_Decoder * const d )
  {
  if( !check_decoder( d ) || d->fatal ) return -1;
  if( d->seeking )
    { d->seeking = false; d->partial_in_size += Rd_purge( d->rdec ); }
  else Rd_finish( d->rdec );
  return 0;
  }


int LZ_decompress_reset( LZ_Decoder * const d )
  {
  if( !check_decoder( d ) ) return -1;
  if( d->lz_decoder )
    { LZd_free( d->lz_decoder ); free( d->lz_decoder ); d->lz_decoder = 0; }
  d->partial_in_size = 0;
  d->partial_out_size = 0;
  Rd_reset( d->rdec );
  d->lz_errno = LZ_ok;
  d->fatal = false;
  d->first_header = true;
  d->seeking = false;
  return 0;
  }


int LZ_decompress_sync_to_member( LZ_Decoder * const d )
  {
  unsigned skipped = 0;
  if( !check_decoder( d ) ) return -1;
  if( d->lz_decoder )
    { LZd_free( d->lz_decoder ); free( d->lz_decoder ); d->lz_decoder = 0; }
  if( Rd_find_header( d->rdec, &skipped ) ) d->seeking = false;
  else
    {
    if( !d->rdec->at_stream_end ) d->seeking = true;
    else { d->seeking = false; d->partial_in_size += Rd_purge( d->rdec ); }
    }
  d->partial_in_size += skipped;
  d->lz_errno = LZ_ok;
  d->fatal = false;
  return 0;
  }


int LZ_decompress_read( LZ_Decoder * const d,
                        uint8_t * const buffer, const int size )
  {
  int result;
  if( !check_decoder( d ) ) return -1;
  if( size < 0 ) return 0;
  if( d->fatal )	/* don't return error until pending bytes are read */
    { if( d->lz_decoder && !Cb_empty( &d->lz_decoder->cb ) ) goto get_data;
      return -1; }
  if( d->seeking ) return 0;

  if( d->lz_decoder && LZd_member_finished( d->lz_decoder ) )
    {
    d->partial_out_size += LZd_data_position( d->lz_decoder );
    LZd_free( d->lz_decoder ); free( d->lz_decoder ); d->lz_decoder = 0;
    }
  if( !d->lz_decoder )
    {
    int rd;
    d->partial_in_size += d->rdec->member_position;
    d->rdec->member_position = 0;
    if( Rd_available_bytes( d->rdec ) < Lh_size + 5 &&
        !d->rdec->at_stream_end ) return 0;
    if( Rd_finished( d->rdec ) && !d->first_header ) return 0;
    rd = Rd_read_data( d->rdec, d->member_header, Lh_size );
    if( rd < Lh_size || Rd_finished( d->rdec ) )	/* End Of File */
      {
      if( rd <= 0 || Lh_check_prefix( d->member_header, rd ) )
        d->lz_errno = LZ_unexpected_eof;
      else
        d->lz_errno = LZ_header_error;
      d->fatal = true;
      return -1;
      }
    if( !Lh_check_magic( d->member_header ) )
      {
      /* unreading the header prevents sync_to_member from skipping a member
         if leading garbage is shorter than a full header; "lgLZIP\x01\x0C" */
      if( Rd_unread_data( d->rdec, rd ) )
        {
        if( d->first_header || !Lh_check_corrupt( d->member_header ) )
          d->lz_errno = LZ_header_error;
        else
          d->lz_errno = LZ_data_error;		/* corrupt header */
        }
      else
        d->lz_errno = LZ_library_error;
      d->fatal = true;
      return -1;
      }
    if( !Lh_check_version( d->member_header ) ||
        !isvalid_ds( Lh_get_dictionary_size( d->member_header ) ) )
      {
      /* Skip a possible "LZIP" leading garbage; "LZIPLZIP\x01\x0C".
         Leave member_pos pointing to the first error. */
      if( Rd_unread_data( d->rdec, 1 + !Lh_check_version( d->member_header ) ) )
        d->lz_errno = LZ_data_error;	/* bad version or bad dict size */
      else
        d->lz_errno = LZ_library_error;
      d->fatal = true;
      return -1;
      }
    d->first_header = false;
    if( Rd_available_bytes( d->rdec ) < 5 )
      {
      /* set position at EOF */
      d->rdec->member_position += Cb_used_bytes( &d->rdec->cb );
      Cb_reset( &d->rdec->cb );
      d->lz_errno = LZ_unexpected_eof;
      d->fatal = true;
      return -1;
      }
    d->lz_decoder = (LZ_decoder *)malloc( sizeof (LZ_decoder) );
    if( !d->lz_decoder || !LZd_init( d->lz_decoder, d->rdec,
                             Lh_get_dictionary_size( d->member_header ) ) )
      {					/* not enough free memory */
      if( d->lz_decoder )
        { LZd_free( d->lz_decoder ); free( d->lz_decoder ); d->lz_decoder = 0; }
      d->lz_errno = LZ_mem_error;
      d->fatal = true;
      return -1;
      }
    d->rdec->reload_pending = true;
    }
  result = LZd_decode_member( d->lz_decoder );
  if( result != 0 )
    {
    if( result == 2 )			/* set input position at EOF */
      { d->rdec->member_position += Cb_used_bytes( &d->rdec->cb );
        Cb_reset( &d->rdec->cb );
        d->lz_errno = LZ_unexpected_eof; }
    else if( result == 6 ) d->lz_errno = LZ_library_error;
    else d->lz_errno = LZ_data_error;
    d->fatal = true;
    if( Cb_empty( &d->lz_decoder->cb ) ) return -1;
    }
get_data:
  return Cb_read_data( &d->lz_decoder->cb, buffer, size );
  }


int LZ_decompress_write( LZ_Decoder * const d,
                         const uint8_t * const buffer, const int size )
  {
  int result;
  if( !check_decoder( d ) || d->fatal ) return -1;
  if( size < 0 ) return 0;

  result = Rd_write_data( d->rdec, buffer, size );
  while( d->seeking )
    {
    int size2;
    unsigned skipped = 0;
    if( Rd_find_header( d->rdec, &skipped ) ) d->seeking = false;
    d->partial_in_size += skipped;
    if( result >= size ) break;
    size2 = Rd_write_data( d->rdec, buffer + result, size - result );
    if( size2 > 0 ) result += size2;
    else break;
    }
  return result;
  }


int LZ_decompress_write_size( LZ_Decoder * const d )
  {
  if( !check_decoder( d ) || d->fatal ) return -1;
  return Rd_free_bytes( d->rdec );
  }


LZ_Errno LZ_decompress_errno( LZ_Decoder * const d )
  {
  if( !d ) return LZ_bad_argument;
  return d->lz_errno;
  }


int LZ_decompress_finished( LZ_Decoder * const d )
  {
  if( !check_decoder( d ) || d->fatal ) return -1;
  return Rd_finished( d->rdec ) &&
         ( !d->lz_decoder || LZd_member_finished( d->lz_decoder ) );
  }


int LZ_decompress_member_finished( LZ_Decoder * const d )
  {
  if( !check_decoder( d ) || d->fatal ) return -1;
  return d->lz_decoder && LZd_member_finished( d->lz_decoder );
  }


int LZ_decompress_member_version( LZ_Decoder * const d )
  {
  if( !check_decoder( d ) ) return -1;
  return Lh_version( d->member_header );
  }


int LZ_decompress_dictionary_size( LZ_Decoder * const d )
  {
  if( !check_decoder( d ) ) return -1;
  return Lh_get_dictionary_size( d->member_header );
  }


unsigned LZ_decompress_data_crc( LZ_Decoder * const d )
  {
  if( check_decoder( d ) && d->lz_decoder )
    return LZd_crc( d->lz_decoder );
  return 0;
  }


unsigned long long LZ_decompress_data_position( LZ_Decoder * const d )
  {
  if( check_decoder( d ) && d->lz_decoder )
    return LZd_data_position( d->lz_decoder );
  return 0;
  }


unsigned long long LZ_decompress_member_position( LZ_Decoder * const d )
  {
  if( !check_decoder( d ) ) return 0;
  return d->rdec->member_position;
  }


unsigned long long LZ_decompress_total_in_size( LZ_Decoder * const d )
  {
  if( !check_decoder( d ) ) return 0;
  return d->partial_in_size + d->rdec->member_position;
  }


unsigned long long LZ_decompress_total_out_size( LZ_Decoder * const d )
  {
  if( !check_decoder( d ) ) return 0;
  if( d->lz_decoder )
    return d->partial_out_size + LZd_data_position( d->lz_decoder );
  return d->partial_out_size;
  }

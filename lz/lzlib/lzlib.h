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

#ifdef __cplusplus
extern "C" {
#endif

/* LZ_API_VERSION was first defined in lzlib 1.8 to 1.
   Since lzlib 1.12, LZ_API_VERSION is defined as (major * 1000 + minor). */

#define LZ_API_VERSION 1015

static const char * const LZ_version_string = "1.15";

typedef enum LZ_Errno
  { LZ_ok = 0,         LZ_bad_argument, LZ_mem_error,
    LZ_sequence_error, LZ_header_error, LZ_unexpected_eof,
    LZ_data_error,     LZ_library_error } LZ_Errno;


int LZ_api_version( void );				/* new in 1.12 */
const char * LZ_version( void );
const char * LZ_strerror( const LZ_Errno lz_errno );

int LZ_min_dictionary_bits( void );
int LZ_min_dictionary_size( void );
int LZ_max_dictionary_bits( void );
int LZ_max_dictionary_size( void );
int LZ_min_match_len_limit( void );
int LZ_max_match_len_limit( void );


/* --------------------- Compression Functions --------------------- */

typedef struct LZ_Encoder LZ_Encoder;

LZ_Encoder * LZ_compress_open( const int dictionary_size,
                               const int match_len_limit,
                               const unsigned long long member_size );
int LZ_compress_close( LZ_Encoder * const encoder );

int LZ_compress_finish( LZ_Encoder * const encoder );
int LZ_compress_restart_member( LZ_Encoder * const encoder,
                                const unsigned long long member_size );
int LZ_compress_sync_flush( LZ_Encoder * const encoder );

int LZ_compress_read( LZ_Encoder * const encoder,
                      uint8_t * const buffer, const int size );
int LZ_compress_write( LZ_Encoder * const encoder,
                       const uint8_t * const buffer, const int size );
int LZ_compress_write_size( LZ_Encoder * const encoder );

LZ_Errno LZ_compress_errno( LZ_Encoder * const encoder );
int LZ_compress_finished( LZ_Encoder * const encoder );
int LZ_compress_member_finished( LZ_Encoder * const encoder );

unsigned long long LZ_compress_data_position( LZ_Encoder * const encoder );
unsigned long long LZ_compress_member_position( LZ_Encoder * const encoder );
unsigned long long LZ_compress_total_in_size( LZ_Encoder * const encoder );
unsigned long long LZ_compress_total_out_size( LZ_Encoder * const encoder );


/* -------------------- Decompression Functions -------------------- */

typedef struct LZ_Decoder LZ_Decoder;

LZ_Decoder * LZ_decompress_open( void );
int LZ_decompress_close( LZ_Decoder * const decoder );

int LZ_decompress_finish( LZ_Decoder * const decoder );
int LZ_decompress_reset( LZ_Decoder * const decoder );
int LZ_decompress_sync_to_member( LZ_Decoder * const decoder );

int LZ_decompress_read( LZ_Decoder * const decoder,
                        uint8_t * const buffer, const int size );
int LZ_decompress_write( LZ_Decoder * const decoder,
                         const uint8_t * const buffer, const int size );
int LZ_decompress_write_size( LZ_Decoder * const decoder );

LZ_Errno LZ_decompress_errno( LZ_Decoder * const decoder );
int LZ_decompress_finished( LZ_Decoder * const decoder );
int LZ_decompress_member_finished( LZ_Decoder * const decoder );

int LZ_decompress_member_version( LZ_Decoder * const decoder );
int LZ_decompress_dictionary_size( LZ_Decoder * const decoder );
unsigned LZ_decompress_data_crc( LZ_Decoder * const decoder );

unsigned long long LZ_decompress_data_position( LZ_Decoder * const decoder );
unsigned long long LZ_decompress_member_position( LZ_Decoder * const decoder );
unsigned long long LZ_decompress_total_in_size( LZ_Decoder * const decoder );
unsigned long long LZ_decompress_total_out_size( LZ_Decoder * const decoder );

#ifdef __cplusplus
}
#endif

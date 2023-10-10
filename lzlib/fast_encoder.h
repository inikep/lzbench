/* Lzlib - Compression library for the lzip format
   Copyright (C) 2009-2022 Antonio Diaz Diaz.

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

struct FLZ_encoder
  {
  struct LZ_encoder_base eb;
  unsigned key4;			/* key made from latest 4 bytes */
  };

static inline void FLZe_reset_key4( struct FLZ_encoder * const fe )
  {
  int i;
  fe->key4 = 0;
  for( i = 0; i < 3 && i < Mb_available_bytes( &fe->eb.mb ); ++i )
    fe->key4 = ( fe->key4 << 4 ) ^ fe->eb.mb.buffer[i];
  }

static inline bool FLZe_update_and_move( struct FLZ_encoder * const fe, int n )
  {
  struct Matchfinder_base * const mb = &fe->eb.mb;
  while( --n >= 0 )
    {
    if( Mb_available_bytes( mb ) >= 4 )
      {
      fe->key4 = ( ( fe->key4 << 4 ) ^ mb->buffer[mb->pos+3] ) & mb->key4_mask;
      mb->pos_array[mb->cyclic_pos] = mb->prev_positions[fe->key4];
      mb->prev_positions[fe->key4] = mb->pos + 1;
      }
    else mb->pos_array[mb->cyclic_pos] = 0;
    if( !Mb_move_pos( mb ) ) return false;
    }
  return true;
  }

static inline bool FLZe_init( struct FLZ_encoder * const fe,
                              const unsigned long long member_size )
  {
  enum { before_size = 0,
         dict_size = 65536,
         /* bytes to keep in buffer after pos */
         after_size = max_match_len,
         dict_factor = 16,
         min_free_bytes = max_marker_size,
         num_prev_positions23 = 0,
         pos_array_factor = 1 };

  return LZeb_init( &fe->eb, before_size, dict_size, after_size, dict_factor,
                    num_prev_positions23, pos_array_factor, min_free_bytes,
                    member_size );
  }

static inline void FLZe_reset( struct FLZ_encoder * const fe,
                               const unsigned long long member_size )
  { LZeb_reset( &fe->eb, member_size ); }

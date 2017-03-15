/*  Lzlib - Compression library for the lzip format
    Copyright (C) 2009-2016 Antonio Diaz Diaz.

    This library is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 2 of the License, or
    (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this library.  If not, see <http://www.gnu.org/licenses/>.

    As a special exception, you may use this file as part of a free
    software library without restriction.  Specifically, if other files
    instantiate templates or use macros or inline functions from this
    file, or you compile this file and link it with other files to
    produce an executable, this file does not by itself cause the
    resulting executable to be covered by the GNU General Public
    License.  This exception does not however invalidate any other
    reasons why the executable file might be covered by the GNU General
    Public License.
*/

struct FLZ_encoder
  {
  struct LZ_encoder_base eb;
  int key4;			/* key made from latest 4 bytes */
  };

static inline void FLZe_reset_key4( struct FLZ_encoder * const fe )
  {
  int i;
  fe->key4 = 0;
  for( i = 0; i < 3 && i < Mb_available_bytes( &fe->eb.mb ); ++i )
    fe->key4 = ( fe->key4 << 4 ) ^ fe->eb.mb.buffer[i];
  }

int FLZe_longest_match_len( struct FLZ_encoder * const fe, int * const distance );

static inline bool FLZe_update_and_move( struct FLZ_encoder * const fe, int n )
  {
  while( --n >= 0 )
    {
    if( Mb_available_bytes( &fe->eb.mb ) >= 4 )
      {
      int newpos;
      fe->key4 = ( ( fe->key4 << 4 ) ^ fe->eb.mb.buffer[fe->eb.mb.pos+3] ) &
                 fe->eb.mb.key4_mask;
      newpos = fe->eb.mb.prev_positions[fe->key4];
      fe->eb.mb.prev_positions[fe->key4] = fe->eb.mb.pos + 1;
      fe->eb.mb.pos_array[fe->eb.mb.cyclic_pos] = newpos;
      }
    else fe->eb.mb.pos_array[fe->eb.mb.cyclic_pos] = 0;
    if( !Mb_move_pos( &fe->eb.mb ) ) return false;
    }
  return true;
  }

static inline bool FLZe_init( struct FLZ_encoder * const fe,
                              const unsigned long long member_size )
  {
  enum { before = 0,
         dict_size = 65536,
         /* bytes to keep in buffer after pos */
         after_size = max_match_len,
         dict_factor = 16,
         num_prev_positions23 = 0,
         pos_array_factor = 1,
         min_free_bytes = max_marker_size };

  return LZeb_init( &fe->eb, before, dict_size, after_size, dict_factor,
                    num_prev_positions23, pos_array_factor, min_free_bytes,
                    member_size );
  }

static inline void FLZe_reset( struct FLZ_encoder * const fe,
                               const unsigned long long member_size )
  { LZeb_reset( &fe->eb, member_size ); }

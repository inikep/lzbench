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

static bool Mb_normalize_pos( struct Matchfinder_base * const mb )
  {
  if( mb->pos > mb->stream_pos )
    { mb->pos = mb->stream_pos; return false; }
  if( !mb->at_stream_end )
    {
    int i;
    const int offset = mb->pos - mb->dictionary_size - mb->before_size;
    const int size = mb->stream_pos - offset;
    memmove( mb->buffer, mb->buffer + offset, size );
    mb->partial_data_pos += offset;
    mb->pos -= offset;
    mb->stream_pos -= offset;
    for( i = 0; i < mb->num_prev_positions; ++i )
      mb->prev_positions[i] -= min( mb->prev_positions[i], offset );
    for( i = 0; i < mb->pos_array_size; ++i )
      mb->pos_array[i] -= min( mb->pos_array[i], offset );
    }
  return true;
  }


static bool Mb_init( struct Matchfinder_base * const mb,
                     const int before, const int dict_size,
                     const int after_size, const int dict_factor,
                     const int num_prev_positions23,
                     const int pos_array_factor )
  {
  const int buffer_size_limit =
    ( dict_factor * dict_size ) + before + after_size;
  unsigned size;
  int i;

  mb->partial_data_pos = 0;
  mb->before_size = before;
  mb->after_size = after_size;
  mb->pos = 0;
  mb->cyclic_pos = 0;
  mb->stream_pos = 0;
  mb->at_stream_end = false;
  mb->flushing = false;

  mb->buffer_size = max( 65536, buffer_size_limit );
  mb->buffer = (uint8_t *)malloc( mb->buffer_size );
  if( !mb->buffer ) return false;
  mb->dictionary_size = dict_size;
  mb->pos_limit = mb->buffer_size - after_size;
  size = 1 << max( 16, real_bits( mb->dictionary_size - 1 ) - 2 );
  if( mb->dictionary_size > 1 << 26 )		/* 64 MiB */
    size >>= 1;
  mb->key4_mask = size - 1;
  mb->num_prev_positions23 = num_prev_positions23;
  size += num_prev_positions23;

  mb->num_prev_positions = size;
  mb->pos_array_size = pos_array_factor * ( mb->dictionary_size + 1 );
  size += mb->pos_array_size;
  if( size * sizeof (int32_t) <= size ) mb->prev_positions = 0;
  else mb->prev_positions = (int32_t *)malloc( size * sizeof (int32_t) );
  if( !mb->prev_positions ) { free( mb->buffer ); return false; }
  mb->pos_array = mb->prev_positions + mb->num_prev_positions;
  for( i = 0; i < mb->num_prev_positions; ++i ) mb->prev_positions[i] = 0;
  return true;
  }


static void Mb_adjust_dictionary_size( struct Matchfinder_base * const mb )
  {
  if( mb->stream_pos < mb->dictionary_size )
    {
    int size;
    mb->buffer_size =
    mb->dictionary_size =
    mb->pos_limit = max( min_dictionary_size, mb->stream_pos );
    size = 1 << max( 16, real_bits( mb->dictionary_size - 1 ) - 2 );
    if( mb->dictionary_size > 1 << 26 )
      size >>= 1;
    mb->key4_mask = size - 1;
    size += mb->num_prev_positions23;
    mb->num_prev_positions = size;
    mb->pos_array = mb->prev_positions + mb->num_prev_positions;
    }
  }


static void Mb_reset( struct Matchfinder_base * const mb )
  {
  int i;
  if( mb->stream_pos > mb->pos )
    memmove( mb->buffer, mb->buffer + mb->pos, mb->stream_pos - mb->pos );
  mb->partial_data_pos = 0;
  mb->stream_pos -= mb->pos;
  mb->pos = 0;
  mb->cyclic_pos = 0;
  mb->at_stream_end = false;
  mb->flushing = false;
  for( i = 0; i < mb->num_prev_positions; ++i ) mb->prev_positions[i] = 0;
  }


     /* End Of Stream mark => (dis == 0xFFFFFFFFU, len == min_match_len) */
static bool LZeb_full_flush( struct LZ_encoder_base * const eb )
  {
  int i;
  const int pos_state = Mb_data_position( &eb->mb ) & pos_state_mask;
  const State state = eb->state;
  File_trailer trailer;
  if( eb->member_finished ||
      Cb_free_bytes( &eb->renc.cb ) < max_marker_size + eb->renc.ff_count + Ft_size )
    return false;
  Re_encode_bit( &eb->renc, &eb->bm_match[state][pos_state], 1 );
  Re_encode_bit( &eb->renc, &eb->bm_rep[state], 0 );
  LZeb_encode_pair( eb, 0xFFFFFFFFU, min_match_len, pos_state );
  Re_flush( &eb->renc );
  Ft_set_data_crc( trailer, LZeb_crc( eb ) );
  Ft_set_data_size( trailer, Mb_data_position( &eb->mb ) );
  Ft_set_member_size( trailer, Re_member_position( &eb->renc ) + Ft_size );
  for( i = 0; i < Ft_size; ++i )
    Cb_put_byte( &eb->renc.cb, trailer[i] );
  return true;
  }


     /* Sync Flush mark => (dis == 0xFFFFFFFFU, len == min_match_len + 1) */
static bool LZeb_sync_flush( struct LZ_encoder_base * const eb )
  {
  int i;
  const int pos_state = Mb_data_position( &eb->mb ) & pos_state_mask;
  const State state = eb->state;
  if( eb->member_finished ||
      Cb_free_bytes( &eb->renc.cb ) < (2 * max_marker_size) + eb->renc.ff_count )
    return false;
  for( i = 0; i < 2; ++i )	/* 2 consecutive markers guarantee decoding */
    {
    Re_encode_bit( &eb->renc, &eb->bm_match[state][pos_state], 1 );
    Re_encode_bit( &eb->renc, &eb->bm_rep[state], 0 );
    LZeb_encode_pair( eb, 0xFFFFFFFFU, min_match_len + 1, pos_state );
    Re_flush( &eb->renc );
    }
  return true;
  }


static void LZeb_reset( struct LZ_encoder_base * const eb,
                        const unsigned long long member_size )
  {
  int i;
  Mb_reset( &eb->mb );
  eb->member_size_limit = member_size - Ft_size - max_marker_size;
  eb->crc = 0xFFFFFFFFU;
  Bm_array_init( eb->bm_literal[0], (1 << literal_context_bits) * 0x300 );
  Bm_array_init( eb->bm_match[0], states * pos_states );
  Bm_array_init( eb->bm_rep, states );
  Bm_array_init( eb->bm_rep0, states );
  Bm_array_init( eb->bm_rep1, states );
  Bm_array_init( eb->bm_rep2, states );
  Bm_array_init( eb->bm_len[0], states * pos_states );
  Bm_array_init( eb->bm_dis_slot[0], len_states * (1 << dis_slot_bits) );
  Bm_array_init( eb->bm_dis, modeled_distances - end_dis_model );
  Bm_array_init( eb->bm_align, dis_align_size );
  Lm_init( &eb->match_len_model );
  Lm_init( &eb->rep_len_model );
  Re_reset( &eb->renc );
  for( i = 0; i < num_rep_distances; ++i ) eb->reps[i] = 0;
  eb->state = 0;
  eb->member_finished = false;
  }

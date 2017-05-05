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

static bool LZd_verify_trailer( struct LZ_decoder * const d )
  {
  File_trailer trailer;
  int size = Rd_read_data( d->rdec, trailer, Ft_size );

  if( size < Ft_size )
    return false;

  return ( Ft_get_data_crc( trailer ) == LZd_crc( d ) &&
           Ft_get_data_size( trailer ) == LZd_data_position( d ) &&
           Ft_get_member_size( trailer ) == d->rdec->member_position );
  }


/* Return value: 0 = OK, 1 = decoder error, 2 = unexpected EOF,
                 3 = trailer error, 4 = unknown marker found,
                 5 = library error. */
static int LZd_decode_member( struct LZ_decoder * const d )
  {
  struct Range_decoder * const rdec = d->rdec;
  State * const state = &d->state;
/*  unsigned long long old_mpos = d->rdec->member_position; */

  if( d->member_finished ) return 0;
  if( !Rd_try_reload( rdec, false ) )
    { if( !rdec->at_stream_end ) return 0; else return 2; }
  if( d->verify_trailer_pending )
    {
    if( Rd_available_bytes( rdec ) < Ft_size && !rdec->at_stream_end )
      return 0;
    d->verify_trailer_pending = false;
    d->member_finished = true;
    if( LZd_verify_trailer( d ) ) return 0; else return 3;
    }

  while( !Rd_finished( rdec ) )
    {
    const int pos_state = LZd_data_position( d ) & pos_state_mask;
/*    const unsigned long long mpos = d->rdec->member_position;
    if( mpos - old_mpos > rd_min_available_bytes ) return 5;
    old_mpos = mpos; */
    if( !Rd_enough_available_bytes( rdec ) )	/* check unexpected eof */
      { if( !rdec->at_stream_end ) return 0; else break; }
    if( !LZd_enough_free_bytes( d ) ) return 0;
    if( Rd_decode_bit( rdec, &d->bm_match[*state][pos_state] ) == 0 )	/* 1st bit */
      {
      const uint8_t prev_byte = LZd_peek_prev( d );
      if( St_is_char( *state ) )
        {
        *state -= ( *state < 4 ) ? *state : 3;
        LZd_put_byte( d, Rd_decode_tree( rdec,
                      d->bm_literal[get_lit_state(prev_byte)], 8 ) );
        }
      else
        {
        *state -= ( *state < 10 ) ? 3 : 6;
        LZd_put_byte( d, Rd_decode_matched( rdec,
                      d->bm_literal[get_lit_state(prev_byte)],
                      LZd_peek( d, d->rep0 ) ) );
        }
      }
    else					/* match or repeated match */
      {
      int len;
      if( Rd_decode_bit( rdec, &d->bm_rep[*state] ) != 0 )	/* 2nd bit */
        {
        if( Rd_decode_bit( rdec, &d->bm_rep0[*state] ) != 0 )	/* 3rd bit */
          {
          unsigned distance;
          if( Rd_decode_bit( rdec, &d->bm_rep1[*state] ) == 0 )	/* 4th bit */
            distance = d->rep1;
          else
            {
            if( Rd_decode_bit( rdec, &d->bm_rep2[*state] ) == 0 )	/* 5th bit */
              distance = d->rep2;
            else
              { distance = d->rep3; d->rep3 = d->rep2; }
            d->rep2 = d->rep1;
            }
          d->rep1 = d->rep0;
          d->rep0 = distance;
          }
        else
          {
          if( Rd_decode_bit( rdec, &d->bm_len[*state][pos_state] ) == 0 )	/* 4th bit */
            { *state = St_set_short_rep( *state );
              LZd_put_byte( d, LZd_peek( d, d->rep0 ) ); continue; }
          }
        *state = St_set_rep( *state );
        len = min_match_len + Rd_decode_len( rdec, &d->rep_len_model, pos_state );
        }
      else					/* match */
        {
        const unsigned rep0_saved = d->rep0;
        int dis_slot;
        len = min_match_len + Rd_decode_len( rdec, &d->match_len_model, pos_state );
        dis_slot = Rd_decode_tree6( rdec, d->bm_dis_slot[get_len_state(len)] );
        if( dis_slot < start_dis_model ) d->rep0 = dis_slot;
        else
          {
          const int direct_bits = ( dis_slot >> 1 ) - 1;
          d->rep0 = ( 2 | ( dis_slot & 1 ) ) << direct_bits;
          if( dis_slot < end_dis_model )
            d->rep0 += Rd_decode_tree_reversed( rdec,
                       d->bm_dis + d->rep0 - dis_slot - 1, direct_bits );
          else
            {
            d->rep0 += Rd_decode( rdec, direct_bits - dis_align_bits ) << dis_align_bits;
            d->rep0 += Rd_decode_tree_reversed4( rdec, d->bm_align );
            if( d->rep0 == 0xFFFFFFFFU )		/* marker found */
              {
              d->rep0 = rep0_saved;
              Rd_normalize( rdec );
              if( len == min_match_len )	/* End Of Stream marker */
                {
                if( Rd_available_bytes( rdec ) < Ft_size && !rdec->at_stream_end )
                  { d->verify_trailer_pending = true; return 0; }
                d->member_finished = true;
                if( LZd_verify_trailer( d ) ) return 0; else return 3;
                }
              if( len == min_match_len + 1 )	/* Sync Flush marker */
                {
                if( Rd_try_reload( rdec, true ) ) { /*old_mpos += 5;*/ continue; }
                else { if( !rdec->at_stream_end ) return 0; else break; }
                }
              return 4;
              }
            }
          }
        d->rep3 = d->rep2; d->rep2 = d->rep1; d->rep1 = rep0_saved;
        *state = St_set_match( *state );
        if( d->rep0 >= d->dictionary_size ||
            ( d->rep0 >= d->cb.put && !d->pos_wrapped ) )
          return 1;
        }
      LZd_copy_block( d, d->rep0, len );
      }
    }
  return 2;
  }

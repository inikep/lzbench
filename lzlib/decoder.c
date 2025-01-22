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

static int LZd_try_check_trailer( LZ_decoder * const d )
  {
  Lzip_trailer trailer;
  if( Rd_available_bytes( d->rdec ) < Lt_size )
    { if( !d->rdec->at_stream_end ) return 0; else return 2; }
  d->check_trailer_pending = false;
  d->member_finished = true;

  if( Rd_read_data( d->rdec, trailer, Lt_size ) == Lt_size &&
      Lt_get_data_crc( trailer ) == LZd_crc( d ) &&
      Lt_get_data_size( trailer ) == LZd_data_position( d ) &&
      Lt_get_member_size( trailer ) == d->rdec->member_position ) return 0;
  return 3;
  }


/* Return value: 0 = OK, 1 = decoder error, 2 = unexpected EOF,
                 3 = trailer error, 4 = unknown marker found,
                 5 = nonzero first LZMA byte found, 6 = library error. */
static int LZd_decode_member( LZ_decoder * const d )
  {
  Range_decoder * const rdec = d->rdec;
  State * const state = &d->state;
  unsigned old_mpos = rdec->member_position;

  if( d->member_finished ) return 0;
  const int tmp = Rd_try_reload( rdec );
  if( tmp > 1 ) return 5;
  if( !tmp ) { if( !rdec->at_stream_end ) return 0; else return 2; }
  if( d->check_trailer_pending ) return LZd_try_check_trailer( d );

  while( !Rd_finished( rdec ) )
    {
    const unsigned mpos = rdec->member_position;
    if( mpos - old_mpos > rd_min_available_bytes ) return 6;
    old_mpos = mpos;
    if( !Rd_enough_available_bytes( rdec ) )	/* check unexpected EOF */
      { if( !rdec->at_stream_end ) return 0;
        if( Cb_empty( &rdec->cb ) ) break; }	/* decode until EOF */
    if( !LZd_enough_free_bytes( d ) ) return 0;
    const int pos_state = LZd_data_position( d ) & pos_state_mask;
    if( Rd_decode_bit( rdec, &d->bm_match[*state][pos_state] ) == 0 ) /* 1st bit */
      {
      /* literal byte */
      Bit_model * const bm = d->bm_literal[get_lit_state(LZd_peek_prev( d ))];
      if( ( *state = St_set_char( *state ) ) < 4 )
        LZd_put_byte( d, Rd_decode_tree8( rdec, bm ) );
      else
        LZd_put_byte( d, Rd_decode_matched( rdec, bm, LZd_peek( d, d->rep0 ) ) );
      continue;
      }
    /* match or repeated match */
    int len;
    if( Rd_decode_bit( rdec, &d->bm_rep[*state] ) != 0 )	/* 2nd bit */
      {
      if( Rd_decode_bit( rdec, &d->bm_rep0[*state] ) == 0 )	/* 3rd bit */
        {
        if( Rd_decode_bit( rdec, &d->bm_len[*state][pos_state] ) == 0 )	/* 4th bit */
          { *state = St_set_shortrep( *state );
            LZd_put_byte( d, LZd_peek( d, d->rep0 ) ); continue; }
        }
      else
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
      *state = St_set_rep( *state );
      len = Rd_decode_len( rdec, &d->rep_len_model, pos_state );
      }
    else					/* match */
      {
      len = Rd_decode_len( rdec, &d->match_len_model, pos_state );
      unsigned distance = Rd_decode_tree6( rdec, d->bm_dis_slot[get_len_state(len)] );
      if( distance >= start_dis_model )
        {
        const unsigned dis_slot = distance;
        const int direct_bits = ( dis_slot >> 1 ) - 1;
        distance = ( 2 | ( dis_slot & 1 ) ) << direct_bits;
        if( dis_slot < end_dis_model )
          distance += Rd_decode_tree_reversed( rdec,
                      d->bm_dis + ( distance - dis_slot ), direct_bits );
        else
          {
          distance +=
            Rd_decode( rdec, direct_bits - dis_align_bits ) << dis_align_bits;
          distance += Rd_decode_tree_reversed4( rdec, d->bm_align );
          if( distance == 0xFFFFFFFFU )		/* marker found */
            {
            Rd_normalize( rdec );
            const unsigned mpos = rdec->member_position;
            if( mpos - old_mpos > rd_min_available_bytes ) return 6;
            old_mpos = mpos;
            if( len == min_match_len )		/* End Of Stream marker */
              {
              d->check_trailer_pending = true;
              return LZd_try_check_trailer( d );
              }
            if( len == min_match_len + 1 )	/* Sync Flush marker */
              {
              rdec->reload_pending = true;
              const int tmp = Rd_try_reload( rdec );
              if( tmp > 1 ) return 5;
              if( tmp ) continue;
              if( !rdec->at_stream_end ) return 0; else break;
              }
            return 4;
            }
          }
        }
      d->rep3 = d->rep2; d->rep2 = d->rep1; d->rep1 = d->rep0; d->rep0 = distance;
      *state = St_set_match( *state );
      if( d->rep0 >= d->dictionary_size ||
          ( d->rep0 >= d->cb.put && !d->pos_wrapped ) ) return 1;
      }
    LZd_copy_block( d, d->rep0, len );
    }
  return 2;
  }

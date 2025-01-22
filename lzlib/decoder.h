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

enum { rd_min_available_bytes = 10 };

typedef struct Range_decoder
  {
  Circular_buffer cb;			/* input buffer */
  unsigned long long member_position;
  uint32_t code;
  uint32_t range;
  bool at_stream_end;
  bool reload_pending;
  } Range_decoder;

static inline bool Rd_init( Range_decoder * const rdec )
  {
  if( !Cb_init( &rdec->cb, 65536 + rd_min_available_bytes ) ) return false;
  rdec->member_position = 0;
  rdec->code = 0;
  rdec->range = 0xFFFFFFFFU;
  rdec->at_stream_end = false;
  rdec->reload_pending = false;
  return true;
  }

static inline void Rd_free( Range_decoder * const rdec )
  { Cb_free( &rdec->cb ); }

static inline bool Rd_finished( const Range_decoder * const rdec )
  { return rdec->at_stream_end && Cb_empty( &rdec->cb ); }

static inline void Rd_finish( Range_decoder * const rdec )
  { rdec->at_stream_end = true; }

static inline bool Rd_enough_available_bytes( const Range_decoder * const rdec )
  { return Cb_used_bytes( &rdec->cb ) >= rd_min_available_bytes; }

static inline unsigned Rd_available_bytes( const Range_decoder * const rdec )
  { return Cb_used_bytes( &rdec->cb ); }

static inline unsigned Rd_free_bytes( const Range_decoder * const rdec )
  { return rdec->at_stream_end ? 0 : Cb_free_bytes( &rdec->cb ); }

static inline unsigned long long Rd_purge( Range_decoder * const rdec )
  {
  const unsigned long long size =
    rdec->member_position + Cb_used_bytes( &rdec->cb );
  Cb_reset( &rdec->cb );
  rdec->member_position = 0; rdec->at_stream_end = true;
  return size;
  }

static inline void Rd_reset( Range_decoder * const rdec )
  { Cb_reset( &rdec->cb );
    rdec->member_position = 0; rdec->at_stream_end = false; }


/* Seek for a member header and update 'get'. Set '*skippedp' to the number
   of bytes skipped. Return true if a valid header is found.
*/
static bool Rd_find_header( Range_decoder * const rdec,
                            unsigned * const skippedp )
  {
  *skippedp = 0;
  while( rdec->cb.get != rdec->cb.put )
    {
    if( rdec->cb.buffer[rdec->cb.get] == lzip_magic[0] )
      {
      unsigned get = rdec->cb.get;
      int i;
      Lzip_header header;
      for( i = 0; i < Lh_size; ++i )
        {
        if( get == rdec->cb.put ) return false;		/* not enough data */
        header[i] = rdec->cb.buffer[get];
        if( ++get >= rdec->cb.buffer_size ) get = 0;
        }
      if( Lh_check( header ) ) return true;
      }
    if( ++rdec->cb.get >= rdec->cb.buffer_size ) rdec->cb.get = 0;
    ++*skippedp;
    }
  return false;
  }


static inline int Rd_write_data( Range_decoder * const rdec,
                                 const uint8_t * const inbuf, const int size )
  {
  if( rdec->at_stream_end || size <= 0 ) return 0;
  return Cb_write_data( &rdec->cb, inbuf, size );
  }

static inline uint8_t Rd_get_byte( Range_decoder * const rdec )
  {
  /* 0xFF avoids decoder error if member is truncated at EOS marker */
  if( Rd_finished( rdec ) ) return 0xFF;
  ++rdec->member_position;
  return Cb_get_byte( &rdec->cb );
  }

static inline int Rd_read_data( Range_decoder * const rdec,
                                uint8_t * const outbuf, const int size )
  {
  const int sz = Cb_read_data( &rdec->cb, outbuf, size );
  if( sz > 0 ) rdec->member_position += sz;
  return sz;
  }

static inline bool Rd_unread_data( Range_decoder * const rdec,
                                   const unsigned size )
  {
  if( size > rdec->member_position || !Cb_unread_data( &rdec->cb, size ) )
    return false;
  rdec->member_position -= size;
  return true;
  }

static int Rd_try_reload( Range_decoder * const rdec )
  {
  if( rdec->reload_pending && Rd_available_bytes( rdec ) >= 5 )
    {
    rdec->reload_pending = false;
    rdec->code = 0;
    rdec->range = 0xFFFFFFFFU;
    /* check first byte of the LZMA stream without reading it */
    if( rdec->cb.buffer[rdec->cb.get] != 0 ) return 2;
    Rd_get_byte( rdec );	/* discard first byte of the LZMA stream */
    int i; for( i = 0; i < 4; ++i )
      rdec->code = (rdec->code << 8) | Rd_get_byte( rdec );
    }
  return !rdec->reload_pending;
  }

static inline void Rd_normalize( Range_decoder * const rdec )
  {
  if( rdec->range <= 0x00FFFFFFU )
    { rdec->range <<= 8; rdec->code = (rdec->code << 8) | Rd_get_byte( rdec ); }
  }

static inline unsigned Rd_decode( Range_decoder * const rdec,
                                  const int num_bits )
  {
  unsigned symbol = 0;
  int i;
  for( i = num_bits; i > 0; --i )
    {
    Rd_normalize( rdec );
    rdec->range >>= 1;
/*    symbol <<= 1; */
/*    if( rdec->code >= rdec->range ) { rdec->code -= rdec->range; symbol |= 1; } */
    const bool bit = rdec->code >= rdec->range;
    symbol <<= 1; symbol += bit;
    rdec->code -= rdec->range & ( 0U - bit );
    }
  return symbol;
  }

static inline unsigned Rd_decode_bit( Range_decoder * const rdec,
                                      Bit_model * const probability )
  {
  Rd_normalize( rdec );
  const uint32_t bound = ( rdec->range >> bit_model_total_bits ) * *probability;
  if( rdec->code < bound )
    {
    rdec->range = bound;
    *probability += ( bit_model_total - *probability ) >> bit_model_move_bits;
    return 0;
    }
  else
    {
    rdec->code -= bound;
    rdec->range -= bound;
    *probability -= *probability >> bit_model_move_bits;
    return 1;
    }
  }

static inline void Rd_decode_symbol_bit( Range_decoder * const rdec,
                         Bit_model * const probability, unsigned * symbol )
  {
  Rd_normalize( rdec );
  *symbol <<= 1;
  const uint32_t bound = ( rdec->range >> bit_model_total_bits ) * *probability;
  if( rdec->code < bound )
    {
    rdec->range = bound;
    *probability += ( bit_model_total - *probability ) >> bit_model_move_bits;
    }
  else
    {
    rdec->code -= bound;
    rdec->range -= bound;
    *probability -= *probability >> bit_model_move_bits;
    *symbol |= 1;
    }
  }

static inline void Rd_decode_symbol_bit_reversed( Range_decoder * const rdec,
                         Bit_model * const probability, unsigned * model,
                         unsigned * symbol, const int i )
  {
  Rd_normalize( rdec );
  *model <<= 1;
  const uint32_t bound = ( rdec->range >> bit_model_total_bits ) * *probability;
  if( rdec->code < bound )
    {
    rdec->range = bound;
    *probability += ( bit_model_total - *probability ) >> bit_model_move_bits;
    }
  else
    {
    rdec->code -= bound;
    rdec->range -= bound;
    *probability -= *probability >> bit_model_move_bits;
    *model |= 1;
    *symbol |= 1 << i;
    }
  }

static inline unsigned Rd_decode_tree6( Range_decoder * const rdec,
                                        Bit_model bm[] )
  {
  unsigned symbol = 1;
  Rd_decode_symbol_bit( rdec, &bm[symbol], &symbol );
  Rd_decode_symbol_bit( rdec, &bm[symbol], &symbol );
  Rd_decode_symbol_bit( rdec, &bm[symbol], &symbol );
  Rd_decode_symbol_bit( rdec, &bm[symbol], &symbol );
  Rd_decode_symbol_bit( rdec, &bm[symbol], &symbol );
  Rd_decode_symbol_bit( rdec, &bm[symbol], &symbol );
  return symbol & 0x3F;
  }

static inline unsigned Rd_decode_tree8( Range_decoder * const rdec,
                                        Bit_model bm[] )
  {
  unsigned symbol = 1;
  Rd_decode_symbol_bit( rdec, &bm[symbol], &symbol );
  Rd_decode_symbol_bit( rdec, &bm[symbol], &symbol );
  Rd_decode_symbol_bit( rdec, &bm[symbol], &symbol );
  Rd_decode_symbol_bit( rdec, &bm[symbol], &symbol );
  Rd_decode_symbol_bit( rdec, &bm[symbol], &symbol );
  Rd_decode_symbol_bit( rdec, &bm[symbol], &symbol );
  Rd_decode_symbol_bit( rdec, &bm[symbol], &symbol );
  Rd_decode_symbol_bit( rdec, &bm[symbol], &symbol );
  return symbol & 0xFF;
  }

static inline unsigned
Rd_decode_tree_reversed( Range_decoder * const rdec,
                         Bit_model bm[], const int num_bits )
  {
  unsigned model = 1;
  unsigned symbol = 0;
  int i;
  for( i = 0; i < num_bits; ++i )
    Rd_decode_symbol_bit_reversed( rdec, &bm[model], &model, &symbol, i );
  return symbol;
  }

static inline unsigned
Rd_decode_tree_reversed4( Range_decoder * const rdec, Bit_model bm[] )
  {
  unsigned model = 1;
  unsigned symbol = 0;
  Rd_decode_symbol_bit_reversed( rdec, &bm[model], &model, &symbol, 0 );
  Rd_decode_symbol_bit_reversed( rdec, &bm[model], &model, &symbol, 1 );
  Rd_decode_symbol_bit_reversed( rdec, &bm[model], &model, &symbol, 2 );
  Rd_decode_symbol_bit_reversed( rdec, &bm[model], &model, &symbol, 3 );
  return symbol;
  }

static inline unsigned Rd_decode_matched( Range_decoder * const rdec,
                                          Bit_model bm[], unsigned match_byte )
  {
  unsigned symbol = 1;
  unsigned mask = 0x100;
  while( true )
    {
    const unsigned match_bit = ( match_byte <<= 1 ) & mask;
    const unsigned bit = Rd_decode_bit( rdec, &bm[symbol+match_bit+mask] );
    symbol <<= 1; symbol += bit;
    if( symbol > 0xFF ) return symbol & 0xFF;
    mask &= ~(match_bit ^ (bit << 8));	/* if( match_bit != bit ) mask = 0; */
    }
  }

static inline unsigned Rd_decode_len( Range_decoder * const rdec,
                                      Len_model * const lm,
                                      const int pos_state )
  {
  Bit_model * bm;
  unsigned mask, offset, symbol = 1;

  if( Rd_decode_bit( rdec, &lm->choice1 ) == 0 )
    { bm = lm->bm_low[pos_state]; mask = 7; offset = 0; goto len3; }
  if( Rd_decode_bit( rdec, &lm->choice2 ) == 0 )
    { bm = lm->bm_mid[pos_state]; mask = 7; offset = len_low_symbols; goto len3; }
  bm = lm->bm_high; mask = 0xFF; offset = len_low_symbols + len_mid_symbols;
  Rd_decode_symbol_bit( rdec, &bm[symbol], &symbol );
  Rd_decode_symbol_bit( rdec, &bm[symbol], &symbol );
  Rd_decode_symbol_bit( rdec, &bm[symbol], &symbol );
  Rd_decode_symbol_bit( rdec, &bm[symbol], &symbol );
  Rd_decode_symbol_bit( rdec, &bm[symbol], &symbol );
len3:
  Rd_decode_symbol_bit( rdec, &bm[symbol], &symbol );
  Rd_decode_symbol_bit( rdec, &bm[symbol], &symbol );
  Rd_decode_symbol_bit( rdec, &bm[symbol], &symbol );
  return ( symbol & mask ) + min_match_len + offset;
  }


enum { lzd_min_free_bytes = max_match_len };

typedef struct LZ_decoder
  {
  Circular_buffer cb;
  unsigned long long partial_data_pos;
  Range_decoder * rdec;
  unsigned dictionary_size;
  uint32_t crc;
  bool check_trailer_pending;
  bool member_finished;
  bool pos_wrapped;
  unsigned rep0;		/* rep[0-3] latest four distances */
  unsigned rep1;		/* used for efficient coding of */
  unsigned rep2;		/* repeated distances */
  unsigned rep3;
  State state;

  Bit_model bm_literal[1<<literal_context_bits][0x300];
  Bit_model bm_match[states][pos_states];
  Bit_model bm_rep[states];
  Bit_model bm_rep0[states];
  Bit_model bm_rep1[states];
  Bit_model bm_rep2[states];
  Bit_model bm_len[states][pos_states];
  Bit_model bm_dis_slot[len_states][1<<dis_slot_bits];
  Bit_model bm_dis[modeled_distances-end_dis_model+1];
  Bit_model bm_align[dis_align_size];

  Len_model match_len_model;
  Len_model rep_len_model;
  } LZ_decoder;

static inline bool LZd_enough_free_bytes( const LZ_decoder * const d )
  { return Cb_free_bytes( &d->cb ) >= lzd_min_free_bytes; }

static inline uint8_t LZd_peek_prev( const LZ_decoder * const d )
  { return d->cb.buffer[((d->cb.put > 0) ? d->cb.put : d->cb.buffer_size)-1]; }

static inline uint8_t LZd_peek( const LZ_decoder * const d,
                                const unsigned distance )
  {
  const unsigned i = ( (d->cb.put > distance) ? 0 : d->cb.buffer_size ) +
                     d->cb.put - distance - 1;
  return d->cb.buffer[i];
  }

static inline void LZd_put_byte( LZ_decoder * const d, const uint8_t b )
  {
  CRC32_update_byte( &d->crc, b );
  d->cb.buffer[d->cb.put] = b;
  if( ++d->cb.put >= d->cb.buffer_size )
    { d->partial_data_pos += d->cb.put; d->cb.put = 0; d->pos_wrapped = true; }
  }

static inline void LZd_copy_block( LZ_decoder * const d,
                                   const unsigned distance, unsigned len )
  {
  unsigned lpos = d->cb.put, i = lpos - distance - 1;
  bool fast, fast2;
  if( lpos > distance )
    {
    fast = len < d->cb.buffer_size - lpos;
    fast2 = fast && len <= lpos - i;
    }
  else
    {
    i += d->cb.buffer_size;
    fast = len < d->cb.buffer_size - i;		/* (i == pos) may happen */
    fast2 = fast && len <= i - lpos;
    }
  if( fast )					/* no wrap */
    {
    const unsigned tlen = len;
    if( fast2 )					/* no wrap, no overlap */
      memcpy( d->cb.buffer + lpos, d->cb.buffer + i, len );
    else
      for( ; len > 0; --len ) d->cb.buffer[lpos++] = d->cb.buffer[i++];
    CRC32_update_buf( &d->crc, d->cb.buffer + d->cb.put, tlen );
    d->cb.put += tlen;
    }
  else for( ; len > 0; --len )
    {
    LZd_put_byte( d, d->cb.buffer[i] );
    if( ++i >= d->cb.buffer_size ) i = 0;
    }
  }

static inline bool LZd_init( LZ_decoder * const d, Range_decoder * const rde,
                             const unsigned dict_size )
  {
  if( !Cb_init( &d->cb, max( 65536, dict_size ) + lzd_min_free_bytes ) )
    return false;
  d->partial_data_pos = 0;
  d->rdec = rde;
  d->dictionary_size = dict_size;
  d->crc = 0xFFFFFFFFU;
  d->check_trailer_pending = false;
  d->member_finished = false;
  d->pos_wrapped = false;
  /* prev_byte of first byte; also for LZd_peek( 0 ) on corrupt file */
  d->cb.buffer[d->cb.buffer_size-1] = 0;
  d->rep0 = 0;
  d->rep1 = 0;
  d->rep2 = 0;
  d->rep3 = 0;
  d->state = 0;

  Bm_array_init( d->bm_literal[0], (1 << literal_context_bits) * 0x300 );
  Bm_array_init( d->bm_match[0], states * pos_states );
  Bm_array_init( d->bm_rep, states );
  Bm_array_init( d->bm_rep0, states );
  Bm_array_init( d->bm_rep1, states );
  Bm_array_init( d->bm_rep2, states );
  Bm_array_init( d->bm_len[0], states * pos_states );
  Bm_array_init( d->bm_dis_slot[0], len_states * (1 << dis_slot_bits) );
  Bm_array_init( d->bm_dis, modeled_distances - end_dis_model + 1 );
  Bm_array_init( d->bm_align, dis_align_size );
  Lm_init( &d->match_len_model );
  Lm_init( &d->rep_len_model );
  return true;
  }

static inline void LZd_free( LZ_decoder * const d ) { Cb_free( &d->cb ); }

static inline bool LZd_member_finished( const LZ_decoder * const d )
  { return d->member_finished && Cb_empty( &d->cb ); }

static inline unsigned LZd_crc( const LZ_decoder * const d )
  { return d->crc ^ 0xFFFFFFFFU; }

static inline unsigned long long
LZd_data_position( const LZ_decoder * const d )
  { return d->partial_data_pos + d->cb.put; }

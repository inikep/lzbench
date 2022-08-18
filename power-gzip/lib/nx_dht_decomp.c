/*
 * NX-GZIP compression accelerator user library
 *
 * Copyright (C) IBM Corporation, 2011-2017
 *
 * Licenses for GPLv2 and Apache v2.0:
 *
 * GPLv2:
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 *
 * Apache v2.0:
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Bulent Abali <abali@us.ibm.com>
 */

/*
  Sample code for decompressing a dynamic huffman table found in a Type 2 deflate block.
  Used for modeling the VHDL behavior. We would use this code to decompress DHT received
  via CPB in a NX compress operation, or decompress DHT in the payload during a NX decompress
  operation.

  To process output of dhtgen, compile this source with
  gcc -D_TEST_MAIN -DZTHEMIS dht_decomp.c -o dht_decomp

  Throw away the first 16 bytes of output of dhtgen. 
  For example, the zero filled first line including 020e is to be thrown away

  abali@css-host-11:~/project/makedht/examples$ xxd dht.bin
  00000000: 0000 0000 0000 0000 0000 0000 0000 020e  ................
  00000010: bdbf eb6e e3c8 b635 fff5 f7f7 f7f7 cfbf  ...n...5........
  00000020: fefe fefe a9bf bfbf bfbf bfbf bfbf bfbf  ................

  You can use the linux command to exclude the first 16 bytes
  tail -c +17 dht.bin > dhtnew.bin
  Now run
  dht_decomp dhtnew.bin

  The defines P9CPB and ZTHEMIS change the format behavior.  If a 16 byte P9
  CPB header is found or not.  If the 3 bit block header is found or not (when
  the input data is from a zlib payload).

*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

/* http://www.gzip.org/zlib/rfc-deflate.html#dyn */
#define HLIT   5
#define HDIST  5
#define HCLEN  4

#undef P9CPB   /* undefine this if reading the output of raw deflate */

#ifdef _TEST_MAIN
#define PRINTF(X) do { fprintf X; } while(0)
#else
#define PRINTF(X) do { ; } while(0)
#endif

typedef struct _ctx {
    unsigned int hold;
    int hold_bits;
    char *dbuf;

    int sum_used_bits;
    int sum2_used_bits; /*  another way to count */

    struct {
	int len;  /*  lsb aligned valid number of bits in code */
	int code; /*  idx and code can share the bits in vhdl */
    } hclen_tree[19];
    int hclen_next_code[8];
    int hclen_bl_count[8];     /*  length counts */

    struct {
	int len;  /*  lsb aligned valid number of bits in the code */
	int idx;
    } ll_tree[286];
    int ll_next_code_base[16];
    int ll_bl_count[16];

    struct {
	int len;  /*  lsb aligned valid number of bits in the code */
	int idx;
    } dist_tree[30];
    int dist_next_code_base[16];
    int dist_bl_count[16];
} ctx;

/*  
  Algorithm: (derived from deflate specification). Main difference is that we only
  do one pass over the long lists to save cycles (see the IDX discussion).
  In contrast deflate spec shows a two pass algoritm.

  get
  hlit 5b
  hdist 5b
  hclen 4b // 3 cycle
  get code lengths (N=0..7) for the code length alphabet (hclen+4)*3b

  for i<(hclen+4) in this order i=16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15
    hclen_bl_count[0]=0 always
    code=0 initially
    tree[i].len = read_len
    tree[i].idx = hclen_bl_count[ read_len ]; 
    the IDX step stores the base address adder 
    to the codemin to be calculated later;  for the hclen computation 
    I actually don't use an IDX because (1) I have the array storage to make a 2nd pass 
    (2) the deflate weird ordering throws off my index calculations;  
    I will use the IDX trick for LL and D computations

    hclen_bl_count[ read_len ] += 1

  endfor // 19 cycle max

  calculate codemin for bl_count[1..7]  // 7 cycles max

  now compute the hclen_tree[i].code for each i // 19 cycle max

  now reading LL lengths
  for j<hlit+257
    read 7 bits from input
    decode: lookup tree[L].code by code, L is the length for the symbol j
    calculate input shift amount
    tree[j].len = L
    tree[j].idx = bl_count[ L ];
    bl_count[ L ] += 1
  end for // 286 max

  now reading distance lengths
  for j<hdist+1
    read 7 bits from input
    decode: lookup tree[D].code by code, D is the length for the symbol j
    calculate input shift amount
    tree[j].len = D
    tree[j].idx = bl_count[ D ];
    bl_count[ D ] += 1
  end for // 30 max

  calculate codemin for ll_bl_count[1..15] and store in to codemin table // 15 cycles max
  calculate codemin for dist_bl_count[1..15] and store in to codemin table // 15 cycles max

  fixme:  total ~391 cycles - half thruput block size is 3128 bytes

*/

/* returns 1 to 7 bits from buf; assumes that the buf had been filled 
*/
static int get_bits( int need_bits, ctx *c )
{
    unsigned int result;
    assert( need_bits < 8 && need_bits > 0 && c->hold_bits < sizeof(int)*8 );
    if( c->hold_bits < need_bits ) {
	c->hold |= ((unsigned int)(*c->dbuf ) & 0xFF) << c->hold_bits;
	c->hold_bits += 8;
	++c->dbuf;
    }
    result = (unsigned int)c->hold & ((1U << need_bits) - 1);
    c->hold >>= need_bits;
    c->hold_bits -= need_bits;
    /*  PRINTF((stderr, "hold 0x%x bits %d\n", c->hold, c->hold_bits)); */
    return result;
}

/* we don't know how many bits we need 
 * so we get the max bits we could use (7 or 15) then shift the used bits amount in 
 * the next call. shf_bits is what we actuall used in the previous cycle therefore
 * we need to discard those many bits from hold first
 */
static int shift_then_get_bits( int need_bits, int shf_bits, ctx *c )
{
    unsigned int result;

    assert( need_bits < 16 && need_bits > 0 && c->hold_bits < sizeof(int)*8 );
    assert( shf_bits < 16 && shf_bits >= 0 );

    /*  Replenish the holding register. */
    /*  I need a one cycle shifter which does these two if-then blocks */
    if( c->hold_bits < (need_bits+shf_bits) ) {
	c->hold |= ((unsigned int)(*c->dbuf ) & 0xFF) << c->hold_bits;
	c->hold_bits += 8;
	++c->dbuf;
    }
    /*  do it again for the max 15 bit codes scenario (never executes for the 7 bit hclen) */
    if( c->hold_bits < (need_bits+shf_bits) ) { 
	c->hold |= ((unsigned int)(*c->dbuf ) & 0xFF) << c->hold_bits;
	c->hold_bits += 8;
	++c->dbuf;
    }

    /*  We used shf_bits in the prev cycle; drop those bits now */
    c->hold >>= shf_bits; 
    c->hold_bits -= shf_bits;
    result = (unsigned int)c->hold & ((1U << need_bits) - 1);
    /*  PRINTF((stderr, "c->hold 0x%x bits %d\n", c->hold, c->hold_bits)); */

    c->sum_used_bits += shf_bits; /*  count total for Craig's DHT defect */

    return result;
}

/* bit reverse
 */
static int reverse( int val, int n ) 
{
    int i, result=0;

    for(i=0; i< n; i++) {
	result = (result << 1) | (val & 1);
	val = val >> 1;
    }
    return result;
}

static int search_hclen_tree( int hcode, ctx *c )
{
    int i;
    PRINTF((stderr, "  search hcode %X\n", hcode));
    for(i=0; i<19; i++) {
	if( c->hclen_tree[i].len ) {
	    if( (hcode & ((1<<c->hclen_tree[i].len)-1)) == (((1<<c->hclen_tree[i].len)-1) & c->hclen_tree[i].code) ) {
		PRINTF((stderr, "  found alp %d, code %X, len %d,\n", i, c->hclen_tree[i].code, c->hclen_tree[i].len )); 
		return i;
	    }
	}
    }
    return -1; /*  not found */
}

/* buf points to the DHT and the total number of bits in DHT
 */
int dht_decomp(char *buf, int buf_bits, ctx *c)
{
    int dbits = buf_bits;
    int i,j;
    int hclen_order[] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 };
    /*  int hclen_table[19]; code length alphabet */
    int used_bits = 0;

    int hbits;
    int code = 0;

    c->dbuf = buf;
    
    
    /*  -------------------------------------------------------------- */
    /*  HLIT: num literal length codes (257 to 286) */
    /*  */
    /*  unsigned int hlit = get_bits( HLIT ) + 257; */

#ifdef P9CPB        
    used_bits = 0;
#elif ZTHEMIS
    used_bits = 0;
#else
    used_bits = 3;  /* when reading from deflate block, discards the type2 block header */
#endif
    
    unsigned int hlit = shift_then_get_bits( 5, used_bits, c ) + 257;
    used_bits = 5; 
    c->sum2_used_bits += used_bits;

    PRINTF((stderr, "HLIT: %d\n", hlit ));

    /*  -------------------------------------------------------------- */
    /*  HDIST: num dist codes (1 to 32) */
    /*  */
    /*  unsigned int hdist = get_bits( HDIST ) + 1; */

    unsigned int hdist = shift_then_get_bits( 5, used_bits, c ) + 1;
    used_bits = 5;
    c->sum2_used_bits += used_bits;
    PRINTF((stderr, "HDIST: %d\n", hdist ));

    /*  -------------------------------------------------------------- */
    /*  HCLEN: num code length codes (4 to 19) */
    /*  */
    /*  unsigned int hclen = get_bits( HCLEN ) + 4; */
    unsigned int hclen = shift_then_get_bits( 4, used_bits, c ) + 4;
    used_bits = 4;
    c->sum2_used_bits += used_bits;
    PRINTF((stderr, "HCLEN: %d\n", hclen ));

    /*  */
    /*  3 cycles to get the counts in the header */
    /*  */

    /*  -------------------------------------------------------------- */
    /*  Clear hclen table; 0 or 1 cycle in parallel */
    /*  */
    for(i=0; i < 19; i++) { 
	c->hclen_tree[ i ].len = 0;
	/*  c->hclen_tree[ i ].idx = 0; */
	c->hclen_tree[ i ].code = 0; /*  do I need this */
    }

    /*  -------------------------------------------------------------- */
    /*  Clear blcount; 0 or 1 cycle */
    /*  */
    for(i=0; i < 8; i++) { 
	c->hclen_bl_count[ i ] = 0;
    }

    /*  -------------------------------------------------------------- */
    /*  Read (HCLEN + 4) x 3 bits from CPB buffer: code lengths for the code  */
    /*  length alphabet in the deflate spec interleaved order  */
    /*  up to 19 cycles */
    /*   */
    for(i=0; i < hclen; i++) { 
	int k = hclen_order[i];
	int len;
	/*  len = get_bits( 3 );  read from cpb buffer */
	
	len = shift_then_get_bits( 3, used_bits, c );  /*  read from cpb buffer */
	used_bits = 3;
	c->sum2_used_bits += used_bits;
	
	c->hclen_tree[k].len = len;  /*  store len */
	/*  c->hclen_tree[k].idx = c->hclen_bl_count[ len ]; where I would have IDX for LL and D */

	if( len ) { /*  bl_count[0]=0 by definition;  */
	    /*  I can check if-then-else this during the codeminimum calc as well */
	    c->hclen_bl_count[ len ] ++ ;
	}
	PRINTF((stderr, "HCLEN: Alphabet symbol %2d, len %d\n", k, len ));
    }
    for(i=0; i < 8; i++) { 
	PRINTF((stderr, "HCLEN bl_count[ %d ]: %d\n", i,  c->hclen_bl_count[ i ] ));
    }    

    /*  -------------------------------------------------------------- */
    /*  hclen code minimum calculation */
    /*  7 cycles */
    /*  */
    c->hclen_next_code[0]=0; /*  unused */
    code = 0;
    for( hbits=1; hbits<8; hbits++) {
	code = (code + c->hclen_bl_count[hbits-1]) << 1;
	c->hclen_next_code[hbits] = code;
    }
    for(i=1; i < 8; i++) { 
	PRINTF((stderr, "HCLEN codeminimums[ %d ]: 0x%-4x, count: %d\n", i,  c->hclen_next_code[ i ], c->hclen_bl_count[i] ));
    }    
    
    /*  -------------------------------------------------------------- */
    /*  Compute the actual code for each of the 0-18 alphabet symbols */
    /*  19 cycles */
    /*   */
    for(i=0; i < 19; i++) { 
	int len = c->hclen_tree[i].len;
	if( len != 0  ) {
	    c->hclen_tree[i].code = reverse( c->hclen_next_code[ len ], len );/* huff code is reversed */
	    c->hclen_next_code[ len ]++;
	}
    }
    for(i=0; i < 19; i++) { 
	if( c->hclen_tree[i].len ) {
	    PRINTF((stderr, "HCLEN: alp %2d code 0x%X len %d\n", i, c->hclen_tree[i].code, c->hclen_tree[i].len ));
	}
    }

    /* --------------------------------------------------------------
       We're done with the hclen table
       now we need to get from DHT the code lengths for the Literal/Len and Dist alphabets
       each length value is 0 to 15 however they are encoded using 1 to 7 bit huff codewords
       found in the hclen_tree[].code above.
       Therefore we retrieve 7 bits each time from CPB.
       then associatively match that 1-7 bits to the hclen_tree[k].code 
       If there is a match, then k is the length of that match (k=0..18)
       Then the IDX is retrieved from next_code[k] and written to the c8t that holds
       the entry for the LZ symbol. Then k is incremented
       Note that k=16,17,18 are the special repeat symbols; I need to fetch
       the argument (bits following 16,17,18) and then loop around until respective
       entries in the c8t are filled.
    */

    /*  -------------------------------------------------------------- */
    /*  clear */
    for(i=1; i < 16; i++) { 
	c->ll_next_code_base[i]=0;
	c->ll_bl_count[i]=0;
	c->dist_next_code_base[i]=0;
	c->dist_bl_count[i]=0;
    }
    
    /* --------------------------------------------------------------
       Read HLIT + HDIST + 258 values from CPB buffer: 
       spec says
       "...The code length repeat codes can cross from HLIT + 257 to the HDIST + 1 code lengths. 
       In other words, all code lengths form a single sequence of HLIT + HDIST + 258 values..."
       
       I had already added 257 and 1 above.
    */
    i = j = 0;
    while( i < hlit+hdist ) {

	int hcode, alp, prev_alp;

	/*  -------------------------------------------------------------- */
	/*  read 7 bits from the cpb buffer, although we may use fewer than 7 */
	/*  according to the decoded value and we must store the actual in used_bits */

	hcode = shift_then_get_bits( 7, used_bits, c ); 
	PRINTF((stderr, "  read from buffer %X\n", hcode ));

	/*  -------------------------------------------------------------- */
	/* look for the matching 1-7 bit pattern in the hclen_tree.
	   struct {
  	     int len;
	     int code;
	   } hclen_tree[19];
	   the match function will return the array index alp (0 to 18)
	   len is the number of valid bits in the variable length code
	   in other words, when alp is valid, then the
	   bottom .len bits of hcode and .code 
	   are matching.  and .len is also used as a shift amount
	   for the next cycle
	*/

	alp = search_hclen_tree( hcode, c );

	if( alp < 0  || alp > 18 ) {
	    PRINTF((stderr, "error: code %X did not match the hclen table\n", alp));
	    used_bits = 0;
	    exit(-1);
	}
	 
	/*  -------------------------------------------------------------- */
	/*  here write the value alp to the c8t array i; that is LL symbol i's length is alp */
	/*  also update the index here. alp=16,17,19 are treated special. */
	/*  Distance lengths follow LL lengths; repeat codes may start in LL and end in D */
	/*  */
	used_bits = c->hclen_tree[alp].len; /*  and the shift amount for the next cycle */
	c->sum2_used_bits += used_bits;

	/*  --------------------------------------------------------------	 */
	/*  now store the length of the LL and DIST codes (0-15 bits) to the c8t array */
	/*  and also the code idx  */

	if( alp < 16 ) {

	    if( i<hlit ) { /*  LL */
		c->ll_tree[i].len = alp;
		c->ll_tree[i].idx = c->ll_bl_count[ alp ];
		PRINTF((stderr, "LL: symbol %3x, len %d, idx %x \n", i, alp, c->ll_tree[i].idx ));
		if( alp ) { /*  bl_count[0]=0 by definition;  */
		    /*  I can check if-then-else this during the codeminimum calc as well */
		    c->ll_bl_count[ alp ] ++ ;
		}
		++ i;
	    }

	    else {  /*  DIST */
		c->dist_tree[j].len = alp;
		c->dist_tree[j].idx = c->dist_bl_count[ alp ];
		PRINTF((stderr, "D: symbol %3x, len %d, idx %x \n", j, alp, c->dist_tree[j].idx ));
		if( alp ) { /*  bl_count[0]=0 by definition;  */
		    c->dist_bl_count[ alp ] ++ ;
		}
		++ j; ++ i;
	    }

	    prev_alp = alp; /*  repeat 16-18 alps need to know what to repeat */
	}

	/*  -------------------------------------------------------------- */
	else if( alp == 16 ) {
	    
	    /*  16: Copy the previous code length 3 - 6 times. */
	    /*  The next 2 bits indicate repeat length */
	    /*  (0 = 3, ... , 3 = 6) */
	    int arg = shift_then_get_bits( 2, used_bits, c );
	    used_bits = 2;
	    c->sum2_used_bits += used_bits;

	    arg = arg+3;
	    PRINTF((stderr, "alp code %d repeat %d times (alp %d)\n", alp, arg, prev_alp ));
	    while( arg-- > 0 ) {

		if( i<hlit ) {  /*  LL */
		    c->ll_tree[i].len = prev_alp;
		    c->ll_tree[i].idx = c->ll_bl_count[ prev_alp ];
		    PRINTF((stderr, "LL: symbol %3x, len %d, idx %x \n", i, prev_alp, c->ll_tree[i].idx ));
		    if( prev_alp ) { /*  bl_count[0]=0 by definition;  */
			/*  I can check if-then-else this during the codeminimum calc as well */
			c->ll_bl_count[ prev_alp ] ++ ;
		    }
		    ++ i;
		}

		else {   /*  DIST */
		    c->dist_tree[j].len = prev_alp;
		    c->dist_tree[j].idx = c->dist_bl_count[ prev_alp ];
		    PRINTF((stderr, "D: symbol %3x, len %d, idx %x \n", j, prev_alp, c->dist_tree[j].idx ));
		    if( prev_alp ) { /*  bl_count[0]=0 by definition;  */
			/*  I can check if-then-else this during the codeminimum calc as well */
			c->dist_bl_count[ prev_alp ] ++ ;
		    }
		    ++ j; ++i;
		}
		/*  prev_alp is the last seen 0-15 alp; no need to update here */
	    }
	}
	
	/*  -------------------------------------------------------------- */
	else if( alp == 17 ) {
	    /*  17: Repeat a code length of 0 for 3 - 10 times. */
	    /*  (3 bits of length) */
	    int arg = shift_then_get_bits( 3, used_bits, c );
	    used_bits = 3;
	    c->sum2_used_bits += used_bits;

	    arg = arg+3;
	    PRINTF((stderr, "alp code %d repeat %d times (alp 0)\n", alp, arg ));	    
	    while( arg-- > 0 ) {

		if( i<hlit ) {  /*  LL */
		    c->ll_tree[i].len = 0;
		    c->ll_tree[i].idx = 0; /*  zero length code is unused */
		    PRINTF((stderr, "LL: symbol %3x, len %d, idx %x \n", i, c->ll_tree[i].len, c->ll_tree[i].idx ));
		    ++ i;
		}

		else {   /*  DIST */
		    c->dist_tree[j].len = 0;
		    c->dist_tree[j].idx = 0; /*  zero length code is unused */
		    PRINTF((stderr, "D: symbol %3x, len %d, idx %x \n", j, c->dist_tree[j].len, c->dist_tree[j].idx ));
		    ++ j; ++i;
		}
		prev_alp = 0; /*  because the last seen alp will be 0 */
	    }
	}
	 
	/*  -------------------------------------------------------------- */
	else if( alp == 18 ) {
	    /*  18: Repeat a code length of 0 for 11 - 138 times */
	    /*  (7 bits of length) */
	    int arg = shift_then_get_bits( 7, used_bits, c );
	    used_bits = 7;
	    c->sum2_used_bits += used_bits;

	    arg = arg+11;
	    PRINTF((stderr, "alp code %d repeat %d times (alp 0)\n", alp, arg ));	    
	    while( arg-- > 0 ) {

		if( i<hlit ) { /*  LL */
		    c->ll_tree[i].len = 0;
		    c->ll_tree[i].idx = 0; /*  zero length code is unused */
		    PRINTF((stderr, "LL: symbol %3x, len %d, idx %x \n", i, c->ll_tree[i].len, c->ll_tree[i].idx ));
		    /*  bl_count[0]=0 by definition;  */
		    ++ i;
		}

		else {   /*  DIST */
		    c->dist_tree[j].len = 0;
		    c->dist_tree[j].idx = 0; /*  zero length code is unused */
		    PRINTF((stderr, "D: symbol %3x, len %d, idx %x \n", j, c->dist_tree[j].len, c->dist_tree[j].idx ));
		    /*  bl_count[0]=0 by definition;  */
		    ++ j; ++i;
		}
		prev_alp = 0;
	    }
	}
    }

    /*  the last code's used bits but not added to the total */
    c->sum_used_bits += used_bits;    

    /*  -------------------------------------------------------------- */
    /*  Calculate codeminimum base (the base code + idx gives the actual code) */
    /*  */
    code = 0;
    for( hbits=1; hbits < 16; hbits++) {
	code = (code + c->ll_bl_count[hbits-1]) << 1;
	c->ll_next_code_base[hbits] = code;
    }
    for(i=1; i < 16; i++) { 
	if( c->ll_bl_count[i] )
	    PRINTF((stderr, "LL codeminimums[ %d ]: 0x%-4x, count: %d\n", i,  c->ll_next_code_base[ i ], c->ll_bl_count[i] ));
    }

    /*  -------------------------------------------------------------- */
    /*  dist and ll codeminimum calcs can go in parallel */
    code = 0;
    for( hbits=1; hbits < 16; hbits++) {
	code = (code + c->dist_bl_count[hbits-1]) << 1;
	c->dist_next_code_base[hbits] = code;
    }
    for(i=1; i < 16; i++) { 
	if( c->dist_bl_count[i] )
	    PRINTF((stderr, "D  codeminimums[ %d ]: 0x%-4x, count: %d\n", i,  c->dist_next_code_base[ i ], c->dist_bl_count[i] ));
    }
    return 0;
}

void dht_decomp_print_zlib_style(ctx *c)
{
    int i;
    fprintf(stderr, "\n\nPrint codes zlib style\n");

    fprintf(stderr, "BL_CODES:\n");
    for(i=0; i < 19; i++) { 
	if( c->hclen_tree[i].len ) {
	    fprintf(stderr, "bl:  %2d  l:  %d  c: 0x%X\n", i,  c->hclen_tree[i].len, c->hclen_tree[i].code );
	}
    }
    /*
      These also show how we would do LZ to Huff mapping in VHDL
      given the LZ symbol i, we lookup it's huff len and idx
      then we lookup the codeminimum base using len, and add idx to it
      we must then reverse the bits to obtain the final huff code for i
    */
    fprintf(stderr, "L_CODES:\n");
    for(i=0; i < 286; i++) { 
	int len = c->ll_tree[i].len;
	int code = c->ll_next_code_base[ len ] + c->ll_tree[i].idx;
	code = reverse( code, len );
	if ( len )
	    fprintf(stderr, "l:  %3d  l:  %d  c: 0x%X\n", i,  len, code );
    }

    fprintf(stderr,"D_CODES:\n");
    for(i=0; i < 30; i++) { 
	int len = c->dist_tree[i].len;
	int code = c->dist_next_code_base[ len ] + c->dist_tree[i].idx;
	code = reverse( code, len );
	if ( len )
	    fprintf(stderr, "d:  %3d  l:  %d  c: 0x%X\n", i,  len, code );
    }
}

static int read_dht(const char *fn, char *dht_buf )
{
    FILE *fp;
    char line[1024];
    char *p,*e;
    long v;
    char hex_byte[10];
    int len=0;
    
    assert( NULL != (fp = fopen(fn,"r")) );

    /* Read the DHT dump from Vishnupriya. DHT starts at line 0090 
       The prior 2 bytes has the DHT length in bits. Example: 
       0080 00 00 00 00 00 00 00 00 00 00 00 00 00 00 03 5b <<DHTLEN at right
       0090 bd ff 80 eb f3 48 ee c7 f1 24 4a 42 12 12 25 39 <<DHT starts at left
    */
    while (NULL != fgets( line, sizeof(line), fp) ) {
	/* line+4 is to skip the 4 digit offset at the beginning of each line */
	for (p=line+4; ; p=e) { 
	    v = strtol (p, &e, 16);  /* hex */
	    if (p == e) break;
	    PRINTF((stderr, "%lx ", v));
	    dht_buf[len++] = v & 0xff;
	}
    }
    PRINTF((stderr, "\n"));

    return len;
}


/* read raw deflate header including the 3 bit block header */
static int read_dht2(const char *fn, char *dht_buf )
{
    FILE *fp;
    char line[1024];
    char *p,*e;
    long v;
    char hex_byte[10];
    int len=0;
    
    assert( NULL != (fp = fopen(fn,"r")) );

    len = fread( dht_buf, 1, 300, fp );

    return len;
}


/* return the number of zero length codes */
static int find_missing_code( char *dht_buf, int dht_len, ctx *c, int pr )
{
    int i,found=0;

    if( pr )
	fprintf(stderr, "\n\nMissing codes:\n");
    
    for(i=0; i < 19; i++) { 
	if( !c->hclen_tree[i].len ) {
	    /* ++found; */
	    if( pr ) fprintf(stderr, "bl:  %2d\n", i );
	}
    }
    for(i=0; i < 286; i++) { 
	if ( !c->ll_tree[i].len ) {
	    ++found;
	    if( pr ) fprintf(stderr, "l:  %3d\n", i );
	}
    }
    for(i=0; i < 30; i++) { 
	if ( !c->dist_tree[i].len ) {
	    ++found;
	    if( pr ) fprintf(stderr, "d:  %3d\n", i );
	}
    }
    return found;
}


static int check_dht( char *dht_buf, int dht_len) 
{
    ctx c;
    int count;
    memset( &c, 0, sizeof(ctx) );
    dht_decomp( dht_buf, dht_len * 8, &c );
    /* look for missing code; assuming all codes must be present */
    count = find_missing_code( dht_buf, dht_len, &c, 0 ) ;
    if( count ) {
	dht_decomp_print_zlib_style(&c);
	find_missing_code( dht_buf, dht_len, &c, 1 );
    }
    fflush(stdout);
    return count;
}


#ifdef _TEST_MAIN
int main(int argc, char **argv)
{
    ctx c;
    char dht_ex[1024];
    int dht_ex_len=0;

    memset( &c, 0, sizeof(ctx) );

#ifdef P9CPB    
    dht_ex_len = read_dht( argv[1], dht_ex );
#else
    dht_ex_len = read_dht2( argv[1], dht_ex );
#endif
    dht_decomp( dht_ex, dht_ex_len * 8, &c );
    dht_decomp_print_zlib_style( &c );
    fprintf(stderr, "Sum of used bits= %d %d\n", c.sum_used_bits, c.sum2_used_bits );
    find_missing_code( dht_ex, dht_ex_len * 8, &c, 1 );
    return 0;
}
#endif


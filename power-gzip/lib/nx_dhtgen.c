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
   Usage: feed the lzcounts in the P9 cpbout memory buffer to dhtgen()
   and get back a dynamic huffman table (DHT) that you can feed to P9
   in the second pass compress.  Optionally, you can manipulate the
   lzcounts with fill_* if you are going to use the DHT as
   canned/cached DHTs that you will repeatedly use.

   Notes:
   unit test
     gcc dhtgen.c -o dhtgen -D_DHTGEN_TEST
   library
     gcc -c dhtgen.c
   regression
     gcc dhtgen.c mtree.c -o dhtgen -D_DHTGEN_TEST -D_RANDOM_TEST
     and run it as ./dhtgen <seed>; see test2.sh as a regression example

   The phrase "Length" is overloaded.  In this file and in few places
   in the Deflate standard Length usually means the number of Huffman
   code bits.  Huffman code lengths are 0 to 15 bits.  Length may also
   refer to an LZ77 length symbol, whose values are 257 to 285.  These
   have no relation to the Huffman code lengths.  The LZ77 Lengths
   will usually appear together with Literal symbols 0 to 255 in the
   same context. In other words LZ77 Literals and Length (LL and
   Lit/Len) values are 0 to 285.
*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <stdint.h>
#include <time.h>
#include <limits.h>

static FILE *dhtgen_log;

#define DHTG_INFO 0x01
#define DHTG_TRC  0x02

static int dhtgen_verbose = 0; /* DHTG_INFO | DHTG_TRC; */

#define dhtg_info  (dhtgen_verbose & DHTG_INFO )
#define dhtg_trace (dhtgen_verbose & DHTG_TRC )

void fill_zero_lzcounts(uint32_t *llhist, uint32_t *dhist, uint32_t val);
void fill_zero_len_dist(uint32_t *llhist, uint32_t *dhist, uint32_t val);

int  dhtgen (
    /* supply the P9 LZ counts here */
    uint32_t  *lhist,
    int num_lhist,
    uint32_t *dhist,
    int num_dhist,
    char *dht,     /* dht returned here; caller is responsible for alloc/free of min 300 bytes */
    int  *dht_num_bytes,     /* number of valid bytes in *dht */
    int  *dht_num_valid_bits,     /* valid bits in the LAST byte; note the value of 0 is encoded as 8 bits */
    int  cpb_header        /* set nonzero if prepending the 16 byte P9 compliant cpbin header with the bit length of dht */
    );

#define outf       stdout

#ifdef _DHTGEN_TEST
#define _DHTGEN_DEBUG
#endif

#define pr_trace(fmt, ...) do { if (dhtg_trace) fprintf (dhtgen_log, "Trace: " fmt, ## __VA_ARGS__); } while (0)
#define pr_info(fmt, ...) do { if (dhtg_info) fprintf (dhtgen_log, "Info: " fmt, ## __VA_ARGS__); } while (0)

#ifdef _DHTGEN_DEBUG
#define ASSERT(X)  assert(X)
#define DBG(X)     do{ X ;} while(0)
#else  /* _DHTGEN_DEBUG */
#define ASSERT(X)  ((void)(X))
#define DBG(X)     do{ ; } while(0)
#endif /* _DHTGEN_DEBUG */

#define NLEN 286
#define NDIS 30
#define CPB_HDR_SZ 16
#define INITIAL_LIMIT (1<<14)

/* These types teach the hardware developer the variable sizes */
typedef uint32_t u1;
typedef uint32_t u3;
typedef uint32_t u5;
typedef uint32_t u7;
typedef uint32_t u9;
typedef uint32_t u10;
typedef uint32_t u16;
typedef uint32_t u24;
typedef uint32_t u32;

typedef struct _leaf_node {
    u9  symbol;   /* a max of NLEN symbols */
    u32 count;
} leaf_node_t;

typedef struct _tree_node {
    leaf_node_t sc; /* tree contains leaves */
    u9 child[2];    /* index of the child */
    u1 is_leaf[2];  /* 1 if child is leaf, 0 if the child is an internal node */
} tree_node_t;

typedef struct _huff_tree {
    tree_node_t *tree; /* binary tree */
    u5 *sym_len;       /* symbol length array */
    u5 max_len;        /* max depth encountered in this tree */
} huff_tree_t;

/* HCLEN is from Deflate spec */
typedef struct _hclen {
    u9 count; /* frequency of the hclen symbol 0 to 18 */
    u5 nbits; /* num code bits, 0 to 7, assigned to the hclen symbol */
    u7 code;  /* huffmanized hclen code  */
} hclen_t;

/* Queue used in the Huffman algorithm */
typedef struct _q
{
    int count, num_elts, head, tail;
    void *arr;
} q_t;

/* *****************************************************************
 * Queuing routines
 * Written as macros to serve different struct types
 * ***************************************************************** */

/* n is the max number of allowed queue elements */
#define init_q( qp, type, n )  \
    ({ (qp)->count = 0; (qp)->num_elts = (n); (qp)->head = (qp)->tail = 0; })

/* adds eltp to the queue and returns the number of queued elements */
#define add_q( qp, type, eltp ) \
    ({	ASSERT( (((qp)->count > 0) && ((qp)->head != (qp)->tail)) ||	\
		(((qp)->count == 0) && ((qp)->head == (qp)->tail)) ||	\
		((qp)->count >= (qp)->num_elts) 	);		\
	((type *)(qp)->arr)[ (qp)->tail ] = *((type *)eltp);		\
	(qp)->tail = ((qp)->tail + 1) % (qp)->num_elts;			\
	++ (qp)->count; })

/* remove eltp from queue and copy in to elt;
   return the number queued elements before delete; 0 is empty and
   and *eltp is invalid) */
#define del_q( qp, type, eltp ) \
    ({ int rc;								\
	if( ((qp)->count == 0) && ((qp)->head == (qp)->tail) ) rc = 0;	\
	else {								\
	    if( !!(eltp) )						\
		*((type *)eltp) = ((type *)(qp)->arr)[ (qp)->head ];	\
	    (qp)->head = ((qp)->head + 1) % (qp)->num_elts;		\
	    rc = (qp)->count; (qp)->count -- ;				\
	} rc;} )

/* inspect head of queue without removing, return non-zero if not empty */
#define headof_q( qp, type, eltp ) \
    ({ int rc;								\
	if( ((qp)->count == 0) && ((qp)->head == (qp)->tail) ) rc = 0;	\
	else {								\
	    *((type *)eltp) = ((type *)(qp)->arr)[ (qp)->head ];	\
	    rc = (qp)->count;						\
	} rc;} )

/* number of occupied elements in the queue */
#define countof_q( qp ) ((qp)->count)

#define print_symbol_list( listp, type, nsym ) do {	\
		int i; ASSERT( !!listp );		\
	for(i=0;i<(nsym);i++) {				\
	    if( !!((type *)(listp))[i].count ) {	\
	      pr_trace ("%3d : %d\n",			\
			((type *)(listp))[i].symbol,	\
			((type *)(listp))[i].count );	\
	    }} pr_trace("\n");} while(0)

static void print_lzcounts(uint32_t *hist, int num, char *kind )
{
    int i;
    ASSERT( hist );
    pr_info ("LZCOUNTS %s\n", kind );
    for(i=0; i<num; i++) {
	if( !!hist[i] ) {
	    pr_info ("%3d : %d\n", i, hist[i] );
	}
    }
    pr_info ("\n\n");
}

/*
   Initialize zero lzcounts to a val.  If DHT will be used once set
   val=0 If DHT will be used repeatedly for different input set val=1
   Setting val != 0 ensures that the DHT contains a symbol for all
   possible symbols at the expense of DHTs being larger
*/

void fill_zero_lzcounts(uint32_t *llhist, uint32_t *dhist, uint32_t val)
{
    int i;
    if( llhist != NULL )
	for(i=0; i<NLEN; i++) {
	    if( ! llhist[i] ) {
		llhist[i] = val;
	    }
	}
    if( dhist != NULL )
	for(i=0; i<NDIS; i++) {
	    if( ! dhist[i] ) {
		dhist[i] = val;
	    }
	}
}

/*
   Handles the nx uninitialized hash table quirk: changes length
   and distance counts to non-zero;  literal counts unchanged
*/
void fill_zero_len_dist(uint32_t *llhist, uint32_t *dhist, uint32_t val)
{
    int i;
    for(i=257; i<NLEN; i++) {
	if( ! llhist[i] ) {
	    llhist[i] = val;
	}
    }
    for(i=0; i<NDIS; i++) {
	if( ! dhist[i] ) {
	    dhist[i] = val;
	}
    }
}

/* *****************************************************************
 * Huffman logic
 * ***************************************************************** */

/*
   returns count of symbols with non-zero count
*/
static int copy_hist_to_leaf_node(leaf_node_t *t, uint32_t *hist, int nsym )
{
    int i,sum=0;
    ASSERT( !!hist && !!t );
    for(i=0; i<nsym; i++) {
	t[i].symbol = i;
	t[i].count = hist[i];
	if( hist[i] ) ++sum ;
    }
    return sum;
}

/*
   Normalize the counts that the sum of counts is less than or equal to
   the limit argument. It sets the max depth to log2(limit) for
   most count distributions (although not a guaranteed upper bound)
*/
static int length_limit(uint32_t *hist, int nsym, int limit)
{
    int i;
    uint64_t divisor, sum=0, sum2=0;
    ASSERT( hist );

    for(i=0; i<nsym; i++) {
	ASSERT( hist[i] >= 0 );
	sum += hist[i];
    }

    divisor = ( sum + limit - 1 ) / limit;

    /* all non-zero values will remain non-zero */
    for(i=0; i<nsym; i++) {
	hist[i] = (hist[i] + divisor - 1) / divisor;
	sum2 += hist[i];
    }
    pr_trace ("sum2 %ld sum %ld %d %ld\n", sum2, sum, nsym, divisor);

    return sum2;
}

/*
   Sort by count; argument is 1 if ascending sort; -1 if descending
   sort.  if qsort_r() is not available, then implement two flavors of
   cmp_count one for ascending and one of descending
*/
static int cmp_count_r(const void *p1, const void *p2, void *arg)
{
    int ascending, p1_count, p2_count;

    ascending = *((int *)arg);
    p1_count = ((leaf_node_t *)p1) -> count;
    p2_count = ((leaf_node_t *)p2) -> count;

    if( ascending ) {
	/* treat zero counts as infinite to eliminate them during sort */
	p1_count = (p1_count) ? p1_count : INT_MAX;
	p2_count = (p2_count) ? p2_count : INT_MAX;
    }

    if( p1_count < p2_count ) {
	/* left to right ascending order */
	return -ascending;
    }
    else if( p1_count == p2_count ) {
	/* smaller symbol goes to left when count are equal; independent of ascent descent */
	if( ((leaf_node_t *)p1) -> symbol < ((leaf_node_t *)p2) -> symbol )
	    return -1;
	else
	    return 1;
    }
    else return ascending;
}

static int cmp_count(const void *p1, const void *p2)
{
    int ascending = 1;
    return cmp_count_r( p1, p2, &ascending);
}

/*
   extract the depth of each symbol
*/
static void tree_walk( huff_tree_t *htree, u9 node, int depth )
{
    u9 left, rite;
    int depth_plus;

    /* saturate at 31 */
    depth_plus = ( depth < 31 ) ? depth+1 : 31;

    left = htree->tree[node].child[0];
    rite = htree->tree[node].child[1];

    if (depth > htree->max_len) htree->max_len = depth;

    ASSERT( left < NLEN && rite < NLEN );

    pr_trace ("tree walk %d count %d left %d right %d\n",
	     htree->tree[node].sc.symbol, htree->tree[node].sc.count, left, rite);

    if( ! htree->tree[node].is_leaf[0] ) {
	/* an internal node node */
	tree_walk( htree, left, depth_plus );
    }
    else {
	htree->sym_len[ left ] = depth;
	pr_trace ("  left child %d, depth %d\n", left, depth );
    }

    if( ! htree->tree[node].is_leaf[1] ) {
	/* an internal node node */
	tree_walk( htree, rite, depth_plus );
    }
    else {
	htree->sym_len[ rite ] = depth;
	pr_trace ("  rite child %d, depth %d\n", rite, depth );
    }
}

/*
   Symbol code lengths assigned by the huffman algorithm
*/
static void print_sym_len(u5 *sym_len, int nsym )
{
    int i;
    ASSERT( sym_len );
    pr_info ("SYMLENS\n" );
    for(i=0; i<nsym; i++) {
	if( !!sym_len[i] ) {
	    pr_info ("%3d : %d\n", i, sym_len[i] );
	}
    }
    pr_info ("\n\n");
}

/*
   Hist are the counts of LZ symbols 0 to len-1.  nz_len is the number
   of nonzero counts in hist; TODO do I really need nz_len?  returns
   the maximum depth found in the huffman tree
*/
static int huffman_tree(uint32_t *hist, int nsym, huff_tree_t *htree)
{
    leaf_node_t leafarr[NLEN]; /* [NLEN]; */
    tree_node_t nodearr[NLEN];
    tree_node_t remaining_node;
    q_t leaf_q, node_q;
    int nz_nsym;

    __builtin_bzero(&remaining_node, sizeof(tree_node_t));

    /* This is where we would copy in the CPBout buffer to dhtgen */

    /* Wikipedia https://en.wikipedia.org/wiki/Huffman_coding 1. Start
       with as many leaves as there are symbols. */
    nz_nsym = copy_hist_to_leaf_node( leafarr, hist, nsym );

    /* 2. Enqueue all leaf nodes into the first queue (by probability
       in increasing order so that the least likely item is in the
       head of the queue). */
    qsort( (void *)leafarr, nsym, sizeof(leaf_node_t), cmp_count );
    DBG( print_symbol_list( leafarr, leaf_node_t, nsym ) );

    /* Leaf contains sorted symbols in ascending order, except the 0
       count symbols are pushed to the back (therefore queue is nz_nsym
       long) */

    /* Setup the leaf queue */
    init_q( &leaf_q, leaf_node_t, nz_nsym );
    leaf_q.arr = leafarr;
    leaf_q.tail = nz_nsym;
    leaf_q.count = nz_nsym;

    /* Setup the tree nodes queue Note: don't need NLEN queue elements
       in principle because there can be at most ceil( NLEN/2 )
       elements in the queue at once.  (imagine all leaf nodes are
       paired in to an internal node and queued; since there are NLEN
       leafs, there can be at most NLEN/2 internal nodes.)  However, I
       need to walk back the tree to find depths; so I may as well use
       an NLEN size queue for that purpose; with NLEN size the queue
       never wraps around; the last node is the tree root, the child
       pointers of each node are the indices in to the queue
    */
    init_q( &node_q, tree_node_t, 2*nz_nsym-1 );
    node_q.arr = nodearr;
    node_q.tail = 0;
    node_q.count = 0;

    /* 3. While there is more than one node in the queues: */
    while( (countof_q( &leaf_q ) + countof_q( &node_q )) > 1 ) {
	leaf_node_t leaf_node;
	tree_node_t tree_node, node[2], new_node;
	int is_leaf[2];
	int idx=0;

	/* 3.1 Dequeue the two nodes with the lowest weight by
	   examining the fronts of both queues. */
	while( idx < 2 ) {
	    int num_leaf, num_node;
	    /* count of queue sizes and the head items */
	    num_leaf = headof_q( &leaf_q, leaf_node_t, &leaf_node );
	    num_node = headof_q( &node_q, tree_node_t, &tree_node );

	    if( num_leaf > 0 && num_node > 0 ) {
		/* Neither queue is empty. When counts are equal
		   break ties by choosing the item in the leaf
		   queue. */
		if( leaf_node.count <= tree_node.sc.count ) {
		    /* remove from leaf queue */
		    node[idx].sc = leaf_node;
		    is_leaf[idx] = 1;
		    ++ idx;
		    del_q( &leaf_q, leaf_node_t, NULL );

		}
		else {
		    /* remove from tree node queue */
		    node[idx] = tree_node;
		    is_leaf[idx] = 0;
		    ++ idx;
		    del_q( &node_q, tree_node_t, NULL );
		}
	    }

	    else if( num_leaf > 0 && num_node == 0 ) {
		/* node queue is empty */
		node[idx].sc = leaf_node;
		is_leaf[idx] = 1;
		++ idx;
		del_q( &leaf_q, leaf_node_t, NULL );
	    }

	    else if( num_leaf == 0 && num_node > 0 ) {
		/* symbol queue is empty */
		node[idx] = tree_node;
		is_leaf[idx] = 0;
		++ idx;
		del_q( &node_q, tree_node_t, NULL );
	    }

	    else { ASSERT(0); }
	}

	pr_trace ( "node %4d %10d %s  node %4d %10d %s\n",
		  node[0].sc.symbol,
		  node[0].sc.count,
		  ((is_leaf[0])? "e" : "i"),
		  node[1].sc.symbol,
		  node[1].sc.count,
		  ((is_leaf[1])? "e" : "i") );

	/*  3.2 Create a new internal node, with the two just-removed
	    nodes as children (either node can be either child) and
	    the sum of their weights as the new weight.  (in the
	    Deflate case left child should be the lexicographically
	    smaller symbol value; but do we really care? We only need
	    code lengths */
	new_node.sc.symbol = node_q.tail;
	new_node.sc.count = node[0].sc.count + node[1].sc.count;
	new_node.child[0] = node[0].sc.symbol;
	new_node.child[1] = node[1].sc.symbol;
	new_node.is_leaf[0] = is_leaf[0];
	new_node.is_leaf[1] = is_leaf[1];

	/*  3.3 Enqueue the new node into the tail of the second
	 *  queue. */
	add_q( &node_q, tree_node_t, &new_node );

	pr_trace ( "ADD node %d count %d left %d right %d\n\n",
		  new_node.sc.symbol,
		  new_node.sc.count,
		  new_node.child[0], /* left */
		  new_node.child[1] ); /* right */

    }

    /* The remaining node is the root node; the tree has now been
     * generated. */
    ASSERT( countof_q( &leaf_q ) == 0 );
    int num_remaining = headof_q( &node_q, tree_node_t, &remaining_node );
    ASSERT( num_remaining == 1 );

    pr_trace ("node %d count %d left %d right %d\n",
		 remaining_node.sc.symbol,
		 remaining_node.sc.count,
		 remaining_node.child[0], /* left */
		 remaining_node.child[1] /* right */
	     );

    /* walk the entire tree starting with the root node */
    htree->tree = nodearr;
    tree_walk( htree, remaining_node.sc.symbol, 1 );

    return htree->max_len;
}

/*
  Run the Huffman algorithm
*/
static void huffmanize( uint32_t *hist, int num_hist, huff_tree_t *tree )
{
    int limit, max_depth, iter;
    iter = 0;
    limit = INITIAL_LIMIT;  /* attempt to limit max length to 15 bits using the log2 estimator;
		       a smaller value here will prevent more cases of depth > 15 but
		       you don't want to eliminate all long codes to prevent the worst
		       case; the iterative procedure will handle this */
    do {
	length_limit( hist, num_hist, limit );
	limit = (limit * 3) / 4;
	/* limit = limit / 2; for the convenience of hardware you can do this */
	tree->max_len = 0;
	max_depth = huffman_tree( hist, num_hist, tree );
	if( max_depth > 15 || iter != 0) {
	    pr_trace ( "LL max depth %d iter %d\n", max_depth, iter ) ;
	    ++iter;
	}
    } while ( max_depth > 15 ); /* if code length exceeds 15 re-run length_limit() */
}

/*
   Hclen codes and their lengths are returned in hclen_tab;
   The .count field is unused in this implementation

   In theory we should run the huffman algorithm on the hclen codes as
   well.  But I will hard code the table here because it is small,
   usually around ~100 bytes which doesn't make a big difference in
   the compressed data size. The hclen lengths I chose are typical for
   many compressed data that I looked at (they jibe with the
   hclen order, except for 16,17,18)

*/

static int encode_hclen(hclen_t *hclen_tab, u5 *sym_len, int num_lsym, int num_dsym )
{
    int i;
    /* Spec says:
       (HCLEN + 4) x 3 bits: code lengths for the code length
       alphabet given just above, in the order: 16, 17, 18,
       0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15

       Hard coded hclen lengths
       3 bit 7 8 9 10
       4 bit 5 6 11 14
       5 bit 0 3 4 12 13 17
       6 bit 2 16 18
       7 bit 1 15

       Huffman codes have MS bit on the right; so I will reverse bits
       in the hex representation
    */
    hclen_t tab[19] = {
	{ 0, 5, 0x3   }, /* 0  00011   11000   */
	{ 0, 7, 0x3f  }, /* 1  0111111 1111110 */
	{ 0, 6, 0xf   }, /* 2  001111  111100  */
	{ 0, 5, 0x13  }, /* 3  10011   11001   */
	{ 0, 5, 0xb   }, /* 4  01011   11010   */
	{ 0, 4, 0x1   }, /* 5  0001    1000    */
	{ 0, 4, 0x9   }, /* 6  1001    1001    */
	{ 0, 3, 0x0   }, /* 7  000     000     */
	{ 0, 3, 0x4   }, /* 8  100     001     */
	{ 0, 3, 0x2   }, /* 9  010     010     */
	{ 0, 3, 0x6   }, /* 10 110     011     */
	{ 0, 4, 0x5   }, /* 11 0101    1010    */
	{ 0, 5, 0x1b  }, /* 12 11011   11011   */
	{ 0, 5, 0x7   }, /* 13 00111   11100   */
	{ 0, 4, 0xd   }, /* 14 1101    1011    */
	{ 0, 7, 0x7f  }, /* 15 1111111 1111111 */
	{ 0, 6, 0x2f  }, /* 16 101111  111101  */
	{ 0, 5, 0x17  }, /* 17 10111   11101   */
	{ 0, 6, 0x1f  }, /* 18 011111  111110  */
    };
    for(i=0; i<19; i++)
	hclen_tab[i] = tab[i];

    return 19;
}

static void print_hclen(hclen_t *hclen_tab)
{
    int i;
    for(i=0;i<19;i++)
	    pr_info( "hclen sym %d nbits %d x'%x\n", i, hclen_tab[i].nbits, hclen_tab[i].code );
}


/*
   macros for writing bits in to the DHT
*/
#define append_bits( data, nbits ) do {		\
	bitbuf |= ((data) << bitptr);		\
	*dht = *dht | (bitbuf & 0xFFL);		\
	bitptr = bitptr + (nbits);		\
	bittotal += (nbits);			\
	pr_trace( "append %x %d\n", (data), nbits );	\
	if ( bitptr > 7 ) {			\
	    /* the byte is full */		\
	    ++ dht; *dht=0;			\
	    bitptr = bitptr - 8;		\
	    bitbuf = bitbuf >> 8;		\
	}                                       \
    } while(0)

#define flush_bits(x) do {			\
	if ( bitptr > 0 ) {			\
	    /* the byte is not empty */		\
	    *dht = *dht | (bitbuf & 0xFFL);	\
	    ++ dht; *dht=0;			\
	}					\
    } while(0)

/*
   RFC 1951 Section 3.2.7

   0 - 15: Represent code lengths of 0 - 15
   16: Copy the previous code length 3 - 6 times.
   The next 2 bits indicate repeat length
   (b'00 -> repeat 3 times, ... , b'11 repeat 6 times)
   For example, the repeating string of 7 fives 5,5,5,5,5,5,5
   may be encoded as 5,16(6)
   For example, the repeating string of 10 fives 5,5,5,5,5,5,5, 5,5,5
   may be encoded as 5,16(6),16(3)
   17: Repeat the length 0 for 3 - 10 times.
   (3 bit repeat value)
   18: Repeat the length 0 for 11 - 138 times
   (7 bits of repeat value)
   For example, 7 sequential 0's
   may be encoded as 17(7), and 138 sequential
   zeros may be encoded as 18(138)

   Returns the number of bits in *dht
*/
static int encode_lengths( char *dht, u5 *sym_len, int num_lsym, int num_dsym )
{
    int i,j, count, cur_idx;
    int state=0, next_state=0;
    u9 cur_len, next_len, new_len;
    hclen_t hclen[19];
    int num_sym_len = num_lsym + num_dsym;
    u16 bitbuf;
    u5 bitptr;
    uint32_t bittotal;
    int nhclen, nhlit, nhdist;
    int horder[19] = { 16, 17, 18,  0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 };

    bittotal = 0;
    bitbuf = 0;
    bitptr = 0;

    nhclen = encode_hclen( hclen, sym_len, num_lsym, num_dsym );

    /*
       write HLIT, HDIST, HCLEN fields in the DHT, Deflate section 3.2.7
       You may have fewer than 257 HLIT symbols but you must still encode
       the first 257 lit/len symbols, and at least 1 dist symbol
    */
    nhlit  = (num_lsym < 257) ? 257 : num_lsym;
    nhdist = (num_dsym < 1) ? 1 : num_dsym;

    num_sym_len = nhlit + nhdist; /* bug fix */

    /* for(i=0; i<(NLEN+NDIS); i++) fprintf(outf, "(%d %d)", (int) i, (int) sym_len[i]);
       fprintf(outf, "\n" );  */

    pr_trace ("hlit hdist hclen %d %d %d\n", nhlit, nhdist, nhclen );
    pr_trace ("hlit hdist hclen %d %d %d encoded\n", nhlit-257, nhdist-1, nhclen-4 );

    *dht = 0;
    append_bits( nhlit-257, 5 );
    append_bits( nhdist-1, 5 );
    append_bits( nhclen-4, 4 );

    /* write hclen symbol's number of bits per Deflate 3.2.7 */
    for(i=0; i<19; i++)
	append_bits( hclen[ horder[i] ].nbits, 3 );

    /* encode the symbols using hclen codes */
    count = 0;
    new_len = next_len = 0;
    cur_len = UINT32_MAX;
    cur_idx = -1;
    for(i=0; i <= num_sym_len; i++) {

	if ( i == num_sym_len )
	    new_len = UINT32_MAX;  /* pushes out the last symbol */
	else
	    new_len = sym_len[i];

	switch( state ) {

	case 0:
	    next_len = new_len;
	    next_state = 1;
	    break;

	case 1:
	    if ( cur_len != new_len ) {
		/* output cur_len */
		pr_trace ( "%2d %d\n", cur_len, cur_idx) ;
		append_bits( hclen[ cur_len ].code, hclen[ cur_len ].nbits );
		next_len = new_len;
		next_state = 1;
	    }
	    else {
		/* cur_len == new_len; repeating symbol len */
		if ( new_len != 0 ) {
		    /* repeating non-zero len */
		    /* output the first non-zero len then go to 16 */
		    pr_trace ( "%2d %d\n", cur_len, cur_idx);
		    append_bits( hclen[ cur_len ].code, hclen[ cur_len ].nbits );
		    next_state = 16;
		    count = 1; /* 1 because 16 excludes the first symbol of the repeats */
		}
		else {
		    /* new_len == 0 */
		    /* repeating zero len */
		    next_state = 17;
		    count = 2;
		}
	    }
	    break;

	case 16:
	    if( new_len != cur_len) {
		/* repeats stopped early */
		if ( count < 3 ) {
		    /* output symbol cur_len count times individually;
		       16 didn't work because needs a minimum three 0's  */
		    for(j=1; j<=count; j++) {
			pr_trace ("%2d %d\n",  cur_len, cur_idx-count+j);
			append_bits( hclen[ cur_len ].code, hclen[ cur_len ].nbits );
		    }
		}
		else {
		    ASSERT( count < 7 );
		    /* output 16 with count (3 to 6 times); count is encoded as 00,01,10,11 */
		    pr_trace ("16 rpt(%d)\n", count);
		    /* first the encoded symbol 16, then it's repeat count argument */
		    append_bits( hclen[ 16 ].code, hclen[ 16 ].nbits );
		    append_bits( count - 3, 2 );
		}
		next_state = 1;
		next_len = new_len;
	    }
	    else {
		/* cur_len continues */
		if ( count == 6)  {
		    /* reached the count limit of 16 but repeats still continue */
		    /* output 16 with count 6 times */
		    pr_trace ( "16 rpt(%d)\n", count);
		    /* first the encoded symbol 16, then it's repeat count argument */
		    append_bits( hclen[ 16 ].code, hclen[ 16 ].nbits );
		    append_bits( count - 3, 2 );
		    count = 1;
		}
		else {
		    count = count + 1;
		}
		next_state = 16; /* to continue with cur_len */
		next_len = new_len;
	    }
	    break;

	case 17:
	    /* encoding a chain of zeros, repeat count is between 1 to 10.
	       17 can encode a zero 3 to 10 times encoded as count-3 in 3 bits */
	    if ( new_len != 0 ) {
		/* zeros have stopped early */
		if ( count < 3 ) {
		    /* output symbol 0 count times individually; 17 needs a minimum three 0's  */
		    for(j=1; j<=count; j++) {
			pr_trace ( "%2d %d\n",  0, cur_idx-count+j);
			append_bits( hclen[ 0 ].code, hclen[ 0 ].nbits );
		    }
		}
		else {
		    /* output 17 with count (3 to 10 times) */
		    pr_trace ("17 rpt(%d)\n", count);
		    append_bits( hclen[ 17 ].code, hclen[ 17 ].nbits );
		    append_bits( count - 3, 3 );
		}
		next_state = 1;
		next_len = new_len;
	    }
	    else {
		/* zeros continue;  (new_len == 0) */
		if ( count == 10)  {
		    /* if more zeros are coming we should encode it with 18 */
		    next_state = 18;
		}
		else {
		    next_state = 17;
		}
		next_len = new_len;
		count = count + 1;
	    }
	    break;

	case 18:
	    /* encoding a chain of zeros with a count between 11 to 138
	       repeat count is encoded as count - 11 in 7 bits  */
	    if (new_len != 0) {
		/* zeros have stopped */
		/* output 18 with count */
		pr_trace ("18 rpt(%d)\n", count);
		append_bits( hclen[ 18 ].code, hclen[ 18 ].nbits );
		append_bits( count - 11, 7 );
		next_state = 1;
		next_len = new_len; /* != 0 */
	    }
	    else {
		/* (new_len == 0) */
		if ( count == 138)  {
		    /* reached the count limit of 18 but zeros still continue */
		    /* output 18 with count */
		    pr_trace ("18 rpt(%d)\n", count);
		    append_bits( hclen[ 18 ].code, hclen[ 18 ].nbits );
		    append_bits( count - 11, 7 );
		    next_state = 17; /* to continue with zeros */
		    next_len = new_len;
		    count = 1;
		}
		else {
		    next_state = 18;
		    next_len = new_len;
		    count = count + 1;
		}
	    }
	    break;
	}
	cur_len = next_len;
	state = next_state;
	cur_idx = cur_idx + 1;
    }
    flush_bits(1); /* push out any remaining byte */
    print_hclen(hclen);

    return bittotal;
}


/*
   In the first 16 bytes of the dht buffer, the dht bit length must be
   declared per P9 spec
*/
static int add_cpb_header(char *dht, int dht_num_bytes, int dht_num_valid_bits)
{
    int invalid_bits;
    int dht_bits;

    memset( dht, 0, CPB_HDR_SZ );
    /* Handle valid bits in the last byte; note that the
       dht_num_valid_bits=0 is encoded as 8 valid bits.  Why?  Because
       it doesn't make sense to have 0 valid bits in the last valid byte
       therefore 0==8 saves 1 bit in the hardware data format.
    */
    invalid_bits = ( dht_num_valid_bits ) ? 8 - dht_num_valid_bits : 0 ;
    dht_bits = 8 * dht_num_bytes - invalid_bits;
    /* bit count is encoded in the rightmost 12 bits; see P9 spec CPB input DHTlen field */
    dht[15] = dht_bits & 0xFF;
    dht[14] = (dht_bits>>8) & 0xFF;
    return dht_num_bytes + CPB_HDR_SZ;
}


/*
   The main library routine producing a DHT in the manner of Deflate
*/
int dhtgen (
    uint32_t *lhist,          /* supply the LZ counts from P9 here */
    int num_lhist,
    uint32_t *dhist,
    int num_dhist,
    char *dht,                 /* dht returned in this buffer; caller is responsible for alloc/free min. 300 bytes */
    int *dht_num_bytes,        /* number of bytes in the dht return buffer */
    int *dht_num_valid_bits,   /* valid bits in the LAST byte; note the value 0 is encoded as 8 bits */
    int cpb_header             /* =1 if prepending the 16byte cpb header that contains the bit length of dht */
    )
{
    huff_tree_t ll_htree, dis_htree;
    u5 lens[NLEN+NDIS];
    char *dhtbuf;
    uint32_t total_bits;

    pr_trace ("num lhist %d dhist %d\n", num_lhist, num_dhist );

    if( cpb_header )
	dhtbuf = dht + CPB_HDR_SZ; /* make room for a header */
    else
	dhtbuf = dht;

    /* Symbol length array to be produced by the Deflate huffman algorithm. */
    memset(lens, 0, sizeof(u5)*(NLEN+NDIS) );

    /* Have 2 separate trees for Length/Literals and Distances
       The two len arrays must be adjacent for the convenience of our coding */
    ll_htree.sym_len = lens;
    /* dis_htree.sym_len = &(lens[num_lhist]); */
    dis_htree.sym_len = &(lens[(num_lhist < 257) ? 257 : num_lhist]);

    print_lzcounts( lhist, num_lhist, "LL" );
    /* The Literal/Length array */
    huffmanize( lhist, num_lhist, &ll_htree );
    print_lzcounts( lhist, num_lhist, "LL Normalized" );
    print_sym_len( ll_htree.sym_len, num_lhist );

    print_lzcounts( dhist, num_dhist, "DIST" );
    /* The Distance array */
    if( num_dhist > 1 ) {
	huffmanize( dhist, num_dhist, &dis_htree );
    }
    else {
	pr_trace ( "bypass huffmanize for distances; num dhist=%d\n", num_dhist);
#ifndef HDIST_FORCE
	/* if no distances then code length = 0 and num dist = 1
	   if one distance then code length = 1 and num dist = 1 */
	/* dis_htree.sym_len[0] = num_dhist ? 1 : 0; */
	dis_htree.sym_len[0] = 1;
	num_dhist = 1;
	/* Note: when no distances are in the counts, it's all literals.
	   Deflate says the following: "If only one distance code is
	   used, it is encoded using one bit, not zero bits; in this
	   case there is a single code length of one, with one unused
	   code.  One distance code of zero bits means that there are
	   no distance codes used at all (the data is all literals)."
	*/
	dhist[0] = 1;
	pr_trace ( "1 distance with length 1; num dhist=%d\n", num_dhist);
#else
	/* force a pair of distances when num distances = 1 */
	num_dhist = 2;
	dis_htree.sym_len[0] = 1;
	dis_htree.sym_len[1] = 1;
	dhist[0] = 1;
	dhist[1] = 1;
	pr_trace ( "2 distance with length 1; num dhist=%d\n", num_dhist);
#endif
    }
    print_lzcounts( dhist, num_dhist, "DIST Normalized" );
    print_sym_len( dis_htree.sym_len, num_dhist );

    /* Now LL and Dist lengths are compacted first then encoded with HCLEN codes  */
    total_bits = encode_lengths( dhtbuf, lens, num_lhist, num_dhist );
    pr_info("total bits %d\n", total_bits);

    /* placeholders */
    *dht_num_bytes = (total_bits + 7) / 8 ;
    *dht_num_valid_bits = total_bits % 8;

    /* add a P9 cpbin header indicating the number of dht bits */
    if( cpb_header ) {
	*dht_num_bytes = add_cpb_header( dht, *dht_num_bytes, *dht_num_valid_bits );
    }

    pr_info( "dht bytes: %d (including cpb hdr), valid bits: %d\n", *dht_num_bytes, *dht_num_valid_bits );

    return 0;
}


#ifdef _DHTGEN_TEST

/* *****************************************************************
 * Unit test utilities  */

/*
   p9 nx has 24 bit saturating counts; ensures that the counts do not
   exceed 2^24 simulating nx like result
*/

static void normalize24(uint32_t *hist, int nsym)
{
    int i,div=1;
    ASSERT( hist );
    /* divide by two repeatedly until all elements
       are less than 1<<24 */
    while( div ) {
	div = 0;
	for(i=0; i<nsym; i++) {
	    if( hist[i] > (1<<24) ) {
		++div;
		break;
	    }
	}
	if( div ) {
	    for(i=0; i<nsym; i++) {
		ASSERT( hist[i] >= 0 );
		hist[i] = (hist[i] + 1) / 2;
	    }
	}
    }
}


#ifdef _RANDOM_TEST  /* for regression only, supplies random lzcounts and computes dht */

/* make random lz history for testing */
extern void make_lzcount(int num_sym, uint32_t *hist, int *num_hist, int *num_nz, unsigned int seed);

/*
   unit test and usage example
*/
int main(int argc, char **argv)
{
    unsigned int i, ret;
    uint32_t lhist[NLEN];
    uint32_t dhist[NDIS];
    int num_nz_lhist;
    int num_nz_dhist;
    int num_lhist;
    int num_dhist;
    int fill;
    int cpb_num_bytes;
    char dht[512];
    int dht_num_bytes=512, dht_num_valid_bits=0;
    FILE *fp;
    char fname[] = "./dht.bin";

    dhtgen_log = stdout;

    /*random seed from command line argument */
    srand( atoi( argv[1] )  );

    /* make random literals and lengths and distances */
    make_lzcount( NLEN, lhist, &num_lhist, &num_nz_lhist, rand() );
    make_lzcount( NDIS, dhist, &num_dhist, &num_nz_dhist, rand() );

    /* Simulate P9nx cpbout */
    normalize24( lhist, num_lhist );
    normalize24( dhist, num_dhist );

    /* You may change zero counts to cover all symbols or all length
       and distances.  Canned DHTs which will be reused should change
       **all** counts to non-zero.  Changing only len_dist is needed
       if you want to avoid the occasional symbol not found error in
       the 2nd pass. Depends on your use case and algorithms
    */
    fill = rand() % 3;  /* simulate */
    switch ( fill ) {
    case 0:
	fill_zero_lzcounts (lhist, dhist, 1) ;
	num_lhist = NLEN;
	num_dhist = NDIS;
	break;
    case 1:
	fill_zero_len_dist (lhist, dhist, 1);
	num_lhist = NLEN;
	num_dhist = NDIS;
	break;
    case 2: /* don't mess with the zero counts */; break;
    }

    /*
       Usage example for dhtgen.  You should copy the buffer "dht" to
       your CPBinput next.
    */

    ret = dhtgen (
	lhist,                     /* lz counts; caller is responsible for alloc/free */
	num_lhist,
	dhist,
	num_dhist,
	dht,                       /* dht returned in this buffer; caller responsible for alloc/free min 300 bytes */
	                           /* reserve 16 bytes for the P9 header */
	&dht_num_bytes,            /* number of bytes in the dht return buffer, starting at dht+16 */
	&dht_num_valid_bits,       /* valid bits in the LAST byte; note the value 0 is encoded as 8 bits */
	1
	);

    if( fname ) {
	/* file for the DHT binary output */
	if( NULL == ( fp = fopen( fname, "w" )) ) {
	    fprintf( stderr, "error: cannot open %s\n", fname );
	    return -1;
	}
	if (fwrite( dht, 1, dht_num_bytes, fp) != dht_num_bytes || ferror(fp)) {
	    return -1;
	}
	fclose( fp );
    }

    return 0;
}


#else /* _RANDOM_TEST */


int print_usage(int argc, char **argv)
{
    fprintf( stderr, "usage:\n");
    fprintf( stderr, "%s [-f|-g] <lzcount> <dht.bin> [-s seed]\n", argv[0]);
    fprintf( stderr, "   <Lzcount> contains a 'symbol : count' pair per line of input.\n");
    fprintf( stderr, "   Lit/Len symbols 0..285 must be followed by Distance symbols 0..29.\n");
    fprintf( stderr, "   Missing symbols have a count of 0 by default.\n");
    fprintf( stderr, "   The optional -f changes all 0 value counts to 1.\n");
    fprintf( stderr, "   The optional -g changes Lengths 257-285 and all Distances with 0 counts to 1.\n");
    fprintf( stderr, "   The optional -s seed chooses a verification scenario.\n");
    fprintf( stderr, "   Human readable output is printed to stdout.\n");
    fprintf( stderr, "   Binary output is dumped to <dht.bin>.\n");
    return -1;
}

/*
   read lzcounts from the file fname and write them to the int arrays
   llhist and dhist for Lit/Len and Distance respectively.
   File format is "%d : %d" per line where the 1st value is the LZ symbol
   and the 2nd value is the count. Literal/Lengths are followed by Distances.
   Symbols are expected to be numerically increasing.
*/

int get_lzcounts(char *fname, uint32_t *llhist, uint32_t *dhist, uint32_t *num_llhist, uint32_t *num_dhist )
{
    int i, lz, prev_lz, count, do_ll, nz_dist;
    FILE *lzf;
    char buf[1024];
    if( NULL == ( lzf = fopen( fname, "r" )) ) {
	    fprintf( dhtgen_log, "cannot open %s\n", fname );
	    return 1;
    }
    /* clear */
    for(i=0; i<NLEN; i++) {
	llhist[i] = 0;
    }
    for(i=0; i<NDIS; i++)  {
	dhist[i] = 0;
    }
    prev_lz=0;
    do_ll=1;
    nz_dist=0;
    *num_llhist = 0;
    *num_dhist = 0;

    while( NULL != fgets( buf, 1023, lzf ) ) {
	sscanf( buf, "%d : %d", &lz, &count );
	if( prev_lz > lz ) { /* detect LL to D transition in the file */
	    do_ll = 0;
	}
	ASSERT( (do_ll==1 && lz >= 0 && lz <= 285) || (do_ll==0 && lz >= 0 && lz <= 29 ) );
	prev_lz = lz;
	if( do_ll ) {
	    llhist[ lz ] = count;
	    if( count > 0 )
	      *num_llhist = lz + 1; /* assumes counts are ordered ascending */
	}
	else {
	    dhist[ lz ] = count;
	    if( count > 0 )
	      *num_dhist = lz + 1; /* assumes counts are ordered ascending */
	}
    }
    if( llhist[256] == 0 )   /* The EOB symbol is always present */
	llhist[256] = 1; /* The EOB symbol is always present */
    if( *num_llhist < 257 )
	*num_llhist = 257;

    fclose( lzf );
    return 0;
}


/*
   test by reading from a file
*/
int main(int argc, char **argv)
{
    int ret;
    uint32_t lhist[NLEN];
    uint32_t dhist[NDIS];

    int c;
    extern char *optarg;
    extern int optind;
    char *lzcount_fn=NULL, *dhtbin_fn=NULL;
    int fflag=0, gflag=0, sflag=0, errflag=0, fidx=0;
    uint32_t num_lhist, num_dhist;
    char dht[512]; /* max dht possible size is 288 bytes or less */
    char cpbtxt[1024]; /* cpbtxt is dht in ascii is about dht_num_bytes/16 * 33 + 33 */
    int dht_num_bytes=512, dht_num_valid_bits=0;

    FILE *fp;
    char fname[] = "./dht.bin";

    dhtgen_log = stdout;

    while ((c = getopt(argc, argv, "fgs:")) != -1) {
	/* parse command line switches */
	switch(c) {
	case 'f':
	    fflag = 1;
	    break;
	case 'g':
	    gflag = 1;
	    break;
	case 's':
	    sflag = atoi(optarg);
	    break;
	case '?':
	default:
	    errflag = 1;
	    break;
	}
    }

    while (optind < argc && fidx < 2) {
	/* Save the two file names on the command line */
	if (fidx==0) lzcount_fn = argv[optind];
	if (fidx==1) dhtbin_fn = argv[optind];
	++fidx;
	++optind;
    };

    if (errflag || (fflag && gflag) || !lzcount_fn || !dhtbin_fn) {
	/* verify arguments */
	print_usage (argc, argv);
	return -1;
    }

    fprintf (dhtgen_log, "fflag: %d gflag: %d sflag: %d in: %s out: %s\n", fflag, gflag, sflag, lzcount_fn, dhtbin_fn);

    if (get_lzcounts (lzcount_fn, lhist, dhist, &num_lhist, &num_dhist)) {
	/* lzcount file parsed */
	return -1;
    }

    if (fflag) {
	/* change all zero counts to one */
	fill_zero_lzcounts (lhist, dhist, 1);
	num_lhist = NLEN;
	num_dhist = NDIS;
    }

    if (gflag) {
	/* change only the len and dist zero counts to one */
	fill_zero_len_dist (lhist, dhist, 1);
	num_lhist = NLEN;
	num_dhist = NDIS;
    }


    /*
       Usage example for dhtgen.  You should copy the buffer "dht" to
       your CPBinput next.
    */

    ret = dhtgen (
	lhist,                     /* lz counts; caller is responsible for alloc/free */
	num_lhist,
	dhist,
	num_dhist,
	dht,                       /* dht returned in this buffer; caller responsible for alloc/free min 316 bytes */
	&dht_num_bytes,            /* number of bytes in the dht return buffer, starting at dht+16 */
	&dht_num_valid_bits,       /* valid bits in the LAST byte; note the value 0 is encoded as 8 bits */
	1                          /* =1 if prepending the 16byte cpb header that contains the bit length of dht */
	);

    if( fname ) {
	/* file for the DHT binary output */
	if( NULL == ( fp = fopen( fname, "w" )) ) {
	    fprintf( dhtgen_log, "error: cannot open %s\n", fname );
	    return -1;
	}
	if (fwrite( dht, 1, dht_num_bytes, fp) != dht_num_bytes || ferror(fp)) {
	    return -1;
	}
	fclose( fp );
    }

    fprintf(dhtgen_log,"bytes: %d valid bits: %d\n", dht_num_bytes, dht_num_valid_bits );
    /* fflush(stderr); */
    /* prints cpbtxt to stdout; to be redirected to a file as needed */
    /* fputs (cpbtxt, stdout); */

    return ret;
}
#endif /* _RANDOM_TEST */


#endif /* _DHTGEN_TEST */

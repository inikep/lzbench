#include "wfLZ.h"

//
// Config
//

// MATCH_DIST cannot exceed 0xffffu
// lowering this improves compression speed but potentially reduces compression ratio
// WFLZ_MAX_MATCH_DIST has a *massive* effect on the speed of Compress()
// WFLZ_MAX_MATCH_DIST_FAST has a minor effect on the speed of CompressFast()
#define WFLZ_MAX_MATCH_DIST      0xffffU
#define WFLZ_MAX_MATCH_DIST_FAST 0xffffU

// number of bytes required = WFLZ_DICTSIZE*sizeof( wfLZ_DictEntry )
// raising this can increase compression ratio slightly, but has a huge impact on the amount of working memory required
// the default value (0xffffU) requires 256KB working memory
#define WFLZ_DICT_SIZE 0xffffU

// when using ChunkCompress() each block will be aligned to this -- makes PS3 SPU transfer convenient
#define WFLZ_CHUNK_PAD 16

#if !defined(SPU) && !defined(WF_LZ_NO_UNALIGNED_ACCESS)
	#define WF_LZ_UNALIGNED_ACCESS
#endif

//
// End Config
//

#define WFLZ_BLOCK_SIZE          4
#define WFLZ_MIN_MATCH_LEN       ( WFLZ_BLOCK_SIZE + 1 )
#define WFLZ_MAX_MATCH_LEN       ( 0xffU-1 ) + WFLZ_MIN_MATCH_LEN

// capped by wfLZ_Block::numLiterals
// this is the maximum length of uncompressible data, if this limit is reached, another block must be emitted
// in practice, raising this helps ratio a very slight amount, but is not worth the cost of making our compression block bigger
#define WFLZ_MAX_SEQUENTIAL_LITERALS 0xffU

typedef uintptr_t ireg_t;
typedef intptr_t  ureg_t;

#define WFLZ_LOG2_8BIT( v )  ( 8 - 90/(((v)/4+14)|1) - 2/((v)/2+1) )
#define WFLZ_LOG2_16BIT( v ) ( 8*((v)>255) + WFLZ_LOG2_8BIT((v) >>8*((v)>255)) ) 
#define WFLZ_LOG2_32BIT( v ) ( 16*((v)>65535L) + WFLZ_LOG2_16BIT((v)*1L >>16*((v)>65535L)) )
#define WFLZ_HASH_SHIFT      ( 31 - WFLZ_LOG2_32BIT( WFLZ_DICT_SIZE ) )

#define WFLZ_HASHPTR( x )    (  ( *( (uint32_t*)x ) * 2654435761U )  >>  ( WFLZ_HASH_SHIFT )  )

typedef struct _wfLZ_Block
{
	uint16_t dist;
	uint8_t  length;
	uint8_t  numLiterals;  // how many literals are there until the next wfLZ_Block
} wfLZ_Block;

typedef struct _wfLZ_Header
{
	char       sig[4];         // this can be WFLZ for a single compressed block, or ZLFW for a block-compressed stream
	uint32_t   compressedSize;
	uint32_t   decompressedSize;
	wfLZ_Block firstBlock;
} wfLZ_Header;

typedef struct _wfLZ_HeaderChunked
{
	char     sig[4];
	uint32_t compressedSize;
	uint32_t decompressedSize;
	uint32_t numChunks;
} wfLZ_HeaderChunked;

typedef struct _wfLZ_ChunkDesc
{
	uint32_t offset;
} wfLZ_ChunkDesc;

typedef struct _wfLZ_DictEntry
{
	const uint8_t* inPos;
} wfLZ_DictEntry;

ureg_t wfLZ_MemCmp( const uint8_t* a, const uint8_t* b, const ureg_t maxLen );
void wfLZ_MemCpy( uint8_t* dst, const uint8_t* src, const uint32_t size );
void wfLZ_MemSet( uint8_t* dst, const uint8_t value, const uint32_t size );
uint32_t wfLZ_RoundUp( const uint32_t value, const uint32_t base ) { return ( value + ( base - 1 ) ) & ~( base - 1 ); }
void wfLZ_EndianSwap16( uint16_t* data ) { *data = ( (*data & 0xFF00) >> 8 ) | ( (*data & 0x00FF) << 8 ); }
void wfLZ_EndianSwap32( uint32_t* data ) { *data = ( (*data & 0xFF000000) >> 24 ) | ( (*data & 0x00FF0000) >> 8 ) | ( (*data & 0x0000FF00) << 8 ) | ( (*data & 0x000000FF) << 24 ); }

#ifndef NULL
	#define NULL 0
#endif

//#define WF_LZ_DBG	// writes compression and decompression logs -- if everything is working, these should match!

#ifdef WF_LZ_DBG
	#include <stdio.h>
	#define WF_LZ_DBG_COMPRESS_FAST_INIT FILE* dbgFh = fopen( "c:/dev/compress-fast.txt", "wb" );
	#define WF_LZ_DBG_COMPRESS_INIT      FILE* dbgFh = fopen( "c:/dev/compress.txt", "wb" );
	#define WF_LZ_DBG_DECOMPRESS_INIT    FILE* dbgFh = fopen( "c:/dev/decompress.txt", "wb" );
	#define WF_LZ_DBG_PRINT( ... )       fprintf( dbgFh, __VA_ARGS__ ); fflush( dbgFh );
	#define WF_LZ_DBG_SHUTDOWN           fclose( dbgFh );

	// may as well collect some stats while we're here...
	uint64_t wfLZ_totalBackTrackDist = 0;
	uint64_t wfLZ_totalBackTrackLength = 0;
	uint64_t wfLZ_numBackTracks = 0;
#else
	#define WF_LZ_DBG_COMPRESS_FAST_INIT
	#define WF_LZ_DBG_COMPRESS_INIT
	#define WF_LZ_DBG_DECOMPRESS_INIT
	#define WF_LZ_DBG_PRINT( ... )
	#define WF_LZ_DBG_SHUTDOWN
#endif

//! wfLZ_GetMaxCompressedSize()

uint32_t wfLZ_GetMaxCompressedSize( const uint32_t inSize )
{
	return
		// header
		sizeof( wfLZ_Header )
		+
		// size of uncompressible data
		(inSize/WFLZ_MAX_SEQUENTIAL_LITERALS + 1) * (WFLZ_MAX_SEQUENTIAL_LITERALS+WFLZ_BLOCK_SIZE)
		+
		// terminating block
		WFLZ_BLOCK_SIZE;
}

//! wfLZ_GetWorkMemSize()

uint32_t wfLZ_GetWorkMemSize()
{
	return (WFLZ_DICT_SIZE+1) * sizeof( wfLZ_DictEntry );
}

//! wfLZ_CompressFast()

uint32_t wfLZ_CompressFast( const uint8_t* WF_RESTRICT const in, const uint32_t inSize, uint8_t* WF_RESTRICT const out, const uint8_t* WF_RESTRICT workMem, const uint32_t swapEndian )
{
	wfLZ_Header header;
	wfLZ_Block* block = &header.firstBlock;
	uint8_t* dst = out + sizeof( wfLZ_Header );
	const uint8_t* WF_RESTRICT src = in;
	ureg_t bytesLeft = (ureg_t)inSize;
	ureg_t numLiterals;
	wfLZ_DictEntry* dict = ( wfLZ_DictEntry* )workMem;

	block->dist = block->length = 0;

	WF_LZ_DBG_COMPRESS_FAST_INIT

	WF_LZ_DBG_PRINT( "wfLZ_CompressFast( %u )\n", inSize );

	// init header
	header.sig[0] = 'W';
	header.sig[1] = 'F';
	header.sig[2] = 'L';
	header.sig[3] = 'Z';
	header.compressedSize = WFLZ_MIN_MATCH_LEN;
	header.decompressedSize = inSize;

	// init dictionary
	wfLZ_MemSet( ( uint8_t* )dict, 0, wfLZ_GetWorkMemSize() );

	// starting literal characters
	{
		const uint8_t* literalsEnd = src + ( WFLZ_MIN_MATCH_LEN > bytesLeft ? bytesLeft : WFLZ_MIN_MATCH_LEN ) ;
		for(
			;
			src != literalsEnd;
			++src, ++dst, --bytesLeft
		)
		{
			if( bytesLeft >= 4 )
			{
				const uint32_t hash = WFLZ_HASHPTR( src );
				dict[ hash ].inPos = src;
			}
			*dst = *src;
			WF_LZ_DBG_PRINT( "  literal [0x%02X] [%c]\n", *src, *src );
		}
		numLiterals = src - in;
	}

	// iterate through input bytes
	while( bytesLeft >= WFLZ_MIN_MATCH_LEN )
	{
		const uint32_t hash = WFLZ_HASHPTR( src );
		const uint8_t** dictEntry = &dict[ hash ].inPos;
		const uint8_t* matchPos = *dictEntry;
		const uint8_t* windowStart = src - WFLZ_MAX_MATCH_DIST_FAST;
		ureg_t matchLength = 0;
		const ureg_t maxMatchLen = WFLZ_MAX_MATCH_LEN > bytesLeft ? bytesLeft : WFLZ_MAX_MATCH_LEN ;

		*dictEntry = src;

		// a match was found, ensure it really is a match and not a hash collision, and determine its length
		if( matchPos != NULL && matchPos >= windowStart )
		{
			matchLength = wfLZ_MemCmp( src, matchPos, maxMatchLen );
		}
		if( matchLength >= WFLZ_MIN_MATCH_LEN )
		{
			const uint32_t matchDist = src - matchPos;

			block->numLiterals = ( uint8_t )numLiterals;
			if( swapEndian != 0 ){ wfLZ_EndianSwap16( &block->dist ); }
			block = ( wfLZ_Block* )dst;
			bytesLeft -= matchLength;
			dst += WFLZ_BLOCK_SIZE;
			src += matchLength;
			block->dist = ( uint16_t )matchDist;
			block->length = ( uint8_t )( matchLength - WFLZ_MIN_MATCH_LEN + 1 );
			numLiterals = 0;

			WF_LZ_DBG_PRINT( "  backtrack [%u] len [%u]\n", matchDist, matchLength );
			#ifdef WF_LZ_DBG
				wfLZ_totalBackTrackDist += matchDist;
				wfLZ_totalBackTrackLength += matchLength;
				++wfLZ_numBackTracks;
			#endif

			header.compressedSize += WFLZ_BLOCK_SIZE;
		}

		// output a literal byte: no entries for this position found, entry is too far away, entry was a hash collision, or the entry did not meet the minimum match length
		else
		{
			// if we've hit the max number of sequential literals, we need to output a compression block header
			if( numLiterals == WFLZ_MAX_SEQUENTIAL_LITERALS )
			{
				block->numLiterals = ( uint8_t )numLiterals;
				if( swapEndian != 0 ){ wfLZ_EndianSwap16( &block->dist ); }
				block = ( wfLZ_Block* )dst;
				dst += WFLZ_BLOCK_SIZE;
				block->dist = block->length = 0;
				numLiterals = 0;
				header.compressedSize += WFLZ_BLOCK_SIZE;
			}

			++numLiterals;
			--bytesLeft;
			WF_LZ_DBG_PRINT( "  literal [0x%02X] [%c]\n", *src, *src );
			*dst++ = *src++;
			++header.compressedSize;
		}
	}
	// output final few bytes as literals, these are not worth compressing
	while( bytesLeft )
	{
		// if we've hit the max number of sequential literals, we need to output a compression block header
		if( numLiterals == WFLZ_MAX_SEQUENTIAL_LITERALS )
		{
			block->numLiterals = ( uint8_t )numLiterals;
			if( swapEndian != 0 ){ wfLZ_EndianSwap16( &block->dist ); }
			block = ( wfLZ_Block* )dst;
			dst += WFLZ_BLOCK_SIZE;
			block->dist = block->length = 0;
			numLiterals = 0;
			header.compressedSize += WFLZ_BLOCK_SIZE;
		}

		++numLiterals;
		--bytesLeft;
		WF_LZ_DBG_PRINT( "  literal [0x%02X] [%c]\n", *src, *src );
		*dst++ = *src++;
		++header.compressedSize;
	}

	// append the 'end' block
	{
		block->numLiterals = ( uint8_t )numLiterals;
		if( swapEndian != 0 ){ wfLZ_EndianSwap16( &block->dist ); }
		block = ( wfLZ_Block* )dst;
		dst += WFLZ_BLOCK_SIZE;
		block->dist = block->length = block->numLiterals = 0;
		header.compressedSize += WFLZ_BLOCK_SIZE;
	}

	// save the header
	if( swapEndian != 0 )
	{
		wfLZ_EndianSwap32( &header.compressedSize );
		wfLZ_EndianSwap32( &header.decompressedSize );
	}
	*( ( wfLZ_Header* )out ) = header;

	WF_LZ_DBG_SHUTDOWN

	return dst - out;
}

//! wfLZ_Compress()

uint32_t wfLZ_Compress( const uint8_t* WF_RESTRICT const in, const uint32_t inSize, uint8_t* WF_RESTRICT const out, const uint8_t* WF_RESTRICT workMem, const uint32_t swapEndian )
{
	wfLZ_Header header;
	wfLZ_Block* block = &header.firstBlock;
	uint8_t* dst = out + sizeof( wfLZ_Header );
	const uint8_t* WF_RESTRICT src = in;
	ureg_t bytesLeft = inSize;
	ureg_t numLiterals = 0;
	wfLZ_DictEntry* dict = ( wfLZ_DictEntry* )workMem;

	block->dist = block->length = 0;

	WF_LZ_DBG_COMPRESS_INIT

	WF_LZ_DBG_PRINT( "wfLZ_Compress( %u )\n", inSize );

	// init header
	header.sig[0] = 'W';
	header.sig[1] = 'F';
	header.sig[2] = 'L';
	header.sig[3] = 'Z';
	header.compressedSize = 0;
	header.decompressedSize = inSize;

	// init dictionary
	wfLZ_MemSet( ( uint8_t* )dict, 0, wfLZ_GetWorkMemSize() );

	// the first bytes are always literal
	{
		const uint8_t* literalsEnd;
		for(
			literalsEnd = src + ( WFLZ_MIN_MATCH_LEN > bytesLeft ? bytesLeft : WFLZ_MIN_MATCH_LEN ) ;
			src != literalsEnd ;
			++dst, ++src, --bytesLeft, ++header.compressedSize, ++numLiterals
		)
		{
			const uint32_t hash = WFLZ_HASHPTR( src );
			dict[ hash ].inPos = src;
			*dst = *src;
			WF_LZ_DBG_PRINT( "  literal [0x%02X] [%c]\n", *src, *src );
		}
	}

	// iterate through input bytes
	while( bytesLeft >= WFLZ_MIN_MATCH_LEN )
	{
		const uint8_t* windowEnd = src - 1;
		const uint8_t* window = windowEnd;
		ureg_t         maxMatchLen;
		ureg_t         bestMatchDist = 0;
		ureg_t         bestMatchLen = 0;
		const uint8_t* windowStart;

		// check hash table for early-fail
		const uint8_t* hashPos;
		const uint32_t hash = WFLZ_HASHPTR( src );
		hashPos = dict[ hash ].inPos;
		dict[ hash ].inPos = src;

		//
		if( hashPos != NULL )
		{
			maxMatchLen = WFLZ_MAX_MATCH_LEN > bytesLeft ? bytesLeft : WFLZ_MAX_MATCH_LEN ;
			windowStart = src - WFLZ_MAX_MATCH_DIST;
			if( windowStart > hashPos ) window = hashPos;
			if( windowStart < in ) windowStart = in;

			// now that we have a search window established for our current position, search it for potential matches
			for( ; window >= windowStart; --window )
			{
				ureg_t matchLen = wfLZ_MemCmp( window, src, maxMatchLen );
				if( matchLen > bestMatchLen )
				{
					bestMatchLen = matchLen;
					bestMatchDist = src - window;
					if( matchLen == maxMatchLen ) { break; }
				}
			}
		}

		// if a match was found, output the corresponding compression block header
		if( bestMatchLen >= WFLZ_MIN_MATCH_LEN )
		{
			block->numLiterals = ( uint8_t )numLiterals;
			if( swapEndian != 0 ){ wfLZ_EndianSwap16( &block->dist ); }
			block = ( wfLZ_Block* )dst;
			bytesLeft -= bestMatchLen;
			dst += WFLZ_BLOCK_SIZE;
			src += bestMatchLen;
			block->dist = ( uint16_t )bestMatchDist;
			block->length = ( uint8_t )( bestMatchLen - WFLZ_MIN_MATCH_LEN + 1 );
			numLiterals = 0;
			WF_LZ_DBG_PRINT( "  backtrack [%u] len [%u]\n", bestMatchDist, bestMatchLen );
			#ifdef WF_LZ_DBG
				wfLZ_totalBackTrackDist += bestMatchDist;
				wfLZ_totalBackTrackLength += bestMatchLen;
				++wfLZ_numBackTracks;
			#endif
			header.compressedSize += WFLZ_BLOCK_SIZE;
		}
		// otherwise, output a literal byte
		else
		{
			// if we've hit the max number of sequential literals, we need to output a compression block header
			if( numLiterals == WFLZ_MAX_SEQUENTIAL_LITERALS )
			{
				block->numLiterals = ( uint8_t )numLiterals;
				if( swapEndian != 0 ){ wfLZ_EndianSwap16( &block->dist ); }
				block = ( wfLZ_Block* )dst;
				dst += WFLZ_BLOCK_SIZE;
				block->dist = block->length = 0;
				numLiterals = 0;
				header.compressedSize += WFLZ_BLOCK_SIZE;
			}

			++numLiterals;
			--bytesLeft;
			WF_LZ_DBG_PRINT( "  literal [0x%02X] [%c]\n", *src, *src );
			*dst++ = *src++;
			++header.compressedSize;
		}
	}

	// output final few bytes as literals, these are not worth compressing
	while( bytesLeft )
	{
		// if we've hit the max number of sequential literals, we need to output a compression block header
		if( numLiterals == WFLZ_MAX_SEQUENTIAL_LITERALS )
		{
			block->numLiterals = ( uint8_t )numLiterals;
			if( swapEndian != 0 ){ wfLZ_EndianSwap16( &block->dist ); }
			block = ( wfLZ_Block* )dst;
			dst += WFLZ_BLOCK_SIZE;
			block->dist = block->length = 0;
			numLiterals = 0;
			header.compressedSize += WFLZ_BLOCK_SIZE;
		}

		++numLiterals;
		--bytesLeft;
		WF_LZ_DBG_PRINT( "  literal [0x%02X] [%c]\n", *src, *src );
		*dst++ = *src++;
		++header.compressedSize;
	}

	// append the 'end' block
	{
		block->numLiterals = ( uint8_t )numLiterals;
		if( swapEndian != 0 ){ wfLZ_EndianSwap16( &block->dist ); }
		block = ( wfLZ_Block* )dst;
		dst += WFLZ_BLOCK_SIZE;
		block->dist = block->length = block->numLiterals = 0;
		header.compressedSize += WFLZ_BLOCK_SIZE;
	}

	// save the header
	if( swapEndian != 0 )
	{
		wfLZ_EndianSwap32( &header.compressedSize );
		wfLZ_EndianSwap32( &header.decompressedSize );
	}
	*( ( wfLZ_Header* )out ) = header;

	WF_LZ_DBG_SHUTDOWN

	return dst - out;
}

//! wfLZ_GetDecompressedSize()

uint32_t wfLZ_GetDecompressedSize( const uint8_t* const in )
{
	wfLZ_Header* header = ( wfLZ_Header* )in;
	if(
		( header->sig[0] == 'W' && header->sig[1] == 'F' && header->sig[2] == 'L' && header->sig[3] == 'Z' )
		||
		( header->sig[0] == 'Z' && header->sig[1] == 'L' && header->sig[2] == 'F' && header->sig[3] == 'W' )
	)
	{
		return header->decompressedSize;
	}
	return 0;
}

//! wfLZ_GetCompressedSize()

uint32_t wfLZ_GetCompressedSize( const uint8_t* const in )
{
	wfLZ_Header* header = ( wfLZ_Header* )in;
	if(
		( header->sig[0] == 'W' && header->sig[1] == 'F' && header->sig[2] == 'L' && header->sig[3] == 'Z' )
		||
		( header->sig[0] == 'Z' && header->sig[1] == 'L' && header->sig[2] == 'F' && header->sig[3] == 'W' )
	)
	{
		return header->compressedSize + sizeof( wfLZ_Header );
	}
	return 0;
}

//! wfLZ_Decompress()

void wfLZ_Decompress( const uint8_t* WF_RESTRICT const in, uint8_t* WF_RESTRICT const out )
{
	wfLZ_Header* header = ( wfLZ_Header* )in;
	uint8_t* WF_RESTRICT dst = out;
	const uint8_t* WF_RESTRICT src = in + sizeof( wfLZ_Header );
	ureg_t numLiterals = header->firstBlock.numLiterals;
	wfLZ_Block* block;
	ureg_t len;
	#ifdef WF_LZ_UNALIGNED_ACCESS
		ureg_t dist;
	#else
		uint16_t dist;
	#endif

	WF_LZ_DBG_DECOMPRESS_INIT
	WF_LZ_DBG_PRINT( "wfLZ_Decompress()\n" );

	for(;;)
	{
		// literals
		if( numLiterals )
		{
			#if 1
				do
				{
					WF_LZ_DBG_PRINT( "  literal [0x%02X] [%c]\n", *src, *src );
					*dst++ = *src++;
					--numLiterals;
				}
				while( numLiterals );
			#else	// good if lots of uncompressible data, but there usually isn't
				wfLZ_MemCpy( dst, src, numLiterals );
				src += numLiterals;
				dst += numLiterals;
			#endif
		}
		else if( dist == 0 && len == 0 ) // we've reached the end of the input
		{
			WF_LZ_DBG_SHUTDOWN
			return;
		}

		// block header
		block = ( wfLZ_Block* )src;
		numLiterals = block->numLiterals;
		#ifndef WF_LZ_UNALIGNED_ACCESS
			( (uint8_t*)&dist )[ 0 ] = ( (uint8_t*)&block->dist )[ 0 ];
			( (uint8_t*)&dist )[ 1 ] = ( (uint8_t*)&block->dist )[ 1 ];
		#else
			dist = block->dist;
		#endif
		len = ( ureg_t )block->length;

		if( len != 0 )
		{
		#ifdef WF_LZ_UNALIGNED_ACCESS
			const uint8_t* WF_RESTRICT cpySrc = dst - dist;
			len += WFLZ_MIN_MATCH_LEN - 1;
			WF_LZ_DBG_PRINT( "  backtrack [%u] len [%u]\n", dist, len );
			if( len <= dist ) // no overlap
			{
				const ureg_t numBlocks = len/sizeof(ureg_t);
				ureg_t i;
				for( i = 0; i != numBlocks; ++i )
				{
					*( (ureg_t*)dst ) = *( (ureg_t*)cpySrc );
					dst    += sizeof(ureg_t);
					cpySrc += sizeof(ureg_t);
				}
				switch( len % sizeof(ureg_t) )
				{
					case 7: *dst++ = *cpySrc++; // compilers smart enough to realize 32 bit ureg_t doesnt need cases 7-4 ?
					case 6: *dst++ = *cpySrc++;
					case 5: *dst++ = *cpySrc++;
					//case 4: *dst++ = *cpySrc++;
					case 4: *( (uint32_t*)dst ) = *( (uint32_t*)cpySrc ); dst += 4; cpySrc += 4; break;
					case 3: *dst++ = *cpySrc++;
					case 2: *dst++ = *cpySrc++;
					case 1: *dst++ = *cpySrc++;
					case 0: break;
				}
			}
			else
			{
				ireg_t n = (len+7) / 8;
				switch( len % 8 )
				{
					case 0: do { *dst++ = *cpySrc++;
					case 7:      *dst++ = *cpySrc++;
					case 6:      *dst++ = *cpySrc++;
					case 5:      *dst++ = *cpySrc++;
					case 4:      *dst++ = *cpySrc++;
					case 3:      *dst++ = *cpySrc++;
					case 2:      *dst++ = *cpySrc++;
					case 1:      *dst++ = *cpySrc++;
					} while( --n > 0 );
				}
			}
		#else
			len += WFLZ_MIN_MATCH_LEN - 1;
			WF_LZ_DBG_PRINT( "  backtrack [%u] len [%u]\n", dist, len );
			const uint8_t* WF_RESTRICT cpySrc = dst - dist;
			ireg_t n = (len+7) / 8;
			switch( len % 8 )
			{
				case 0: do { *dst++ = *cpySrc++;
				case 7:      *dst++ = *cpySrc++;
				case 6:      *dst++ = *cpySrc++;
				case 5:      *dst++ = *cpySrc++;
				case 4:      *dst++ = *cpySrc++;
				case 3:      *dst++ = *cpySrc++;
				case 2:      *dst++ = *cpySrc++;
				case 1:      *dst++ = *cpySrc++;
				} while( --n > 0 );
			}
		#endif
		}
		src += WFLZ_BLOCK_SIZE;
	}
}

//! wfLZ_GetHeaderSize()

uint32_t wfLZ_GetHeaderSize( const uint8_t* const in )
{
	if( in[0] == 'Z' && in[1] == 'L' && in[2] == 'F' && in[3] == 'W' )
	{
		const wfLZ_HeaderChunked* const header = ( const wfLZ_HeaderChunked* )in;
		return sizeof( wfLZ_HeaderChunked ) + sizeof( wfLZ_ChunkDesc )*header->numChunks;
	}
	if( in[0] == 'W' && in[1] == 'F' && in[2] == 'L' && in[3] == 'Z' )
	{
		return sizeof( wfLZ_Header );
	}
	return 0;
}

//! LZC_GetMaxChunkCompressedSize()

uint32_t wfLZ_GetMaxChunkCompressedSize( const uint32_t inSize, const uint32_t blockSize )
{
	const uint32_t numChunks = ( (inSize-1) / blockSize ) + 1;
	return
		wfLZ_RoundUp( wfLZ_GetMaxCompressedSize( blockSize ), WFLZ_CHUNK_PAD )*numChunks
		+
		wfLZ_RoundUp( sizeof( wfLZ_ChunkDesc ) * numChunks, WFLZ_CHUNK_PAD )
		+
		sizeof( wfLZ_HeaderChunked )
	;
}

//! LZC_ChunkCompress()

uint32_t wfLZ_ChunkCompress( uint8_t* in, const uint32_t inSize, const uint32_t blockSize, uint8_t* out, const uint8_t* workMem, const uint32_t swapEndian, const uint32_t useFastCompress )
{
	wfLZ_HeaderChunked* header;
	wfLZ_ChunkDesc* block;
	uint32_t bytesLeft;

	const uint32_t numChunks = ( (inSize-1) / blockSize ) + 1;
	uint32_t totalCompressedSize = 0;

	header = ( wfLZ_HeaderChunked* )out;
	block = ( wfLZ_ChunkDesc* )( out+sizeof( wfLZ_HeaderChunked ) );
	totalCompressedSize += wfLZ_RoundUp( sizeof( wfLZ_HeaderChunked ) + sizeof( wfLZ_ChunkDesc )*numChunks, WFLZ_CHUNK_PAD );
	wfLZ_MemSet( out, 0, totalCompressedSize );
	out += totalCompressedSize;

	for( bytesLeft = inSize; bytesLeft != 0; /**/ )
	{
		const uint32_t decompressedSize = bytesLeft >= blockSize ? blockSize : bytesLeft ;
		uint32_t compressedSize = useFastCompress == 0 ? wfLZ_Compress( in, decompressedSize, out, workMem, swapEndian ) : wfLZ_CompressFast( in, decompressedSize, out, workMem, swapEndian );
		uint32_t pad = wfLZ_RoundUp( compressedSize, WFLZ_CHUNK_PAD ) - compressedSize;
		wfLZ_MemSet( out + compressedSize, 0, pad );
		compressedSize += pad;
		block->offset = totalCompressedSize;

		if( swapEndian != 0 )
		{
			wfLZ_EndianSwap32( &block->offset );
		}

		++block;
		bytesLeft           -= decompressedSize;
		in                  += decompressedSize;
		out                 += compressedSize;
		totalCompressedSize += compressedSize;
	}

	header->sig[0]           = 'Z';
	header->sig[1]           = 'L';
	header->sig[2]           = 'F';
	header->sig[3]           = 'W';
	header->decompressedSize = inSize;
	header->numChunks        = numChunks;
	header->compressedSize   = totalCompressedSize - sizeof( wfLZ_HeaderChunked );
	if( swapEndian != 0 )
	{
		wfLZ_EndianSwap32( &header->decompressedSize );
		wfLZ_EndianSwap32( &header->compressedSize );
		wfLZ_EndianSwap32( &header->numChunks );
	}

	return totalCompressedSize;
}

//! wfLZ_GetNumChunks()

uint32_t wfLZ_GetNumChunks( const uint8_t* const in )
{
	const wfLZ_HeaderChunked* const header = ( const wfLZ_HeaderChunked* const )in;
	if( header->sig[0] == 'Z' && header->sig[1] == 'L' && header->sig[2] == 'F' && header->sig[3] == 'W' )
	{
		return header->numChunks;
	}
	return 0;
}

//! wfLZ_ChunkDecompressCallback()

void wfLZ_ChunkDecompressCallback( uint8_t* in, void( *chunkCallback )( void* ) )
{
	uint32_t chunkIdx;
	//wfLZ_ChunkDesc* chunk;
	wfLZ_HeaderChunked* header = ( wfLZ_HeaderChunked* )in;
	const uint32_t numChunks = header->numChunks;
	in += sizeof( wfLZ_HeaderChunked );

	//chunk = ( wfLZ_ChunkDesc* )in;
	in += sizeof( wfLZ_ChunkDesc ) * numChunks;

	for( chunkIdx = 0; chunkIdx != numChunks; ++chunkIdx )
	{
		chunkCallback( in );
		in += wfLZ_RoundUp( wfLZ_GetCompressedSize( in ), WFLZ_CHUNK_PAD );
	}
}

//! wfLZ_ChunkDecompressLoop()

uint8_t* wfLZ_ChunkDecompressLoop( uint8_t* in, uint32_t** chunkDesc )
{
	wfLZ_HeaderChunked* header = ( wfLZ_HeaderChunked* )in;
	wfLZ_ChunkDesc* chunks = ( wfLZ_ChunkDesc* )( in + sizeof( wfLZ_HeaderChunked ) );
	if( *chunkDesc == NULL )
	{
		*chunkDesc = ( uint32_t* )chunks;
	}
	else
	{
		++*chunkDesc;
		if( *chunkDesc == ( uint32_t* )chunks + header->numChunks ) { return NULL; }
	}
	return in + **chunkDesc;
}

/*!
Utility functions below, not exposed publicly
*/

//! wfLZ_MemCmp()
/*!
Returns the number of sequential matching characters.  Not the same as memcmp!!!
*/

ureg_t wfLZ_MemCmp( const uint8_t* a, const uint8_t* b, const ureg_t maxLen )
{
#ifdef WF_LZ_UNALIGNED_ACCESS
	ureg_t matched = 0;
	const ureg_t numBlocks = maxLen/sizeof(ureg_t);
	ureg_t i;

	for( i = 0; i != numBlocks; ++i )
	{
		if( *((ureg_t*)a) != *((ureg_t*)b) )
		{
			for( i = 0; i != (sizeof(ureg_t)-1); ++i )
			{
				if( *a != *b ) { break; }
				++a;
				++b;
				++matched;
			}
			return matched;
		}
		else
		{
			matched += sizeof(ureg_t);
			a       += sizeof(ureg_t);
			b       += sizeof(ureg_t);
		}
	}
	{
		const ureg_t remain = maxLen % sizeof(ureg_t);
		for( i = 0; i != remain; ++i )
		{
			if( *a != *b ) { break; }
			++a;
			++b;
			++matched;
		}
	}
	return matched;
#else
	ureg_t matched = 0;
	while( *a++ == *b++ && matched < maxLen ) ++matched;
	return matched;
#endif
}

//! wfLZ_MemCpy()

void wfLZ_MemCpy( uint8_t* dst, const uint8_t* src, const uint32_t size )
{
#ifdef WF_LZ_UNALIGNED_ACCESS
	int32_t n = (size+7) / 8;
	switch( size % 8 )
	{
		case 0: do { *dst++ = *src++;
		case 7:      *dst++ = *src++;
		case 6:      *dst++ = *src++;
		case 5:      *dst++ = *src++;
		case 4:      *dst++ = *src++;
		case 3:      *dst++ = *src++;
		case 2:      *dst++ = *src++;
		case 1:      *dst++ = *src++;
		} while(--n > 0);
	}
#else
	uint32_t i;
	for( i = 0; i != size; ++i ) *dst++ = *src++;
#endif
}

//! wfLZ_MemSet()

void wfLZ_MemSet( uint8_t* dst, const uint8_t value, const uint32_t size )
{
	uint32_t i;
	for( i = 0; i != size; ++i ) *dst++ = value;
}
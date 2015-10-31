#pragma once
#ifndef WF_LZ_H
#define WF_LZ_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined _MSC_VER && _MSC_VER < 1600
	typedef signed char        int8_t;
	typedef short              int16_t;
	typedef int                int32_t;
	typedef long long          int64_t;
	typedef unsigned char      uint8_t;
	typedef unsigned short     uint16_t;
	typedef unsigned int       uint32_t;
	typedef unsigned long long uint64_t;
	#if defined __x86_64 || defined __x86_64__ || defined _M_X64
		typedef uint64_t uintptr_t;
		typedef int64_t  intptr_t;
	#else
		typedef uint32_t uintptr_t;
		typedef int32_t  intptr_t;
	#endif
#else
	#include <stdint.h>
#endif
	
#ifndef WF_RESTRICT
	#if defined _MSC_VER || defined __ARMCC_VERSION || defined __GHS_VERSION_NUMBER || defined __GNUC__ || defined __psp2__ || defined __SNC__
		#define WF_RESTRICT __restrict
	#elif __STDC_VERSION__ >= 19901L
		#define WF_RESTRICT restrict
	#else
		#define WF_RESTRICT
	#endif
#endif

//! wfLZ_GetMaxCompressedSize()
/*! Use this to figure out the maximum size for your compression buffer */
extern uint32_t wfLZ_GetMaxCompressedSize( const uint32_t inSize );

//! wfLZ_GetWorkMemSize()
/*! Returns the minimum size for workMem passed to wfLZ_CompressFast and wfLZ_Compress */
extern uint32_t wfLZ_GetWorkMemSize();

//! wfLZ_CompressFast()
/* Returns the size of the compressed data
* CompressFast greatly speeds up compression, but potentially reduces compression ratio.
* swapEndian = 0, compression and decompression are carried out on processors of the same endianness
*/
uint32_t wfLZ_CompressFast( const uint8_t* WF_RESTRICT const in, const uint32_t inSize, uint8_t* WF_RESTRICT const out, const uint8_t* WF_RESTRICT workMem, const uint32_t swapEndian );

//! wfLZ_Compress()
/*! Returns the size of the compressed data
* This is mostly a reference compressor, it is extremely slow.
*/
extern uint32_t wfLZ_Compress( const uint8_t* WF_RESTRICT const in, const uint32_t inSize, uint8_t* WF_RESTRICT const out, const uint8_t* WF_RESTRICT workMem, const uint32_t swapEndian );

//! wfLZ_GetDecompressedSize()
/*! Returns 0 if the data does not appear to be valid WFLZ */
extern uint32_t wfLZ_GetDecompressedSize( const uint8_t* const in );

//! wfLZ_GetCompressedSize()
/*! Returns 0 if the data does not appear to be valid WFLZ */
extern uint32_t wfLZ_GetCompressedSize( const uint8_t* const in );

//! wfLZ_Decompress()
/*! Use wfLZ_GetDecompressedSize to allocate an output buffer of the correct size */
extern void wfLZ_Decompress( const uint8_t* WF_RESTRICT const in, uint8_t* WF_RESTRICT const out );

//! wfLZ_GetHeaderSize()
/*! Returns 0 if the data does not appear to be valid WFLZ */
uint32_t wfLZ_GetHeaderSize( const uint8_t* const in );

/*! Example Usage
	uint8_t* workMem = ( uint8_t* )malloc( wfLZ_GetWorkMemSize() );
	uint8_t* compressed = ( uint8_t* )malloc( wfLZ_GetMaxCompressedSize( decompressedSize ) );
	uint32_t compressedSize = wfLZ_CompressFast( decompressed, decompressedSize, compressed, workMem, 0 );

	....

	uint32_t decompressedSize = wfLZ_GetDecompressedSize( compressed );
	uint8_t* decompressed = ( uint8_t* )malloc( decompressedSize );
	wfLZ_Decompress( compressed, decompressed );
*/

//! Chunk-based Compression
/*!
Chunk compression is an easy way to parallelize decompression.  Input is broken into chunks that can be decompressed independently.
Compression ratio will suffer a little bit.
*/

//! wfLZ_GetMaxChunkCompressedSize()
extern uint32_t wfLZ_GetMaxChunkCompressedSize( const uint32_t inSize, const uint32_t blockSize );

//! wfLZ_ChunkCompress()
/*!
* blockSize must be a multiple of WFLZ_CHUNK_PAD
* useFastCompress = 0, use Compress() instead of CompressFast()
* TODO: Would be nice to have parallelized compression functions for this
*/
extern uint32_t wfLZ_ChunkCompress( uint8_t* in, const uint32_t inSize, const uint32_t blockSize, uint8_t* out, const uint8_t* workMem, const uint32_t swapEndian, const uint32_t useFastCompress );

//! wfLZ_GetNumChunks()
/*!
* Returns 0 if data appears invalid
*/
uint32_t wfLZ_GetNumChunks( const uint8_t* const in );

//! wfLZ_ChunkDecompressCallback()
/*! Example Usage
	void ChunkCB( void* block )
	{
		uint32_t decompressedSize = wfLZ_GetDecompressedSize( compressed );
		uint8_t* decompressed = ( uint8_t* )malloc( decompressedSize );
		wfLZ_Decompress( block, decompressed );
	}
	uint8_t* chunkCompressedData = wfLZ_ChunkCompress( ... );
	wfLZ_ChunkDecompressCallback( chunkCompressedData, ChunkCB );

*/
void wfLZ_ChunkDecompressCallback( uint8_t* in, void( *chunkCallback )( void* ) );

//! wfLZ_ChunkDecompressLoop()
/*! Example Usage
	uint8_t* chunkCompressedData = wfLZ_ChunkCompress( ... );
	uint32_t decompressedSize = wfLZ_GetDecompressedSize( compressed );
	uint8_t* decompressed = ( uint8_t* )malloc( decompressedSize );
	uint32_t* chunk = NULL;
	while( uint8_t* compressedBlock = wfLZ_ChunkDecompressLoop( chunkCompressedData, &chunk ) )
	{
		wfLZ_Decompress( compressedBlock, decompressed );
		const u32 blockSize = wfLZ_GetDecompressedSize( compressedBlock );
		decompressed += blockSize;
	}
*/
uint8_t* wfLZ_ChunkDecompressLoop( uint8_t* in, uint32_t** chunkDesc );

#ifdef __cplusplus
}
#endif

#endif // WF_LZ_H

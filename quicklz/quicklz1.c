// Fast data compression library
// Copyright (C) 2006-2011 Lasse Mikkel Reinhold
// lar@quicklz.com
//
// QuickLZ can be used for free under the GPL 1, 2 or 3 license (where anything 
// released into public must be open source) or under a commercial license if such 
// has been acquired (see http://www.quicklz.com/order.html). The commercial license 
// does not cover derived or ported versions created by third parties under GPL.

// 1.5.0 final

#define QLZ_COMPRESSION_LEVEL 1
#define QLZ_STREAMING_BUFFER 0 

#include "quicklz.h"

#if QLZ150_VERSION_MAJOR != 1 || QLZ150_VERSION_MINOR != 5 || QLZ150_VERSION_REVISION != 0
	#error quicklz.c and quicklz.h have different versions
#endif

#if (defined(__X86__) || defined(__i386__) || defined(i386) || defined(_M_IX86) || defined(__386__) || defined(__x86_64__) || defined(_M_X64))
	#define X86X64
#endif

#define MINOFFSET 2
#define UNCONDITIONAL_MATCHLEN 6
#define UNCOMPRESSED_END 4
#define CWORD_LEN 4

#if QLZ_COMPRESSION_LEVEL == 1 && defined QLZ_PTR_64 && QLZ_STREAMING_BUFFER == 0
	#define OFFSET_BASE source
	#define CAST (ui32)(size_t)
#else
	#define OFFSET_BASE 0
	#define CAST
#endif

int QLZ_GET_SETTING(int setting)
{
	switch (setting)
	{
		case 0: return QLZ_COMPRESSION_LEVEL;
		case 1: return sizeof(qlz150_state_compress);
		case 2: return sizeof(qlz150_state_decompress);
		case 3: return QLZ_STREAMING_BUFFER;
#ifdef QLZ_MEMORY_SAFE
		case 6: return 1;
#else
		case 6: return 0;
#endif
		case 7: return QLZ150_VERSION_MAJOR;
		case 8: return QLZ150_VERSION_MINOR;
		case 9: return QLZ150_VERSION_REVISION;
	}
	return -1;
}

static int same(const unsigned char *src, size_t n)
{
	while(n > 0 && *(src + n) == *src)
		n--;
	return n == 0 ? 1 : 0;
}

static void RESET_TABLE_COMPRESS(qlz150_state_compress *state)
{
	int i;
	for(i = 0; i < QLZ_HASH_VALUES; i++)
	{
#if QLZ_COMPRESSION_LEVEL == 1
		state->hash[i].offset = 0;
#else
		state->hash_counter[i] = 0;
#endif
	}
}

static void RESET_TABLE_DECOMPRESS(qlz150_state_decompress *state)
{
	int i;
	(void)state;
	(void)i;
}

static __inline ui32 HASH_FUNC(ui32 i)
{
	return ((i >> 12) ^ i) & (QLZ_HASH_VALUES - 1);
}

static __inline ui32 FAST_READ(void const *src, ui32 bytes)
{
#ifndef X86X64
	unsigned char *p = (unsigned char*)src;
	switch (bytes)
	{
		case 4:
			return(*p | *(p + 1) << 8 | *(p + 2) << 16 | *(p + 3) << 24);
		case 3: 
			return(*p | *(p + 1) << 8 | *(p + 2) << 16);
		case 2:
			return(*p | *(p + 1) << 8);
		case 1: 
			return(*p);
	}
	return 0;
#else
	if (bytes >= 1 && bytes <= 4)
		return *((ui32*)src);
	else
		return 0;
#endif
}

static __inline ui32 HASHAT(const unsigned char *src)
{
	ui32 fetch, hash;
	fetch = FAST_READ(src, 3);
	hash = HASH_FUNC(fetch);
	return hash;
}

static __inline void FAST_WRITE(ui32 f, void *dst, size_t bytes)
{
#ifndef X86X64
	unsigned char *p = (unsigned char*)dst;

	switch (bytes)
	{
		case 4: 
			*p = (unsigned char)f;
			*(p + 1) = (unsigned char)(f >> 8);
			*(p + 2) = (unsigned char)(f >> 16);
			*(p + 3) = (unsigned char)(f >> 24);
			return;
		case 3:
			*p = (unsigned char)f;
			*(p + 1) = (unsigned char)(f >> 8);
			*(p + 2) = (unsigned char)(f >> 16);
			return;
		case 2:
			*p = (unsigned char)f;
			*(p + 1) = (unsigned char)(f >> 8);
			return;
		case 1:
			*p = (unsigned char)f;
			return;
	}
#else
	switch (bytes)
	{
		case 4: 
			*((ui32*)dst) = f;
			return;
		case 3:
			*((ui32*)dst) = f;
			return;
		case 2:
			*((ui16 *)dst) = (ui16)f;
			return;
		case 1:
			*((unsigned char*)dst) = (unsigned char)f;
			return;
	}
#endif
}


size_t QLZ_SIZE_DECOMPRESSED(const char *source)
{
	ui32 n, r;
	n = (((*source) & 2) == 2) ? 4 : 1;
	r = FAST_READ(source + 1 + n, n);
	r = r & (0xffffffff >> ((4 - n)*8));
	return r;
}

size_t QLZ_SIZE_COMPRESSED(const char *source)
{
	ui32 n, r;
	n = (((*source) & 2) == 2) ? 4 : 1;
	r = FAST_READ(source + 1, n);
	r = r & (0xffffffff >> ((4 - n)*8));
	return r;
}

size_t QLZ_SIZE_HEADER(const char *source)
{
	size_t n = 2*((((*source) & 2) == 2) ? 4 : 1) + 1;
	return n;
}


static __inline void MEMCPY_UP(unsigned char *dst, const unsigned char *src, ui32 n)
{
	// Caution if modifying memcpy_up! Overlap of dst and src must be special handled.
#ifndef X86X64
	unsigned char *end = dst + n;
	while(dst < end)
	{
		*dst = *src;
		dst++;
		src++;
	}
#else
	ui32 f = 0;
	do
	{
		*(ui32 *)(dst + f) = *(ui32 *)(src + f);
		f += MINOFFSET + 1;
	}
	while (f < n);
#endif
}

static __inline void UPDATE_HASH(qlz150_state_decompress *state, const unsigned char *s)
{
	ui32 hash;
	hash = HASHAT(s);
	state->hash[hash].offset = s;
	state->hash_counter[hash] = 1;
	(void)state;
	(void)s;
}

static void UPDATE_HASH_UPTO(qlz150_state_decompress *state, unsigned char **lh, const unsigned char *max)
{
	while(*lh < max)
	{
		(*lh)++;
		UPDATE_HASH(state, *lh);
	}
}

static size_t QLZ_COMPRESS_CORE(const unsigned char *source, unsigned char *destination, size_t size, qlz150_state_compress *state)
{
	const unsigned char *last_byte = source + size - 1;
	const unsigned char *src = source;
	unsigned char *cword_ptr = destination;
	unsigned char *dst = destination + CWORD_LEN;
	ui32 cword_val = 1U << 31;
	const unsigned char *last_matchstart = last_byte - UNCONDITIONAL_MATCHLEN - UNCOMPRESSED_END; 
	ui32 fetch = 0;
	unsigned int lits = 0;

	(void) lits;

	if(src <= last_matchstart)
		fetch = FAST_READ(src, 3);
	
	while(src <= last_matchstart)
	{
		if ((cword_val & 1) == 1)
		{
			// store uncompressed if compression ratio is too low
			if (src > source + (size >> 1) && dst - destination > src - source - ((src - source) >> 5))
				return 0;

			FAST_WRITE((cword_val >> 1) | (1U << 31), cword_ptr, CWORD_LEN);

			cword_ptr = dst;
			dst += CWORD_LEN;
			cword_val = 1U << 31;
			fetch = FAST_READ(src, 3);
		}

		{
			const unsigned char *o;
			ui32 hash, cached;

			hash = HASH_FUNC(fetch);
			cached = fetch ^ state->hash[hash].cache;
			state->hash[hash].cache = fetch;

			o = state->hash[hash].offset + OFFSET_BASE;
			state->hash[hash].offset = CAST(src - OFFSET_BASE);

#ifdef X86X64
			if ((cached & 0xffffff) == 0 && o != OFFSET_BASE && (src - o > MINOFFSET || (src == o + 1 && lits >= 3 && src > source + 3 && same(src - 3, 6))))
			{
				if(cached != 0)
				{
#else
			if (cached == 0 && o != OFFSET_BASE && (src - o > MINOFFSET || (src == o + 1 && lits >= 3 && src > source + 3 && same(src - 3, 6))))
			{
				if (*(o + 3) != *(src + 3))
				{
#endif
					hash <<= 4;
					cword_val = (cword_val >> 1) | (1U << 31);
					FAST_WRITE((3 - 2) | hash, dst, 2);
					src += 3;
					dst += 2;
				}
				else
				{
					const unsigned char *old_src = src;
					size_t matchlen;
					hash <<= 4;

					cword_val = (cword_val >> 1) | (1U << 31);
					src += 4;

					if(*(o + (src - old_src)) == *src)
					{
						src++;
						if(*(o + (src - old_src)) == *src)
						{
							size_t q = last_byte - UNCOMPRESSED_END - (src - 5) + 1;
							size_t remaining = q > 255 ? 255 : q;
							src++;	
							while(*(o + (src - old_src)) == *src && (size_t)(src - old_src) < remaining)
								src++;
						}
					}

					matchlen = src - old_src;
					if (matchlen < 18)
					{
						FAST_WRITE((ui32)(matchlen - 2) | hash, dst, 2);
						dst += 2;
					} 
					else
					{
						FAST_WRITE((ui32)(matchlen << 16) | hash, dst, 3);
						dst += 3;
					}
				}
				fetch = FAST_READ(src, 3);
				lits = 0;
			}
			else
			{
				lits++;
				*dst = *src;
				src++;
				dst++;
				cword_val = (cword_val >> 1);
#ifdef X86X64
				fetch = FAST_READ(src, 3);
#else
				fetch = (fetch >> 8 & 0xffff) | (*(src + 2) << 16);
#endif
			}
		}
	}
	while (src <= last_byte)
	{
		if ((cword_val & 1) == 1)
		{
			FAST_WRITE((cword_val >> 1) | (1U << 31), cword_ptr, CWORD_LEN);
			cword_ptr = dst;
			dst += CWORD_LEN;
			cword_val = 1U << 31;
		}
		if (src <= last_byte - 3)
		{
			ui32 hash, fetch;
			fetch = FAST_READ(src, 3);
			hash = HASH_FUNC(fetch);
			state->hash[hash].offset = CAST(src - OFFSET_BASE);
			state->hash[hash].cache = fetch;
		}
		*dst = *src;
		src++;
		dst++;
		cword_val = (cword_val >> 1);
	}

	while((cword_val & 1) != 1)
		cword_val = (cword_val >> 1);

	FAST_WRITE((cword_val >> 1) | (1U << 31), cword_ptr, CWORD_LEN);

	// min. size must be 9 bytes so that the qlz_size functions can take 9 bytes as argument
	return dst - destination < 9 ? 9 : dst - destination;
}

static size_t QLZ_DECOMPRESS_CORE(const unsigned char *source, unsigned char *destination, size_t size, qlz150_state_decompress *state, const unsigned char *history)
{
	const unsigned char *src = source + QLZ_SIZE_HEADER((const char *)source);
	unsigned char *dst = destination;
	const unsigned char *last_destination_byte = destination + size - 1;
	ui32 cword_val = 1;
	const unsigned char *last_matchstart = last_destination_byte - UNCONDITIONAL_MATCHLEN - UNCOMPRESSED_END;
	unsigned char *last_hashed = destination - 1;
	const unsigned char *last_source_byte = source + QLZ_SIZE_COMPRESSED((const char *)source) - 1;
	static const ui32 bitlut[16] = {4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0};

	(void) last_source_byte;
	(void) last_hashed;
	(void) state;
	(void) history;

	for(;;) 
	{
		ui32 fetch;

		if (cword_val == 1)
		{
#ifdef QLZ_MEMORY_SAFE
			if(src + CWORD_LEN - 1 > last_source_byte)
				return 0;
#endif
			cword_val = FAST_READ(src, CWORD_LEN);
			src += CWORD_LEN;
		}

#ifdef QLZ_MEMORY_SAFE
			if(src + 4 - 1 > last_source_byte)
				return 0;
#endif

		fetch = FAST_READ(src, 4);

		if ((cword_val & 1) == 1)
		{
			ui32 matchlen;
			const unsigned char *offset2;

#if QLZ_COMPRESSION_LEVEL == 1
			ui32 hash;
			cword_val = cword_val >> 1;
			hash = (fetch >> 4) & 0xfff;
			offset2 = (const unsigned char *)(size_t)state->hash[hash].offset;

			if((fetch & 0xf) != 0)
			{
				matchlen = (fetch & 0xf) + 2;
				src += 2;
			}
			else
			{
				matchlen = *(src + 2);
				src += 3;							
			}	

#elif QLZ_COMPRESSION_LEVEL == 2
			ui32 hash;
			unsigned char c;
			cword_val = cword_val >> 1;
			hash = (fetch >> 5) & 0x7ff;
			c = (unsigned char)(fetch & 0x3);
			offset2 = state->hash[hash].offset[c];

			if((fetch & (28)) != 0)
			{
				matchlen = ((fetch >> 2) & 0x7) + 2;
				src += 2;
			}
			else
			{
				matchlen = *(src + 2);
				src += 3;							
			}	

#elif QLZ_COMPRESSION_LEVEL == 3
			ui32 offset;
			cword_val = cword_val >> 1;
			if ((fetch & 3) == 0)
			{
				offset = (fetch & 0xff) >> 2;
				matchlen = 3;
				src++;
			}
			else if ((fetch & 2) == 0)
			{
				offset = (fetch & 0xffff) >> 2;
				matchlen = 3;
				src += 2;
			}
			else if ((fetch & 1) == 0)
			{
				offset = (fetch & 0xffff) >> 6;
				matchlen = ((fetch >> 2) & 15) + 3;
				src += 2;
			}
			else if ((fetch & 127) != 3)
			{
				offset = (fetch >> 7) & 0x1ffff;
				matchlen = ((fetch >> 2) & 0x1f) + 2;
				src += 3;
			}
			else
			{
				offset = (fetch >> 15);
				matchlen = ((fetch >> 7) & 255) + 3;
				src += 4;
			}

			offset2 = dst - offset;
#endif
	
#ifdef QLZ_MEMORY_SAFE
			if(offset2 < history || offset2 > dst - MINOFFSET - 1)
				return 0;

			if(matchlen > (ui32)(last_destination_byte - dst - UNCOMPRESSED_END + 1))
				return 0;
#endif

			MEMCPY_UP(dst, offset2, matchlen);
			dst += matchlen;

#if QLZ_COMPRESSION_LEVEL <= 2
			UPDATE_HASH_UPTO(state, &last_hashed, dst - matchlen);
			last_hashed = dst - 1;
#endif
		}
		else
		{
			if (dst < last_matchstart)
			{
				unsigned int n = bitlut[cword_val & 0xf];
#ifdef X86X64
				*(ui32 *)dst = *(ui32 *)src;
#else
				MEMCPY_UP(dst, src, 4);
#endif
				cword_val = cword_val >> n;
				dst += n;
				src += n;
#if QLZ_COMPRESSION_LEVEL <= 2
				UPDATE_HASH_UPTO(state, &last_hashed, dst - 3);		
#endif
			}
			else
			{			
				while(dst <= last_destination_byte)
				{
					if (cword_val == 1)
					{
						src += CWORD_LEN;
						cword_val = 1U << 31;
					}
#ifdef QLZ_MEMORY_SAFE
					if(src >= last_source_byte + 1)
						return 0;
#endif
					*dst = *src;
					dst++;
					src++;
					cword_val = cword_val >> 1;
				}

#if QLZ_COMPRESSION_LEVEL <= 2
				UPDATE_HASH_UPTO(state, &last_hashed, last_destination_byte - 3); // todo, use constant
#endif
				return size;
			}

		}
	}
}

size_t QLZ_COMPRESS(const void *source, char *destination, size_t size, qlz150_state_compress *state)
{
	size_t r;
	ui32 compressed;
	size_t base;

	if(size == 0 || size > 0xffffffff - 400)
		return 0;

	if(size < 216)
		base = 3;
	else
		base = 9;

#if QLZ_STREAMING_BUFFER > 0
	if (state->stream_counter + size - 1 >= QLZ_STREAMING_BUFFER)
#endif
	{
		RESET_TABLE_COMPRESS(state);
		r = base + QLZ_COMPRESS_CORE((const unsigned char *)source, (unsigned char*)destination + base, size, state);
#if QLZ_STREAMING_BUFFER > 0
		RESET_TABLE_COMPRESS(state);
#endif
		if(r == base)
		{
			memcpy(destination + base, source, size);
			r = size + base;
			compressed = 0;
		}
		else
		{
			compressed = 1;
		}
		state->stream_counter = 0;
	}
#if QLZ_STREAMING_BUFFER > 0
	else
	{
		unsigned char *src = state->stream_buffer + state->stream_counter;

		memcpy(src, source, size);
		r = base + QLZ_COMPRESS_CORE(src, (unsigned char*)destination + base, size, state);

 		if(r == base)
		{
			memcpy(destination + base, src, size);
			r = size + base;
			compressed = 0;
			RESET_TABLE_COMPRESS(state);
		}
		else
		{
			compressed = 1;
		}
		state->stream_counter += size;
	}
#endif
	if(base == 3)
	{
		*destination = (unsigned char)(0 | compressed);
		*(destination + 1) = (unsigned char)r;
		*(destination + 2) = (unsigned char)size;
	}
	else
	{
		*destination = (unsigned char)(2 | compressed);
		FAST_WRITE((ui32)r, destination + 1, 4);
		FAST_WRITE((ui32)size, destination + 5, 4);
	}
	
	*destination |= (QLZ_COMPRESSION_LEVEL << 2);
	*destination |= (1 << 6);
	*destination |= ((QLZ_STREAMING_BUFFER == 0 ? 0 : (QLZ_STREAMING_BUFFER == 100000 ? 1 : (QLZ_STREAMING_BUFFER == 1000000 ? 2 : 3))) << 4);

// 76543210
// 01SSLLHC

	return r;
}

size_t QLZ_DECOMPRESS(const char *source, void *destination, qlz150_state_decompress *state)
{
	size_t dsiz = QLZ_SIZE_DECOMPRESSED(source);

#if QLZ_STREAMING_BUFFER > 0
	if (state->stream_counter + QLZ_SIZE_DECOMPRESSED(source) - 1 >= QLZ_STREAMING_BUFFER) 
#endif
	{
		if((*source & 1) == 1)
		{
			RESET_TABLE_DECOMPRESS(state);
			dsiz = QLZ_DECOMPRESS_CORE((const unsigned char *)source, (unsigned char *)destination, dsiz, state, (const unsigned char *)destination);
		}
		else
		{
			memcpy(destination, source + QLZ_SIZE_HEADER(source), dsiz);
		}
		state->stream_counter = 0;
		RESET_TABLE_DECOMPRESS(state);
	}
#if QLZ_STREAMING_BUFFER > 0
	else
	{
		unsigned char *dst = state->stream_buffer + state->stream_counter;
		if((*source & 1) == 1)
		{
			dsiz = QLZ_DECOMPRESS_CORE((const unsigned char *)source, dst, dsiz, state, (const unsigned char *)state->stream_buffer);
		}
		else
		{
			memcpy(dst, source + QLZ_SIZE_HEADER(source), dsiz);
			RESET_TABLE_DECOMPRESS(state);
		}
		memcpy(destination, dst, dsiz);
		state->stream_counter += dsiz;
	}
#endif
	return dsiz;
}


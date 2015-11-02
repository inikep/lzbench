// crush.cpp
// Written and placed in the public domain by Ilya Muravyov
//

#ifdef _MSC_VER
#define _CRT_SECECURE_NO_WARNINGS
#define _CRT_DISABLE_PERFCRIT_LOCKS
#else
#define _FILE_OFFSET_BITS 64
#define _ftelli64 ftello64
#endif
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

namespace crush
{

// Bit I/O
//

uint8_t* g_inbuf;
uint8_t* g_outbuf;
int g_inbuf_pos;
int g_outbuf_pos;
int bit_buf;
int bit_count;

inline void init_bits(uint8_t* inbuf, uint8_t* outbuf)
{
	bit_count=bit_buf=g_inbuf_pos=g_outbuf_pos=0;
	g_inbuf = inbuf;
	g_outbuf = outbuf;
}

inline void put_bits(int n, int x)
{
	bit_buf|=x<<bit_count;
	bit_count+=n;
	while (bit_count>=8)
	{
		g_outbuf[g_outbuf_pos++] = bit_buf;
		bit_buf>>=8;
		bit_count-=8;
	}
}

inline void flush_bits()
{
	put_bits(7, 0);
	bit_count=bit_buf=0;
}

inline int get_bits(int n)
{
	while (bit_count<n)
	{
		bit_buf|=g_inbuf[g_inbuf_pos++]<<bit_count;
		bit_count+=8;
	}
	const int x=bit_buf&((1<<n)-1);
	bit_buf>>=n;
	bit_count-=n;
	return x;
}

// LZ77
//

const int W_BITS=21; // Window size (17..23)
const int W_SIZE=1<<W_BITS;
const int W_MASK=W_SIZE-1;
const int SLOT_BITS=4;
const int NUM_SLOTS=1<<SLOT_BITS;

const int A_BITS=2; // 1 xx
const int B_BITS=2; // 01 xx
const int C_BITS=2; // 001 xx
const int D_BITS=3; // 0001 xxx
const int E_BITS=5; // 00001 xxxxx
const int F_BITS=9; // 00000 xxxxxxxxx
const int A=1<<A_BITS;
const int B=(1<<B_BITS)+A;
const int C=(1<<C_BITS)+B;
const int D=(1<<D_BITS)+C;
const int E=(1<<E_BITS)+D;
const int F=(1<<F_BITS)+E;
const int MIN_MATCH=3;
const int MAX_MATCH=(F-1)+MIN_MATCH;

const int TOO_FAR=1<<16;

const int HASH1_LEN=MIN_MATCH;
const int HASH2_LEN=MIN_MATCH+1;
const int HASH1_BITS=21;
const int HASH2_BITS=24;
const int HASH1_SIZE=1<<HASH1_BITS;
const int HASH2_SIZE=1<<HASH2_BITS;
const int HASH1_MASK=HASH1_SIZE-1;
const int HASH2_MASK=HASH2_SIZE-1;
const int HASH1_SHIFT=(HASH1_BITS+(HASH1_LEN-1))/HASH1_LEN;
const int HASH2_SHIFT=(HASH2_BITS+(HASH2_LEN-1))/HASH2_LEN;

inline int update_hash1(int h, int c)
{
	return ((h<<HASH1_SHIFT)+c)&HASH1_MASK;
}

inline int update_hash2(int h, int c)
{
	return ((h<<HASH2_SHIFT)+c)&HASH2_MASK;
}

inline int get_min(int a, int b)
{
	return a<b?a:b;
}

inline int get_max(int a, int b)
{
	return a>b?a:b;
}

inline int get_penalty(int a, int b)
{
	int p=0;
	while (a>b)
	{
		a>>=3;
		++p;
	}
	return p;
}

uint32_t compress(int level, uint8_t* buf, int size, uint8_t* outbuf)
{
	static int head[HASH1_SIZE+HASH2_SIZE];
	static int prev[W_SIZE];

	const int max_chain[]={4, 256, 1<<12};

	{
		for (int i=0; i<HASH1_SIZE+HASH2_SIZE; ++i)
			head[i]=-1;

		int h1=0;
		int h2=0;
		for (int i=0; i<HASH1_LEN; ++i)
			h1=update_hash1(h1, buf[i]);
		for (int i=0; i<HASH2_LEN; ++i)
			h2=update_hash2(h2, buf[i]);

		init_bits(NULL, outbuf);

		int p=0;
		while (p<size)
		{
			int len=MIN_MATCH-1;
			int offset=W_SIZE;

			const int max_match=get_min(MAX_MATCH, size-p);
			const int limit=get_max(p-W_SIZE, 0);

			if (head[h1]>=limit)
			{
				int s=head[h1];
				if (buf[s]==buf[p])
				{
					int l=0;
					while (++l<max_match)
						if (buf[s+l]!=buf[p+l])
							break;
					if (l>len)
					{
						len=l;
						offset=p-s;
					}
				}
			}

			if (len<MAX_MATCH)
			{
				int chain_len=max_chain[level];
				int s=head[h2+HASH1_SIZE];

				while ((chain_len--!=0)&&(s>=limit))
				{
					if ((buf[s+len]==buf[p+len])&&(buf[s]==buf[p]))
					{	
						int l=0;
						while (++l<max_match)
							if (buf[s+l]!=buf[p+l])
								break;
						if (l>len+get_penalty((p-s)>>4, offset))
						{
							len=l;
							offset=p-s;
						}
						if (l==max_match)
							break;
					}
					s=prev[s&W_MASK];
				}
			}

			if ((len==MIN_MATCH)&&(offset>TOO_FAR))
				len=0;

			if ((level>=2)&&(len>=MIN_MATCH)&&(len<max_match))
			{
				const int next_p=p+1;
				const int max_lazy=get_min(len+4, max_match);

				int chain_len=max_chain[level];
				int s=head[update_hash2(h2, buf[next_p+(HASH2_LEN-1)])+HASH1_SIZE];

				while ((chain_len--!=0)&&(s>=limit))
				{
					if ((buf[s+len]==buf[next_p+len])&&(buf[s]==buf[next_p]))
					{
						int l=0;
						while (++l<max_lazy)
							if (buf[s+l]!=buf[next_p+l])
								break;
						if (l>len+get_penalty(next_p-s, offset))
						{
							len=0;
							break;
						}
						if (l==max_lazy)
							break;
					}
					s=prev[s&W_MASK];
				}
			}

			if (len>=MIN_MATCH) // Match
			{
				put_bits(1, 1);

				const int l=len-MIN_MATCH;
				if (l<A)
				{
					put_bits(1, 1); // 1
					put_bits(A_BITS, l);
				}
				else if (l<B)
				{
					put_bits(2, 1<<1); // 01
					put_bits(B_BITS, l-A);
				}
				else if (l<C)
				{
					put_bits(3, 1<<2); // 001
					put_bits(C_BITS, l-B);
				}
				else if (l<D)
				{
					put_bits(4, 1<<3); // 0001
					put_bits(D_BITS, l-C);
				}
				else if (l<E)
				{
					put_bits(5, 1<<4); // 00001
					put_bits(E_BITS, l-D);
				}
				else
				{
					put_bits(5, 0); // 00000
					put_bits(F_BITS, l-E);
				}

				--offset;
				int log=W_BITS-NUM_SLOTS;
				while (offset>=(2<<log))
					++log;
				put_bits(SLOT_BITS, log-(W_BITS-NUM_SLOTS));
				if (log>(W_BITS-NUM_SLOTS))
					put_bits(log, offset-(1<<log));
				else
					put_bits(W_BITS-(NUM_SLOTS-1), offset);
			}
			else // Literal
			{
				len=1;
				put_bits(9, buf[p]<<1); // 0 xxxxxxxx
			}

			while (len--!=0) // Insert new strings
			{
				head[h1]=p;
				prev[p&W_MASK]=head[h2+HASH1_SIZE];
				head[h2+HASH1_SIZE]=p;
				++p;
				h1=update_hash1(h1, buf[p+(HASH1_LEN-1)]);
				h2=update_hash2(h2, buf[p+(HASH2_LEN-1)]);
			}
		}

		flush_bits();
	}

	return g_outbuf_pos;
}

uint32_t decompress(uint8_t* inbuf, uint8_t* outbuf, int outsize)
{
    if ((outsize<1))
    {
        fprintf(stderr, "File corrupted: size=%d\n", outsize);
        return 0;
    }

    init_bits(inbuf, NULL);

    int p=0;
    while (p<outsize)
    {
        if (get_bits(1))
        {
            int len;
            if (get_bits(1))
                len=get_bits(A_BITS);
            else if (get_bits(1))
                len=get_bits(B_BITS)+A;
            else if (get_bits(1))
                len=get_bits(C_BITS)+B;
            else if (get_bits(1))
                len=get_bits(D_BITS)+C;
            else if (get_bits(1))
                len=get_bits(E_BITS)+D;
            else
                len=get_bits(F_BITS)+E;

            const int log=get_bits(SLOT_BITS)+(W_BITS-NUM_SLOTS);
            int s=~(log>(W_BITS-NUM_SLOTS)
                ?get_bits(log)+(1<<log)
                :get_bits(W_BITS-(NUM_SLOTS-1)))+p;
            if (s<0)
            {
                fprintf(stderr, "File corrupted: s=%d p=%d outsize=%d\n", s, p, outsize);
                return 0;
            }

            outbuf[p++]=outbuf[s++];
            outbuf[p++]=outbuf[s++];
            outbuf[p++]=outbuf[s++];
            while (len--!=0)
                outbuf[p++]=outbuf[s++];
        }
        else
            outbuf[p++]=get_bits(8);
    }

    return p;
}

} // namespace crush

/*
int main(int argc, char* argv[])
{
	using namespace crush;

	const clock_t start=clock();

	if (argc!=4)
	{
		fprintf(stderr,
			"CRUSH by Ilya Muravyov, v1.00\n"
			"Usage: CRUSH command infile outfile\n"
			"Commands:\n"
			"  c[f,x] Compress (Fast..Max)\n"
			"  d      Decompress\n");
		exit(1);
	}

	FILE* in=fopen(argv[2], "rb");
	if (!in)
	{
		perror(argv[2]);
		exit(1);
	}
	FILE* out=fopen(argv[3], "wb");
	if (!out)
	{
		perror(argv[3]);
		exit(1);
	}

	fseek(in, 0L, SEEK_END);
	uint32_t insize = ftell(in);
	rewind(in);

	uint8_t* inbuf = (uint8_t*) malloc(insize);
	if (!inbuf)
	{
		perror("out of mem");
		exit(1);
	}

	fread(inbuf, 1, insize, in);

	int outsize;
	if (*argv[1]=='d')
		outsize = *((uint32_t*)(inbuf));
	else
		outsize = insize;

	uint8_t* outbuf = (uint8_t*) malloc(outsize);
	if (!outbuf)
	{
		perror("out of mem");
		exit(1);
	}

	
	if (*argv[1]=='c')
	{
		printf("Compressing %s... %d bytes\n", argv[2], insize);
		int level = argv[1][1]=='f'?0:(argv[1][1]=='x'?2:1);

		fwrite(&insize, 1, sizeof(insize), out); // Little-endian
		outsize = compress(level, inbuf, insize, outbuf);
		fwrite(outbuf, 1, outsize, out);
	}
	else if (*argv[1]=='d')
	{
		printf("Decompressing %s... %d bytes\n", argv[2], insize);
		decompress(inbuf+sizeof(uint32_t), outbuf, outsize);
		fwrite(outbuf, 1, outsize, out);
	}
	else
	{
		fprintf(stderr, "Unknown command: %s\n", argv[1]);
		exit(1);
	}

	printf("%lld -> %lld in %gs\n", _ftelli64(in), _ftelli64(out),
		double(clock()-start)/CLOCKS_PER_SEC);

	fclose(in);
	fclose(out);

	return 0;
}

*/

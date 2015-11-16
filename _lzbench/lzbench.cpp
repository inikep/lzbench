/*
(C) 2011-2015 by Przemyslaw Skibinski (inikep@gmail.com)

    LICENSE

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation; either version 3 of
    the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    General Public License for more details at
    Visit <http://www.gnu.org/copyleft/gpl.html>.

*/

#define _CRT_SECURE_NO_WARNINGS
#define PROGNAME "lzbench"
#define PROGVERSION "0.8.1"
#define LZBENCH_DEBUG(level, fmt, args...) if (verbose >= level) printf(fmt, ##args)

#define MAX(a,b) ((a)>(b))?(a):(b)
#ifndef MIN
	#define MIN(a,b) ((a)<(b)?(a):(b))
#endif

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(WIN64) || defined(_WIN64)
	#define WINDOWS
#endif

#define __STDC_FORMAT_MACROS // now PRIu64 will work
#include <inttypes.h> // now PRIu64 will work
#define _FILE_OFFSET_BITS 64  // turn off_t into a 64-bit type for ftello() and fseeko()
#define PAD_SIZE (16*1024)

#include <vector>
#include <numeric>
#include <algorithm> // sort
#include <string>
#include <stdlib.h> 
#include <stdio.h> 
#include <stdint.h> 
#include <string.h> 
#include "lzbench.h"

#ifdef WINDOWS
	#include <windows.h>
	#define InitTimer(x) if (!QueryPerformanceFrequency(&x)) { printf("QueryPerformance not present"); };
	#define GetTime(x) QueryPerformanceCounter(&x); 
	#define GetDiffTime(ticksPerSecond, start_ticks, end_ticks) (1000*(end_ticks.QuadPart - start_ticks.QuadPart)/ticksPerSecond.QuadPart)
	void uni_sleep(UINT usec) { Sleep(usec); };
	#ifndef __GNUC__
		#define fseeko64 _fseeki64 
		#define ftello64 _ftelli64
	#endif
	#define PROGOS "Windows"
#else
    #include <stdarg.h> // va_args
	#include <time.h>   
	#include <unistd.h>
	#include <sys/resource.h>
	typedef struct timespec LARGE_INTEGER;
	#define InitTimer(x) 
	#define GetTime(x) if(clock_gettime( CLOCK_REALTIME, &x) == -1 ){ printf("clock_gettime error"); };
	#define GetDiffTime(ticksPerSecond, start_ticks, end_ticks) (1000*( end_ticks.tv_sec - start_ticks.tv_sec ) + ( end_ticks.tv_nsec - start_ticks.tv_nsec )/1000000)
	void uni_sleep(uint32_t usec) { usleep(usec * 1000); };
	#define PROGOS "Linux"
#endif

#define ITERS(count) for(int ii=0; ii<count; ii++)


typedef struct string_table
{
    std::string column1;
    float column2, column3, column5;
    uint64_t column4;
    string_table(std::string c1, float c2, float c3, size_t c4, float c5) : column1(c1), column2(c2), column3(c3), column4(c4), column5(c5) {}
} string_table_t;

struct less_using_1st_column { inline bool operator() (const string_table_t& struct1, const string_table_t& struct2) {  return (struct1.column1 < struct2.column1); } };
struct less_using_2nd_column { inline bool operator() (const string_table_t& struct1, const string_table_t& struct2) {  return (struct1.column2 > struct2.column2); } };
struct less_using_3rd_column { inline bool operator() (const string_table_t& struct1, const string_table_t& struct2) {  return (struct1.column3 > struct2.column3); } };
struct less_using_4th_column { inline bool operator() (const string_table_t& struct1, const string_table_t& struct2) {  return (struct1.column4 < struct2.column4); } };
struct less_using_5th_column { inline bool operator() (const string_table_t& struct1, const string_table_t& struct2) {  return (struct1.column5 < struct2.column5); } };

std::vector<string_table_t> results;
bool turbobench_format = false;
int verbose = 0;


void format(std::string& s,const char* formatstring, ...) 
{
   char buff[1024];
   va_list args;
   va_start(args, formatstring);

#ifdef WIN32
   _vsnprintf( buff, sizeof(buff), formatstring, args);
#else
   vsnprintf( buff, sizeof(buff), formatstring, args);
#endif

   va_end(args);

   s=buff;
} 


void print_row(string_table_t& row)
{
    if (turbobench_format)
    {
        printf("%12" PRId64" %6.1f%9.2f%9.2f  %s\n", row.column4, row.column5, row.column2, row.column3, row.column1.c_str());
        return;
    }

    printf("| %-27s ", row.column1.c_str());
    if (row.column2 < 10) printf("|%6.2f MB/s ", row.column2); else printf("|%6d MB/s ", (int)row.column2);
    if (!row.column3)
        printf("|      ERROR ");
    else
        if (row.column3 < 10) printf("|%6.2f MB/s ", row.column3); else printf("|%6d MB/s ", (int)row.column3); 
    printf("|%12" PRId64 " |%6.2f |\n", row.column4, row.column5);
}


void print_stats(const compressor_desc_t* desc, int level, std::vector<uint32_t> &ctime, std::vector<uint32_t> &dtime, uint32_t insize, uint32_t outsize, bool decomp_error, int cspeed)
{
    std::string column1;
    std::sort(ctime.begin(), ctime.end());
    std::sort(dtime.begin(), dtime.end());

    uint32_t cmili_fastest = ctime[0] + (ctime[0] == 0);
    uint32_t dmili_fastest = dtime[0] + (dtime[0] == 0);
    uint32_t cmili_med = ctime[ctime.size()/2] + (ctime[ctime.size()/2] == 0);
    uint32_t dmili_med = dtime[dtime.size()/2] + (dtime[dtime.size()/2] == 0);
    uint32_t cmili_avg = std::accumulate(ctime.begin(),ctime.end(),0) / ctime.size();
    uint32_t dmili_avg = std::accumulate(dtime.begin(),dtime.end(),0) / dtime.size();
    if (cmili_avg == 0) cmili_avg = 1;
    if (dmili_avg == 0) dmili_avg = 1;

    if (cspeed > insize / cmili_fastest / 1024) { LZBENCH_DEBUG(9, "%s FULL slower than %d MB/s\n", desc->name, insize / cmili_fastest / 1024); return; } 

    if (desc->first_level == 0 && desc->last_level==0)
        format(column1, "%s %s", desc->name, desc->version);
    else
        format(column1, "%s %s level %d", desc->name, desc->version, level);

    results.push_back(string_table_t(column1, insize / cmili_fastest / 1024.0, (decomp_error)?0:(insize / dmili_fastest / 1024.0), outsize, outsize * 100.0 / insize));
    print_row(results[results.size()-1]);

    ctime.clear();
    dtime.clear();
};


size_t common(uint8_t *p1, uint8_t *p2)
{
	size_t size = 0;

	while (*(p1++) == *(p2++))
        size++;

	return size;
}


void add_time(std::vector<uint32_t> &ctime, std::vector<uint32_t> &dtime, uint32_t comp_time, uint32_t decomp_time)
{
	ctime.push_back(comp_time);
	dtime.push_back(decomp_time);
}


int64_t lzbench_compress(compress_func compress, size_t chunk_size, std::vector<size_t> &compr_lens, uint8_t *inbuf, size_t insize, uint8_t *outbuf, size_t outsize, size_t param1, size_t param2, char* workmem)
{
    int64_t clen;
    size_t part, sum = 0;
    uint8_t *start = inbuf;
    compr_lens.clear();
    
    while (insize > 0)
    {
        part = MIN(insize, chunk_size);
        clen = compress((char*)inbuf, part, (char*)outbuf, outsize, param1, param2, workmem);
		LZBENCH_DEBUG(5,"ENC part=%d clen=%d in=%d\n", (int)part, (int)clen, (int)(inbuf-start));

        if (clen <= 0 || clen == part)
        {
            memcpy(outbuf, inbuf, part);
            clen = part;
        }
        
        inbuf += part;
        insize -= part;
        outbuf += clen;
        outsize -= clen;
        compr_lens.push_back(clen);
        sum += clen;
    }
    return sum;
}


int64_t lzbench_decompress(compress_func decompress, size_t chunk_size, std::vector<size_t> &compr_lens, uint8_t *inbuf, size_t insize, uint8_t *outbuf, size_t outsize, uint8_t *origbuf, size_t param1, size_t param2, char* workmem)
{
    int64_t dlen;
    int num=0;
    size_t part, sum = 0;
    uint8_t *outstart = outbuf;

    while (insize > 0)
    {
        part = compr_lens[num++];
        if (part > insize) return 0;
        if (part == MIN(chunk_size,outsize)) // uncompressed
        {
            memcpy(outbuf, inbuf, part);
            dlen = part;
        }
        else
        {
            dlen = decompress((char*)inbuf, part, (char*)outbuf, MIN(chunk_size,outsize), param1, param2, workmem);
        }
		LZBENCH_DEBUG(5, "DEC part=%d dlen=%d out=%d\n", (int)part, (int)dlen, (int)(outbuf - outstart));
        if (dlen <= 0) return dlen;

        inbuf += part;
        insize -= part;
        outbuf += dlen;
        outsize -= dlen;
        sum += dlen;
    }
    
    return sum;
}


void lzbench_test(const compressor_desc_t* desc, int level, int cspeed, size_t chunk_size, int iters, uint8_t *inbuf, size_t insize, uint8_t *compbuf, size_t comprsize, uint8_t *decomp, LARGE_INTEGER ticksPerSecond, size_t param1, size_t param2)
{
    LARGE_INTEGER start_ticks, end_ticks;
    int64_t complen=0, decomplen;
    std::vector<uint32_t> ctime, dtime;
    std::vector<size_t> compr_lens;
    bool decomp_error = false;
    char* workmem = NULL;

    if (!desc->compress || !desc->decompress) goto done;
    if (desc->init) workmem = desc->init(chunk_size);

    LZBENCH_DEBUG(1, "*** trying %s insize=%d comprsize=%d chunk_size=%d\n", desc->name, (int)insize, (int)comprsize, (int)chunk_size);

    if (cspeed > 0)
    {
        uint32_t part = MIN(100*1024,chunk_size);
        GetTime(start_ticks);
        int64_t clen = desc->compress((char*)inbuf, part, (char*)compbuf, comprsize, param1, param2, workmem);
        GetTime(end_ticks);
        uint32_t milisec = GetDiffTime(ticksPerSecond, start_ticks, end_ticks);
  //      printf("\nclen=%d milisec=%d %s\n", clen, milisec, desc->name);
        if (clen>0 && milisec>=3) // longer than 3 milisec = slower than 33 MB/s
        {
            part = part / milisec / 1024; // speed in MB/s
    //        printf("%s = %d MB/s, %d\n", desc->name, part, clen);
            if (part < cspeed) { LZBENCH_DEBUG(9, "%s (100K) slower than %d MB/s\n", desc->name, part); goto done; }
        }
    }
    
    ITERS(iters)
    {
        GetTime(start_ticks);
        complen = lzbench_compress(desc->compress, chunk_size, compr_lens, inbuf, insize, compbuf, comprsize, param1, param2, workmem);
        GetTime(end_ticks);
        
        uint32_t milisec = GetDiffTime(ticksPerSecond, start_ticks, end_ticks);
        if (complen>0 && milisec>=3) // longer than 3 milisec
        {
            if (insize / milisec / 1024 < cspeed) { LZBENCH_DEBUG(9, "%s 1ITER slower than %d MB/s\n", desc->name, (uint32_t)(insize / milisec / 1024)); goto done; }
        }

        GetTime(start_ticks);
        decomplen = lzbench_decompress(desc->decompress, chunk_size, compr_lens, compbuf, complen, decomp, insize, inbuf, param1, param2, workmem);
        GetTime(end_ticks);


        add_time(ctime, dtime, milisec, GetDiffTime(ticksPerSecond, start_ticks, end_ticks)); 

        if (insize != decomplen)
        {   
            decomp_error = true; 
            LZBENCH_DEBUG(1, "ERROR: inlen[%d] != outlen[%d]\n", (int32_t)insize, (int32_t)decomplen);
        }
        
        if (memcmp(inbuf, decomp, insize) != 0)
        {
            decomp_error = true; 

            size_t cmn = common(inbuf, decomp);
            LZBENCH_DEBUG(1, "ERROR in %s: common=%d/%d\n", desc->name, (int32_t)cmn, (int32_t)insize);
            
            if (verbose >= 10)
            {
                char text[256];
                snprintf(text, sizeof(text), "%s_failed", desc->name);
                cmn /= chunk_size;
                size_t err_size = MIN(insize, (cmn+1)*chunk_size);
                err_size -= cmn*chunk_size;
                printf("ERROR: fwrite %d-%d to %s\n", (int32_t)(cmn*chunk_size), (int32_t)(cmn*chunk_size+err_size), text);
                FILE *f = fopen(text, "wb");
                if (f) fwrite(inbuf+cmn*chunk_size, 1, err_size, f), fclose(f);
                exit(0);
            }
        }

        memset(decomp, 0, insize); // clear output buffer
        uni_sleep(1); // give processor to other processes
        
        if (decomp_error) break;
    }
    print_stats(desc, level, ctime, dtime, insize, complen, decomp_error, cspeed);
done:
    if (desc->deinit) desc->deinit(workmem);
};


void lzbench_test_with_params(char *namesWithParams, int cspeed, size_t chunk_size, int iters, uint8_t *inbuf, size_t insize, uint8_t *compbuf, size_t comprsize, uint8_t *decomp, LARGE_INTEGER ticksPerSecond)
{
    const char delimiters[] = "/";
    const char delimiters2[] = ",";
    char *copy, *copy2, *token, *token2, *token3, *save_ptr, *save_ptr2;

    copy = (char*)strdup(namesWithParams);
    token = strtok_r(copy, delimiters, &save_ptr);

    while (token != NULL) 
    {
        if (strcmp(token, "fast")==0)
        {
            lzbench_test_with_params(compr_fast, cspeed, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond);
            token = strtok_r(NULL, delimiters, &save_ptr);
            continue;
        }

        if (strcmp(token, "opt")==0)
        {
            lzbench_test_with_params(compr_opt, cspeed, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond);
            token = strtok_r(NULL, delimiters, &save_ptr);
            continue;
        }

        if (strcmp(token, "all")==0)
        {
            lzbench_test_with_params(compr_all, cspeed, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond);
            token = strtok_r(NULL, delimiters, &save_ptr);
            continue;
        }

        if (strcmp(token, "ucl")==0)
        {
            lzbench_test_with_params(compr_ucl, cspeed, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond);
            token = strtok_r(NULL, delimiters, &save_ptr);
            continue;
        }

        if (strcmp(token, "lzo")==0)
        {
            lzbench_test_with_params(compr_lzo, cspeed, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond);
            token = strtok_r(NULL, delimiters, &save_ptr);
            continue;
        }

        if (strcmp(token, "lzo1b")==0)
        {
            lzbench_test_with_params(compr_lzo1b, cspeed, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond);
            token = strtok_r(NULL, delimiters, &save_ptr);
            continue;
        }

        if (strcmp(token, "lzo1c")==0)
        {
            lzbench_test_with_params(compr_lzo1c, cspeed, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond);
            token = strtok_r(NULL, delimiters, &save_ptr);
            continue;
        }

        if (strcmp(token, "lzo1f")==0)
        {
            lzbench_test_with_params(compr_lzo1f, cspeed, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond);
            token = strtok_r(NULL, delimiters, &save_ptr);
            continue;
        }

        if (strcmp(token, "lzo1x")==0)
        {
            lzbench_test_with_params(compr_lzo1x, cspeed, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond);
            token = strtok_r(NULL, delimiters, &save_ptr);
            continue;
        }

        if (strcmp(token, "lzo1y")==0)
        {
            lzbench_test_with_params(compr_lzo1y, cspeed, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond);
            token = strtok_r(NULL, delimiters, &save_ptr);
            continue;
        }

        copy2 = (char*)strdup(token);
        LZBENCH_DEBUG(1, "params = %s\n", token);
        token2 = strtok_r(copy2, delimiters2, &save_ptr2);

        if (token2)
        {
            token3 = strtok_r(NULL, delimiters2, &save_ptr2);
            do
            {
                bool found = false;
                for (int i=1; i<LZBENCH_COMPRESSOR_COUNT; i++)
                {
                    if (strcmp(comp_desc[i].name, token2) == 0)
                    {
                        found = true;
    //                        printf("%s %s %s\n", token2, comp_desc[i].version, token3);
                        if (!token3)
                        {                          
                            for (int level=comp_desc[i].first_level; level<=comp_desc[i].last_level; level++)
                                lzbench_test(&comp_desc[i], level, cspeed, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, level, 0);
                        }
                        else
                            lzbench_test(&comp_desc[i], atoi(token3), cspeed, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, atoi(token3), 0);
                        break;
                    }
                }
                if (!found) printf("NOT FOUND: %s %s\n", token2, token3);
                token3 = strtok_r(NULL, delimiters2, &save_ptr2);
            }
            while (token3 != NULL);
        }

        free(copy2);
        
        token = strtok_r(NULL, delimiters, &save_ptr);
    }

    free(copy);
}


void lzbenchmark(FILE* in, char* encoder_list, int iters, uint32_t chunk_size, int cspeed)
{
	std::vector<uint32_t> ctime, dtime;
	LARGE_INTEGER ticksPerSecond, start_ticks, mid_ticks, end_ticks;
	uint32_t comprsize, insize;
	uint8_t *inbuf, *compbuf, *decomp;

	InitTimer(ticksPerSecond);

	fseek(in, 0L, SEEK_END);
	insize = ftell(in);
	rewind(in);

	comprsize = insize + insize/6 + PAD_SIZE; // for pithy

//	printf("insize=%lld comprsize=%lld\n", insize, comprsize);
	inbuf = (uint8_t*)malloc(insize + PAD_SIZE);
	compbuf = (uint8_t*)malloc(comprsize);
	decomp = (uint8_t*)calloc(1, insize + PAD_SIZE);

	if (!inbuf || !compbuf || !decomp)
	{
		printf("Not enough memory!");
		exit(1);
	}

	insize = fread(inbuf, 1, insize, in);
	if (chunk_size > insize) chunk_size = insize;

	ITERS(iters)
	{
		GetTime(start_ticks);
		memcpy(compbuf, inbuf, insize);
		GetTime(mid_ticks);
		memcpy(decomp, compbuf, insize);
		GetTime(end_ticks);
		add_time(ctime, dtime, GetDiffTime(ticksPerSecond, start_ticks, mid_ticks), GetDiffTime(ticksPerSecond, mid_ticks, end_ticks));
	}
    printf("| Compressor name             | Compression| Decompress.| Compr. size | Ratio |\n");
	print_stats(&comp_desc[0], 0, ctime, dtime, insize, insize, false, 0);

    lzbench_test_with_params(encoder_list?encoder_list:compr_fast, cspeed, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond);

	free(inbuf);
	free(compbuf);
	free(decomp);

	if (chunk_size > 10 * (1<<20))
		printf("done... (%d iterations, chunk_size=%d MB, min_compr_speed=%d MB)\n", iters, chunk_size >> 20, cspeed);
	else
		printf("done... (%d iterations, chunk_size=%d KB, min_compr_speed=%d MB)\n", iters, chunk_size >> 10, cspeed);
}


void test_compressor(char* filename)
{
	uint32_t comprsize, insize, outsize, decompsize;
	char *inbuf, *compbuf, *decomp;
    FILE* in;
    
	if (!(in=fopen(filename, "rb"))) {
		perror(filename);
		exit(1);
	}
    
	fseek(in, 0L, SEEK_END);
	insize = ftell(in);
	rewind(in);

    comprsize = insize + 2048;
//	printf("insize=%lld comprsize=%lld\n", insize, comprsize);
	inbuf = (char*)malloc(insize + 2048);
	compbuf = (char*)malloc(comprsize);
	decomp = (char*)calloc(1, insize + 2048);

	if (!inbuf || !compbuf || !decomp)
	{
		printf("Not enough memory!");
		exit(1);
	}

	insize = fread(inbuf, 1, insize, in);

    outsize = lzbench_zstdhc_compress(inbuf, insize, compbuf, comprsize, 9, 0, 0);
    printf("insize=%d outsize=%d\n", insize, outsize);
    decompsize = lzbench_zstdhc_decompress(compbuf, outsize, decomp, insize, 0, 0, 0);
    printf("insize=%d outsize=%d\n", outsize, decompsize);

    fclose(in);
}




int main( int argc, char** argv) 
{
	FILE *in;
	uint32_t iterations, chunk_size, cspeed;
    char* encoder_list = NULL;
    int sort_col = 0;

	iterations = 1;
	chunk_size = 1 << 31;
	cspeed = 0;

//    test_compressor(argv[1]);
//    exit(0);


#ifdef WINDOWS
//	SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
	SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
#else
	setpriority(PRIO_PROCESS, 0, -20);
#endif

	printf(PROGNAME " " PROGVERSION " (%d-bit " PROGOS ")   Assembled by P.Skibinski\n", (uint32_t)(8 * sizeof(uint8_t*)));

	while ((argc>1)&&(argv[1][0]=='-')) {
		switch (argv[1][1]) {
	case 'i':
		iterations=atoi(argv[1]+2);
		break;
	case 'c':
		sort_col = atoi(argv[1] + 2);
		break;
	case 'b':
		chunk_size = atoi(argv[1] + 2) << 10;
		break;
	case 'e':
		encoder_list = strdup(argv[1] + 2);
		break;
	case 's':
		cspeed = atoi(argv[1] + 2);
		break;
	case 't':
		turbobench_format = true;
		break;
	case 'v':
		verbose = atoi(argv[1] + 2);
		break;
	case '-': // --help
	case 'h':
        break;
        
	default:
		fprintf(stderr, "unknown option: %s\n", argv[1]);
		exit(1);
		}
		argv++;
		argc--;
	}

	if (argc<2) {
		fprintf(stderr, "usage: " PROGNAME " [options] input\n");
		fprintf(stderr, " -bX: set block/chunk size to X KB (default = %d KB)\n", chunk_size>>10);
		fprintf(stderr, " -cX: sort results by column number X\n");
		fprintf(stderr, " -eX: X = compressors separated by '/' with parameters specified after ','\n");
		fprintf(stderr, " -iX: number of iterations (default = %d)\n", iterations);
		fprintf(stderr, " -sX: use only compressors with compression speed over X MB (default = %d MB)\n", cspeed);

        fprintf(stderr,"\nExample usage:\n");
        fprintf(stderr,"  " PROGNAME " -ebrotli filename - selects all levels of brotli\n");
        fprintf(stderr,"  " PROGNAME " -ebrotli,2,5/zstd filename - selects levels 2 & 5 of brotli and zstd\n");

        printf("\nAvailable compressors for -e option:\n");
        printf("all - alias for all available compressors\n");
        printf("fast - alias for compressors with compression speed over 100 MB/s\n");
        printf("opt - compressors with optimal parsing (slow compression, fast decompression)\n");
        printf("lzo / ucl - aliases for all levels of given compressors\n");
        for (int i=1; i<LZBENCH_COMPRESSOR_COUNT; i++)
        {
            printf("%s %s\n", comp_desc[i].name, comp_desc[i].version);
        }
                    
		exit(1);
	}

	if (!(in=fopen(argv[1], "rb"))) {
		perror(argv[1]);
		exit(1);
	}

	lzbenchmark(in, encoder_list, iterations, chunk_size, cspeed);

    if (encoder_list) free(encoder_list);

	fclose(in);


    if (sort_col <= 0) return 0;

    printf("\nThe results sorted by column number %d:\n", sort_col);
    printf("| Compressor name             | Compression| Decompress.| Compr. size | Ratio |\n");

    switch (sort_col)
    {
        default:
        case 1: std::sort(results.begin(), results.end(), less_using_1st_column()); break;
        case 2: std::sort(results.begin(), results.end(), less_using_2nd_column()); break;
        case 3: std::sort(results.begin(), results.end(), less_using_3rd_column()); break;
        case 4: std::sort(results.begin(), results.end(), less_using_4th_column()); break;
        case 5: std::sort(results.begin(), results.end(), less_using_5th_column()); break;
    }

    for (std::vector<string_table_t>::iterator it = results.begin(); it!=results.end(); it++)
    {
        print_row(*it);
    }
}




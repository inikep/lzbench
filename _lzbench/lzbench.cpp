/*
(C) 2011-2016 by Przemyslaw Skibinski (inikep@gmail.com)

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

#include "lzbench.h"
#include <numeric>
#include <algorithm> // sort
#include <stdlib.h> 
#include <stdio.h> 
#include <stdint.h> 
#include <string.h>


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

void print_header(lzbench_params_t *params)
{
    switch (params->textformat)
    {
        case CSV:
            printf("Compressor name,Compression speed,Decompression speed,Compressed size,Ratio,Filename\n"); break;
        case TURBOBENCH:
            printf("  Compressed  Ratio   Cspeed   Dspeed  Compressor name Filename\n"); break;
        case TEXT:
            printf("Compressor name         Compress. Decompress. Compr. size  Ratio Filename\n"); break;
        case MARKDOWN:
            printf("| Compressor name         | Compression| Decompress.| Compr. size | Ratio | Filename |\n"); 
            printf("| ---------------         | -----------| -----------| ----------- | ----- | -------- |\n"); 
            break;
    }
}

void print_row(lzbench_params_t *params, string_table_t& row)
{
    switch (params->textformat)
    {
        case CSV:
            printf("%s,%.2f,%.2f,%" PRId64 ",%.2f,%s\n", row.column1.c_str(), row.column2, row.column3, row.column4, row.column5, row.filename.c_str()); break;
        case TURBOBENCH:
            printf("%12" PRId64 " %6.1f%9.2f%9.2f  %22s %s\n", row.column4, row.column5, row.column2, row.column3, row.column1.c_str(), row.filename.c_str()); break;
        case TEXT:
            printf("%-23s", row.column1.c_str());
            if (row.column2 < 10) printf("%6.2f MB/s", row.column2); else printf("%6d MB/s", (int)row.column2);
            if (!row.column3)
                printf("      ERROR");
            else
                if (row.column3 < 10) printf("%6.2f MB/s", row.column3); else printf("%6d MB/s", (int)row.column3); 
            printf("%12" PRId64 " %6.2f %s\n", row.column4, row.column5, row.filename.c_str());
            break;
        case MARKDOWN:
            printf("| %-23s ", row.column1.c_str());
            if (row.column2 < 10) printf("|%6.2f MB/s ", row.column2); else printf("|%6d MB/s ", (int)row.column2);
            if (!row.column3)
                printf("|      ERROR ");
            else
                if (row.column3 < 10) printf("|%6.2f MB/s ", row.column3); else printf("|%6d MB/s ", (int)row.column3); 
            printf("|%12" PRId64 " |%6.2f | %-s|\n", row.column4, row.column5, row.filename.c_str());
            break;
    }
}


void print_stats(lzbench_params_t *params, const compressor_desc_t* desc, int level, std::vector<float> &cspd, std::vector<float> &dspd, size_t insize, size_t outsize, bool decomp_error)
{
    std::string column1;
    std::sort(cspd.begin(), cspd.end());
    std::sort(dspd.begin(), dspd.end());
    float cspeed, dspeed;
    
    switch (params->timetype)
    {
        case FASTEST: 
            cspeed = cspd[cspd.size()-1];
            dspeed = dspd[dspd.size()-1];
            break;
        case AVERAGE: 
            cspeed = std::accumulate(cspd.begin(),cspd.end(),0) / cspd.size();
            dspeed = std::accumulate(dspd.begin(),dspd.end(),0) / dspd.size();
            break;
        case MEDIAN: 
            cspeed = cspd[cspd.size()/2];
            dspeed = dspd[dspd.size()/2];
            break;
    }

    if (desc->first_level == 0 && desc->last_level==0)
        format(column1, "%s %s", desc->name, desc->version);
    else
        format(column1, "%s %s -%d", desc->name, desc->version, level);

    params->results.push_back(string_table_t(column1, cspeed, (decomp_error)?0:dspeed, outsize, outsize * 100.0 / insize, params->in_filename));
    print_row(params, params->results[params->results.size()-1]);

    cspd.clear();
    dspd.clear();
};


size_t common(uint8_t *p1, uint8_t *p2)
{
	size_t size = 0;

	while (*(p1++) == *(p2++))
        size++;

	return size;
}


inline int64_t lzbench_compress(lzbench_params_t *params, size_t chunk_size, compress_func compress, std::vector<size_t> &compr_lens, uint8_t *inbuf, size_t insize, uint8_t *outbuf, size_t outsize, size_t param1, size_t param2, char* workmem)
{
    int64_t clen;
    size_t outpart, part, sum = 0;
    uint8_t *start = inbuf;
    compr_lens.clear();
    outpart = GET_COMPRESS_BOUND(chunk_size);

    while (insize > 0)
    {
        part = MIN(insize, chunk_size);
        if (outpart > outsize) outpart = outsize;

        clen = compress((char*)inbuf, part, (char*)outbuf, outpart, param1, param2, workmem);
        LZBENCH_DEBUG(9,"ENC part=%d clen=%d in=%d\n", (int)part, (int)clen, (int)(inbuf-start));

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


inline int64_t lzbench_decompress(lzbench_params_t *params, size_t chunk_size, compress_func decompress, std::vector<size_t> &compr_lens, uint8_t *inbuf, size_t insize, uint8_t *outbuf, size_t outsize, uint8_t *origbuf, size_t param1, size_t param2, char* workmem)
{
    int64_t dlen;
    int num=0;
    size_t part, sum = 0;
    uint8_t *outstart = outbuf;

    while (insize > 0)
    {
        part = compr_lens[num++];
        if (part > insize) return 0;
        if (part == MIN(chunk_size, outsize)) // uncompressed
        {
            memcpy(outbuf, inbuf, part);
            dlen = part;
        }
        else
        {
            dlen = decompress((char*)inbuf, part, (char*)outbuf, MIN(chunk_size, outsize), param1, param2, workmem);
        }
        LZBENCH_DEBUG(9, "DEC part=%d dlen=%d out=%d\n", (int)part, (int)dlen, (int)(outbuf - outstart));
        if (dlen <= 0) return dlen;

        inbuf += part;
        insize -= part;
        outbuf += dlen;
        outsize -= dlen;
        sum += dlen;
    }
    
    return sum;
}


void lzbench_test(lzbench_params_t *params, const compressor_desc_t* desc, int level, uint8_t *inbuf, size_t insize, uint8_t *compbuf, size_t comprsize, uint8_t *decomp, bench_rate_t rate, size_t param1)
{
    float speed;
    int i, total_c_iters, total_d_iters;
    bench_timer_t loop_ticks, start_ticks, end_ticks, timer_ticks;
    int64_t complen=0, decomplen;
    uint64_t nanosec, total_nanosec;
    std::vector<float> cspeed, dspeed;
    std::vector<size_t> compr_lens;
    bool decomp_error = false;
    char* workmem = NULL;
    size_t param2 = desc->additional_param;
    size_t chunk_size = (params->chunk_size > insize) ? insize : params->chunk_size;

    if (desc->max_block_size != 0 && chunk_size > desc->max_block_size) chunk_size = desc->max_block_size;
    if (!desc->compress || !desc->decompress) goto done;
    if (desc->init) workmem = desc->init(chunk_size, param1);

    LZBENCH_DEBUG(1, "*** trying %s insize=%d comprsize=%d chunk_size=%d\n", desc->name, (int)insize, (int)comprsize, (int)chunk_size);

    if (params->cspeed > 0)
    {
        size_t part = MIN(100*1024, chunk_size);
        GetTime(start_ticks);
        int64_t clen = desc->compress((char*)inbuf, part, (char*)compbuf, comprsize, param1, param2, workmem);
        GetTime(end_ticks);
        nanosec = GetDiffTime(rate, start_ticks, end_ticks);
        if (clen>0 && nanosec>=1000)
        {
            part = (part / nanosec); // speed in MB/s
            if (part < params->cspeed) { LZBENCH_DEBUG(5, "%s (100K) slower than %d MB/s nanosec=%d\n", desc->name, (uint32_t)part, (uint32_t)nanosec); goto done; }
        }
    }

    total_c_iters = 0;
    GetTime(timer_ticks);
    do
    {
        i = 0;
        uni_sleep(1); // give processor to other processes
        GetTime(loop_ticks);
        do
        {
            GetTime(start_ticks);
            complen = lzbench_compress(params, chunk_size, desc->compress, compr_lens, inbuf, insize, compbuf, comprsize, param1, param2, workmem);
            GetTime(end_ticks);
            nanosec = GetDiffTime(rate, start_ticks, end_ticks);
            if (nanosec >= 10) cspeed.push_back((float)insize/nanosec);
            i++;
        }
        while (GetDiffTime(rate, loop_ticks, end_ticks) < params->cloop_time);

        nanosec = GetDiffTime(rate, loop_ticks, end_ticks);
        speed = (float)insize*i/nanosec;
        cspeed.push_back(speed);
        LZBENCH_DEBUG(8, "%s nanosec=%d\n", desc->name, (int)nanosec);

        if ((uint32_t)speed < params->cspeed) { LZBENCH_DEBUG(5, "%s slower than %d MB/s\n", desc->name, (uint32_t)speed); return; } 

        total_nanosec = GetDiffTime(rate, timer_ticks, end_ticks);
        total_c_iters += i;
        if (total_c_iters >= params->c_iters && total_nanosec > (params->cmintime*1000)) break;
        printf("%s compr iter=%d time=%.2fs speed=%.2f MB/s     \r", desc->name, total_c_iters, total_nanosec/1000000.0, speed);
    }
    while (true);


    total_d_iters = 0;
    GetTime(timer_ticks);
    do
    {
        i = 0;
        uni_sleep(1); // give processor to other processes
        GetTime(loop_ticks);
        do
        {
            GetTime(start_ticks);
            decomplen = lzbench_decompress(params, chunk_size, desc->decompress, compr_lens, compbuf, complen, decomp, insize, inbuf, param1, param2, workmem);
            GetTime(end_ticks);
            nanosec = GetDiffTime(rate, start_ticks, end_ticks);
            if (nanosec >= 10) dspeed.push_back((float)insize/nanosec);
            i++;
        }
        while (GetDiffTime(rate, loop_ticks, end_ticks) < params->dloop_time);

        nanosec = GetDiffTime(rate, loop_ticks, end_ticks);
        dspeed.push_back((float)insize*i/nanosec);
        LZBENCH_DEBUG(9, "%s dnanosec=%d\n", desc->name, (int)nanosec);

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
            
            if (params->verbose >= 10)
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

        if (decomp_error) break;
        
        total_nanosec = GetDiffTime(rate, timer_ticks, end_ticks);
        total_d_iters += i;
        if (total_d_iters >= params->d_iters && total_nanosec > (params->dmintime*1000)) break;
        printf("%s decompr iter=%d time=%.2fs speed=%.2f MB/s     \r", desc->name, total_d_iters, total_nanosec/1000000.0, (float)insize*i/nanosec);
    }
    while (true);

 //   printf("total_c_iters=%d total_d_iters=%d            \n", total_c_iters, total_d_iters);
    print_stats(params, desc, level, cspeed, dspeed, insize, complen, decomp_error);

done:
    if (desc->deinit) desc->deinit(workmem);
};


void lzbench_test_with_params(lzbench_params_t *params, char *namesWithParams, uint8_t *inbuf, size_t insize, uint8_t *compbuf, size_t comprsize, uint8_t *decomp, bench_rate_t rate)
{
    const char delimiters[] = "/";
    const char delimiters2[] = ",";
    char *copy, *copy2, *token, *token2, *token3, *save_ptr, *save_ptr2;

    copy = (char*)strdup(namesWithParams);
    token = strtok_r(copy, delimiters, &save_ptr);

    while (token != NULL) 
    {
        for (int i=0; i<LZBENCH_ALIASES_COUNT; i++)
        {
            if (strcmp(token, alias_desc[i].name)==0)
            {
                lzbench_test_with_params(params, (char*)alias_desc[i].params, inbuf, insize, compbuf, comprsize, decomp, rate);
                goto next_token; 
           }
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
                                lzbench_test(params, &comp_desc[i], level, inbuf, insize, compbuf, comprsize, decomp, rate, level);
                        }
                        else
                            lzbench_test(params, &comp_desc[i], atoi(token3), inbuf, insize, compbuf, comprsize, decomp, rate, atoi(token3));
                        break;
                    }
                }
                if (!found) printf("NOT FOUND: %s %s\n", token2, token3);
                token3 = strtok_r(NULL, delimiters2, &save_ptr2);
            }
            while (token3 != NULL);
        }

        free(copy2);
next_token:
        token = strtok_r(NULL, delimiters, &save_ptr);
    }

    free(copy);
}


void lzbenchmark(lzbench_params_t* params, FILE* in, char* encoder_list, bool first_time)
{
	bench_rate_t rate;
	size_t comprsize, insize;
	uint8_t *inbuf, *compbuf, *decomp;

	InitTimer(rate);

	fseeko(in, 0L, SEEK_END);
	insize = ftello(in);
	rewind(in);

	comprsize = GET_COMPRESS_BOUND(insize);

//	printf("insize=%llu comprsize=%llu %llu\n", insize, comprsize, MAX(MEMCPY_BUFFER_SIZE, insize));
	inbuf = (uint8_t*)malloc(insize + PAD_SIZE);
	compbuf = (uint8_t*)malloc(comprsize);
	decomp = (uint8_t*)calloc(1, insize + PAD_SIZE);

	if (!inbuf || !compbuf || !decomp)
	{
		printf("Not enough memory!");
		exit(1);
	}

	insize = fread(inbuf, 1, insize, in);

    if (first_time)
    {
        if (insize < (1<<20)) printf("WARNING: For small files with memcpy and fast compressors you can expect the cache effect causing much higher compression and decompression speed\n");

        print_header(params);

        lzbench_params_t params_memcpy;
        memcpy(&params_memcpy, params, sizeof(lzbench_params_t));
        params_memcpy.cmintime = params_memcpy.dmintime = 0;
        params_memcpy.c_iters = params_memcpy.d_iters = 0;
        params_memcpy.cloop_time = params_memcpy.dloop_time = DEFAULT_LOOP_TIME;
        lzbench_test(&params_memcpy, &comp_desc[0], 0, inbuf, insize, compbuf, insize, decomp, rate, 0);
    }
    
    lzbench_test_with_params(params, encoder_list?encoder_list:(char*)alias_desc[1].params, inbuf, insize, compbuf, comprsize, decomp, rate);

	free(inbuf);
	free(compbuf);
	free(decomp);
}


int main( int argc, char** argv) 
{
	FILE *in;
    char* encoder_list = NULL;
    int sort_col = 0;
    lzbench_params_t params;

    memset(&params, 0, sizeof(lzbench_params_t));
    params.timetype = FASTEST;
    params.textformat = TEXT;
    params.verbose = 0;
	params.chunk_size = (1ULL << 31) - (1ULL << 31)/6;
	params.cspeed = 0;
    params.c_iters = params.d_iters = 1;
    params.cmintime = 10*DEFAULT_LOOP_TIME/1000; // 1 sec
    params.dmintime = 5*DEFAULT_LOOP_TIME/1000; // 0.5 sec
    params.cloop_time = params.dloop_time = DEFAULT_LOOP_TIME;

#ifdef WINDOWS
	SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
#else
	setpriority(PRIO_PROCESS, 0, -20);
#endif

	printf(PROGNAME " " PROGVERSION " (%d-bit " PROGOS ")   Assembled by P.Skibinski\n", (uint32_t)(8 * sizeof(uint8_t*)));

	while ((argc>1)&&(argv[1][0]=='-')) {
		switch (argv[1][1]) {
	case 'b':
		params.chunk_size = atoi(argv[1] + 2) << 10;
		break;
	case 'c':
		sort_col = atoi(argv[1] + 2);
		break;
	case 'e':
		encoder_list = strdup(argv[1] + 2);
		break;
	case 'i':
		params.c_iters=atoi(argv[1]+2);
		break;
	case 'j':
		params.d_iters=atoi(argv[1]+2);
		break;
	case 'o':
        params.textformat = (textformat_e)atoi(argv[1] + 2);
		break;
	case 'p':
        params.timetype = (timetype_e)atoi(argv[1] + 2);
		break;
	case 's':
		params.cspeed = atoi(argv[1] + 2);
		break;
	case 't':
		params.cmintime = 1000*atoi(argv[1] + 2);
        params.cloop_time = (params.cmintime)?DEFAULT_LOOP_TIME:0;
		break;
	case 'u':
		params.dmintime = 1000*atoi(argv[1] + 2);
        params.dloop_time = (params.dmintime)?DEFAULT_LOOP_TIME:0;
		break;
	case 'v':
		params.verbose = atoi(argv[1] + 2);
		break;
	case '-': // --help
	case 'h':
        break;
    case 'l':
        printf("\nAvailable compressors for -e option:\n");
        printf("all - alias for all available compressors\n");
        printf("fast - alias for compressors with compression speed over 100 MB/s\n");
        printf("opt - compressors with optimal parsing (slow compression, fast decompression)\n");
        printf("lzo / ucl - aliases for all levels of given compressors\n");
        for (int i=1; i<LZBENCH_COMPRESSOR_COUNT; i++)
        {
            if (comp_desc[i].compress)
            {
                if (comp_desc[i].first_level < comp_desc[i].last_level)
                    printf("%s %s [%d-%d]\n", comp_desc[i].name, comp_desc[i].version, comp_desc[i].first_level, comp_desc[i].last_level);
                else
                    printf("%s %s\n", comp_desc[i].name, comp_desc[i].version);
            }
        }
        return 0;
    default:
		fprintf(stderr, "unknown option: %s\n", argv[1]);
		exit(1);
		}
		argv++;
		argc--;
	}

	if (argc<2) {
		fprintf(stderr, "usage: " PROGNAME " [options] input_file [input_file2] [input_file3]\n\nwhere [options] are:\n");
		fprintf(stderr, " -bX  set block/chunk size to X KB (default = MIN(filesize,%d KB))\n", (int)(params.chunk_size>>10));
		fprintf(stderr, " -cX  sort results by column number X\n");
		fprintf(stderr, " -eX  X = compressors separated by '/' with parameters specified after ','\n");
		fprintf(stderr, " -iX  set min. number of compression iterations (default = %d)\n", params.c_iters);
		fprintf(stderr, " -jX  set min. number of decompression iterations (default = %d)\n", params.d_iters);
		fprintf(stderr, " -l   list of available compressors and aliases\n");
        fprintf(stderr, " -oX  output text format 1=Markdown, 2=text, 3=CSV (default = %d)\n", params.textformat);
		fprintf(stderr, " -pX  print time for all iterations: 1=fastest 2=average 3=median (default = %d)\n", params.timetype);
		fprintf(stderr, " -sX  use only compressors with compression speed over X MB (default = %d MB)\n", params.cspeed);
		fprintf(stderr, " -tX  set min. time in seconds for compression (default = %.1f)\n", params.cmintime/1000.0);
 		fprintf(stderr, " -uX  set min. time in seconds for decompression (default = %.1f)\n", params.dmintime/1000.0);
        fprintf(stderr,"\nExample usage:\n");
        fprintf(stderr,"  " PROGNAME " -ebrotli filename - selects all levels of brotli\n");
        fprintf(stderr,"  " PROGNAME " -ebrotli,2,5/zstd filename - selects levels 2 & 5 of brotli and zstd\n");                    
		exit(0);
	}
    
    bool first_time = true;
    while (argc > 1)
    {
        if (!(in=fopen(argv[1], "rb"))) {
            perror(argv[1]);
        } else {
            char* pch = strrchr(argv[1], '\\');
            params.in_filename = pch ? pch+1 : argv[1];
            lzbenchmark(&params, in, encoder_list, first_time);
            first_time = false;
            fclose(in);
        }
        argv++;
        argc--;
    }

	if (params.chunk_size > 10 * (1<<20))
		printf("done... (cIters=%d dIters=%d cTime=%.1f dTime=%.1f chunkSize=%dMB cSpeed=%dMB)\n", params.c_iters, params.d_iters, params.cmintime/1000.0, params.dmintime/1000.0, (int)(params.chunk_size >> 20), params.cspeed);
	else
		printf("done... (cIters=%d dIters=%d cTime=%.1f dTime=%.1f chunkSize=%dKB cSpeed=%dMB)\n", params.c_iters, params.d_iters, params.cmintime/1000.0, params.dmintime/1000.0, (int)(params.chunk_size >> 10), params.cspeed);


    if (encoder_list) free(encoder_list);

    if (sort_col <= 0) return 0;

    printf("\nThe results sorted by column number %d:\n", sort_col);
    print_header(&params);

    switch (sort_col)
    {
        default:
        case 1: std::sort(params.results.begin(), params.results.end(), less_using_1st_column()); break;
        case 2: std::sort(params.results.begin(), params.results.end(), less_using_2nd_column()); break;
        case 3: std::sort(params.results.begin(), params.results.end(), less_using_3rd_column()); break;
        case 4: std::sort(params.results.begin(), params.results.end(), less_using_4th_column()); break;
        case 5: std::sort(params.results.begin(), params.results.end(), less_using_5th_column()); break;
    }

    for (std::vector<string_table_t>::iterator it = params.results.begin(); it!=params.results.end(); it++)
    {
        print_row(&params, *it);
    }
}




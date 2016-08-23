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


std::vector<std::string> split(const std::string &text, char sep) {
  std::vector<std::string> tokens;
  std::size_t start = 0, end = 0;
  while (text[start] == sep) start++;
  while ((end = text.find(sep, start)) != std::string::npos) {
    tokens.push_back(text.substr(start, end - start));
    start = end + 1;
  }
  tokens.push_back(text.substr(start));
  return tokens;
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


void print_speed(lzbench_params_t *params, string_table_t& row)
{
    float cspeed, dspeed, ratio;
    cspeed = row.column5 * 1000.0 / row.column2;
    dspeed = (!row.column3) ? 0 : (row.column5 * 1000.0 / row.column3);
    ratio = row.column4 * 100.0 / row.column5;
    
    switch (params->textformat)
    {
        case CSV:
            printf("%s,%.2f,%.2f,%llu,%.2f,%s\n", row.column1.c_str(), cspeed, dspeed, (unsigned long long)row.column4, ratio, row.filename.c_str()); break;
        case TURBOBENCH:
            printf("%12llu %6.1f%9.2f%9.2f  %22s %s\n", (unsigned long long)row.column4, ratio, cspeed, dspeed, row.column1.c_str(), row.filename.c_str()); break;
        case TEXT:
            printf("%-23s", row.column1.c_str());
            if (cspeed < 10) printf("%6.2f MB/s", cspeed); else printf("%6d MB/s", (int)cspeed);
            if (!dspeed)
                printf("      ERROR");
            else
                if (dspeed < 10) printf("%6.2f MB/s", dspeed); else printf("%6d MB/s", (int)dspeed); 
            printf("%12llu %6.2f %s\n", (unsigned long long)row.column4, ratio, row.filename.c_str());
            break;
        case MARKDOWN:
            printf("| %-23s ", row.column1.c_str());
            if (cspeed < 10) printf("|%6.2f MB/s ", cspeed); else printf("|%6d MB/s ", (int)cspeed);
            if (!dspeed)
                printf("|      ERROR ");
            else
                if (dspeed < 10) printf("|%6.2f MB/s ", dspeed); else printf("|%6d MB/s ", (int)dspeed); 
            printf("|%12llu |%6.2f | %-s|\n", (unsigned long long)row.column4, ratio, row.filename.c_str());
            break;
    }
}


void print_time(lzbench_params_t *params, string_table_t& row)
{
    float ratio = row.column4 * 100.0 / row.column5;
    uint64_t ctime = row.column2 / 1000;
    uint64_t dtime = row.column3 / 1000;

    switch (params->textformat)
    {
        case CSV:
            printf("%s,%llu,%llu,%llu,%.2f,%s\n", row.column1.c_str(), (unsigned long long)ctime, (unsigned long long)dtime, (unsigned long long)row.column4, ratio, row.filename.c_str()); break;
        case TURBOBENCH:
            printf("%12llu %6.1f%9llu%9llu  %22s %s\n", (unsigned long long)row.column4, ratio, (unsigned long long)ctime, (unsigned long long)dtime, row.column1.c_str(), row.filename.c_str()); break;
        case TEXT:
            printf("%-23s", row.column1.c_str());
            printf("%8llu us", (unsigned long long)ctime);
            if (!dtime)
                printf("      ERROR");
            else
                printf("%8llu us", (unsigned long long)dtime); 
            printf("%12llu %6.2f %s\n", (unsigned long long)row.column4, ratio, row.filename.c_str());
            break;
        case MARKDOWN:
            printf("| %-23s ", row.column1.c_str());
            printf("|%8llu us ", (unsigned long long)ctime);
            if (!dtime)
                printf("|      ERROR ");
            else
                printf("|%8llu us ", (unsigned long long)dtime); 
            printf("|%12llu |%6.2f | %-s|\n", (unsigned long long)row.column4, ratio, row.filename.c_str());
            break;
    }
}


void print_stats(lzbench_params_t *params, const compressor_desc_t* desc, int level, std::vector<uint64_t> &ctime, std::vector<uint64_t> &dtime, size_t insize, size_t outsize, bool decomp_error)
{
    std::string column1;
    std::sort(ctime.begin(), ctime.end());
    std::sort(dtime.begin(), dtime.end());
    uint64_t best_ctime, best_dtime;
    
    switch (params->timetype)
    {
        default:
        case FASTEST: 
            best_ctime = ctime[0];
            best_dtime = dtime[0];
            break;
        case AVERAGE: 
            best_ctime = std::accumulate(ctime.begin(),ctime.end(),0) / ctime.size();
            best_dtime = std::accumulate(dtime.begin(),dtime.end(),0) / dtime.size();
            break;
        case MEDIAN: 
            best_ctime = ctime[ctime.size()/2];
            best_dtime = dtime[dtime.size()/2];
            break;
    }

    if (desc->first_level == 0 && desc->last_level==0)
        format(column1, "%s %s", desc->name, desc->version);
    else
        format(column1, "%s %s -%d", desc->name, desc->version, level);

    params->results.push_back(string_table_t(column1, best_ctime, (decomp_error)?0:best_dtime, outsize, insize, params->in_filename));
    if (params->show_speed)
        print_speed(params, params->results[params->results.size()-1]);
    else
        print_time(params, params->results[params->results.size()-1]);

    ctime.clear();
    dtime.clear();
}


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
        LZBENCH_PRINT(9, "ENC part=%d clen=%d in=%d\n", (int)part, (int)clen, (int)(inbuf-start));

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


inline int64_t lzbench_decompress(lzbench_params_t *params, size_t chunk_size, compress_func decompress, std::vector<size_t> &compr_lens, uint8_t *inbuf, size_t insize, uint8_t *outbuf, size_t outsize, size_t param1, size_t param2, char* workmem)
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
        LZBENCH_PRINT(9, "DEC part=%d dlen=%d out=%d\n", (int)part, (int)dlen, (int)(outbuf - outstart));
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
    std::vector<uint64_t> ctime, dtime;
    std::vector<size_t> compr_lens;
    bool decomp_error = false;
    char* workmem = NULL;
    size_t param2 = desc->additional_param;
    size_t chunk_size = (params->chunk_size > insize) ? insize : params->chunk_size;

    LZBENCH_PRINT(5, "*** trying %s insize=%d comprsize=%d chunk_size=%d\n", desc->name, (int)insize, (int)comprsize, (int)chunk_size);

    if (desc->max_block_size != 0 && chunk_size > desc->max_block_size) chunk_size = desc->max_block_size;
    if (!desc->compress || !desc->decompress) goto done;
    if (desc->init) workmem = desc->init(chunk_size, param1);

    if (params->cspeed > 0)
    {
        size_t part = MIN(100*1024, chunk_size);
        GetTime(start_ticks);
        int64_t clen = desc->compress((char*)inbuf, part, (char*)compbuf, comprsize, param1, param2, workmem);
        GetTime(end_ticks);
        nanosec = GetDiffTime(rate, start_ticks, end_ticks)/1000;
        if (clen>0 && nanosec>=1000)
        {
            part = (part / nanosec); // speed in MB/s
            if (part < params->cspeed) { LZBENCH_PRINT(7, "%s (100K) slower than %d MB/s nanosec=%d\n", desc->name, (uint32_t)part, (uint32_t)nanosec); goto done; }
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
            if (nanosec >= 10000) ctime.push_back(nanosec);
            i++;
        }
        while (GetDiffTime(rate, loop_ticks, end_ticks) < params->cloop_time);

        nanosec = GetDiffTime(rate, loop_ticks, end_ticks);
        ctime.push_back(nanosec/i);
        speed = (float)insize*i*1000/nanosec;
        LZBENCH_PRINT(8, "%s nanosec=%d\n", desc->name, (int)nanosec);

        if ((uint32_t)speed < params->cspeed) { LZBENCH_PRINT(7, "%s slower than %d MB/s\n", desc->name, (uint32_t)speed); return; } 

        total_nanosec = GetDiffTime(rate, timer_ticks, end_ticks);
        total_c_iters += i;
        if ((total_c_iters >= params->c_iters) && (total_nanosec > ((uint64_t)params->cmintime*1000000))) break;
        LZBENCH_PRINT(2, "%s compr iter=%d time=%.2fs speed=%.2f MB/s     \r", desc->name, total_c_iters, total_nanosec/1000000000.0, speed);
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
            decomplen = lzbench_decompress(params, chunk_size, desc->decompress, compr_lens, compbuf, complen, decomp, insize, param1, param2, workmem);
            GetTime(end_ticks);
            nanosec = GetDiffTime(rate, start_ticks, end_ticks);
            if (nanosec >= 10000) dtime.push_back(nanosec);
            i++;
        }
        while (GetDiffTime(rate, loop_ticks, end_ticks) < params->dloop_time);

        nanosec = GetDiffTime(rate, loop_ticks, end_ticks);
        dtime.push_back(nanosec/i);
        LZBENCH_PRINT(9, "%s dnanosec=%d\n", desc->name, (int)nanosec);

        if (insize != decomplen)
        {   
            decomp_error = true; 
            LZBENCH_PRINT(5, "ERROR: inlen[%d] != outlen[%d]\n", (int32_t)insize, (int32_t)decomplen);
        }
        
        if (memcmp(inbuf, decomp, insize) != 0)
        {
            decomp_error = true; 

            size_t cmn = common(inbuf, decomp);
            LZBENCH_PRINT(5, "ERROR in %s: common=%d/%d\n", desc->name, (int32_t)cmn, (int32_t)insize);
            
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
                exit(1);
            }
        }

        memset(decomp, 0, insize); // clear output buffer

        if (decomp_error) break;
        
        total_nanosec = GetDiffTime(rate, timer_ticks, end_ticks);
        total_d_iters += i;
        if ((total_d_iters >= params->d_iters) && (total_nanosec > ((uint64_t)params->dmintime*1000000))) break;
        LZBENCH_PRINT(2, "%s decompr iter=%d time=%.2fs speed=%.2f MB/s     \r", desc->name, total_d_iters, total_nanosec/1000000000.0, (float)insize*i*1000/nanosec);
    }
    while (true);

 //   printf("total_c_iters=%d total_d_iters=%d            \n", total_c_iters, total_d_iters);
    print_stats(params, desc, level, ctime, dtime, insize, complen, decomp_error);

done:
    if (desc->deinit) desc->deinit(workmem);
}


void lzbench_test_with_params(lzbench_params_t *params, const char *namesWithParams, uint8_t *inbuf, size_t insize, uint8_t *compbuf, size_t comprsize, uint8_t *decomp, bench_rate_t rate)
{
    std::vector<std::string> cnames, cparams;

	if (!namesWithParams) return;

    cnames = split(namesWithParams, '/');

    for (int k=0; k<cnames.size(); k++)
        LZBENCH_PRINT(5, "cnames[%d] = %s\n", k, cnames[k].c_str());

    for (int k=0; k<cnames.size(); k++)
    {
        for (int i=0; i<LZBENCH_ALIASES_COUNT; i++)
        {
            if (strcmp(cnames[k].c_str(), alias_desc[i].name)==0)
            {
                lzbench_test_with_params(params, alias_desc[i].params, inbuf, insize, compbuf, comprsize, decomp, rate);
                goto next_k;
            }
        }

        LZBENCH_PRINT(5, "params = %s\n", cnames[k].c_str());
        cparams = split(cnames[k].c_str(), ',');
        if (cparams.size() >= 1)
        {
            int j=1;
            do {
                bool found = false;
                for (int i=1; i<LZBENCH_COMPRESSOR_COUNT; i++)
                {
                    if (strcmp(comp_desc[i].name, cparams[0].c_str()) == 0)
                    {
                        found = true;
                       // printf("%s %s %s\n", cparams[0].c_str(), comp_desc[i].version, cparams[j].c_str());
                        if (j >= cparams.size())
                        {                          
                            for (int level=comp_desc[i].first_level; level<=comp_desc[i].last_level; level++)
                                lzbench_test(params, &comp_desc[i], level, inbuf, insize, compbuf, comprsize, decomp, rate, level);
                        }
                        else
                            lzbench_test(params, &comp_desc[i], atoi(cparams[j].c_str()), inbuf, insize, compbuf, comprsize, decomp, rate, atoi(cparams[j].c_str()));
                        break;
                    }
                }
                if (!found) printf("NOT FOUND: %s %s\n", cparams[0].c_str(), (j<cparams.size()) ? cparams[j].c_str() : NULL);
                j++;
            }
            while (j < cparams.size());
        }
next_k:
        continue;
    }
}


void lzbench_alloc(lzbench_params_t* params, FILE* in, char* encoder_list, bool first_time)
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
    
    lzbench_test_with_params(params, encoder_list?encoder_list:alias_desc[0].params, inbuf, insize, compbuf, comprsize, decomp, rate);

    free(inbuf);
    free(compbuf);
    free(decomp);
}


void usage(lzbench_params_t* params)
{
    fprintf(stderr, "usage: " PROGNAME " [options] input_file [input_file2] [input_file3]\n\nwhere [options] are:\n");
    fprintf(stderr, " -bX  set block/chunk size to X KB (default = MIN(filesize,%d KB))\n", (int)(params->chunk_size>>10));
    fprintf(stderr, " -cX  sort results by column number X\n");
    fprintf(stderr, " -eX  X = compressors separated by '/' with parameters specified after ',' (deflt=fast)\n");
    fprintf(stderr, " -iX  set min. number of compression iterations (default = %d)\n", params->c_iters);
    fprintf(stderr, " -jX  set min. number of decompression iterations (default = %d)\n", params->d_iters);
    fprintf(stderr, " -l   list of available compressors and aliases\n");
    fprintf(stderr, " -oX  output text format 1=Markdown, 2=text, 3=CSV (default = %d)\n", params->textformat);
    fprintf(stderr, " -pX  print time for all iterations: 1=fastest 2=average 3=median (default = %d)\n", params->timetype);
    fprintf(stderr, " -r   disable real-time process priority\n");
    fprintf(stderr, " -sX  use only compressors with compression speed over X MB (default = %d MB)\n", params->cspeed);
    fprintf(stderr, " -tX  set min. time in seconds for compression (default = %.1f)\n", params->cmintime/1000.0);
    fprintf(stderr, " -uX  set min. time in seconds for decompression (default = %.1f)\n", params->dmintime/1000.0);
    fprintf(stderr, " -v   disable progress information\n");
    fprintf(stderr, " -z   show (de)compression times instead of speed\n");
    fprintf(stderr,"\nExample usage:\n");
    fprintf(stderr,"  " PROGNAME " -ezstd filename = selects all levels of zstd\n");
    fprintf(stderr,"  " PROGNAME " -ebrotli,2,5/zstd filename = selects levels 2 & 5 of brotli and zstd\n");
    fprintf(stderr,"  " PROGNAME " -t3 -u5 fname = 3 sec compression and 5 sec decompression loops\n");
    fprintf(stderr,"  " PROGNAME " -t0 -u0 -i3 -j5 -elz5 fname = 3 compression and 5 decompression iter.\n");
    fprintf(stderr,"  " PROGNAME " -t0u0i3j5 -elz5 fname = the same as above with aggregated parameters\n");
    exit(0);
}


int main( int argc, char** argv)
{
    FILE *in;
    char* encoder_list = NULL;
    int sort_col = 0, real_time = 1;
    lzbench_params_t lzparams;
    lzbench_params_t* params = &lzparams;

    memset(params, 0, sizeof(lzbench_params_t));
    params->timetype = FASTEST;
    params->textformat = TEXT;
    params->show_speed = 1;
    params->verbose = 2;
    params->chunk_size = (1ULL << 31) - (1ULL << 31)/6;
    params->cspeed = 0;
    params->c_iters = params->d_iters = 1;
    params->cmintime = 10*DEFAULT_LOOP_TIME/1000000; // 1 sec
    params->dmintime = 5*DEFAULT_LOOP_TIME/1000000; // 0.5 sec
    params->cloop_time = params->dloop_time = DEFAULT_LOOP_TIME;

    printf(PROGNAME " " PROGVERSION " (%d-bit " PROGOS ")   Assembled by P.Skibinski\n", (uint32_t)(8 * sizeof(uint8_t*)));

    while ((argc>1) && (argv[1][0]=='-')) {
    char* argument = argv[1]+1; 
    while (argument[0] != 0) {
        char* numPtr = argument + 1;
        unsigned number = 0;
        while ((*numPtr >='0') && (*numPtr <='9')) { number *= 10;  number += *numPtr - '0'; numPtr++; }
        switch (argument[0])
        {
        case 'b':
            params->chunk_size = number << 10;
            break;
        case 'c':
            sort_col = number;
            break;
        case 'e':
            encoder_list = strdup(argument + 1);
            numPtr += strlen(numPtr);
            break;
        case 'i':
            params->c_iters = number;
            break;
        case 'j':
            params->d_iters = number;
            break;
        case 'o':
            params->textformat = (textformat_e)number;
            break;
        case 'p':
            params->timetype = (timetype_e)number;
            break;
        case 'r':
            real_time = 0;
            break;
        case 's':
            params->cspeed = number;
            break;
        case 't':
            params->cmintime = 1000*number;
            params->cloop_time = (params->cmintime)?DEFAULT_LOOP_TIME:0;
            break;
        case 'u':
            params->dmintime = 1000*number;
            params->dloop_time = (params->dmintime)?DEFAULT_LOOP_TIME:0;
            break;
        case 'v':
            params->verbose = number;
            break;
        case 'z':
            params->show_speed = 0;
            break;
        case '-': // --help
        case 'h':
            usage(params);
            break;
        case 'l':
            printf("\nAvailable compressors for -e option:\n");
            printf("all - alias for all available compressors\n");
            printf("fast - alias for compressors with compression speed over 100 MB/s (default)\n");
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
        argument = numPtr;
    }
    argv++;
    argc--;
    }

    LZBENCH_PRINT(5, "params: chunk_size=%d c_iters=%d d_iters=%d cspeed=%d cmintime=%d dmintime=%d encoder_list=%s\n", (int)params->chunk_size, params->c_iters, params->d_iters, params->cspeed, params->cmintime, params->dmintime, encoder_list);

    if (argc<2) usage(params);

    if (real_time)
    {
    #ifdef WINDOWS
        SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
    #else
        setpriority(PRIO_PROCESS, 0, -20);
    #endif
    } else {
        printf("The real-time process priority disabled\n");
    }

    bool first_time = true;
    while (argc > 1)
    {
        if (!(in=fopen(argv[1], "rb"))) {
            perror(argv[1]);
        } else {
            char* pch = strrchr(argv[1], '\\');
            params->in_filename = pch ? pch+1 : argv[1];
            lzbench_alloc(params, in, encoder_list, first_time);
            first_time = false;
            fclose(in);
        }
        argv++;
        argc--;
    }

    if (params->chunk_size > 10 * (1<<20))
        printf("done... (cIters=%d dIters=%d cTime=%.1f dTime=%.1f chunkSize=%dMB cSpeed=%dMB)\n", params->c_iters, params->d_iters, params->cmintime/1000.0, params->dmintime/1000.0, (int)(params->chunk_size >> 20), params->cspeed);
    else
        printf("done... (cIters=%d dIters=%d cTime=%.1f dTime=%.1f chunkSize=%dKB cSpeed=%dMB)\n", params->c_iters, params->d_iters, params->cmintime/1000.0, params->dmintime/1000.0, (int)(params->chunk_size >> 10), params->cspeed);


    if (encoder_list) free(encoder_list);

    if (sort_col <= 0) return 0;

    printf("\nThe results sorted by column number %d:\n", sort_col);
    print_header(params);

    switch (sort_col)
    {
        default:
        case 1: std::sort(params->results.begin(), params->results.end(), less_using_1st_column()); break;
        case 2: std::sort(params->results.begin(), params->results.end(), less_using_2nd_column()); break;
        case 3: std::sort(params->results.begin(), params->results.end(), less_using_3rd_column()); break;
        case 4: std::sort(params->results.begin(), params->results.end(), less_using_4th_column()); break;
        case 5: std::sort(params->results.begin(), params->results.end(), less_using_5th_column()); break;
    }

    for (std::vector<string_table_t>::iterator it = params->results.begin(); it!=params->results.end(); it++)
    {
        if (params->show_speed)
            print_speed(params, *it);
        else
            print_time(params, *it);
    }
}




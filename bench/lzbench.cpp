/*
 * Copyright (c) Przemyslaw Skibinski <inikep@gmail.com>
 * All rights reserved.
 *
 * This source code is dual-licensed under the GPLv2 and GPLv3 licenses.
 * For additional details, refer to the LICENSE file located in the root
 * directory of this source tree.
 */

#include "lzbench.h"
#include "util.h"
#include "cpuid1.h"
#include "threadpool.h"

#include <numeric>
#include <algorithm> // sort
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <thread> // this_thread::yield


int g_exit_result = 0; // global variable


int istrcmp(const char *str1, const char *str2)
{
    int c1, c2;
    while (1) {
        c1 = tolower((unsigned char)(*str1++));
        c2 = tolower((unsigned char)(*str2++));
        if (c1 == 0 || c1 != c2) return c1 == c2 ? 0 : c1 > c2 ? 1 : -1;
    }
}


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

    s = buff;
}


std::vector<std::string> split(const std::string &text, char sep)
{
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
            if (params->show_speed)
                printf("Compressor name,Compression speed,Decompression speed,Original size,Compressed size,Ratio,Filename\n");
            else
                printf("Compressor name,Compression time in us,Decompression time in us,Original size,Compressed size,Ratio,Filename\n"); break;
            break;
        case TURBOBENCH:
            printf("  Compressed  Ratio   Cspeed   Dspeed         Compressor name Filename\n");
            break;
        case TEXT:
        case TEXT_FULL:
        {
            std::string threads_format;
            if (params->threads > 0 && params->codec_threads > 0)
                threads_format = "  C,D;I Threads";
            else if (params->threads > 0)
                threads_format = " C,D Threads";
            else if (params->codec_threads > 0)
                threads_format = "I_Threads";
            else
                threads_format = "       ";

            if (params->textformat == TEXT)
                printf("Compressor name %s Compress. Decompress. Compr. size  Ratio Filename\n", threads_format.c_str());
            else
                printf("Compressor name %s Compress. Decompress.  Orig. size  Compr. size  Ratio Filename\n", threads_format.c_str());
            break;
        }
        case MARKDOWN:
        {
            std::string threads_format, line_format;
            if (params->threads > 0 && params->codec_threads > 0) {
                threads_format = "   |C,D;I Threads"; line_format = "   | ----------- ";
            } else if (params->threads > 0) {
                threads_format = "  |C,D Threads"; line_format = "  | --------- ";
            } else if (params->codec_threads > 0) {
                threads_format = " |I_Threads"; line_format = " | ------- ";
            } else {
                threads_format = line_format = "      ";
            }
            printf("| Compressor name   %s| Compression| Decompress.| Compr. size | Ratio | Filename |\n", threads_format.c_str());
            printf("| ---------------   %s| -----------| -----------| ----------- | ----- | -------- |\n", line_format.c_str());
            break;
        }
        case MARKDOWN2:
            printf("| Compressor name         | Ratio | Compression| Decompress.|\n");
            printf("| ---------------         | ------| -----------| ---------- |\n");
            break;
    }
}


void print_speed(lzbench_params_t *params, string_table_t& row)
{
    float cspeed, dspeed, ratio;
    cspeed = (!row.col2_ctime) ? 0 : (row.col5_origsize * 1000.0 / row.col2_ctime);
    dspeed = (!row.col3_dtime) ? 0 : (row.col5_origsize * 1000.0 / row.col3_dtime);
    ratio = row.col4_comprsize * 100.0 / row.col5_origsize;

    switch (params->textformat)
    {
        case CSV:
            printf("%s,%.2f,%.2f,%llu,%llu,%.2f,%s\n", row.col1_algname.c_str(), cspeed, dspeed, (unsigned long long)row.col5_origsize, (unsigned long long)row.col4_comprsize, ratio, row.col6_filename.c_str()); break;
        case TURBOBENCH:
            printf("%12llu %6.1f%9.2f%9.2f  %22s %s\n", (unsigned long long)row.col4_comprsize, ratio, cspeed, dspeed, row.col1_algname.c_str(), row.col6_filename.c_str()); break;
        case TEXT:
        case TEXT_FULL:
            printf("%-23s", row.col1_algname.c_str());
            if (params->threads > 0)
                printf("%2d,%2d", row.usedCompThreads, row.usedDecompThreads);
            if ((params->threads > 0) && (params->codec_threads > 0))
                printf(";");
            if (params->codec_threads > 0)
                printf("%2d", row.usedCodecThreads);

            if (cspeed) {
                if (cspeed < 10) printf("%6.2f MB/s", cspeed);
                else if (cspeed < 100) printf("%6.1f MB/s", cspeed);
                else printf("%6d MB/s", (int)cspeed);
            } else {
                printf("      ERROR");
            }
            if (dspeed) {
                if (dspeed < 10) printf("%6.2f MB/s", dspeed);
                else if (dspeed < 100) printf("%6.1f MB/s", dspeed);
                else printf("%6d MB/s", (int)dspeed);
            } else {
                printf("      ERROR");
            }
            if (params->textformat == TEXT_FULL)
                printf("%12llu %12llu %6.2f %s\n", (unsigned long long) row.col5_origsize, (unsigned long long)row.col4_comprsize, ratio, row.col6_filename.c_str());
            else
                printf("%12llu %6.2f %s\n", (unsigned long long)row.col4_comprsize, ratio, row.col6_filename.c_str());
            break;
        case MARKDOWN:
            printf("| %-23s ", row.col1_algname.c_str());
            if (params->threads > 0 || params->codec_threads > 0) {
                printf("| ");
                if (params->threads > 0)
                    printf("%2d,%2d", row.usedCompThreads, row.usedDecompThreads);
                if ((params->threads > 0) && (params->codec_threads > 0))
                    printf(";");
                if (params->codec_threads > 0)
                    printf("%2d", row.usedCodecThreads);
                printf(" ");
            }

            if (cspeed) {
                if (cspeed < 10) printf("|%6.2f MB/s ", cspeed);
                else if (cspeed < 100) printf("|%6.1f MB/s ", cspeed);
                else printf("|%6d MB/s ", (int)cspeed);
            } else {
                printf("|      ERROR ");
            }
            if (dspeed) {
                if (dspeed < 10) printf("|%6.2f MB/s ", dspeed);
                else if (dspeed < 100) printf("|%6.1f MB/s ", dspeed);
                else printf("|%6d MB/s ", (int)dspeed);
            } else {
                printf("|      ERROR ");
            }
            printf("|%12llu |%6.2f | %-s|\n", (unsigned long long)row.col4_comprsize, ratio, row.col6_filename.c_str());
            break;
        case MARKDOWN2:
            ratio = 1.0*row.col5_origsize / row.col4_comprsize;
            printf("| %-23s |%6.3f ", row.col1_algname.c_str(), ratio);
            if (cspeed) {
                if (cspeed < 10) printf("|%6.2f MB/s ", cspeed);
                else if (cspeed < 100) printf("|%6.1f MB/s ", cspeed);
                else printf("|%6d MB/s ", (int)cspeed);
            } else {
                printf("|      ERROR ");
            }
            if (dspeed) {
                if (dspeed < 10) printf("|%6.2f MB/s ", dspeed);
                else if (dspeed < 100) printf("|%6.1f MB/s ", dspeed);
                else printf("|%6d MB/s ", (int)dspeed);
            } else {
                printf("|      ERROR ");
            }
            printf("|\n");
            break;
    }
}


void print_time(lzbench_params_t *params, string_table_t& row)
{
    float ratio = row.col4_comprsize * 100.0 / row.col5_origsize;
    uint64_t ctime = row.col2_ctime / 1000;
    uint64_t dtime = row.col3_dtime / 1000;

    switch (params->textformat)
    {
        case CSV:
            printf("%s,%llu,%llu,%llu,%llu,%.2f,%s\n", row.col1_algname.c_str(), (unsigned long long)ctime, (unsigned long long)dtime,  (unsigned long long) row.col5_origsize, (unsigned long long)row.col4_comprsize, ratio, row.col6_filename.c_str()); break;
        case TURBOBENCH:
            printf("%12llu %6.1f%9llu%9llu  %22s %s\n", (unsigned long long)row.col4_comprsize, ratio, (unsigned long long)ctime, (unsigned long long)dtime, row.col1_algname.c_str(), row.col6_filename.c_str()); break;
        case TEXT:
        case TEXT_FULL:
            printf("%-23s", row.col1_algname.c_str());
            printf("%8llu us", (unsigned long long)ctime);
            if (!dtime)
                printf("      ERROR");
            else
                printf("%8llu us", (unsigned long long)dtime);
            if (params->textformat == TEXT_FULL)
                printf("%12llu %12llu %6.2f %s\n", (unsigned long long) row.col5_origsize, (unsigned long long)row.col4_comprsize, ratio, row.col6_filename.c_str());
            else
                printf("%12llu %6.2f %s\n", (unsigned long long)row.col4_comprsize, ratio, row.col6_filename.c_str());
            break;
        case MARKDOWN:
        case MARKDOWN2:
            printf("| %-23s ", row.col1_algname.c_str());
            printf("|%8llu us ", (unsigned long long)ctime);
            if (!dtime)
                printf("|      ERROR ");
            else
                printf("|%8llu us ", (unsigned long long)dtime);
            printf("|%12llu |%6.2f | %-s|\n", (unsigned long long)row.col4_comprsize, ratio, row.col6_filename.c_str());
            break;
    }
}


void print_stats(lzbench_params_t *params, const compressor_desc_t* desc, int level, std::vector<uint64_t> &ctime, std::vector<uint64_t> &dtime, size_t insize, size_t outsize, bool comp_error, bool decomp_error, int used_comp_threads, int used_decomp_threads, int used_codec_threads)
{
    std::string col1_algname;
    std::sort(ctime.begin(), ctime.end());
    std::sort(dtime.begin(), dtime.end());
    uint64_t best_ctime, best_dtime;

    switch (params->timetype)
    {
        default:
        case FASTEST:
            best_ctime = ctime.empty()?0:ctime[0];
            best_dtime = dtime.empty()?0:dtime[0];
            break;
        case AVERAGE:
            best_ctime = ctime.empty()?0:std::accumulate(ctime.begin(),ctime.end(),(uint64_t)0) / ctime.size();
            best_dtime = dtime.empty()?0:std::accumulate(dtime.begin(),dtime.end(),(uint64_t)0) / dtime.size();
            break;
        case MEDIAN:
            best_ctime = ctime.empty()?0:(ctime[(ctime.size()-1)/2] + ctime[ctime.size()/2]) / 2;
            best_dtime = dtime.empty()?0:(dtime[(dtime.size()-1)/2] + dtime[dtime.size()/2]) / 2;
            break;
    }

    if (desc->first_level == 0 && desc->last_level==0)
        format(col1_algname, "%s", desc->name_version);
    else
        format(col1_algname, "%s -%d", desc->name_version, level);

    LZBENCH_PRINT(9, "ALL best_ctime=%llu best_dtime=%llu\n", (uint64)((comp_error)?0:best_ctime), (uint64)((decomp_error)?0:best_dtime));
    params->results.push_back(string_table_t(col1_algname, (comp_error)?0:best_ctime, (decomp_error)?0:best_dtime, outsize, insize, params->in_filename, used_comp_threads, used_decomp_threads, used_codec_threads));
    if (params->show_speed)
        print_speed(params, params->results[params->results.size()-1]);
    else
        print_time(params, params->results[params->results.size()-1]);

    fflush(stdout);
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

/*
 * Allocate a buffer of size bytes using malloc (or equivalent call returning a buffer
 * that can be passed to free). Touches each page so that the each page is actually
 * physically allocated and mapped into the process.
 */
void *alloc_and_touch(size_t size, bool must_zero) {
    void *buf = must_zero ? calloc(1, size) : malloc(size);
    volatile char zero = 0;
    for (size_t i = 0; i < size; i += MIN_PAGE_SIZE) {
        static_cast<char * volatile>(buf)[i] = zero;
    }
    return buf;
}

#ifndef DISABLE_THREADING
inline int64_t lzbench_compress_mt(ThreadPool& pool, lzbench_params_t *params, const std::vector<size_t>& chunk_sizes, compress_func compress, std::vector<size_t> &compr_sizes, uint8_t *inbuf, uint8_t *outbuf, size_t outsize, codec_options_t *codec_options, std::vector<char*> &workmems)
{
    int64_t clen;
    size_t outpart, part, csum = 0, dsum = 0;
    uint8_t *start = inbuf;
    size_t cscount = chunk_sizes.size();

    compr_sizes.resize(cscount);

    for (size_t i=0; i<cscount; i++)
    {
        part = chunk_sizes[i];
        outpart = GET_COMPRESS_BOUND(part);
        if (outpart > outsize) outpart = outsize;
        compr_sizes[i] = outpart;

        pool.enqueue({true, i, inbuf, part, outbuf, outpart, compress, codec_options, &workmems }); // result goes to pool.compSizes[i]

        inbuf += part;
        outbuf += outpart;
        outsize -= outpart;

        LZBENCH_PRINT(9, "ENCMT enqueue=%zu/%zu part=%zu/%zu in=%llu\n", i, cscount, part, outpart, (uint64)(inbuf-start));
    }

    pool.waitForCompletion();  // Wait for all compression tasks to finish

    for (size_t i=0; i<cscount; i++)
    {
        outpart = compr_sizes[i]; // max compressed size
        clen = compr_sizes[i] = pool.compSizes[i];

        if (clen > outpart) {
            LZBENCH_PRINT(0, "Compression ERROR: incompressible data in=%zu out=%lld/%zu (in_bytes=%zu out_bytes=%zu)\n", chunk_sizes[i], (int64)clen, outpart, dsum, csum);
            return 0;
        }

        if (clen <= 0) {
            LZBENCH_PRINT(0, "ERROR: compression error in=%zu out=%lld/%zu (in_bytes=%zu out_bytes=%zu)\n", chunk_sizes[i], (int64)clen, outpart, dsum, csum);
            return 0;
        }
        csum += clen;
        dsum += chunk_sizes[i];
        LZBENCH_PRINT(9, "ENCMT chunk=%zu/%zu part=%zu clen=%lld out=%zu->%zu\n", i, cscount, chunk_sizes[i], (int64)clen, dsum, csum);
    }

    return csum;
}


inline int64_t lzbench_decompress_mt(ThreadPool& pool, lzbench_params_t *params, const std::vector<size_t>& chunk_sizes, compress_func decompress, std::vector<size_t> &compr_sizes, uint8_t *inbuf, uint8_t *outbuf, codec_options_t *codec_options, std::vector<char*> &workmems)
{
    int64_t dlen;
    size_t part, csum = 0, dsum = 0;
    uint8_t *outstart = outbuf;
    size_t cscount = compr_sizes.size();

    for (size_t i=0; i<cscount; i++)
    {
        part = compr_sizes[i];
        dlen = chunk_sizes[i];
        pool.enqueue({false, i, inbuf, part, outbuf, chunk_sizes[i], decompress, codec_options, &workmems}); // result goes to pool.chunkSizes[i]

        inbuf += GET_COMPRESS_BOUND(dlen);
        outbuf += dlen;
        LZBENCH_PRINT(9, "DECMT enqueue=%zu/%zu part=%zu dlen=%lld in=%llu\n", i, cscount, part, (int64)dlen, (uint64)(outbuf - outstart));
    }

    pool.waitForCompletion();  // Wait for all compression tasks to finish

    for (size_t i=0; i<cscount; i++)
    {
        part = compr_sizes[i];
        dlen = pool.chunkSizes[i];
        if (dlen <= 0) {
            LZBENCH_PRINT(9, "DECMT chunk=%zu/%zu part=%zu dlen=%lld out=%llu\n", i, cscount, part, (int64)dlen, (uint64)(outbuf - outstart));
            return dlen;
        }
        csum += part;
        dsum += dlen;
        LZBENCH_PRINT(9, "DECMT chunk=%zu/%zu part=%zu dlen=%lld out=%zu->%zu\n", i, cscount, part, (int64)dlen, csum, dsum);
    }

    return dsum;
}
#endif // #ifndef DISABLE_THREADING

inline int64_t lzbench_compress(lzbench_params_t *params, const std::vector<size_t>& chunk_sizes, compress_func compress, std::vector<size_t> &compr_sizes, uint8_t *inbuf, uint8_t *outbuf, size_t outsize, codec_options_t *codec_options)
{
    int64_t clen;
    size_t outpart, part, sum = 0;
    uint8_t *start = inbuf;
    size_t cscount = chunk_sizes.size();

    compr_sizes.resize(cscount);

    for (size_t i=0; i<cscount; i++)
    {
        part = chunk_sizes[i];
        outpart = GET_COMPRESS_BOUND(part);
        if (outpart > outsize) outpart = outsize;

        clen = compress((char*)inbuf, part, (char*)outbuf, outpart, codec_options);

        if (clen <= 0)
        {
            LZBENCH_PRINT(9, "ERROR: part=%zu clen=%lld in=%llu out=%zu\n", part, (int64)clen, (uint64)(inbuf-start), sum);
            LZBENCH_PRINT(0, "ERROR: compression error in=%zu out=%lld/%zu (in_bytes=%llu out_bytes=%llu)\n", part, (int64)clen, outpart, (uint64)(inbuf+part-start), (uint64)sum+clen);
            return 0;
        }

        inbuf += part;
        outbuf += clen;
        outsize -= clen;
        compr_sizes[i] = clen;
        sum += clen;
        LZBENCH_PRINT(9, "ENC chunk=%zu/%zu part=%zu clen=%lld out=%llu->%zu\n", i, cscount, chunk_sizes[i], (int64)clen, (uint64)(inbuf-start), sum);
    }
    return sum;
}


inline int64_t lzbench_decompress(lzbench_params_t *params, const std::vector<size_t>& chunk_sizes, compress_func decompress, std::vector<size_t> &compr_sizes, uint8_t *inbuf, uint8_t *outbuf, codec_options_t *codec_options)
{
    int64_t dlen;
    size_t part, sum = 0;
    uint8_t *outstart = outbuf;
    int cscount = compr_sizes.size();

    for (int i=0; i<cscount; i++)
    {
        part = compr_sizes[i];

#ifndef NDEBUG
        if (params->verbose >= 10) {
            FILE *f = fopen("last_to_decomp", "wb");
            if (f) fwrite(inbuf, 1, part, f), fclose(f);
        }
#endif

        dlen = decompress((char*)inbuf, part, (char*)outbuf, chunk_sizes[i], codec_options);

        if (dlen <= 0) {
            LZBENCH_PRINT(9, "DEC part=%zu dlen=%lld out=%llu\n",part, (int64)dlen, (uint64)(outbuf - outstart));
            return dlen;
        }

        inbuf += part;
        outbuf += dlen;
        sum += dlen;
        LZBENCH_PRINT(9, "DEC part=%zu dlen=%lld out=%llu\n", part, (int64)dlen, (uint64)(outbuf - outstart));
    }

    return sum;
}


void lzbench_process_single_codec(ThreadPool& pool, int numThreads, lzbench_params_t *params, size_t max_chunk_size, const std::vector<size_t> &chunk_sizes, const compressor_desc_t* desc, int level, uint8_t *inbuf, size_t insize, uint8_t *compbuf, size_t comprsize, uint8_t *decomp, bench_rate_t rate, int param1)
{
    int codec_threads = params->codec_threads;

    if (!(desc->mt_mode & BENCH_POOL_MT)) numThreads = 1;  // No support for lzbench's external thread pool
    if (!(desc->mt_mode & INTERNAL_MT)) codec_threads = 1; // No support for internal (built-in) multithreading

    float speed;
    int i, total_c_iters, total_d_iters;
    bench_timer_t loop_ticks, start_ticks, end_ticks, timer_ticks;
    int64_t complen=0, decomplen;
    uint64_t nanosec, total_nanosec;
    std::vector<uint64_t> ctime, dtime;
    std::vector<size_t> compr_sizes;
    bool comp_error = false, decomp_error = false;
    int param2 = desc->additional_param;
    size_t compThreadsUsed = (numThreads <= 1), decompThreadsUsed = (numThreads <= 1), codecThreadsUsed = (codec_threads);
    std::vector<char*> workmems(numThreads, nullptr);


    LZBENCH_PRINT(5, "*** trying %s insize=%zu comprsize=%zu chunk_size=%zu\n", desc->name, insize, comprsize, max_chunk_size);

    if (!desc->compress || !desc->decompress) return;
    if (level < desc->first_level || level > desc->last_level) {
        LZBENCH_PRINT(0, "ERROR in %s: level %d out of range (%d, %d)\n", desc->name, level, desc->first_level, desc->last_level);
        return;
    }

    if (desc->init) {
        for (int i = 0; i < numThreads; i++) {
            if (desc->init) {
                workmems[i] = desc->init(max_chunk_size, param1, param2);
            }
        }
    }

    codec_options_t codec_options { param1, param2, workmems[0], (codec_threads <= 1) ? 1 : codec_threads };

    if (params->cspeed > 0)
    {
        size_t part = MIN(100*1024, max_chunk_size);
        GetTime(start_ticks);
        int64_t clen = desc->compress((char*)inbuf, part, (char*)compbuf, GET_COMPRESS_BOUND(part), &codec_options);
        GetTime(end_ticks);
        nanosec = GetDiffTime(rate, start_ticks, end_ticks)/1000;
        if (clen>0 && nanosec>=1000)
        {
            part = (part / nanosec); // speed in MB/s
            if (part < params->cspeed) { LZBENCH_PRINT(7, "%s (100K) slower than %zu MB/s nanosec=%llu\n", desc->name, part, (uint64)nanosec); goto done; }
        }
    }

    LZBENCH_PRINT(5, "%s chunk_sizes=%d\n", desc->name, (int)chunk_sizes.size());

    total_c_iters = 0;
    GetTime(timer_ticks);

    do
    {
        i = 0;
        std::this_thread::yield(); // give processor to other processes
        GetTime(loop_ticks);
        do
        {
            GetTime(start_ticks);

#ifndef DISABLE_THREADING
            if (numThreads > 1) {
                complen = lzbench_compress_mt(pool, params, chunk_sizes, desc->compress, compr_sizes, inbuf, compbuf, comprsize, &codec_options, workmems);
            }
            else
#endif // #ifndef DISABLE_THREADING
            {
                complen = lzbench_compress(params, chunk_sizes, desc->compress, compr_sizes, inbuf, compbuf, comprsize, &codec_options);
            }
            if (complen == 0) {
               comp_error = true;
               g_exit_result = 10; // lzbench will return 10 to shell
               goto stats;
            }

            GetTime(end_ticks);
            nanosec = GetDiffTime(rate, start_ticks, end_ticks);
            if (nanosec >= 10000) ctime.push_back(nanosec);
            i++;
        }
        while (GetDiffTime(rate, loop_ticks, end_ticks) < params->cloop_time);

        nanosec = GetDiffTime(rate, loop_ticks, end_ticks);
        ctime.push_back(nanosec/i);
        speed = (float)insize*i*1000/nanosec;

        if ((uint32_t)speed < params->cspeed) { LZBENCH_PRINT(7, "%s slower than %llu MB/s\n", desc->name, (uint64)speed); return; }

        total_nanosec = GetDiffTime(rate, timer_ticks, end_ticks);
        total_c_iters += i;
        LZBENCH_PRINT(8, "ENC %s nanosec=%llu iters=%d/%d\n", desc->name, (uint64)nanosec, total_c_iters, params->c_iters);
        if ((total_c_iters >= params->c_iters) && (total_nanosec > ((uint64_t)params->cmintime*1000000))) break;
        LZBENCH_STDERR(2, "%s compr iter=%d time=%.2fs speed=%.2f MB/s     \r", desc->name, total_c_iters, total_nanosec/1000000000.0, speed);
    }
    while (true);


    total_d_iters = 0;
    GetTime(timer_ticks);
    if (!params->compress_only)
    do
    {
        i = 0;
        std::this_thread::yield(); // give processor to other processes
        GetTime(loop_ticks);
        do
        {
            GetTime(start_ticks);
#ifndef DISABLE_THREADING
            if (numThreads > 1) {
                decomplen = lzbench_decompress_mt(pool, params, chunk_sizes, desc->decompress, compr_sizes, compbuf, decomp, &codec_options, workmems);
            }
            else
#endif // #ifndef DISABLE_THREADING
            {
                decomplen = lzbench_decompress(params, chunk_sizes, desc->decompress, compr_sizes, compbuf, decomp, &codec_options);
            }
            GetTime(end_ticks);
            nanosec = GetDiffTime(rate, start_ticks, end_ticks);
            if (nanosec >= 10000) dtime.push_back(nanosec);
            i++;
        }
        while (GetDiffTime(rate, loop_ticks, end_ticks) < params->dloop_time);

        nanosec = GetDiffTime(rate, loop_ticks, end_ticks);
        dtime.push_back(nanosec/i);

        if (insize != decomplen)
        {
            decomp_error = true;
            LZBENCH_PRINT(0, "ERROR in %s: decompressed size mismatch in_bytes[%zu] != out_bytes[%lld]\n", desc->name, insize, (int64)decomplen);
        }

        if (memcmp(inbuf, decomp, insize) != 0)
        {
            decomp_error = true;

            size_t cmn = common(inbuf, decomp);
            LZBENCH_PRINT(0, "ERROR in %s: decompressed bytes common=%zu/%zu\n", desc->name, cmn, insize);

            if (params->verbose >= 10)
            {
                char text[256];
                snprintf(text, sizeof(text), "%s_failed", desc->name);
                cmn /= max_chunk_size;
                size_t err_size = MIN(insize, (cmn+1)*max_chunk_size);
                err_size -= cmn*max_chunk_size;
                printf("ERROR: fwrite %llu-%llu to %s\n", (uint64)(cmn*max_chunk_size), (uint64)(cmn*max_chunk_size+err_size), text);
                FILE *f = fopen(text, "wb");
                if (f) fwrite(decomp+cmn*max_chunk_size, 1, err_size, f), fclose(f);
                exit(1);
            }
        }

        memset(decomp, 0, insize); // clear output buffer

        if (decomp_error) {
            g_exit_result = 11; // lzbench will return 11 to shell
            break;
        }

        total_nanosec = GetDiffTime(rate, timer_ticks, end_ticks);
        total_d_iters += i;
        LZBENCH_PRINT(9, "DEC %s dnanosec=%llu iters=%d/%d\n", desc->name, (uint64)nanosec, i, params->d_iters);
        if ((total_d_iters >= params->d_iters) && (total_nanosec > ((uint64_t)params->dmintime*1000000))) break;
        LZBENCH_STDERR(2, "%s decompr iter=%d time=%.2fs speed=%.2f MB/s     \r", desc->name, total_d_iters, total_nanosec/1000000000.0, (float)insize*i*1000/nanosec);
    }
    while (true);

stats:
#ifndef DISABLE_THREADING
    for (size_t i = 0; i < numThreads; ++i) {
        //fprintf(stdout, "T%zu=%zu/%zu ", i, pool.comptasksDone[i], pool.decomptasksDone[i]);
        if (pool.comptasksDone[i] > 0) compThreadsUsed++;
        if (pool.decomptasksDone[i] > 0) decompThreadsUsed++;
    }
    pool.clear();
#endif // #ifndef DISABLE_THREADING
    print_stats(params, desc, level, ctime, dtime, insize, complen, comp_error, decomp_error, compThreadsUsed, decompThreadsUsed, codecThreadsUsed);

done:
    if (desc->deinit) {
        for (int i = 0; i < numThreads; i++) {
            if (workmems[i]) desc->deinit(workmems[i]);
        }
    }
}


int lzbench_process_codec_list(lzbench_params_t *params, size_t max_chunk_size, std::vector<size_t> &chunk_sizes, const char *namesWithParams, uint8_t *inbuf, size_t insize, uint8_t *compbuf, size_t comprsize, uint8_t *decomp, bench_rate_t rate)
{
    std::vector<std::string> cnames, cparams;
    int numThreads = params->threads > 0 ? params->threads : 1;
#ifndef DISABLE_THREADING
    ThreadPool pool(numThreads, chunk_sizes.size());
#else
    ThreadPool pool;
#endif // #ifndef DISABLE_THREADING

    if (!namesWithParams) return numThreads;

    LZBENCH_PRINT(5, "*** lzbench_process_codec_list insize=%zu comprsize=%zu\n", insize, comprsize);

    cnames = split(namesWithParams, '/');

    for (int k=0; k<cnames.size(); k++)
        LZBENCH_PRINT(5, "cnames[%d] = %s\n", k, cnames[k].c_str());

    for (int k=0; k<cnames.size(); k++)
    {
        for (int i=0; i<LZBENCH_ALIASES_COUNT; i++)
        {
            if (istrcmp(cnames[k].c_str(), alias_desc[i].name)==0)
            {
                lzbench_process_codec_list(params, max_chunk_size, chunk_sizes, alias_desc[i].params, inbuf, insize, compbuf, comprsize, decomp, rate);
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
                for (int i=0; i<LZBENCH_COMPRESSOR_COUNT; i++)
                {
                    if (istrcmp(comp_desc[i].name, cparams[0].c_str()) == 0)
                    {
                        found = true;
                       // printf("%s %s %s\n", cparams[0].c_str(), comp_desc[i].version, cparams[j].c_str());
                        if (j >= cparams.size())
                        {
                            for (int level=comp_desc[i].first_level; level<=comp_desc[i].last_level; level++)
                                lzbench_process_single_codec(pool, numThreads, params, max_chunk_size, chunk_sizes, &comp_desc[i], level, inbuf, insize, compbuf, comprsize, decomp, rate, level);
                        }
                        else
                            lzbench_process_single_codec(pool, numThreads, params, max_chunk_size, chunk_sizes, &comp_desc[i], atoi(cparams[j].c_str()), inbuf, insize, compbuf, comprsize, decomp, rate, atoi(cparams[j].c_str()));
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

    return numThreads;
}


void lzbench_process_mem_blocks(lzbench_params_t *params, size_t max_chunk_size, std::vector<size_t> &file_sizes, const char *namesWithParams, uint8_t *inbuf, size_t insize, bench_rate_t rate)
{
    uint8_t *compbuf, *decomp;
    size_t comprsize;
    std::vector<size_t> chunk_sizes;

    if (max_chunk_size > 0) {
        max_chunk_size = (max_chunk_size > insize) ? insize : max_chunk_size;
    } else {
        max_chunk_size = insize;
    }

    for (int i=0; i<file_sizes.size(); i++) {
        size_t tmpsize = file_sizes[i];
        while (tmpsize > 0)
        {
            chunk_sizes.push_back(MIN(tmpsize, max_chunk_size));
            tmpsize -= MIN(tmpsize, max_chunk_size);
        }
    }

    comprsize = GET_COMPRESS_BOUND(insize) + chunk_sizes.size() * PAD_SIZE;
    compbuf = (uint8_t*)alloc_and_touch(comprsize, false);
    decomp = (uint8_t*)alloc_and_touch(insize + PAD_SIZE, true);

    if (!compbuf || !decomp)
    {
        printf("Not enough memory, please use -m option!\n");
        g_exit_result=3;
        return;
    }

    LZBENCH_PRINT(5, "file_sizes=%zu chunk_sizes=%zu\n", file_sizes.size(), chunk_sizes.size());

    int numThreads = lzbench_process_codec_list(params, max_chunk_size, chunk_sizes, namesWithParams, inbuf, insize, compbuf, comprsize, decomp, rate);

    if (chunk_sizes.size() > 1 || numThreads > 1)
        LZBENCH_PRINT(3, "[Summary] Files=%zu Chunks=%zu ChunkSize=%zu Threads=%d\n", file_sizes.size(), chunk_sizes.size(), max_chunk_size, numThreads);

    free(compbuf);
    free(decomp);
}


int lzbench_join(lzbench_params_t* params, const char** inFileNames, unsigned ifnIdx, char* encoder_list)
{
    bench_rate_t rate;
    size_t insize, inpos, totalsize;
    uint8_t *inbuf;
    std::vector<size_t> file_sizes;
    std::string text;
    FILE* in;
    const char* pch;

    totalsize = UTIL_getTotalFileSize(inFileNames, ifnIdx);
    if (totalsize == 0) {
        printf("Could not find input files\n");
        return 1;
    }

    inbuf = (uint8_t*)alloc_and_touch(totalsize + PAD_SIZE, false);

    if (!inbuf)
    {
        printf("Not enough memory, please use -m option!\n");
        return 2;
    }

    InitTimer(rate);
    inpos = 0;

    for (int i=0; i<ifnIdx; i++)
    {
        if (UTIL_isDirectory(inFileNames[i])) {
            fprintf(stderr, "warning: use -r to process directories (%s)\n", inFileNames[i]);
            continue;
        }

        if (!(in=fopen(inFileNames[i], "rb"))) {
            perror(inFileNames[i]);
            continue;
        }

        fseeko(in, 0L, SEEK_END);
        insize = ftello(in);
        rewind(in);

        if (inpos + insize > totalsize) { printf("inpos + insize > totalsize\n"); goto _clean; };
        insize = fread(inbuf+inpos, 1, insize, in);
        file_sizes.push_back(insize);
        inpos += insize;
        fclose(in);
    }

    if (file_sizes.size() == 0)
        goto _clean;

    format(text, "%d files", (int)file_sizes.size());
    params->in_filename = text.c_str();

    LZBENCH_PRINT(5, "totalsize=%zu inpos=%zu\n", totalsize, inpos);
    totalsize = inpos;

    print_header(params);
    lzbench_process_mem_blocks(params, params->chunk_size, file_sizes, encoder_list?encoder_list:alias_desc[0].params, inbuf, totalsize, rate);

_clean:
    free(inbuf);

    return g_exit_result;
}


int lzbench_main(lzbench_params_t* params, const char** inFileNames, unsigned ifnIdx, char* encoder_list)
{
    bench_rate_t rate;
    size_t insize, real_insize;
    uint8_t *inbuf;
    std::vector<size_t> file_sizes;
    FILE* in;
    const char* pch;

    for (int i=0; i<ifnIdx; i++)
    {
        if (UTIL_isDirectory(inFileNames[i])) {
            fprintf(stderr, "warning: use -r to process directories (%s)\n", inFileNames[i]);
            continue;
        }

        if (!(in=fopen(inFileNames[i], "rb"))) {
            perror(inFileNames[i]);
            continue;
        }

        pch = strrchr(inFileNames[i], '\\');
        params->in_filename = pch ? pch+1 : inFileNames[i];

        InitTimer(rate);

        fseeko(in, 0L, SEEK_END);
        real_insize = ftello(in);
        rewind(in);

        if (i == 0) print_header(params);

        if (params->mem_limit && real_insize > params->mem_limit)
            insize = params->mem_limit;
        else
            insize = real_insize;

        inbuf = (uint8_t*)alloc_and_touch(insize + PAD_SIZE, false);

        if (!inbuf) {
            printf("Not enough memory, please use -m option!");
            return 3;
        }

        if (params->random_read){
          unsigned long long pos = 0;
          if (params->chunk_size > 0 && real_insize > params->chunk_size){
            pos = (rand() % (real_insize / params->chunk_size)) * params->chunk_size;
            insize = params->chunk_size;
            fseeko(in, pos, SEEK_SET);
          } else {
            insize = real_insize;
          }
          printf("Seeking to: %llu, reading %llu bytes\n", pos, (unsigned long long)insize);
        }

        insize = fread(inbuf, 1, insize, in);

        if (insize == 0) {
            LZBENCH_PRINT(2, "[Warning] File %s is empty and will be ignored\n", inFileNames[i]);
            continue;
        }

        size_t max_chunk_size = params->chunk_size;
        if (params->chunk_size == 0 && params->threads > 1) {
            max_chunk_size = (insize + params->threads - 1) / params->threads;
            LZBENCH_PRINT(5, "set max_chunk_size=%zu insize=%zu params->threads=%d\n", max_chunk_size, insize, params->threads);
        }

        if (params->mem_limit && real_insize > params->mem_limit) {
            int i;
            std::string partname;
            const char* filename = params->in_filename;
            for (i=1; insize > 0; i++)
            {
                format(partname, "%s part %d", filename, i);
                params->in_filename = partname.c_str();
                file_sizes.push_back(insize);
                lzbench_process_mem_blocks(params, max_chunk_size, file_sizes, encoder_list?encoder_list:alias_desc[0].params, inbuf, insize, rate);
                file_sizes.clear();
                insize = fread(inbuf, 1, insize, in);
            }
        } else {
            file_sizes.push_back(insize);
            lzbench_process_mem_blocks(params, max_chunk_size, file_sizes, encoder_list?encoder_list:alias_desc[0].params, inbuf, insize, rate);
            file_sizes.clear();
        }

        fclose(in);
        free(inbuf);
    }

    return g_exit_result;
}


void usage(lzbench_params_t* params)
{
    fprintf(stdout, "lzbench - in-memory benchmark of open-source compressors\n\n");
    fprintf(stdout, "usage: " PROGNAME " [options] [input]\n\nwhere [input] is a file/s or a directory and [options] are:\n");
    fprintf(stdout, "  -b#   set block/chunk size to # KB, 0=disabled {default: %llu}\n", (uint64)(params->chunk_size>>10));
    fprintf(stdout, "  -c#   sort results by column # (1=algname, 2=ctime, 3=dtime, 4=comprsize)\n");
    fprintf(stdout, "  -e#   #=compressors separated by '/' with parameters specified after ',' {fast}\n");
    fprintf(stdout, "  -h    display this help and exit\n");
    fprintf(stdout, "  -I#   use # internal threads (if compressor supports it)\n");
    fprintf(stdout, "  -iX,Y set min. number of compression and decompression iterations {%d, %d}\n", params->c_iters, params->d_iters);
    fprintf(stdout, "  -j    join files in memory but compress them independently (for many small files)\n");
    fprintf(stdout, "  -l    list of available compressors and aliases\n");
    fprintf(stdout, "  -m#   set memory limit to # MB {no limit}\n");
    fprintf(stdout, "  -o#   output text format 1=Markdown, 2=text, 3=text+origSize, 4=CSV {%d}\n", params->textformat);
    fprintf(stdout, "  -p#   print time for all iterations: 1=fastest 2=average 3=median {%d}\n", params->timetype);
    fprintf(stdout, "  -q    suppress progress information (-qq supresses more)\n");
    fprintf(stdout, "  -R    read block/chunk size from random blocks (to estimate for large files)\n");
#ifdef UTIL_HAS_CREATEFILELIST
    fprintf(stdout, "  -r    operate recursively on directories\n");
#endif
    fprintf(stdout, "  -s#   use only compressors with compression speed over # MB {%d MB}\n", params->cspeed);
    fprintf(stdout, "  -T#   use # thread pool threads (works with -b to split input into blocks)\n");
    fprintf(stdout, "  -tX,Y set min. time in seconds for compression and decompression {%.0f, %.0f}\n", params->cmintime/1000.0, params->dmintime/1000.0);
    fprintf(stdout, "  -v    be verbose (-vv gives more)\n");
    fprintf(stdout, "  -V    output version information and exit\n");
    fprintf(stdout, "  -x    disable real-time process priority\n");
    fprintf(stdout, "  -z    show (de)compression times instead of speed\n");
    fprintf(stdout, "\nExample usage:\n");
    fprintf(stdout, "  " PROGNAME " -ezstd filename = selects all levels of zstd\n");
    fprintf(stdout, "  " PROGNAME " -ebrotli,2,5/zstd filename = selects levels 2 & 5 of brotli and zstd\n");
    fprintf(stdout, "  " PROGNAME " -t3,5 fname = 3 sec compression and 5 sec decompression loops\n");
    fprintf(stdout, "  " PROGNAME " -t0,0 -i3,5 fname = 3 compression and 5 decompression iterations\n");
    fprintf(stdout, "  " PROGNAME " -o1c4 fname = output markdown format and sort by 4th column\n");
    fprintf(stdout, "  " PROGNAME " -j -r dirname/ = recursively select and join files in given directory\n");
}

void show_version()
{
    fprintf(stdout,
            "" PROGNAME " " PROGVERSION "\n"
            "Copyright (C) 2011-2025 Przemyslaw Skibinski\n"
            "License GPL v2 or v3: GNU GPL version 2 or 3 <http://gnu.org/licenses/gpl.html>\n"
            "This is free software: you are free to change and redistribute it.\n"
            "There is NO WARRANTY, to the extent permitted by law.\n" );
}

#define xstr(s) str(s)
#define str(s) #s

static const char *get_compiler_information(void)
{
#if defined __clang__
    return "Clang " xstr(__clang_major__) "." xstr(__clang_minor__) "." xstr(__clang_patchlevel__);
#elif defined __GNUC__
    //return "gcc " __VERSION__;
    return "GCC " xstr(__GNUC__) "." xstr(__GNUC_MINOR__) "." xstr(__GNUC_PATCHLEVEL__);
#elif defined _MSC_VER
    return "MSVC " xstr(_MSC_VER);
#else
    return "unknown compiler";
#endif
}

char* cpu_brand_string(void)
{
    uint32_t mx[4], i, a, b, c, d;

    #if (defined(__i386__) || defined(__x86_64__))
    char* cpu_brand_str = (char*)calloc(1, 3*sizeof(mx)+1);
    if (!cpu_brand_str)
        return NULL;

    __cpuid(CPUID_EXTENDED, a, b, c, d); // check availability of extended functions
    if (a >= CPUID_BRANDSTRINGEND)
        {
        for(i=0; i<=2; i++)
            {
            cpuid_string(CPUID_BRANDSTRING+i, (uint32_t*)mx);
            strncpy(cpu_brand_str+sizeof(mx)*i, (char*)mx, sizeof(mx));
            }
        }
    else
        return NULL; // CPUID_EXTENDED unsupported by cpu

    cpu_brand_str[3*sizeof(mx)] = '\0';
    return cpu_brand_str;
    #else
    return NULL;
    #endif // (defined(__i386__) || defined(__x86_64__))
}


int main( int argc, char** argv)
{
    FILE *in;
    char* encoder_list = NULL;
    int result = 0, sort_col = 0, real_time = 1;
    lzbench_params_t lzparams;
    lzbench_params_t* params = &lzparams;
    const char** inFileNames = (const char**) calloc(argc, sizeof(char*));
    unsigned ifnIdx = 0;
    bool join = false;
    char* cpu_brand = NULL;
    const char* compiler_info = NULL;
#ifdef UTIL_HAS_CREATEFILELIST
    const char** extendedFileList = NULL;
    char* fileNamesBuf = NULL;
    unsigned fileNamesNb, recursive = 0;
#endif

    if (inFileNames==NULL) {
        LZBENCH_PRINT(2, "Allocation error : not enough memory%c\n", ' ');
        return 1;
    }

    memset(params, 0, sizeof(lzbench_params_t));
    params->timetype = FASTEST;
    params->textformat = TEXT;
    params->show_speed = 1;
    params->verbose = 2;
    params->chunk_size = 0;
    params->cspeed = 0;
    params->c_iters = params->d_iters = 1;
    params->cmintime = 10*DEFAULT_LOOP_TIME/1000000; // 1 sec
    params->dmintime = 20*DEFAULT_LOOP_TIME/1000000; // 2 sec
    params->cloop_time = params->dloop_time = DEFAULT_LOOP_TIME;
    params->threads = 0;
    params->codec_threads = 0;


    while ((argc>1) && (argv[1][0]=='-')) {
    char* argument = argv[1]+1;
    if (!strcmp(argument, "-compress-only")) params->compress_only = 1;
    else while (argument[0] != 0) {
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
            if (*numPtr == ',')
            {
                numPtr++;
                number = 0;
                while ((*numPtr >='0') && (*numPtr <='9')) { number *= 10;  number += *numPtr - '0'; numPtr++; }
                params->d_iters = number;
            }
            break;
        case 'j':
            join = true;
            break;
        case 'm':
            params->mem_limit = number << 18; /*  total memory usage = mem_limit * 4  */
            if (params->textformat == TEXT) params->textformat = TEXT_FULL;
            break;
        case 'o':
            params->textformat = (textformat_e)number;
            if (params->textformat == CSV) params->verbose = 0;
            break;
        case 'p':
            params->timetype = (timetype_e)number;
            break;
        case 'q':
            if (params->verbose > 0) params->verbose--;
            break;
#ifdef UTIL_HAS_CREATEFILELIST
        case 'r':
            recursive = 1;
            break;
#endif
        case 'R':
            params->random_read = 1;
            srand(time(NULL));
            break;
        case 's':
            params->cspeed = number;
            break;
        case 't':
            params->cmintime = 1000*number;
            params->cloop_time = (params->cmintime)?DEFAULT_LOOP_TIME:0;
            if (*numPtr == ',')
            {
                numPtr++;
                number = 0;
                while ((*numPtr >='0') && (*numPtr <='9')) { number *= 10;  number += *numPtr - '0'; numPtr++; }
                params->dmintime = 1000*number;
                params->dloop_time = (params->dmintime)?DEFAULT_LOOP_TIME:0;
            }
            break;
#ifndef DISABLE_THREADING
        case 'T':
            params->threads = number;
            break;
        case 'I':
            params->codec_threads = number;
            break;
#endif // #ifndef DISABLE_THREADING
        case 'u':
            params->dmintime = 1000*number;
            params->dloop_time = (params->dmintime)?DEFAULT_LOOP_TIME:0;
            break;
        case 'v':
            if (number > 0)
                params->verbose = number;
            else
                params->verbose++;
            break;
        case 'x':
            real_time = 0;
            break;
        case 'z':
            params->show_speed = 0;
            break;
        case '-': // --help
        case 'h':
            usage(params);
            goto _clean;
        case 'V':
            show_version();
            goto _clean;
        case 'l':
            printf("Available compressors for -e option:\n");
            for (int i=0; i<LZBENCH_COMPRESSOR_COUNT; i++)
            {
                if (comp_desc[i].compress)
                {
                    if (comp_desc[i].first_level < comp_desc[i].last_level)
                        printf("%s = %s; levels=[%d-%d]", comp_desc[i].name, comp_desc[i].name_version, comp_desc[i].first_level, comp_desc[i].last_level);
                    else
                        printf("%s = %s", comp_desc[i].name, comp_desc[i].name_version);

                    if (comp_desc[i].mt_mode == FULL_THREADING)
                        printf("; threading=-I,-T");
                    else if (comp_desc[i].mt_mode & INTERNAL_MT)
                        printf("; threading=-I");
                    else if (comp_desc[i].mt_mode & BENCH_POOL_MT)
                        ;//printf("threading=-T");
                    else
                        printf("; threading=none");
                    printf("\n");
                }
            }

            printf("\nAvailable aliases for -e option:\n");
            for (int i=0; i<LZBENCH_ALIASES_COUNT; i++)
            {
                if (alias_desc[i].description)
                    printf("%s: %s\n%s = %s\n\n", alias_desc[i].name, alias_desc[i].description, alias_desc[i].name, alias_desc[i].params);
            }

            return 0;
        default:
            fprintf(stderr, "unknown option: %s\n", argv[1]);
            result = 1; goto _clean;
        }
        argument = numPtr;
    }
    argv++;
    argc--;
    }

    while (argc > 1) {
        inFileNames[ifnIdx++] = argv[1];
        argv++;
        argc--;
    }

    cpu_brand = cpu_brand_string();
    compiler_info = get_compiler_information();
    LZBENCH_PRINT(2, PROGNAME " " PROGVERSION " | %s | %d-bit " PROGOS " | %s\n\n", compiler_info, (uint32_t)(8 * sizeof(uint8_t*)), cpu_brand ? cpu_brand : "");
    LZBENCH_PRINT(5, "params: chunk_size=%llu c_iters=%d d_iters=%d cspeed=%d cmintime=%d dmintime=%d encoder_list=%s\n", (uint64)params->chunk_size, params->c_iters, params->d_iters, params->cspeed, params->cmintime, params->dmintime, encoder_list);

    if (ifnIdx < 1)  { usage(params); goto _clean; }

    if (real_time)
    {
        SET_HIGH_PRIORITY;
    } else {
        LZBENCH_STDERR(2, "The real-time process priority disabled%c\n", ' ');
    }


#ifdef UTIL_HAS_CREATEFILELIST
    if (recursive) {  /* at this stage, filenameTable is a list of paths, which can contain both files and directories */
        extendedFileList = UTIL_createFileList(inFileNames, ifnIdx, &fileNamesBuf, &fileNamesNb);
        if (extendedFileList) {
            unsigned u;
            for (u=0; u<fileNamesNb; u++) LZBENCH_PRINT(4, "%u %s\n", u, extendedFileList[u]);
            free((void*)inFileNames);
            inFileNames = extendedFileList;
            ifnIdx = fileNamesNb;
        }
    }
#endif

    /* Main function */
    if (join)
        result = lzbench_join(params, inFileNames, ifnIdx, encoder_list);
    else
        result = lzbench_main(params, inFileNames, ifnIdx, encoder_list);

    if (params->chunk_size > 10 * (1<<20)) {
        LZBENCH_STDERR(2, "[Params] cIters=%d dIters=%d cTime=%.1f dTime=%.1f chunkSize=%lluMB cSpeed=%dMB\n", params->c_iters, params->d_iters, params->cmintime/1000.0, params->dmintime/1000.0, (uint64)(params->chunk_size >> 20), params->cspeed);
    } else {
        LZBENCH_STDERR(2, "[Params] cIters=%d dIters=%d cTime=%.1f dTime=%.1f chunkSize=%lluKB cSpeed=%dMB\n", params->c_iters, params->d_iters, params->cmintime/1000.0, params->dmintime/1000.0, (uint64)(params->chunk_size >> 10), params->cspeed);
    }

    if (sort_col <= 0) goto _clean;

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

_clean:
    if (encoder_list)
        free(encoder_list);
#ifdef UTIL_HAS_CREATEFILELIST
    if (extendedFileList)
        UTIL_freeFileList(extendedFileList, fileNamesBuf);
    else
#endif
    free((void*)inFileNames);
    if (cpu_brand)
        free(cpu_brand);
    return result;
}

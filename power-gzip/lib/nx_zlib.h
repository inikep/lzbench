/*
 * NX-GZIP compression accelerator user library
 * implementing zlib library interfaces
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
 * Authors: Bulent Abali <abali@us.ibm.com>
 *          Xiao Lei Hu  <xlhu@cn.ibm.com>
 *
 */

/** @file nx_zlib.h
 *  @brief Provides libnxz own API
 */

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <assert.h>
#include <errno.h>
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <endian.h>
#include <pthread.h>
#include <sys/platform/ppc.h>
#include <zlib.h>
#include "nxu.h"
#include "nx_dbg.h"

#ifndef _NX_ZLIB_H
#define _NX_ZLIB_H

/*! \def GZIP_AUTO
    \brief Use software compression/decompression for inputs smaller than a
threshold and use NX otherwise.
*/
#define GZIP_AUTO	0x00
/*! \def GZIP_SW
    \brief Use software compression/decompression.
*/
#define GZIP_SW 	0x01
/*! \def GZIP_NX
    \brief Use NX compression/decompression.
*/
#define GZIP_NX 	0x02
/*! \def GZIP_MIX
    \brief Compress like GZIP_AUTO but mix software and NX with specific ratio for inputs bigger than the threshold, use software for decompression.
*/
#define GZIP_MIX	0x03
/*! \def GZIP_MIX2
    \brief Use NX to compress and software to decompress.
*/
#define GZIP_MIX2	0x04

struct selector {
	uint8_t inflate;
	uint8_t deflate;
};

#define COMPRESS_THRESHOLD	(1024)
#define DECOMPRESS_THRESHOLD	(1024)

#define NX_MIN(X,Y) (((X)<(Y))?(X):(Y))
#define NX_MAX(X,Y) (((X)>(Y))?(X):(Y))

#define ASSERT(X) assert(X)

#ifndef __unused
#  define __unused __attribute__((unused))
#endif

#define likely(x)    __builtin_expect(!!(x), 1)
#define unlikely(x)  __builtin_expect(!!(x), 0)

/* debug flags for libnx */
#define NX_VERBOSE_LIBNX_MASK   0x000000ff
#define NX_DEVICES_MAX 256

/* deflate header */
#define HEADER_RAW   0
#define HEADER_ZLIB  1
#define HEADER_GZIP  2

#ifndef MAX_WBITS
#  define MAX_WBITS 15
#endif
#ifndef DEF_WBITS
#  define DEF_WBITS MAX_WBITS
#endif

#define NXQWSZ  (sizeof(nx_qw_t))

#ifdef NX_LOG_SOURCE_TARGET
void nx_print_dde(nx_dde_t *ddep, const char *msg);
#endif

#define zlib_version zlibVersion()
extern const char *zlibVersion OF((void));

/* common config variables for all streams */
struct nx_config_t {
	long     page_sz;
	int      line_sz;
	int      stored_block_len;
	uint32_t max_byte_count_low;
	uint32_t max_byte_count_high;
	uint32_t max_byte_count_current;
	uint32_t max_source_dde_count;
	uint32_t max_target_dde_count;
	uint32_t max_vas_reuse_count;
	uint32_t per_job_len;          /* less than suspend limit */
	uint32_t strm_def_bufsz;
	uint32_t soft_copy_threshold;  /* choose memcpy or hwcopy */
	uint32_t cache_threshold; /** Cache input before processing */
	int 	 deflate_fifo_in_len;
	int 	 deflate_fifo_out_len;
	int      window_max;
	int      timeout_pgfaults;
	int      verbose;
	int      mlock_nx_crb_csb;
	int      dht;
	uint8_t  nx_ratio; /** ratio from 0 to 100 indicating 0% to 100% of nx
			    * over sw */
	int      strategy_override; /** Force use of an specific deflate
				     * strategy.  0 is fixed huffman, 1 is
				     * dynamic huffman */
	struct selector mode; /** mode selector: selects between software
				* and hardware compression. */
};
typedef struct nx_config_t *nx_configp_t;
extern struct nx_config_t nx_config;

/* NX device handle */
struct nx_dev_t {
	int lock;          /* crb serializer */
	int nx_errno;
	int socket_id;     /* one NX-gzip per cpu socket */
	int nx_id;         /* unique */
	int open_cnt;      /* number of active users */
	int use_cnt;       /* total number of users */
	pid_t creator_pid; /* PID of the process that allocated this handle */

	/* https://github.com/sukadev/linux/blob/vas-kern-v8.1/tools/testing/selftests/powerpc/user-nx842/compress.c#L514 */
	struct {
		int16_t version;
		int16_t id;
		int64_t flags;
		void *paste_addr;
		int fd;
		void *vas_handle;
	}; /* vas */
};
typedef struct nx_dev_t *nx_devp_t;
#define NX_DEVICES_MAX 256

/* save recent header bytes for hcrc calculations */
typedef struct ckbuf_t { char buf[128]; } ckbuf_t;

#define MAGIC1 0x1234567887654321ull

/* z_stream equivalent of NX hardware */
typedef struct nx_stream_s {
        /* parameters for the supported functions */
	uint64_t        magic1;
	int             level;          /* compression level */
	int             method;         /* must be Z_DEFLATED for zlib */
	int             windowBits;     /* also encodes zlib/gzip/raw */

	int             memLevel;       /* 1...9 (default=8) */
	int             strategy;       /* force compression algorithm */

	/* stream data management */
	unsigned char   *next_in;       /* next input byte */
	uint32_t        avail_in;       /* # of bytes available at next_in */
	unsigned long   total_in;       /* total nb of inp read so far */

	unsigned char   *next_out;      /* next obyte should be put there */
	uint32_t        avail_out;      /* remaining free space at next_out */
	unsigned long   total_out;      /* total nb of bytes output so far */

	/* private area */
	uint32_t        adler;          /* one of adler32 or crc32 */

	uint32_t        adler32;        /* machine generated */
	uint32_t        crc32;          /* checksums of bytes
                                         * compressed then written to
                                         * the stream out. note that
                                         * this interpretation is
                                         * different than zlib.h which
                                         * says checksums are
                                         * immediately updated upon
                                         * reading from the input
                                         * stream. Checksums will reflect
					 * the true values only after
					 * the stream is finished or fully
					 * flushed to the output */
	uint64_t        checksum_set;   /* nx wrap function code helper */

	int             header_len;

	unsigned char   trailer[9];     /* temp storage for tail bytes */
	int             trailer_len;

	uint64_t        total_time;     /* stream's total time running */

	uint16_t        hcrc16;         /* stored in the gzip header */
	uint32_t        cksum;          /* running checksum of the header */
	ckbuf_t         ckbuf;          /* hcrc16 helpers */
	int             ckidx;

	int             inf_state;
	int             inf_held;
	int		resuming;
	int		history_len;
	int		last_comp_ratio;
	int		is_final;
	int		invoke_cnt;  /* the times to invoke nx inflate or nx deflate */
	void		*dhthandle;

	z_streamp       zstrm;          /* point to the parent  */

	gz_headerp      gzhead;         /* where to save gzip header information */
	int             gzflags;        /* FLG */
	unsigned int    length;

	int             zlib_cmf;
	int             zlib_flg;

	unsigned int    dict_len;
	unsigned int    dict_alloc_len;
	uint32_t        dict_id;
	char            *dict;


	int             status;         /* stream status */

	nx_devp_t       nxdevp;         /* nx hardware device */
	int             wrap;           /* 0 raw, 1 zlib, 2 gzip */
	long            page_sz;

        int             sync_point;     /* is inflate currently at a sync point? */
	int             need_stored_block;
	long            last_ratio;     /* compression ratio; 500
					 * means 50% */

	/* fifo_in is the saved amount from last deflate() call
	   fifo_out is the overflowed amount from last deflate()
	   call */
	char            *fifo_in;       /** user input collects here */
	char            *fifo_out;      /** user output overflows here */

	int32_t         len_in;         /* fifo_in length */
	int32_t         used_in;        /* fifo_in used bytes */
	int32_t         cur_in;         /* fifo_in starting offset */

	int32_t         len_out;
	int32_t         used_out;
	int32_t         cur_out;

	/* return status */
	int             nx_cc;          /* nx return codes */
	uint32_t        nx_ce;          /* completion extension Fig.6-7 */
	int             z_rc;           /* libz return codes */

	uint32_t        spbc;
	/** \brief Target Processed Byte Count
	 * \details Amount of target data bytes an accelerator has written in
	 * processing this CRB.
	 */
	uint32_t        tpbc;
	uint32_t        tebc;

	/* nx commands */
	int             flush;

	uint32_t        dry_run;        /* compress by this amount
					 * do not update pointers */

	/* nx command and parameter block; one command at a time per stream */
	nx_gzip_crb_cpb_t *nxcmdp;
	nx_gzip_crb_cpb_t nxcmd0;

	/* base, history, fifo_in first, and last, next_in */
	nx_dde_t        *ddl_in;
	nx_dde_t        dde_in[5]  __attribute__ ((aligned (128)));

	/* base, next_out, fifo_out */
	nx_dde_t        *ddl_out;
	nx_dde_t        dde_out[4] __attribute__ ((aligned (128)));

	/* software zlib switch and pointer */

	/* true means stream can be switched between sw and hw */
	char            switchable;

	void		*sw_stream;

} nx_stream;
typedef struct nx_stream_s *nx_streamp;

static inline int has_nx_state(z_streamp strm)
{
	nx_streamp nx_state;

	if (strm == NULL) return 0;
	nx_state = (struct nx_stream_s *)(strm->state);
	if (nx_state == NULL) return 0;

	return (nx_state->magic1 == MAGIC1);
}

static inline int use_nx_inflate(z_streamp strm)
{
	uint64_t rnd;
	assert(strm != NULL);

	if(nx_config.mode.inflate == GZIP_NX) return 1;
	if(nx_config.mode.inflate == GZIP_SW) return 0;

	/* #1 Threshold */
	if(strm->avail_in <= DECOMPRESS_THRESHOLD) return 0;

	if(nx_config.mode.inflate == GZIP_AUTO) return 1;

	/* #2 Percentage */
	rnd = __ppc_get_timebase();
	if( rnd%100 < nx_config.nx_ratio){ /* use nx to nx_ratio */
		return 1; /* nx */
	}else{
		return 0;
	}
}

static inline int use_nx_deflate(z_streamp strm)
{
	assert(strm != NULL);

        if(nx_config.mode.deflate == GZIP_NX) return 1;
        if(nx_config.mode.deflate == GZIP_SW) return 0;

	/* #1 Threshold */
	if(strm->avail_in <= COMPRESS_THRESHOLD) return 0;
	return 1;
}

/* stream pointers and lengths manipulated */
#define update_stream_out(s,b) do{(s)->next_out += (b); (s)->total_out += (b); (s)->avail_out -= (b);}while(0)
#define update_stream_in(s,b)  do{(s)->next_in  += (b); (s)->total_in  += (b); (s)->avail_in  -= (b);}while(0)

#define copy_stream_in(d,s)  do{(d)->next_in  = (s)->next_in;  (d)->total_in  = (s)->total_in;  (d)->avail_in  = (s)->avail_in;}while(0)
#define copy_stream_out(d,s) do{(d)->next_out = (s)->next_out; (d)->total_out = (s)->total_out; (d)->avail_out = (s)->avail_out;}while(0)

/* Fifo buffer management. NX has scatter gather capability.
   We treat the fifo queue in two steps: from current head (or tail) to
   the fifo end referred to as "first" and from 0 to the current tail (or head)
   referred to as "last". To add sz bytes to the fifo
   1. test fifo_free_bytes >= sz
   2. get fifo_free_first_bytes and fifo_free_last_bytes amounts
   3. get fifo_free_first_offset and fifo_free_last_offset addresses
   4. append to fifo_free_first_offset; increase 'used'
   5. if any data remaining, append to fifo_free_last_offset

   To remove sz bytes from the fifo
   1. test fifo_used_bytes >= sz
   2. get fifo_used_first_bytes and fifo_used_last_bytes
   3. get fifo_used_first_offset and fifo_used_last_offset
   4. remove from fifo_used_first_offset; increase 'cur' mod 'fifolen', decrease 'used'
   5. if more data to go, remove from fifo_used_last_offset
*/
#define fifo_used_bytes(used) (used)
#define fifo_free_bytes(used, len) ((len)-(used))
// amount of free bytes in the first and last parts
#define fifo_free_first_bytes(cur, used, len)  ((((cur)+(used))<=(len))? (len)-((cur)+(used)): 0)
#define fifo_free_last_bytes(cur, used, len)   ((((cur)+(used))<=(len))? (cur): (len)-(used))
// amount of used bytes in the first and last parts
#define fifo_used_first_bytes(cur, used, len)  ((((cur)+(used))<=(len))? (used) : (len)-(cur))
#define fifo_used_last_bytes(cur, used, len)   ((((cur)+(used))<=(len))? 0: ((used)+(cur))-(len))
// first and last free parts start here
#define fifo_free_first_offset(cur, used)      ((cur)+(used))
#define fifo_free_last_offset(cur, used, len)  fifo_used_last_bytes(cur, used, len)
// first and last used parts start here
#define fifo_used_first_offset(cur)            (cur)
#define fifo_used_last_offset(cur)             (0)

/* for appending bytes in to the stream */
#define nx_put_byte(s,b)  do { if ((s)->avail_out > 0)			\
		{ *((s)->next_out++) = (b); --(s)->avail_out; ++(s)->total_out; \
		  *((s)->zstrm->next_out++) = (b); --(s)->zstrm->avail_out; ++(s)->zstrm->total_out; } \
		else { *((s)->fifo_out + (s)->cur_out + (s)->used_out) = (b); ++(s)->used_out; } } while(0)

/* nx_inflate_get_byte is used for header processing.  It goes to
   inf_return when bytes are not sufficient */
#define nx_inflate_get_byte(s,b) \
	do { if ((s)->avail_in == 0) goto inf_return; b = (s)->ckbuf.buf[(s)->ckidx++] = *((s)->next_in); \
		update_stream_in(s,1); update_stream_in(s->zstrm, 1);\
		if ((s)->gzflags & 0x02) {			\
			/* when the buffer is near full do a partial checksum */ \
			(s)->cksum = crc32((s)->cksum, (const unsigned char *)(s)->ckbuf.buf, (s)->ckidx); \
			(s)->ckidx = 0; }\
	} while(0)

#define print_dbg_info(s, line) \
do { prt_info(\
"== %s:%d avail_in %ld total_in %ld \
used_in %ld cur_in %ld \
avail_out %ld total_out %ld \
used_out %ld cur_out %ld \
len_in %ld len_out %ld flush %d\n", __FUNCTION__, line, \
(long)(s)->avail_in, (long)(s)->total_in,	\
(long)(s)->used_in, (long)(s)->cur_in,		\
(long)(s)->avail_out, (long)(s)->total_out,	\
(long)(s)->used_out, (long)(s)->cur_out,	\
(long)(s)->len_in, (long)(s)->len_out, (s)->flush);	\
} while (0)


/* inflate states */
typedef enum {
	inf_state_header = 0,
	inf_state_gzip_id1,
	inf_state_gzip_id2,
	inf_state_gzip_cm,
	inf_state_gzip_flg,
	inf_state_gzip_mtime,
	inf_state_gzip_xfl,
	inf_state_gzip_os,
	inf_state_gzip_xlen,
	inf_state_gzip_extra,
	inf_state_gzip_name,
	inf_state_gzip_comment,
	inf_state_gzip_hcrc, /* 12 */
	inf_state_zlib_id1,
	inf_state_zlib_flg,
	inf_state_zlib_dict,
	inf_state_zlib_dictid,
	inf_state_inflate, /* 17 */
	inf_state_data_error,
	inf_state_mem_error,
	inf_state_buf_error,
	inf_state_stream_error,
} inf_state_t;

#define ZLIB_SIZE_SLOTS 256	/* Each slot represents 4KiB, the last
				   slot is represending everything
				   which larger or equal 1024KiB */

struct zlib_stats {
	unsigned long deflateInit;
	unsigned long deflate;
	unsigned long deflate_sw;
	unsigned long deflate_nx;
	unsigned long deflate_avail_in[ZLIB_SIZE_SLOTS];
	unsigned long deflate_avail_out[ZLIB_SIZE_SLOTS];
	unsigned long deflateReset;
	unsigned long deflate_total_in[ZLIB_SIZE_SLOTS];
	unsigned long deflate_total_out[ZLIB_SIZE_SLOTS];
	unsigned long deflateSetDictionary;
	unsigned long deflateSetHeader;
	unsigned long deflateParams;
	unsigned long deflateBound;
	unsigned long deflatePrime;
	unsigned long deflateCopy;
	unsigned long deflateEnd;
	unsigned long compress;

	unsigned long inflateInit;
	unsigned long inflate;
	unsigned long inflate_sw;
	unsigned long inflate_nx;
	unsigned long inflate_avail_in[ZLIB_SIZE_SLOTS];
	unsigned long inflate_avail_out[ZLIB_SIZE_SLOTS];
	unsigned long inflateReset;
	unsigned long inflateReset2;
	unsigned long inflate_total_in[ZLIB_SIZE_SLOTS];
	unsigned long inflate_total_out[ZLIB_SIZE_SLOTS];
	unsigned long inflateSetDictionary;
	unsigned long inflateGetDictionary;
	unsigned long inflateGetHeader;
	unsigned long inflateSync;
	unsigned long inflatePrime;
	unsigned long inflateCopy;
	unsigned long inflateEnd;

	unsigned long uncompress;

	uint64_t deflate_len;
	uint64_t deflate_time;

	uint64_t inflate_len;
	uint64_t inflate_time;

};

extern pthread_mutex_t zlib_stats_mutex;
extern struct zlib_stats zlib_stats;
static inline void zlib_stats_inc(unsigned long *count)
{
        if (!nx_gzip_gather_statistics())
                return;

        pthread_mutex_lock(&zlib_stats_mutex);
        *count = *count + 1;
        pthread_mutex_unlock(&zlib_stats_mutex);
}

#ifndef ARRAY_SIZE
#  define ARRAY_SIZE(a)	 (sizeof((a)) / sizeof((a)[0]))
#endif

/* gzip_vas.c */
extern void *nx_fault_storage_address;
extern void *nx_function_begin(int function, int pri);
extern int nx_function_end(void *vas_handle);
extern uint64_t nx_wait_ticks(uint64_t ticks, uint64_t accumulated_ticks, int do_sleep);

/* zlib crc32.c and adler32.c */
extern unsigned long nx_crc32_combine(unsigned long crc1, unsigned long crc2, uint64_t len2);
extern unsigned long nx_adler32_combine(unsigned long adler1, unsigned long adler2, off_t len2);
extern unsigned long nx_crc32(unsigned long crc, const unsigned char *buf, uint64_t len);
extern unsigned long nx_adler32(unsigned long adler, const char *buf, unsigned int len);

/* nx_zlib.c */
extern nx_devp_t nx_open(int nx_id);
extern int nx_close(nx_devp_t nxdevp);
extern int nx_touch_pages(void *buf, long buf_len, long page_len, int wr);
extern void *nx_alloc_buffer(uint32_t len, long alignment, int lock);
extern void nx_free_buffer(void *buf, uint32_t len, int unlock);
extern int nx_submit_job(nx_dde_t *src, nx_dde_t *dst, nx_gzip_crb_cpb_t *cmdp, void *handle);
extern int nx_append_dde(nx_dde_t *ddl, void *addr, uint32_t len);
extern int nx_touch_pages_dde(nx_dde_t *ddep, long buf_sz, long page_sz, int wr);
extern int nx_copy(char *dst, char *src, uint64_t len, uint32_t *crc, uint32_t *adler, nx_devp_t nxdevp);
extern void nx_hw_init(void);
extern void nx_hw_done(void);

/* nx_deflate.c */
extern int nx_deflateInit_(z_streamp strm, int level, const char *version, int stream_size);
extern int nx_deflateInit2_(z_streamp strm, int level, int method, int windowBits,
		int memLevel __unused, int strategy, const char *version __unused, int stream_size __unused);
#define nx_deflateInit(strm, level) nx_deflateInit_((strm), (level), ZLIB_VERSION, (int)sizeof(z_stream))
#define nx_deflateInit2(strm, level, method, windowBits, memLevel, strategy) \
	nx_deflateInit2_((strm), (level), (method), (windowBits), (memLevel), \
			(strategy), ZLIB_VERSION, (int)sizeof(z_stream))
extern int nx_deflate(z_streamp strm, int flush);
extern int nx_deflateEnd(z_streamp strm);
extern unsigned long nx_deflateBound(z_streamp strm, unsigned long sourceLen);
extern int nx_deflateSetDictionary(z_streamp strm, const unsigned char *dictionary,
				uint dictLength);

/* nx_inflate.c */
extern int nx_inflateInit_(z_streamp strm, const char *version, int stream_size);
extern int nx_inflateInit2_(z_streamp strm, int windowBits, const char *version, int stream_size);
#define nx_inflateInit(strm) nx_inflateInit_((strm), ZLIB_VERSION, (int)sizeof(z_stream))
#define nx_inflateInit2(strm, windowBits)				\
	nx_inflateInit2_((strm), (windowBits), ZLIB_VERSION, (int)sizeof(z_stream))
extern int nx_inflate(z_streamp strm, int flush);
extern int nx_inflateEnd(z_streamp strm);
extern int nx_inflateSyncPoint(z_streamp strm);
extern int nx_inflateSetDictionary(z_streamp strm, const unsigned char *dictionary,
				uint dictLength);

/* nx_compress.c */
extern int nx_compress2(Bytef *dest, uLongf *destLen, const Bytef *source, uLong sourceLen, int level);
extern int nx_compress(Bytef *dest, uLongf *destLen, const Bytef *source, uLong sourceLen);
extern uLong nx_compressBound(uLong sourceLen);

/* nx_uncompr.c */
extern int nx_uncompress2(Bytef *dest, uLongf *destLen, const Bytef *source, uLong *sourceLen);
extern int nx_uncompress(Bytef *dest, uLongf *destLen, const Bytef *source, uLong sourceLen);

/* nx_dht.c */
extern void *dht_begin(char *ifile, char *ofile);
extern void dht_end(void *handle);
extern int dht_lookup(nx_gzip_crb_cpb_t *cmdp, int request, void *handle);
extern void *dht_copy(void *handle);

/* sw_zlib.c */
extern int sw_zlib_init(void);
extern void sw_zlib_close(void);
extern const char *sw_zlibVersion(void);
extern int sw_deflateInit_(z_streamp strm, int level, const char* version, int stream_size);
extern int sw_deflateInit2_(z_streamp strm, int level, int method, int windowBits,
			int memLevel, int strategy,    const char *version, int stream_size);
extern int sw_deflate(z_streamp strm, int flush);
extern int sw_deflateEnd(z_streamp strm);
extern int sw_deflateReset(z_streamp strm);
extern int sw_deflateResetKeep(z_streamp strm);
extern int sw_deflateSetHeader(z_streamp strm, gz_headerp head);
extern uLong sw_deflateBound(z_streamp strm, uLong sourceLen);
extern int sw_deflateSetDictionary(z_streamp strm, const Bytef *dictionary, uInt  dictLength);
extern int sw_deflateCopy(z_streamp dest, z_streamp source);
extern int sw_uncompress(Bytef *dest, uLongf *destLen, const Bytef *source, uLong sourceLen);
#if ZLIB_VERNUM >= 0x1290
extern int sw_uncompress2(Bytef *dest, uLongf *destLen, const Bytef *source, uLong *sourceLen);
#endif

extern int sw_inflateInit_(z_streamp strm, const char *version, int stream_size);
extern int sw_inflateInit2_(z_streamp strm, int  windowBits, const char *version, int stream_size);
extern int sw_inflateReset(z_streamp strm);
extern int sw_inflateReset2(z_streamp strm, int windowBits);
extern int sw_inflateResetKeep(z_streamp strm);
extern int sw_inflateSetDictionary(z_streamp strm, const Bytef *dictionary, uInt  dictLength);
extern int sw_inflate(z_streamp strm, int flush);
extern int sw_inflateEnd(z_streamp strm);
extern int sw_inflateCopy(z_streamp dest, z_streamp source);
extern int sw_inflateGetHeader(z_streamp strm, gz_headerp head);
extern int sw_inflateSyncPoint(z_streamp strm);
extern int sw_compress(Bytef *dest, uLongf *destLen, const Bytef *source, uLong sourceLen);
extern int sw_compress2(Bytef *dest, uLongf *destLen, const Bytef *source, uLong sourceLen, int level);
extern uLong sw_compressBound(uLong sourceLen);



#endif /* _NX_ZLIB_H */

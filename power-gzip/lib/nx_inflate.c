/*
 * NX-GZIP compression accelerator user library
 * implementing zlib compression library interfaces
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

/** @file nx_inflate.c
 * \brief Implement the inflate function for the NX GZIP accelerator and
 * related functions.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <assert.h>
#include <errno.h>
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <endian.h>
#include "zlib.h"
#include "copy-paste.h"
#include "nxu.h"
#include "nx.h"
#include "nx-gzip.h"
#include "nx_zlib.h"
#include "nx_dbg.h"

/** \brief Fixed 32K history length
 * \details The maximum distance in the deflate standard is 32768 Bytes.
 */
#define INF_HIS_LEN (1<<15)
#define INF_MAX_DICT_LEN  INF_HIS_LEN

#define INF_MIN_INPUT_LEN 300 /* greater than 288 for dht plus 3 bit header plus 1 byte */
/** \brief Maximum Compression Ratio
 * https://stackoverflow.com/a/42865320/5504692
 */
#define INF_MAX_COMPRESSION_RATIO 1032
#define INF_MAX_EXPANSION_BYTES (INF_MIN_INPUT_LEN * INF_MAX_COMPRESSION_RATIO)

/* move the overflow from the current fifo head-32KB to the fifo_out
   buffer beginning. Fifo_out starts with 32KB history then */
#define fifo_out_len_check(s)		  \
do { if ((s)->cur_out > (s)->len_out/2) { \
	memmove((s)->fifo_out, (s)->fifo_out + (s)->cur_out - INF_HIS_LEN, INF_HIS_LEN + (s)->used_out); \
	(s)->cur_out = INF_HIS_LEN; } \
} while(0)

#define fifo_in_len_check(s)		\
do { if ((s)->cur_in > (s)->len_in/2) { \
	memmove((s)->fifo_in, (s)->fifo_in + (s)->cur_in, (s)->used_in); \
	(s)->cur_in = 0; } \
} while(0)

static int nx_inflate_(nx_streamp s, int flush);

int nx_inflateResetKeep(z_streamp strm)
{
	nx_streamp s;
	if (strm == Z_NULL)
		return Z_STREAM_ERROR;
	s = (nx_streamp) strm->state;
	strm->total_in = strm->total_out = s->total_in = 0;
	strm->msg = Z_NULL;
	s->gzhead = NULL;
	return Z_OK;
}

int nx_inflateReset(z_streamp strm)
{
	nx_streamp s;
	if (strm == Z_NULL)
		return Z_STREAM_ERROR;

	s = (nx_streamp) strm->state;
	strm->msg = Z_NULL;

	if (s->wrap)
		s->adler = s->wrap & 1;

	s->total_in = s->total_out = 0;

	s->used_in = s->used_out = 0;
	s->cur_in = 0;
	s->cur_out = INF_HIS_LEN; /* keep a 32k gap here */
	s->inf_state = 0;
	s->resuming = 0;
	s->history_len = 0;
	s->is_final = 0;
	s->trailer_len = 0;

	s->nxcmdp  = &s->nxcmd0;

	s->crc32 = INIT_CRC;
	s->adler32 = INIT_ADLER;
	s->ckidx = 0;
	s->cksum = INIT_CRC;

	s->total_time = 0;

	return nx_inflateResetKeep(strm);
}

static int nx_inflateReset2(z_streamp strm, int windowBits)
{
	int wrap;
	nx_streamp s;

	if (strm == Z_NULL) return Z_STREAM_ERROR;
	s = (nx_streamp) strm->state;
	if (s == NULL) return Z_STREAM_ERROR;

	/* Note: NX-GZIP does not do windows smaller than 32KB;
	   silently accept all window sizes */

	/* extract wrap request from windowBits parameter */
	if (windowBits < 0) {
		wrap = HEADER_RAW;
		windowBits = -windowBits;
	}
	else if (windowBits >= 8 && windowBits <= 15)
		wrap = HEADER_ZLIB;
	else if (windowBits >= 8+16 && windowBits <= 15+16)
		wrap = HEADER_GZIP;
	else if (windowBits >= 8+32 && windowBits <= 15+32)
		wrap = HEADER_ZLIB | HEADER_GZIP; /* auto detect header */
	else if (windowBits == 0) {
		/* zlib.h states "can also be zero to request that
		   inflate use the window size in the zlib header of
		   the compressed stream. */
		wrap = HEADER_ZLIB;
		windowBits = 15;
	}
	else return Z_STREAM_ERROR;

	s->wrap = wrap;
	s->windowBits = windowBits;

	return nx_inflateReset(strm);
}

int nx_inflateInit2_(z_streamp strm, int windowBits, const char *version, int stream_size)
{
	int ret;
	nx_streamp s;
	nx_devp_t h;

	prt_info("%s:%d strm %p\n", __FUNCTION__, __LINE__, strm);

	if (version == Z_NULL || version[0] != ZLIB_VERSION[0] ||
	    stream_size != (int)(sizeof(z_stream)))
		return Z_VERSION_ERROR;

	if (strm == Z_NULL) return Z_STREAM_ERROR;

	strm->msg = Z_NULL; /* in case we return an error */

	h = nx_open(-1); /* if want to pick specific NX device, set env NX_GZIP_DEV_NUM */
	if (!h) {
		prt_err("cannot open NX device\n");
		return Z_STREAM_ERROR;
	}

	s = nx_alloc_buffer(sizeof(*s), nx_config.page_sz, nx_config.mlock_nx_crb_csb);
	if (s == NULL) return Z_MEM_ERROR;
	memset(s, 0, sizeof(*s));

	s->magic1  = MAGIC1;
	s->zstrm   = strm;
	s->nxcmdp  = &s->nxcmd0;
	s->page_sz = nx_config.page_sz;
	s->nxdevp  = h;
	s->gzhead  = NULL;
	s->ddl_in  = s->dde_in;
	s->ddl_out = s->dde_out;
	s->sync_point = 0;

	/* small input data will be buffered here */
	s->fifo_in = NULL;

	/* overflow buffer */
	s->fifo_out = NULL;

	strm->state = (void *) s;

	s->switchable = 0;
	s->sw_stream = NULL;

	ret = nx_inflateReset2(strm, windowBits);
	if (ret != Z_OK) {
		prt_err("nx_inflateReset2\n");
		goto reset_err;
	}

	return ret;

reset_err:
	if (s)
		nx_free_buffer(s, 0, 0);
	strm->state = Z_NULL;
	return ret;
}

int nx_inflateInit_(z_streamp strm, const char *version, int stream_size)
{
	return nx_inflateInit2_(strm, DEF_WBITS, version, stream_size);
}

int nx_inflateEnd(z_streamp strm)
{
	nx_streamp s;
	void *temp = NULL;
	int rc;

	prt_info("%s:%d strm %p\n", __FUNCTION__, __LINE__, strm);

	if (strm == Z_NULL) return Z_STREAM_ERROR;
	s = (nx_streamp) strm->state;
	if (s == NULL) return Z_STREAM_ERROR;

	/* In case call inflateEnd without a inflate call.  */
	if(s->sw_stream){
		temp  = (void *)strm->state;
		strm->state = s->sw_stream;
		rc = sw_inflateEnd(strm);
		prt_info("call sw_inflateEnd to release sw resource,rc=%d\n",rc);
		strm->state = temp;
		s->sw_stream = NULL;
	}

	/* TODO add here Z_DATA_ERROR if the stream was freed
	   prematurely (when some input or output was discarded). */


	nx_free_buffer(s->fifo_in, s->len_in, 0);
	nx_free_buffer(s->fifo_out, s->len_out, 0);
	nx_free_buffer(s->dict, s->dict_alloc_len, 0);
	nx_close(s->nxdevp);

	nx_free_buffer(s, sizeof(*s), nx_config.mlock_nx_crb_csb);

	return Z_OK;
}

int nx_inflate(z_streamp strm, int flush)
{
	int rc = Z_OK, in, out;
	nx_streamp s;
	void *temp = NULL;

	if (strm == Z_NULL) return Z_STREAM_ERROR;
	s = (nx_streamp) strm->state;
	if (s == NULL) return Z_STREAM_ERROR;

	/* check for sw deflate first*/
	if(has_nx_state(strm) && s->switchable && (0 == use_nx_inflate(strm))){
		/*Use software zlib, switch the sw and hw state*/
		s = (nx_streamp) strm->state;
		s->switchable = 0; /* decided to use sw zlib and not switchable */
		temp  = s->sw_stream;  /* save the sw pointer */
		s->sw_stream = NULL;

		rc = nx_inflateEnd(strm); /* free the hw resource */
		prt_info("call nx_inflateEnd to clean the hw resource,rc=%d\n",rc);
		strm->state = temp;  /* restore the sw pointer */
		prt_info("call software inflate,len=%d\n", strm->avail_in);
		rc = sw_inflate(strm,flush);
		prt_info("call software inflate, rc=%d\n", rc);
		return rc;
	}else if(s->sw_stream){
		/*decide to use nx here, release the sw resource */
		temp  = (void *)strm->state;
		strm->state = s->sw_stream;

		rc = sw_inflateEnd(strm);
		prt_info("call sw_inflateEnd to clean the sw resource,rc=%d\n",rc);
		strm->state = temp;
		s->sw_stream = NULL;
	}

	s->switchable = 0;


	if (flush == Z_BLOCK || flush == Z_TREES) {
		strm->msg = (char *)"Z_BLOCK or Z_TREES not implemented";
		prt_err("Z_BLOCK or Z_TREES not implemented!\n");
		return Z_STREAM_ERROR;
	}

	if (s->fifo_out == NULL) {
		/* overflow buffer is about 40% of s->avail_in */
		s->len_out = (INF_HIS_LEN*2 + (s->zstrm->avail_in * 40)/100);
		/* for the max possible expansion of inflate input */
		s->len_out = NX_MAX( INF_MAX_EXPANSION_BYTES, s->len_out);
		s->len_out = NX_MAX( INF_HIS_LEN << 3, s->len_out );
		if (NULL == (s->fifo_out = nx_alloc_buffer(s->len_out, nx_config.page_sz, 0))) {
			prt_err("nx_alloc_buffer for inflate fifo_out\n");
			return Z_MEM_ERROR;
		}
	}

	/* copy in from user stream to internal structures */
	copy_stream_in(s, s->zstrm);
	copy_stream_out(s, s->zstrm);

	/* Account for progress */
	in = s->avail_in;
	out = s->avail_out;

inf_forever:
	/* inflate state machine */

	switch (s->inf_state) {
		unsigned int c, copy;

	case inf_state_header:

		prt_info("%d: inf_state %d\n", __LINE__, s->inf_state);

		if (s->wrap == (HEADER_ZLIB | HEADER_GZIP)) {
			/* auto detect zlib/gzip */
			nx_inflate_get_byte(s, c);
			if (c == 0x1f) {
				/* looks like gzip; see rfc1952 ID2 and ID2 fields */
				s->inf_state = inf_state_gzip_id2;
				s->wrap = HEADER_GZIP;
			}
			else if (((c & 0x0f) == 0x08) && ( ((c >> 4) & 0x0f) < 8)) {
				/* looks like zlib; see rfc1950 CMF fields, CM and CINFO */
				s->inf_state = inf_state_zlib_flg;
				s->zlib_cmf = c;
				s->wrap = HEADER_ZLIB;
			}
			else {
				strm->msg = (char *)"incorrect header";
				s->inf_state = inf_state_data_error;
			}
		}
		else if (s->wrap == HEADER_ZLIB) {
			/* look for a zlib header */
			s->inf_state = inf_state_zlib_id1;
			if (s->gzhead != NULL) {
				/* this should be an error */
				s->gzhead->done = -1;
			}
		}
		else if (s->wrap == HEADER_GZIP) {
			/* look for a gzip header */
			if (s->gzhead != NULL)
				s->gzhead->done = 0;
			s->inf_state = inf_state_gzip_id1;
		}
		else {
			/* raw inflate doesn't use checksums but we do
			 * it anyway since comes for free */
			s->crc32 = INIT_CRC;
			s->adler32 = INIT_ADLER;
			s->inf_state = inf_state_inflate; /* go to inflate proper */
		}
		break;

	case inf_state_gzip_id1:

		prt_info("%d: inf_state %d\n", __LINE__, s->inf_state);

		nx_inflate_get_byte(s, c);
		if (c != 0x1f) {
			strm->msg = (char *)"incorrect gzip header";
			s->inf_state = inf_state_data_error;
			break;
		}

		s->inf_state = inf_state_gzip_id2;
		/* fall thru */

	case inf_state_gzip_id2:

		prt_info("%d: inf_state %d\n", __LINE__, s->inf_state);

		nx_inflate_get_byte(s, c);
		if (c != 0x8b) {
			strm->msg = (char *)"incorrect gzip header";
			s->inf_state = inf_state_data_error;
			break;
		}

		s->inf_state = inf_state_gzip_cm;
		/* fall thru */

	case inf_state_gzip_cm:

		prt_info("%d: inf_state %d\n", __LINE__, s->inf_state);

		nx_inflate_get_byte(s, c);
		if (c != 0x08) {
			strm->msg = (char *)"unknown compression method";
			s->inf_state = inf_state_data_error;
			break;
		}

		s->inf_state = inf_state_gzip_flg;
		/* fall thru */

	case inf_state_gzip_flg:

		prt_info("%d: inf_state %d\n", __LINE__, s->inf_state);

		nx_inflate_get_byte(s, c);
		s->gzflags = c;
		prt_info("%d: s->gzflags=0x%x\n",__LINE__,s->gzflags);

		if ((s->gzflags & 0xe0) != 0) { /* reserved bits are set */
			strm->msg = (char *)"unknown header flags set";
			s->inf_state = inf_state_data_error;
			break;
		}

		if (s->gzhead != NULL) {
			/* FLG field of the file says this is compressed text */
			s->gzhead->text = (int) (s->gzflags & 1);
			s->gzhead->time = 0;
		}

		s->inf_held = 0;
		s->inf_state = inf_state_gzip_mtime;
		/* fall thru */

	case inf_state_gzip_mtime:

		prt_info("%d: inf_state %d\n", __LINE__, s->inf_state);

		while (s->inf_held < 4) { /* need 4 bytes for MTIME */
			nx_inflate_get_byte(s, c);
			if (s->gzhead != NULL) {
				s->gzhead->time = c << (8 * s->inf_held) | s->gzhead->time;
			}
			++ s->inf_held;
		}
		s->inf_held = 0;
		if (s->gzhead != NULL) {
			assert( ((s->gzhead->time & (1<<31)) == 0) );
		}
		/* assertion is a reminder for endian check; either
		   fires right away or in the year 2038 if we're still
		   alive */

		s->inf_state = inf_state_gzip_xfl;
		/* fall thru */

	case inf_state_gzip_xfl:

		prt_info("%d: inf_state %d\n", __LINE__, s->inf_state);

		nx_inflate_get_byte(s, c);
		if (s->gzhead != NULL)
			s->gzhead->xflags = c;

		s->inf_state = inf_state_gzip_os;
		/* fall thru */

	case inf_state_gzip_os:

		prt_info("%d: inf_state %d\n", __LINE__, s->inf_state);

		nx_inflate_get_byte(s, c);
		if (s->gzhead != NULL)
			s->gzhead->os = c;

		s->inf_held = 0;
		s->length = 0;
		s->inf_state = inf_state_gzip_xlen;
		/* fall thru */

	case inf_state_gzip_xlen:

		prt_info("%d: inf_state %d\n", __LINE__, s->inf_state);

		if (s->gzflags & 0x04) { /* fextra was set */
			while (s->inf_held < 2) {
				nx_inflate_get_byte(s, c);
				s->length = s->length | (c << (s->inf_held * 8));
				++ s->inf_held;
			}

			s->length = le32toh(s->length);
			if (s->gzhead != NULL)
				s->gzhead->extra_len = s->length;
		}
		else if (s->gzhead != NULL)
			s->gzhead->extra = NULL;
		s->inf_held = 0;
		s->inf_state = inf_state_gzip_extra;
		/* fall thru */

	case inf_state_gzip_extra:

		prt_info("%d: inf_state %d\n", __LINE__, s->inf_state);

		if (s->gzflags & 0x04) { /* fextra was set */
			copy = s->length;
			if (copy > s->avail_in) copy = s->avail_in;
			if (copy) {
				if (s->gzhead != NULL &&
				    s->gzhead->extra != NULL) {
					unsigned int len = s->gzhead->extra_len - s->length;
					memcpy(s->gzhead->extra + len, s->next_in,
					       len + copy > s->gzhead->extra_max ?
					       s->gzhead->extra_max - len : copy);
				}
				if (s->gzflags & 0x02) /* fhcrc was set */
					s->cksum = crc32(s->cksum, s->next_in, copy);
				update_stream_in(s, copy);
				s->length -= copy;
			}
			if (s->length) goto inf_return; /* more extra data to copy */
		}

		s->length = 0;
		s->inf_state = inf_state_gzip_name;
		/* fall thru */

	case inf_state_gzip_name:

		prt_info("%d: inf_state %d\n", __LINE__, s->inf_state);

		if (s->gzflags & 0x08) { /* fname was set */
			if (s->avail_in == 0) goto inf_return;
			copy = 0;
			do {
				c = (unsigned int)(s->next_in[copy++]);
				if (s->gzhead != NULL &&
				    s->gzhead->name != NULL &&
				    s->length < s->gzhead->name_max )
					s->gzhead->name[s->length++] = (char) c;
			} while (!!c && copy < s->avail_in);
			if (s->gzflags & 0x02) /* fhcrc was set */
				s->cksum = crc32(s->cksum, s->next_in, copy);
			update_stream_in(s, copy);
			if (!!c) goto inf_return; /* need more name */
		}
		else if (s->gzhead != NULL)
			s->gzhead->name = NULL;

		s->length = 0;
		s->inf_state = inf_state_gzip_comment;
		/* fall thru */

	case inf_state_gzip_comment:

		prt_info("%d: inf_state %d\n", __LINE__, s->inf_state);

		if (s->gzflags & 0x10) { /* fcomment was set */
			if (s->avail_in == 0) goto inf_return;
			copy = 0;
			do {
				c = (unsigned int)(s->next_in[copy++]);
				if (s->gzhead != NULL &&
				    s->gzhead->comment != NULL &&
				    s->length < s->gzhead->comm_max )
					s->gzhead->comment[s->length++] = (char) c;
			} while (!!c && copy < s->avail_in);
			if (s->gzflags & 0x02) /* fhcrc was set */
				s->cksum = crc32(s->cksum, s->next_in, copy);
			update_stream_in(s, copy);
			if (!!c) goto inf_return; /* need more comment */
		}
		else if (s->gzhead != NULL)
			s->gzhead->comment = NULL;

		s->length = 0;
		s->inf_held = 0;
		s->inf_state = inf_state_gzip_hcrc;
		/* fall thru */

	case inf_state_gzip_hcrc:

		prt_info("%d: inf_state %d\n", __LINE__, s->inf_state);

		if (s->gzflags & 0x02) { /* fhcrc was set */
			uint32_t checksum = s->cksum; /*check sum for data before hcrc16 field.*/

			while( s->inf_held < 2 ) {
				nx_inflate_get_byte(s, c);
				s->hcrc16 = s->hcrc16 | (c << (s->inf_held * 8));
				++ s->inf_held;
			}
			s->hcrc16 = le16toh(s->hcrc16);
			s->gzhead->hcrc = 1;
			s->gzhead->done = 1;

			/* Compare stored and compute hcrc checksums here */
			if (s->hcrc16 != (checksum & 0xffff)) {
				strm->msg = (char *)"header crc mismatch";
				s->inf_state = inf_state_data_error;
				break;
			}
		}
		else if (s->gzhead != NULL){
			s->gzhead->hcrc = 0;
			s->gzhead->done = 1;
		}

		s->inf_held = 0;
		s->adler = s->crc32 = INIT_CRC;
		s->inf_state = inf_state_inflate; /* go to inflate proper */

		break;

	case inf_state_zlib_id1:

		prt_info("%d: inf_state %d\n", __LINE__, s->inf_state);

		nx_inflate_get_byte(s, c);
		if ((c & 0x0f) != 0x08) {
			strm->msg = (char *)"unknown compression method";
			s->inf_state = inf_state_data_error;
			break;
		} else if (((c >> 4) & 0x0f) >= 8) {
			strm->msg = (char *)"invalid window size";
			s->inf_state = inf_state_data_error;
			break;
		}
		else {
			s->inf_state = inf_state_zlib_flg; /* zlib flg field */
			s->zlib_cmf = c;
		}
		/* fall thru */

	case inf_state_zlib_flg:

		prt_info("%d: inf_state %d\n", __LINE__, s->inf_state);

		nx_inflate_get_byte(s, c);
		if ( ((s->zlib_cmf * 256 + c) % 31) != 0 ) {
			strm->msg = (char *)"incorrect header check";
			s->inf_state = inf_state_data_error;
			break;
		}
		if (c & 1<<5) {
			s->inf_state = inf_state_zlib_dictid;
			s->dict_id = 0;
			s->dict_len = 0;
		}
		else {
			s->inf_state = inf_state_inflate; /* go to inflate proper */
			s->adler = s->adler32 = INIT_ADLER;
		}
		s->inf_held = 0;
		break;

	case inf_state_zlib_dictid:

		prt_info("%d: inf_state %d\n", __LINE__, s->inf_state);

		while (s->inf_held < 4) {
			nx_inflate_get_byte(s, c);
			s->dict_id = (s->dict_id << 8) | (c & 0xff);
			++ s->inf_held;
		}
		prt_info("need dictionary %x\n", s->dict_id);
		strm->adler = s->dict_id; /* asking user to supply this dict with dict_id */
		s->inf_state = inf_state_zlib_dict;
		s->inf_held = 0;
		s->dict_len = 0;

	case inf_state_zlib_dict:

		prt_info("%d: inf_state %d\n", __LINE__, s->inf_state);

		if (s->dict_len == 0) {
			return Z_NEED_DICT;
		}
		s->adler = s->adler32 = INIT_ADLER;
		s->inf_state = inf_state_inflate; /* go to inflate proper */

	case inf_state_inflate:

		prt_info("%d: inf_state %d\n", __LINE__, s->inf_state);

		rc = nx_inflate_(s, flush);
		goto inf_return;

	case inf_state_data_error:

		prt_info("%d: inf_state %d\n", __LINE__, s->inf_state);

		rc = Z_DATA_ERROR;
		goto inf_return;

	case inf_state_mem_error:

		prt_info("%d: inf_state %d\n", __LINE__, s->inf_state);

		rc = Z_MEM_ERROR;
		break;

	case inf_state_buf_error:

		prt_info("%d: inf_state %d\n", __LINE__, s->inf_state);

		rc = Z_BUF_ERROR;
		break;

	default:

		prt_info("%d: inf_state %d\n", __LINE__, s->inf_state);

		rc = Z_STREAM_ERROR;
		break;
	}
	goto inf_forever;

inf_return:

	/* copy out to user stream */
	copy_stream_in(s->zstrm, s);
	copy_stream_out(s->zstrm, s);

	/* Following zlib behaviour. If there is no progress Z_BUF_ERROR
	 * is returned. */
	in -= s->avail_in;
	out -= s->avail_out;
	if (in == 0 && out == 0 && rc == Z_OK) return Z_BUF_ERROR;

	/* if flush is Z_FINISH we cannot return Z_OK. */
	if (flush == Z_FINISH && rc == Z_OK) return Z_BUF_ERROR;
	return rc;
}

static inline void nx_inflate_update_checksum(nx_streamp s)
{
	nx_gzip_crb_cpb_t *cmdp = s->nxcmdp;

	s->crc32 = get32(cmdp->cpb, out_crc);
	s->adler32 = get32(cmdp->cpb, out_adler);

	if (s->wrap == HEADER_GZIP)
		s->zstrm->adler = s->adler = s->crc32;
	else if (s->wrap == HEADER_ZLIB)
		s->zstrm->adler = s->adler = s->adler32;
}

/* 0 is verify only, 1 is copy only, 2 is both copy and verify */
static int nx_inflate_verify_checksum(nx_streamp s, int copy)
{
	nx_gzip_crb_cpb_t *cmdp = s->nxcmdp;
	unsigned char *tail;
	uint32_t cksum, isize;

	if (copy > 0) {
		/* to handle the case of crc and isize spanning fifo_in
		 * and next_in */
		int need, got;
		if (s->wrap == HEADER_GZIP)
			need = 8;
		else if (s->wrap == HEADER_ZLIB)
			need = 4;
		else
			need = 0;

		/* if partial copy exist from previous calls */
		need = NX_MAX( NX_MIN(need - s->trailer_len, need), 0 );

		/* copy need bytes from fifo_in */
		got = NX_MIN(s->used_in, need);
		if (got > 0) {
			memcpy(s->trailer, s->fifo_in + s->cur_in, got);
			s->trailer_len   = got;
			s->used_in      -= got;
			s->cur_in       += got;
			fifo_in_len_check(s);
		}

		/* copy any remaining from next_in */
		got = NX_MIN(s->avail_in, need - got);
		if (got > 0) {
			memcpy(s->trailer + s->trailer_len, s->next_in, got);
			s->trailer_len    += got;
			update_stream_in(s, got);
		}
		if (copy == 1)
			return Z_OK; /* copy only */
	}

	tail = s->trailer;

	if (s->wrap == HEADER_GZIP) {
		if (s->trailer_len == 8) {
			/* crc32 and isize are present; compare checksums */
			cksum = (tail[0] | tail[1]<<8 | tail[2]<<16 | tail[3]<<24);
			isize = (tail[4] | tail[5]<<8 | tail[6]<<16 | tail[7]<<24);

			prt_info("computed checksum %08x isize %08x\n", cmdp->cpb.out_crc, (uint32_t)(s->total_out % (1ULL<<32)));
			prt_info("stored   checksum %08x isize %08x\n", cksum, isize);

			nx_inflate_update_checksum(s);

			if (cksum == cmdp->cpb.out_crc && isize == (uint32_t)(s->total_out % (1ULL<<32)) )
				return Z_STREAM_END;
			else {
				prt_info("checksum or isize mismatch\n");
				return Z_STREAM_ERROR;
			}
		}
		else return Z_OK; /* didn't receive all */
	}
	else if (s->wrap == HEADER_ZLIB) {
		if (s->trailer_len == 4) {
			/* adler32 is present; compare checksums */
			cksum = (tail[0] | tail[1]<<8 | tail[2]<<16
				 | ((uint32_t) tail[3])<<24);

			prt_info("computed checksum %08x\n", cmdp->cpb.out_adler);
			prt_info("stored   checksum %08x\n", cksum);

			nx_inflate_update_checksum(s);

			if (cksum == cmdp->cpb.out_adler)
				return Z_STREAM_END;
			else {
				prt_info("checksum mismatch\n");
				return Z_STREAM_ERROR;
			}
		}
		else return Z_OK; /* didn't receive all */
	}
	/* raw data does not check crc */
	return Z_STREAM_END;
}

/* Overlay any dictionary on top of the inflate history and calculate
   lengths rounding up to 16 byte integrals */
static int nx_amend_history_with_dict(nx_streamp s)
{
	int dlen = NX_MIN(s->dict_len, INF_MAX_DICT_LEN);
	int hlen, padding, nx_history_len;
	nx_gzip_crb_cpb_t *cmdp = s->nxcmdp;

	ASSERT(s->history_len >= 0);

	prt_info("amend before: dict_len %d hist_len %d\n", dlen, s->history_len);
	if (s->history_len + dlen >= INF_HIS_LEN) {
		/* dictionary occupies most of the window, and history is
		 * reduced */
		hlen = INF_MAX_DICT_LEN - dlen;
	} else {
		/* history will be padded to guarantee that hlen + dlen is as
		 * multiple of 16 */
		padding = NXQWSZ - (s->history_len + dlen) % NXQWSZ;
		hlen = s->history_len + padding;
	}
	prt_info("amend after: dict_len %d hist_len %d\n", dlen, hlen);

	/* sum is integral of 16 */
	nx_history_len = hlen + dlen;
	ASSERT( (nx_history_len % 16) == 0 );

	cmdp->cpb.in_histlen = 0;
	putnn(cmdp->cpb, in_histlen, nx_history_len / NXQWSZ);
	ASSERT(!!s->dict && !!s->fifo_out);

	/* add hlen bytes from the end of the history */
	if (hlen > 0)
		nx_append_dde(s->ddl_in, s->fifo_out + (s->cur_out - hlen), hlen);

	/* add dlen bytes from the end of the dictionary */
	if (dlen > 0)
		nx_append_dde(s->ddl_in, s->dict + s->dict_len - dlen, dlen);

	return nx_history_len;
}


static int copy_data_to_fifo_in(nx_streamp s) {
	uint32_t free_space, read_sz;

	if (s->fifo_in == NULL) {
		s->len_in = nx_config.cache_threshold * 2;
		if (NULL == (s->fifo_in = nx_alloc_buffer(s->len_in, nx_config.page_sz, 0))) {
			prt_err("nx_alloc_buffer for inflate fifo_in\n");
			return Z_MEM_ERROR;
		}
	}

	/* reset fifo head to reduce unnecessary wrap arounds */
	s->cur_in = (s->used_in == 0) ? 0 : s->cur_in;
	fifo_in_len_check(s);
	free_space = s->len_in - s->cur_in - s->used_in;

	read_sz = NX_MIN(free_space, s->avail_in);
	if (read_sz > 0) {
		/* copy from next_in to the offset cur_in + used_in */
		memcpy(s->fifo_in + s->cur_in + s->used_in, s->next_in, read_sz);
		update_stream_in(s, read_sz);
		s->used_in = s->used_in + read_sz;
	}

	return Z_OK;
}

/** \brief Internal implementation of inflate.
 *
 * @param s nx_streamp to be processed.
 * @param flush Determines when uncompressed bytes are added to next_out.
 */
static int nx_inflate_(nx_streamp s, int flush)
{
	/* queuing, file ops, byte counting */
	uint32_t write_sz, source_sz, target_sz;
	long loop_cnt = 0, loop_max = 0xffff;

	/* inflate benefits from large jobs; memcopies must be amortized */
	uint32_t inflate_per_job_len = 64 * nx_config.per_job_len;

	/* nx hardware */
	uint32_t sfbt = 0, subc = 0, spbc, tpbc, nx_ce, fc;

	nx_gzip_crb_cpb_t *cmdp = s->nxcmdp;
	nx_dde_t *ddl_in = s->ddl_in;
	nx_dde_t *ddl_out = s->ddl_out;

	uint64_t ticks_total = 0;
	int cc, rc, timeout_pgfaults, partial_bits=0;
	/** \brief Includes dictionary and history going in to nx-gzip
	 */
	int nx_history_len;

	/**
	 * \dot inflate() machine state
	 * digraph state {
	 * error_check [shape=box];
	 * copy_fifo_out_to_next_out [shape=box];
	 * restart_nx [shape=box];
	 * ok_cc3 [shape=box];
	 * offsets_state [shape=box];
	 * start -> error_check;
	 * error_check -> copy_fifo_out_to_next_out;
	 * error_check -> error;
	 * copy_fifo_out_to_next_out -> restart_nx;
	 * restart_nx -> restart_nx [label="cc == ERR_NX_AT_FAULT\
	 *  || cc == ERR_NX_TARGET_SPACE" fontsize=8];
	 * restart_nx -> offsets_state [label="cc == ERR_NX_OK"];
	 * restart_nx -> ok_cc3;
	 * restart_nx -> error;
	 * ok_cc3 -> offsets_state;
	 * offsets_state -> copy_fifo_out_to_next_out;
	 * offsets_state -> return [label="Needs more data"];
	 * }
	 * \enddot
	 */

	timeout_pgfaults = nx_config.timeout_pgfaults;

	print_dbg_info(s, __LINE__);

	if (s->avail_in == 0 && s->used_in == 0 && s->avail_out == 0 && s->used_out == 0)
		return Z_STREAM_END;

	if (s->is_final == 1 && s->used_out == 0) {
		/* returning from avail_out==0 */
		return nx_inflate_verify_checksum(s, 2); /* copy and verify */
	}

	/* duplicating zlib behavior */
	if ((s->avail_in > 0 && s->next_in == NULL) || (s->next_out == NULL))
		return Z_STREAM_ERROR;

	if (s->next_in != NULL && s->next_out != NULL && s->avail_out == 0)
		return Z_BUF_ERROR;

copy_fifo_out_to_next_out:

	if (++loop_cnt == loop_max) {
		prt_err("cannot make progress; too many loops loop_cnt = %ld\n", (long)loop_cnt);
		/* This should never happen.  If we reach this line, there is
		   a bug in this code.  Return Z_STREAM_ERROR, which is the
		   closest error possible in this scenario.  */
		return Z_STREAM_ERROR;
	}

	/* if fifo_out is not empty, first copy contents to next_out.
	 * Remember to keep up to last 32KB as the history in fifo_out. */
	if (s->used_out > 0) {
		write_sz = NX_MIN(s->used_out, s->avail_out);
		if (write_sz > 0) {
			memcpy(s->next_out, s->fifo_out + s->cur_out, write_sz);
			update_stream_out(s, write_sz);
			s->used_out -= write_sz;
			s->cur_out += write_sz;
			fifo_out_len_check(s);
		}
		print_dbg_info(s, __LINE__);

		if (s->used_out > 0 && s->avail_out == 0) {
			prt_info("need more avail_out\n");
			return Z_OK; /* Need more space to write to */
		}

		if (s->is_final == 1) {
			return nx_inflate_verify_checksum(s, 2);
		}
	}

	assert(s->used_out == 0);

	/* If there is no input (s->avail_in == 0 && s->used_in == 0) or there
	   is no space for output (s->avail_out == 0), return. */
	if (s->avail_out == 0 || (s->avail_in == 0 && s->used_in == 0)) return Z_OK;
	if (s->used_out == 0 && s->avail_in == 0 && s->used_in == 0) return Z_OK;
	/* we should flush all data to next_out here, s->used_out should be 0 */

	/* used_in is the data amount waiting in fifo_in; avail_in is
	   the data amount waiting in the user buffer next_in.
	   Cache the input in fifo_in until we have enough data in order to send
	   to the NX GZIP accelerator.
	   Avoid executing this code when avail_in is 0.  That means either the
	   end of the stream or the end of the current request.  There is
	   nothing to copy anyway.
	   Likewise when flush is either Z_FINISH or Z_SYNC_FLUSH.  In these
	   cases, inflate is expected to provide an output and copying data to
	   fifo_in would just add unnecessary delays.
	   The following code is not just an optimization, it is also required
	   by NX because it may refuse to start processing a stream if the input
	   is not large enough.  */
	if (s->avail_in > 0
	    && (s->avail_in + s->used_in < nx_config.cache_threshold)
	    && s->avail_out > 0
	    && flush != Z_FINISH && flush != Z_SYNC_FLUSH) {
		/* We haven't accumulated enough data. Cache any input data
		   provided and wait for the application to send more in order
		   to reduce the amount of requests sent to the accelerator. */
		return copy_data_to_fifo_in(s);
	}
	print_dbg_info(s, __LINE__);

	/* Reset the sync point status */
	s->sync_point = 0;

	/* NX decompresses input data */

	/* address/len lists */
	clearp_dde(ddl_in);
	clearp_dde(ddl_out);

	nx_history_len = s->history_len;

	/* FC, CRC, HistLen, Table 6-6 */
	if (s->resuming || (s->dict_len > 0)) {
		/* Resuming a partially decompressed input.  The key
		   to resume is supplying the max 32KB dictionary
		   (history) to NX, which is basically the last 32KB
		   or less of the output earlier produced. And also
		   make sure partial checksums are carried forward
		*/
		fc = GZIP_FC_DECOMPRESS_RESUME;

		/* Crc of prev job passed to the job to be resumed */
		put32(cmdp->cpb, in_crc, s->crc32);
		put32(cmdp->cpb, in_adler, s->adler32);

		/* Round up the sizes to quadword. Section 2.10
		   Rounding up will not segfault because
		   nx_alloc_buffer has padding at the beginning */

		if (s->dict_len > 0) {
			/* lays dict on top of hist */
			nx_history_len = nx_amend_history_with_dict(s);

			if (s->wrap == HEADER_ZLIB) {
				/* in the raw mode pass crc as is; in the zlib mode
				   initialize them */
				put32(cmdp->cpb, in_crc, INIT_CRC );
				put32(cmdp->cpb, in_adler, INIT_ADLER);
				put32(cmdp->cpb, out_crc, INIT_CRC );
				put32(cmdp->cpb, out_adler, INIT_ADLER);
			}

			s->last_comp_ratio = NX_MAX( NX_MIN(1000UL, s->last_comp_ratio), 100L );

			print_dbg_info(s, __LINE__);
		}
		else {
			/* no dictionary here */
			ASSERT( s->dict_len == 0 );
			nx_history_len = (nx_history_len + NXQWSZ - 1) / NXQWSZ;
			putnn(cmdp->cpb, in_histlen, nx_history_len);
			nx_history_len = nx_history_len * NXQWSZ; /* convert to bytes */

			if (nx_history_len > 0) {
				/* deflate history goes in first */
				ASSERT(s->cur_out >= nx_history_len);
				nx_append_dde(ddl_in, s->fifo_out + (s->cur_out - nx_history_len), nx_history_len);
			}
			print_dbg_info(s, __LINE__);
		}
	}
	else {
		/* First decompress job */
		fc = GZIP_FC_DECOMPRESS;

		nx_history_len = s->history_len = 0;
		/* writing a 0 clears out subc as well */
		cmdp->cpb.in_histlen = 0;

		/* initialize the crc values */
		put32(cmdp->cpb, in_crc, INIT_CRC );
		put32(cmdp->cpb, in_adler, INIT_ADLER);
		put32(cmdp->cpb, out_crc, INIT_CRC );
		put32(cmdp->cpb, out_adler, INIT_ADLER);

		/* We use the most recently measured compression ratio
		   as a heuristic to estimate the input and output
		   sizes. If we give too much input, the target buffer
		   overflows and NX cycles are wasted, and then we
		   must retry with smaller input size. 1000 is 100% */
		s->last_comp_ratio = 1000UL;
	}

	/* clear then copy fc to the crb */
	cmdp->crb.gzip_fc = 0;
	putnn(cmdp->crb, gzip_fc, fc);

	/*
	 * NX source buffers
	 */
	/* buffered user input is next */
	if (s->fifo_in != NULL)
		nx_append_dde(ddl_in, s->fifo_in + s->cur_in, s->used_in);
	/* then current user input */
	nx_append_dde(ddl_in, s->next_in, s->avail_in);
	source_sz = getp32(ddl_in, ddebc); /* total bytes going in to engine */
	ASSERT( source_sz > nx_history_len );

	/*
	 * NX target buffers
	 */
	ASSERT(s->used_out == 0);

	uint32_t len_next_out = s->avail_out;
	nx_append_dde(ddl_out, s->next_out, len_next_out); /* decomp in to user buffer */

	/* overflow, used_out == 0 required by definition, +used_out below is unnecessary */
	nx_append_dde(ddl_out, s->fifo_out + s->cur_out + s->used_out, s->len_out - s->cur_out - s->used_out);
	target_sz = len_next_out + s->len_out - s->cur_out - s->used_out;

	prt_info("len_next_out %d len_out %d cur_out %d used_out %d source_sz %d history_len %d\n",
		 len_next_out, s->len_out, s->cur_out, s->used_out, source_sz, nx_history_len);

	/* We want exactly the History size amount of 32KB to overflow
	   in to fifo_out.  If overflow is less, the history spans
	   next_out and fifo_out and must be copied in to fifo_out to
	   setup history for the next job, and the fifo_out fraction is
	   also copied back to user's next_out before the next job.
	   If overflow is more, all the overflow must be copied back
	   to user's next_out before the next job. We want to minimize
	   these copies (memcpy) for performance. Therefore, the
	   heuristic here will estimate the source size for the
	   desired target size */

	/* avail_out plus 32 KB history plus a bit of overhead */
	uint32_t target_sz_expected = len_next_out + INF_HIS_LEN + (INF_HIS_LEN >> 2);

	target_sz_expected = NX_MIN(target_sz_expected, inflate_per_job_len);

	/* e.g. if we want 100KB at the output and if the compression
	   ratio is 10% we want 10KB if input */
	uint32_t source_sz_expected = (uint32_t)(((uint64_t)target_sz_expected * s->last_comp_ratio + 1000L)/1000UL);

	prt_info("target_sz_expected %d source_sz_expected %d source_sz %d last_comp_ratio %d nx_history_len %d\n", target_sz_expected, source_sz_expected, source_sz, s->last_comp_ratio, nx_history_len);

	/* do not include input side history in the estimation */
	source_sz = source_sz - nx_history_len;

	ASSERT(source_sz > 0);

	source_sz = NX_MIN(source_sz, source_sz_expected);

	/* add the history back */
	source_sz = source_sz + nx_history_len;

restart_nx:

	putp32(ddl_in, ddebc, source_sz);

	/* fault in pages */
	nx_touch_pages_dde(ddl_in, source_sz, nx_config.page_sz, 0);
	nx_touch_pages_dde(ddl_out, target_sz, nx_config.page_sz, 1);
	nx_touch_pages( (void *)cmdp, sizeof(nx_gzip_crb_cpb_t), nx_config.page_sz, 0);

	/*
	 * send job to NX
	 */
	cc = nx_submit_job(ddl_in, ddl_out, cmdp, s->nxdevp);

	switch (cc) {

	case ERR_NX_AT_FAULT:

		/* We touched the pages ahead of time. In the most
		   common case we shouldn't be here. But may be some
		   pages were paged out. Kernel should have placed the
		   faulting address to fsaddr */
		print_dbg_info(s, __LINE__);

		prt_warn("ERR_NX_AT_FAULT: crb.csb.fsaddr %p source_sz %d ",
			 (void *)cmdp->crb.csb.fsaddr, source_sz);
		prt_warn("target_sz %d\n", target_sz);
#ifdef NX_LOG_SOURCE_TARGET
		nx_print_dde(ddl_in, "source");
		nx_print_dde(ddl_out, "target");
#endif
		if (ticks_total == 0) {
			/* try once with exact number of pages */
			ticks_total = nx_wait_ticks(500, ticks_total, 0);
			goto restart_nx;
		}
		else {
			/* if still faulting try fewer input pages *
			   assuming memory outage; */
			ASSERT( source_sz > nx_history_len );

			/* We insist on having a minimum of
			   INF_MIN_INPUT_LEN and
			   INF_MAX_EXPANSION_BYTES memory present;
			   that is about 2 pages minimum for source and
			   and 6 pages for target; if the system does not
			   have 8 free pages then the loop will last forever */
			source_sz = source_sz - nx_history_len;
			if (source_sz > (2 * INF_MIN_INPUT_LEN))
				source_sz = (source_sz + 1) / 2;
			else if (source_sz > INF_MIN_INPUT_LEN)
				source_sz = INF_MIN_INPUT_LEN;

			/* else if caller gave fewer source bytes, keep it as is */
			source_sz = source_sz + nx_history_len;

			if (target_sz > (2 * INF_MAX_EXPANSION_BYTES))
				target_sz = (target_sz + 1) / 2;
			else if (target_sz > INF_MAX_EXPANSION_BYTES)
				target_sz = INF_MAX_EXPANSION_BYTES;

			ticks_total = nx_wait_ticks(500, ticks_total, 0);
			if (ticks_total > (timeout_pgfaults * nx_get_freq())) {
			   /* TODO what to do when page faults are too many?
			    * Kernel MM would have killed the process. */
				prt_err("Cannot make progress; too many page");
				prt_err(" faults cc= %d\n", cc);
			}
			else {
				prt_warn("ERR_NX_AT_FAULT: more retry\n");
				goto restart_nx;
			}
		}

	case ERR_NX_DATA_LENGTH:
		/* Not an error in the most common case; it just says
		   there is trailing data that we must examine */

		/* CC=3 CE(1)=0 CE(0)=1 indicates partial completion
		   Fig.6-7 and Table 6-8 */
		nx_ce = get_csb_ce_ms3b(cmdp->crb.csb);

		if (!csb_ce_termination(nx_ce) &&
		    csb_ce_partial_completion(nx_ce)) {
			/* check CPB for more information
			   spbc and tpbc are valid */
			sfbt = getnn(cmdp->cpb, out_sfbt); /* Table 6-4 */
			subc = getnn(cmdp->cpb, out_subc); /* Table 6-4 */
			spbc = get32(cmdp->cpb, out_spbc_decomp);
			tpbc = get32(cmdp->crb.csb, tpbc);
			ASSERT(target_sz >= tpbc);
			goto ok_cc3; /* not an error */
		}
		else {
			/* History length error when CE(1)=1 CE(0)=0.
			   We have a bug */
			rc = Z_DATA_ERROR;
			prt_err("history length error cc= %d\n", cc);
			goto err5;
		}

	case ERR_NX_TARGET_SPACE:
		/* Target buffer not large enough; retry smaller input
		   data; give at least 1 byte. SPBC/TPBC are not valid */
		ASSERT( source_sz > nx_history_len );
		source_sz = ((source_sz - nx_history_len + 1) / 2) + nx_history_len;

		source_sz = source_sz - nx_history_len;
		/* reduce large source down to minimum viable; if
		   source is already small don't change it */
		if (source_sz > (2 * INF_MIN_INPUT_LEN))
			source_sz = (source_sz + 1) / 2;
		else if (source_sz > INF_MIN_INPUT_LEN)
			source_sz = INF_MIN_INPUT_LEN;

		/* else if caller gave fewer source bytes, keep it as is */
		source_sz = source_sz + nx_history_len;

		/* do not change target size because we allocated a
		   minimum of INF_MAX_EXPANSION_BYTES which should
		   cover the max expansion of INF_MIN_INPUT_LEN
		   bytes */

		prt_info("ERR_NX_TARGET_SPACE; retry with smaller input data src %d hist %d\n", source_sz, nx_history_len);
		goto restart_nx;

	case ERR_NX_OK:

		/* This should not happen for gzip or zlib formatted data;
		 * we need trailing crc and isize */
		prt_info("ERR_NX_OK\n");
		spbc = get32(cmdp->cpb, out_spbc_decomp);
		tpbc = get32(cmdp->crb.csb, tpbc);
		ASSERT(target_sz >= tpbc);
		ASSERT(spbc >= s->history_len);
		source_sz = spbc - nx_history_len;
		goto offsets_state;

	default:
		prt_err("error: cc = %u, cc = 0x%x\n", cc, cc);
		char* csb = (char*) (&cmdp->crb.csb);
		for(int i = 0; i < 4; i++) /* dump first 32 bits of csb */
			prt_err("CSB: 0x %02x %02x %02x %02x\n", csb[0], csb[1], csb[2], csb[3]);
		rc = Z_DATA_ERROR;
		goto err5;
	}

ok_cc3:

	prt_info("cc3: sfbt %x subc %d\n", sfbt, subc);
	print_dbg_info(s, __LINE__);

	ASSERT(spbc > nx_history_len);
	source_sz = spbc - nx_history_len;

	partial_bits = 0;

	/* Table 6-4: Source Final Block Type (SFBT) describes the
	   last processed deflate block and clues the software how to
	   resume the next job.  SUBC indicates how many input bits NX
	   consumed but did not process.  SPBC indicates how many
	   bytes of source were given to the accelerator including
	   history bytes.
	*/
	switch (sfbt) {
		char* last_byte;
		int dhtlen;

	case 0b0000: /* Deflate final EOB received */

		/* Calculating the checksum start position. */
		source_sz = source_sz - subc / 8;
		s->is_final = 1;
		break;

		/* Resume decompression cases are below. Basically
		   indicates where NX has suspended and how to resume
		   the input stream */

	case 0b1000: /* Within a literal block; use rembytecount */
	case 0b1001: /* Within a literal block; use rembytecount; bfinal=1 */

		/* Supply the partially processed source byte again */
		source_sz = source_sz - ((subc + 7) / 8);
		partial_bits = subc;

		/* SUBC LS 3bits: number of bits in the first source
		 * byte need to be processed. */
		/* 000 means all 8 bits;  Table 6-3 */
		/* Clear subc, histlen, sfbt, rembytecnt, dhtlen  */
		cmdp->cpb.in_subc = 0;
		cmdp->cpb.in_sfbt = 0;
		putnn(cmdp->cpb, in_subc, subc % 8);
		putnn(cmdp->cpb, in_sfbt, sfbt);
		putnn(cmdp->cpb, in_rembytecnt, getnn( cmdp->cpb, out_rembytecnt));
		break;

	case 0b1010: /* Within a FH block; */
	case 0b1011: /* Within a FH block; bfinal=1 */

		source_sz = source_sz - ((subc + 7) / 8);
		partial_bits = subc;

		/* Clear subc, histlen, sfbt, rembytecnt, dhtlen */
		cmdp->cpb.in_subc = 0;
		cmdp->cpb.in_sfbt = 0;
		putnn(cmdp->cpb, in_subc, subc % 8);
		putnn(cmdp->cpb, in_sfbt, sfbt);
		break;

	case 0b1100: /* Within a DH block; */
	case 0b1101: /* Within a DH block; bfinal=1 */

		source_sz = source_sz - ((subc + 7) / 8);
		partial_bits = subc;

		/* Clear subc, histlen, sfbt, rembytecnt, dhtlen */
		cmdp->cpb.in_subc = 0;
		cmdp->cpb.in_sfbt = 0;
		putnn(cmdp->cpb, in_subc, subc % 8);
		putnn(cmdp->cpb, in_sfbt, sfbt);

		dhtlen = getnn(cmdp->cpb, out_dhtlen);
		putnn(cmdp->cpb, in_dhtlen, dhtlen);
		ASSERT(dhtlen >= 42);

		/* Round up to a qword */
		dhtlen = (dhtlen + 127) / 128;

		/* Clear any unused bits in the last qword */
		/* cmdp->cpb.in_dht[dhtlen-1].dword[0] = 0; */
		/* cmdp->cpb.in_dht[dhtlen-1].dword[1] = 0; */

		while (dhtlen > 0) { /* Copy dht from cpb.out to cpb.in */
			--dhtlen;
			cmdp->cpb.in_dht[dhtlen] = cmdp->cpb.out_dht[dhtlen];
		}
		break;

	case 0b1110: /* Within a block header; bfinal=0; */
		     /* Also given if source data exactly ends (SUBC=0) with EOB code */
		     /* with BFINAL=0. Means the next byte will contain a block header. */

		/* Find if the last byte is in fifo_in or next_in. */
		last_byte = source_sz <= s->used_in ?
			&s->fifo_in[s->cur_in + source_sz - 1] :
			(char*) &s->next_in[source_sz - s->used_in - 1];

		/* inflateSyncPoint needs to detect the situation when we are in
		 * the middle of processing a literal block and run out of input
		 * right before LEN/NLEN fields. So the engine will have
		 * processed the 3-bit header and between 0 and 7 padding bits.
		 * However, according to the docs (Table 5-3) there are
		 * two other situations that may yield the same SFBT and SUBC
		 * combination:
		 *   1. Last block ended and FC was *_DECOMPRESS_*_SINGLE_BLK_N_SUSPEND
		 *   2. Stopped in the middle of processing DH block header.
		 *
		 * The first case is easily ignored after testing the value of
		 * fc.  The second case requires to analyse the value of the
		 * last 1 or 2 bytes of the input stream in order to distinguish
		 * it with the case that has a sync point. */

		if (subc >= 3 && subc <= 10
		    && fc != GZIP_FC_DECOMPRESS_SINGLE_BLK_N_SUSPEND
		    && fc != GZIP_FC_DECOMPRESS_RESUME_SINGLE_BLK_N_SUSPEND) {

			/* A sync point happens when the last SUBC input bits
			 * are all zero (all-zero header and padding bits).
			 * These should correspond to:
			 *   BFINAL == 0b0 (1 bit) | BTYPE == 0b00 (2 bits) |
			 *             padding (0-7 bits). */
			if(subc <= 8) {
				/* If SUBC is less than 8, it means the BTYPE
				 * field was completely contained in the last
				 * byte, so we don't have to check the one
				 * before that. */
				s->sync_point = !(*last_byte & (char)(0xff<<(8-subc)));
			} else {
				s->sync_point = *last_byte == 0 &&
					!(*(last_byte-1) & (char)(0xff<<(16-subc)));
			}

		}

		prt_info("sync point check: s->sync_point %d fc %x subc %d"
			 " last_byte 0x%x last_byte-1 0x%x\n",
			s->sync_point, fc, subc, *last_byte, *(last_byte-1));
	case 0b1111: /* within a block header with BFINAL=1. */

		/** If source_sz becomes 0, the source could not be processed,
		 * and we have to increase the source before
		 * \ref NotEnoughSource "the next iteration".
		 */
		source_sz = source_sz - ((subc + 7) / 8);
		partial_bits = subc;
		prt_info("source_sz %d partial_bits %d in_histlen %d in_subc %d"
			 " in_sfbt %d in_rembytecnt %d\n",
			 source_sz, partial_bits,
			 getnn(cmdp->cpb, in_histlen),
			 getnn(cmdp->cpb, in_subc),
			 getnn(cmdp->cpb, in_sfbt),
			 getnn(cmdp->cpb, in_rembytecnt));

		/* Clear subc, histlen, sfbt, rembytecnt, dhtlen */
		cmdp->cpb.in_subc = 0;
		cmdp->cpb.in_sfbt = 0;
		putnn(cmdp->cpb, in_subc, subc % 8);
		putnn(cmdp->cpb, in_sfbt, sfbt);
	}

offsets_state:

	print_dbg_info(s, __LINE__);
	prt_info("== %d source_sz %d used_in %d cur_in %d\n", __LINE__, source_sz, s->used_in, s->cur_in );

	/* Adjust the source and target buffer offsets and lengths  */
	/* source_sz is the real used in size */
	if (source_sz > s->used_in) {
		update_stream_in(s, source_sz - s->used_in);
		s->used_in = 0;
	}
	else {
		s->used_in -= source_sz;
		s->cur_in  += source_sz;
		fifo_in_len_check(s);
	}

	print_dbg_info(s, __LINE__);
	prt_info("== %d source_sz %d used_in %d cur_in %d\n", __LINE__, source_sz, s->used_in, s->cur_in );

	nx_inflate_update_checksum(s);

	int overflow_len = tpbc - len_next_out;
	if (overflow_len <= 0) { /* there is no overflow */
		assert(s->used_out == 0);
		if (s->is_final == 0) {
			int need_len = NX_MIN(INF_HIS_LEN, tpbc);
			/* Copy the tail of data in next_out as the history to
			   the current head of fifo_out. Size is 32KB commonly
			   but can be less if the engine produce less than
			   32KB.  Note that cur_out-32KB already contains the
			   history of the previous operation. The new history
			   is appended after the old history */
			memcpy(s->fifo_out + s->cur_out, s->next_out + tpbc - need_len, need_len);
			s->cur_out += need_len;
			fifo_out_len_check(s);
		}
		update_stream_out(s, tpbc);

		print_dbg_info(s, __LINE__);
	}
	else if (overflow_len > 0 && overflow_len < INF_HIS_LEN){
		int need_len = INF_HIS_LEN - overflow_len;
		need_len = NX_MIN(need_len, len_next_out);
		int len;
		/* When overflow is less than the history len e.g. the
		   history is now spanning next_out and fifo_out */
		if (len_next_out + overflow_len > INF_HIS_LEN) {
			len = INF_HIS_LEN - overflow_len;
			memcpy(s->fifo_out + s->cur_out - len, s->next_out + len_next_out - len, len);
		}
		else {
			len = INF_HIS_LEN - (len_next_out + overflow_len);
			/* len_next_out is the amount engine wrote next_out. */
			/* Shifts fifo_out contents backwards towards
			   the beginning. Use memmove because memory may
			   overlap. */
			memmove(s->fifo_out + s->cur_out - len_next_out - len,
				s->fifo_out + s->cur_out - len, len);
			/* copies from next_out to the gap opened in
			   fifo_out as a result of previous memmove. Also use
			   memmove because memory may overlap. */
			memmove(s->fifo_out + s->cur_out - len_next_out,
				s->next_out, len_next_out);
		}

		s->used_out += overflow_len;
		update_stream_out(s, len_next_out);

		print_dbg_info(s, __LINE__);
	}
	else { /* overflow_len > 1<<15 */
		s->used_out += overflow_len;
		update_stream_out(s, len_next_out);

		print_dbg_info(s, __LINE__);
	}

	print_dbg_info(s, __LINE__);

	s->history_len = NX_MIN(s->total_out + s->used_out, INF_HIS_LEN);

	prt_info("== %d source_sz %d tpbc %d last_comp_ratio %d\n", __LINE__, source_sz, tpbc, s->last_comp_ratio);

	if (source_sz != 0 || tpbc != 0) {
		/* if both are zero estimate is probably wrong; so keep it as is */
		s->last_comp_ratio = (1000UL * ((uint64_t)source_sz + 1)) / ((uint64_t)tpbc + 1);
		s->last_comp_ratio = NX_MAX( NX_MIN(1000UL, s->last_comp_ratio), 1 ); /* bounds check */
	}
	/** \anchor NotEnoughSource
	 * However, if not enough source is available, we must give more.
	 * Assume the worst case and use 100%.
	 */
	else if (cc == ERR_NX_DATA_LENGTH && sfbt == 0xe && subc > 0)
		s->last_comp_ratio = 1000UL;

	prt_info("== %d source_sz %d tpbc %d last_comp_ratio %d\n", __LINE__, source_sz, tpbc, s->last_comp_ratio);

	if (!s->is_final) s->resuming = 1;

	/* raw mode will set this again for the next dictionary */
	s->dict_len = 0;

	print_dbg_info(s, __LINE__);

	if (s->is_final == 1 || cc == ERR_NX_OK) {

		print_dbg_info(s, __LINE__);

		/* copy trailer bytes to temp storage */
		nx_inflate_verify_checksum(s, 1);
		/* update total_in */
		s->total_in = s->total_in - s->used_in; /* garbage past cksum ????? */
		s->is_final = 1;
		/* s->used_in = 0; */
		if (s->used_out == 0) {
			/* final state and everything copied to next_out */
			print_dbg_info(s, __LINE__);
			/* return Z_STREAM_END if all cksum bytes
			 * available otherwise Z_OK */
			return nx_inflate_verify_checksum(s, 0);
		}
		else {
			goto copy_fifo_out_to_next_out;
		}
	}

	print_dbg_info(s, __LINE__);
	prt_info("== %d flush %d is_final %d last_comp_ratio %d\n", __LINE__, s->flush, s->is_final, s->last_comp_ratio);

	if (s->avail_out > 0) {
		if (((s->used_in + s->avail_in) > ((partial_bits + 7) / 8))) {
			/* if more input is available than the required bits */
			print_dbg_info(s, __LINE__);
			goto copy_fifo_out_to_next_out;
		}

		/* Not enough input to keep going */

		if (s->avail_in > 0) {
			/* Cache whatever input is left and let the user provide
			 * more in the next call. */
			return copy_data_to_fifo_in(s);
		}
	}

	/* need more bits from user */
	print_dbg_info(s, __LINE__);
	return Z_OK;

err5:
	prt_err("rc %d\n", rc);
	return rc;
}

int nx_inflateSetDictionary(z_streamp strm, const Bytef *dictionary, uInt dictLength)
{
	nx_streamp s;
	uint32_t adler;
	int cc;

	if (dictionary == NULL || strm == NULL)
		return Z_STREAM_ERROR;

	if (NULL == (s = (nx_streamp) strm->state))
		return Z_STREAM_ERROR;

	if (s->wrap == HEADER_GZIP) {
		/* gzip doesn't allow dictionaries; */
		prt_err("inflateSetDictionary error: gzip format does not permit dictionary\n");
		return Z_STREAM_ERROR;
	}

	if (s->inf_state != inf_state_zlib_dict && s->wrap == HEADER_ZLIB ) {
		prt_err("inflateSetDictionary error: inflate did not ask for a dictionary\n");
		return Z_STREAM_ERROR;
	}

	do {
		if (s->dict != NULL) {
			if(dictLength > s->dict_alloc_len) /* need to resize? */
				nx_free_buffer(s->dict, s->dict_alloc_len, 0);
			else
				break; /* Skip allocation */
		}

		/* one time allocation until inflateEnd() */
		s->dict_alloc_len = NX_MAX( INF_MAX_DICT_LEN, dictLength);
		/* we don't need larger than INF_MAX_DICT_LEN in
		   principle; however nx_copy needs a target buffer to
		   be able to compute adler32 */
		if (NULL == (s->dict = nx_alloc_buffer(s->dict_alloc_len, s->page_sz, 0))) {
			s->dict_alloc_len = 0;
			return Z_MEM_ERROR;
		}
		s->dict_len = 0;
	} while(0);

	/* copy dictionary in and also calculate it's checksum */
	adler = INIT_ADLER;
	cc = nx_copy(s->dict, (char *)dictionary, dictLength, NULL, &adler, s->nxdevp);
	if (cc != ERR_NX_OK) {
		prt_err("nx_copy dictionary error\n");
		return Z_STREAM_ERROR;
	}

	/* Got here due to inflate() returning Z_NEED_DICT which should
	   have saved the dict_id found in the zlib header to
	   s->dict_id; raw blocks do not have a dictionary id */

	if (s->dict_id != adler && s->wrap == HEADER_ZLIB) {
		prt_err("supplied dictionary ID does not match the inflate header\n");
		return Z_DATA_ERROR;
	}
	s->dict_len = dictLength;

	return Z_OK;


	/*
	   Notes: if there is historical data in fifo_out, I need to
	   truncate it by the dictlen amount (see the amend comment)

	   zlib.h: inflate

	   If a preset dictionary is needed after this call (see
	   inflateSetDictionary below), inflate sets strm->adler to
	   the Adler-32 checksum of the dictionary chosen by the
	   compressor and returns Z_NEED_DICT; otherwise it sets
	   strm->adler to the Adler-32 checksum of all output produced
	   so far (that is, total_out bytes) and returns Z_OK,
	   Z_STREAM_END or an error code as described below.  At the
	   end of the stream, inflate() checks that its computed
	   Adler-32 checksum is equal to that saved by the compressor
	   and returns Z_STREAM_END only if the checksum is correct.

	   inflateSetDictionary

	   Initializes the decompression dictionary from the given
	   uncompressed byte sequence.  This function must be called
	   immediately after a call of inflate, if that call returned
	   Z_NEED_DICT.  The dictionary chosen by the compressor can
	   be determined from the Adler-32 value returned by that call
	   of inflate.  The compressor and decompressor must use
	   exactly the same dictionary (see deflateSetDictionary).
	   For raw inflate, this function can be called at any time to
	   set the dictionary.  If the provided dictionary is smaller
	   than the window and there is already data in the window,
	   then the provided dictionary will amend what's there.  The
	   application must insure that the dictionary that was used
	   for compression is provided.

	   inflateSetDictionary returns Z_OK if success,
	   Z_STREAM_ERROR if a parameter is invalid (e.g.  dictionary
	   being Z_NULL) or the stream state is inconsistent,
	   Z_DATA_ERROR if the given dictionary doesn't match the
	   expected one (incorrect Adler-32 value).
	   inflateSetDictionary does not perform any decompression:
	   this will be done by subsequent calls of inflate().
	*/
}

int nx_inflateCopy(z_streamp dest, z_streamp source)
{
	nx_streamp s, d;

	prt_info("%s:%d dst %p src %p\n", __FUNCTION__, __LINE__,dest, source);

	if (dest == NULL || source == NULL)
		return Z_STREAM_ERROR;

	if (source->state == NULL)
		return Z_STREAM_ERROR;

	s = (nx_streamp) source->state;

	/* z_stream copy */
	memcpy((void *)dest, (const void *)source, sizeof(z_stream));

	/* allocate nx specific struct for dest */
	if (NULL == (d = nx_alloc_buffer(sizeof(*d), nx_config.page_sz, nx_config.mlock_nx_crb_csb)))
		goto mem_error;

	d->dict = d->fifo_in = d->fifo_out = NULL;

	/* source nx state copied to dest nx state */
	memcpy(d, s, sizeof(*s));

	/* dest points to its child nx_stream struct */
	dest->state = (void *)d;

	/* nx overflow underflow buffers */
	if (s->fifo_out != NULL) {
		if (NULL == (d->fifo_out = nx_alloc_buffer(s->len_out, nx_config.page_sz, 0)))
			goto mem_error;
		memcpy(d->fifo_out, s->fifo_out, s->len_out);
	}

	if (s->fifo_in != NULL) {
		if (NULL == (d->fifo_in = nx_alloc_buffer(s->len_in, nx_config.page_sz, 0)))
			goto mem_error;
		memcpy(d->fifo_in, s->fifo_in, s->len_in);
	}

	if (s->dict != NULL) {
		if (NULL == (d->dict = nx_alloc_buffer(s->dict_alloc_len, nx_config.page_sz, 0)))
			goto mem_error;
		memcpy(d->dict, s->dict, s->dict_alloc_len);
	}

	d->zstrm = dest;  /* pointer to parent */

	return Z_OK;

mem_error:

	prt_info("%s:%d memory error\n", __FUNCTION__, __LINE__);

	if (d->dict != NULL)
		nx_free_buffer(d->dict, d->dict_alloc_len, 0);
	if (d->fifo_in != NULL)
		nx_free_buffer(d->fifo_in, d->len_in, 0);
	if (d->fifo_out != NULL)
		nx_free_buffer(d->fifo_out, d->len_out, 0);
	if (d != NULL)
		nx_free_buffer(d, sizeof(*d), nx_config.mlock_nx_crb_csb);

	return Z_MEM_ERROR;
}

int nx_inflateGetHeader(z_streamp strm, gz_headerp head)
{
	nx_streamp s;

	prt_info("%s:%d strm %p gzhead %p\n", __FUNCTION__, __LINE__, strm, head);

	if (strm == NULL)
		return Z_STREAM_ERROR;

	if (strm->state == NULL)
		return Z_STREAM_ERROR;

	s = (nx_streamp) strm->state;

	if (s->wrap != HEADER_GZIP)
		return Z_STREAM_ERROR;

	s->gzhead = head;
	head->done = 0;

	return Z_OK;
}

int nx_inflateSyncPoint(z_streamp strm)
{
	nx_streamp s;

	if (strm == NULL)
		return Z_STREAM_ERROR;

	if (strm->state == NULL)
		return Z_STREAM_ERROR;

	s = (nx_streamp) strm->state;

	return s->sync_point;
}

#ifdef ZLIB_API
int inflateInit_(z_streamp strm, const char *version, int stream_size)
{
	return inflateInit2_(strm, DEF_WBITS, version, stream_size);
}

int inflateInit2_(z_streamp strm, int windowBits, const char *version, int stream_size)
{
	int rc;
	void *temp = NULL;
	nx_streamp s;

	/* statistic */
	zlib_stats_inc(&zlib_stats.inflateInit);

	strm->state = NULL;
	if(nx_config.mode.inflate == GZIP_AUTO ||
	   nx_config.mode.inflate == GZIP_MIX){
		/* Call sw and nx initialization.  */
		rc = sw_inflateInit2_(strm, windowBits, version, stream_size);
		if(rc != Z_OK) return rc;

		/*If the stream has been initialized by sw*/
		if(strm->state && (0 == has_nx_state(strm))){
			temp = (void *)strm->state; /*record the sw context*/
			strm->state = NULL;
			prt_info("this stream has been initialized by sw\n");
		}

		rc = nx_inflateInit2_(strm, windowBits, version, stream_size);
		if(rc != Z_OK){
			sw_inflateEnd(strm);
			return rc;
		}

		if(temp){ /* recorded sw context*/
			s = (nx_streamp) strm->state;
			s->sw_stream = temp;
			s->switchable = 1;
		}
	}else if(nx_config.mode.inflate == GZIP_NX){
		rc = nx_inflateInit2_(strm, windowBits, version, stream_size);
	}else{
		rc = sw_inflateInit2_(strm, windowBits, version, stream_size);
	}
	return rc;
}
int inflateReset(z_streamp strm)
{
	int rc;

	if (0 == has_nx_state(strm)){
		rc = sw_inflateReset(strm);
	}else{
		rc = nx_inflateReset(strm);
	}

	return rc;
}

int inflateReset2(z_streamp strm, int windowBits)
{
	int rc;

	if (0 == has_nx_state(strm)){
		rc = sw_inflateReset2(strm,windowBits);
	}else{
		rc = nx_inflateReset2(strm,windowBits);
	}

	return rc;
}

int inflateResetKeep(z_streamp strm)
{
	int rc;

	if (0 == has_nx_state(strm)){
		rc = sw_inflateResetKeep(strm);
	}else{
		rc = nx_inflateResetKeep(strm);
	}

	return rc;
}


int inflateEnd(z_streamp strm)
{
	int rc;

	/* statistic */
	zlib_stats_inc(&zlib_stats.inflateEnd);

	if (0 == has_nx_state(strm)){
		rc = sw_inflateEnd(strm);
	}else{
		rc =nx_inflateEnd(strm);
	}

	return rc;
}
int inflate(z_streamp strm, int flush)
{
	int rc;
	unsigned int avail_in_slot, avail_out_slot;
	uint64_t t1=0, t2, t_diff;
	unsigned int avail_in=0, avail_out=0;

	if (strm == NULL || strm->state == NULL) return Z_STREAM_ERROR;

	/* statistic */
	if (nx_gzip_gather_statistics()) {
		avail_in = strm->avail_in;
		avail_out = strm->avail_out;
		t1 = nx_get_time();
	}

	if (0 == has_nx_state(strm)){
		prt_info("call sw_inflate,len=%d\n", strm->avail_in);
		rc = sw_inflate(strm, flush);
		prt_info("call sw_inflate,rc=%d\n", rc);
	}else{
		rc = nx_inflate(strm, flush);
	}

	/* statistic */
	if (nx_gzip_gather_statistics() && (rc == Z_OK || rc == Z_STREAM_END)) {

		avail_in_slot = avail_in / 4096;
		if (avail_in_slot >= ZLIB_SIZE_SLOTS)
			avail_in_slot = ZLIB_SIZE_SLOTS - 1;
		zlib_stats_inc(&zlib_stats.inflate_avail_in[avail_in_slot]);

		avail_out_slot = avail_out / 4096;
		if (avail_out_slot >= ZLIB_SIZE_SLOTS)
			avail_out_slot = ZLIB_SIZE_SLOTS - 1;
		zlib_stats_inc(&zlib_stats.inflate_avail_out[avail_out_slot]);
		zlib_stats_inc(&zlib_stats.inflate);
		if (0 == has_nx_state(strm)){
			zlib_stats_inc(&zlib_stats.inflate_sw);
		}else{
			zlib_stats_inc(&zlib_stats.inflate_nx);
		}

		__atomic_fetch_add(&zlib_stats.inflate_len, avail_in,  __ATOMIC_RELAXED);

		t2 = nx_get_time();
		t_diff = nx_time_to_us(nx_time_diff(t1,t2));

		__atomic_fetch_add(&zlib_stats.inflate_time, t_diff,  __ATOMIC_RELAXED);
	}

	return rc;
}
int inflateSetDictionary(z_streamp strm, const Bytef *dictionary, uInt dictLength)
{
	int rc;

	if (0 == has_nx_state(strm)){
		rc = sw_inflateSetDictionary(strm, dictionary, dictLength);
	}else{
		rc = nx_inflateSetDictionary(strm, dictionary, dictLength);
	}

	return rc;
}

int inflateCopy(z_streamp dest, z_streamp source)
{
	int rc;

	if (0 == has_nx_state(source)){
		rc = sw_inflateCopy(dest, source);
	}else{
		rc = nx_inflateCopy(dest, source);
	}

	return rc;
}

int inflateGetHeader(z_streamp strm, gz_headerp head)
{
	int rc;

	if (0 == has_nx_state(strm)){
		rc = sw_inflateGetHeader(strm, head);
	}else{
		rc = nx_inflateGetHeader(strm, head);
	}

	return rc;
}

int inflateSyncPoint(z_streamp strm)
{
	int rc;

	if (0 == has_nx_state(strm))
		rc = sw_inflateSyncPoint(strm);
	else
		rc = nx_inflateSyncPoint(strm);

	return rc;
}

#endif

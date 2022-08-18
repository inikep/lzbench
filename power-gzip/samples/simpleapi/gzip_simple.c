#include "gzip_simple.h"

__attribute__((constructor)) void nx_init(void)
{
	nx_overflow_buffer = malloc(OVERFLOW_BUFFER_SIZE);
	memset(nx_overflow_buffer, 0, OVERFLOW_BUFFER_SIZE);

	int ctr = 0;
	for (ctr = 0; ctr < NX_MAX_DEVICES; ctr++) {
		nx_devices[ctr] = malloc(sizeof(p9_simple_handle_t));
		nx_devices[ctr]->vas_handle =
			nx_function_begin(NX_FUNC_COMP_GZIP, ctr);
		if (nx_devices[ctr]->vas_handle == NULL) {
			fprintf(stderr, "device handle open error chip id %d\n",
				ctr);
			exit(-1);
		}
		nx_devices[ctr]->open_count = 0;
		nx_devices[ctr]->chipId = ctr;
	}

	/* add a signal action */
	sigact.sa_handler = 0;
	sigact.sa_sigaction = sigsegv_handler;
	sigact.sa_flags = SA_SIGINFO;
	sigact.sa_restorer = 0;
	sigemptyset(&sigact.sa_mask);
	sigaction(SIGSEGV, &sigact, NULL);
}

__attribute__((destructor)) void nx_deinit(void)
{
	int ctr = 0;
	int retval = 0;
	for (ctr = 0; ctr < NX_MAX_DEVICES; ctr++) {
		retval = nx_function_end(nx_devices[ctr]->vas_handle);
		if (retval < 0) {
			fprintf(stderr,
				"device handle close error chip id %d\n", ctr);
			exit(-1);
		}
		if (nx_devices[ctr]->open_count != 0) {

			fprintf(stderr,
				" device handles are not properly closed chip id=%d open count=%d \n",
				ctr, nx_devices[ctr]->open_count);
		}
		free(nx_devices[ctr]);
	}


	free(nx_overflow_buffer);
}


/******************/
/*utility functions*/
/******************/
int findChip(int cpuId)
{
	if (cpuId < 80) {
		return 0;
	} else {
		return 1;
	}
}


static void intcopy(int value, char *buffer)
{
	buffer[3] = (value >> 24) & 0xFF;
	buffer[2] = (value >> 16) & 0xFF;
	buffer[1] = (value >> 8) & 0xFF;
	buffer[0] = value & 0xFF;
}

/*
   Touch specified number of pages in supplied buffer
 */
static int nx_touch_pages(void *buf, long buf_len, long page_len, int wr)
{
	char *begin = buf;
	char *end = (char *)buf + buf_len - 1;
	volatile char t;

	assert(buf_len >= 0 && !!buf);

	NXPRT(fprintf(stderr, "touch %p %p len 0x%lx wr=%d\n", buf,
		      buf + buf_len, bu f_len, wr));

	if (buf_len <= 0 || buf == NULL)
		return -1;
	do {
		t = *begin;
		if (wr)
			*begin = t;
		begin = begin + page_len;
	} while (begin < end);

	/* when buf_sz is small or buf tail is in another page */
	t = *end;
	if (wr)
		*end = t;

	return 0;
}

/*
   Touch specified number of pages represented in number bytes
   beginning from the first buffer in a dde list.
   Do not touch the pages past buf_sz-th byte's page.

   Set buf_sz = 0 to touch all pages described by the ddep.
*/
static int nx_touch_pages_dde(nx_dde_t *ddep, long buf_sz, long page_sz, int wr)
{
	volatile char t;
	uint32_t indirect_count;
	uint32_t buf_len;
	long total;
	uint64_t buf_addr;
	nx_dde_t *dde_list;
	int i;

	assert(!!ddep);

	nx_touch_pages((void *)ddep, sizeof(nx_dde_t), page_sz, 0);

	indirect_count = getpnn(ddep, dde_count);

	// prt_trace("nx_touch_pages_dde dde_count %d request len 0x%lx\n",
	// indirect_count, buf_sz);

	if (indirect_count == 0) {
		/* direct dde */
		buf_len = getp32(ddep, ddebc);
		buf_addr = getp64(ddep, ddead);

		// prt_trace("touch direct ddebc 0x%x ddead %p\n", buf_len,
		// (void *)buf_addr);

		if (buf_sz == 0)
			nx_touch_pages((void *)buf_addr, buf_len, page_sz, wr);
		else
			nx_touch_pages((void *)buf_addr,
				       NX_MIN(buf_len, buf_sz), page_sz, wr);

		return ERR_NX_OK;
	}

	/* indirect dde */
	if (indirect_count > MAX_DDE_COUNT)
		return ERR_NX_EXCESSIVE_DDE;

	/* first address of the list */
	dde_list = (nx_dde_t *)getp64(ddep, ddead);

	if (buf_sz == 0)
		buf_sz = getp32(ddep, ddebc);

	total = 0;
	for (i = 0; i < indirect_count; i++) {
		buf_len = get32(dde_list[i], ddebc);
		buf_addr = get64(dde_list[i], ddead);
		total += buf_len;

		nx_touch_pages((void *)&(dde_list[i]), sizeof(nx_dde_t),
			       page_sz, 0);

		// prt_trace("touch loop len 0x%x ddead %p total 0x%lx\n",
		// buf_len, (void *)buf_addr, total);

		/* touching fewer pages than encoded in the ddebc */
		if (total > buf_sz) {
			buf_len = NX_MIN(buf_len, total - buf_sz);
			nx_touch_pages((void *)buf_addr, buf_len, page_sz, wr);
			// prt_trace("touch loop break len 0x%x ddead %p\n",
			// buf_len, (void *)buf_addr);
			break;
		}
		nx_touch_pages((void *)buf_addr, buf_len, page_sz, wr);
	}
	return ERR_NX_OK;
}


/*
   Adds an (address, len) pair to the list of ddes (ddl) and updates
   the base dde.  ddl[0] is the only dde in a direct dde which
   contains a single (addr,len) pair.  For more pairs, ddl[0] becomes
   the indirect (base) dde that points to a list of direct ddes.
   See Section 6.4 of the NX-gzip user manual for DDE description.
   Addr=NULL, len=0 clears the ddl[0].  Returns the total number of
   bytes in ddl.  Caller is responsible for allocting the array of
   nx_dde_t *ddl.  If N addresses are required in the scatter-gather
   list, the ddl array must have N+1 entries minimum.
*/
int nx_append_dde(nx_dde_t *ddl, void *addr, uint32_t len)
{
	uint32_t ddecnt;
	uint32_t bytes;

	if (addr == NULL && len == 0) {
		clearp_dde(ddl);
		return 0;
	}

	// prt_trace("%d: nx_append_dde addr %p len %x\n", __LINE__, addr, len);

	/* number of ddes in the dde list ; == 0 when it is a direct dde */
	ddecnt = getpnn(ddl, dde_count);
	bytes = getp32(ddl, ddebc);

	/* NXPRT( fprintf(stderr, "%d: get dde_count %d ddebc %d\n", __LINE__,
	 * ddecnt, bytes ) ); */

	if (ddecnt == 0 && bytes == 0) {
		/* first dde is unused; make it a direct dde */
		bytes = len;
		putp32(ddl, ddebc, bytes);
		putp64(ddl, ddead, (uint64_t)addr);

		/* NXPRT( fprintf(stderr, "%d: put ddebc %d ddead %p\n",
		 * __LINE__, bytes, (void *)addr ) ); */
	} else if (ddecnt == 0) {
		/* converting direct to indirect dde */
		/* ddl[0] becomes head dde of ddl */
		/* copy direct to indirect first */
		ddl[1] = ddl[0];

		/* add the new dde next */
		clear_dde(ddl[2]);
		put32(ddl[2], ddebc, len);
		put64(ddl[2], ddead, (uint64_t)addr);

		/* ddl head points to 2 direct ddes */
		ddecnt = 2;
		putpnn(ddl, dde_count, ddecnt);
		bytes = bytes + len;
		putp32(ddl, ddebc, bytes);
		/* pointer to the first direct dde */
		putp64(ddl, ddead, (uint64_t)&ddl[1]);
	} else {
		/* append a dde to an existing indirect ddl */
		++ddecnt;
		clear_dde(ddl[ddecnt]);
		put64(ddl[ddecnt], ddead, (uint64_t)addr);
		put32(ddl[ddecnt], ddebc, len);

		putpnn(ddl, dde_count, ddecnt);
		bytes = bytes + len;
		putp32(ddl, ddebc, bytes); /* byte sum of all dde */
	}
	return bytes;
}

void sigsegv_handler(int sig, siginfo_t *info, void *ctx)
{
	fprintf(stderr, "%d: Got signal %d si_code %d, si_addr %p\n", getpid(),
		sig, info->si_code, info->si_addr);
	nx_fault_storage_address = info->si_addr;
	// exit(0);
}

static void nx_print_dde(nx_dde_t *ddep, const char *msg)
{
	uint32_t indirect_count;
	uint32_t buf_len;
	uint64_t buf_addr;
	nx_dde_t *dde_list;
	int i;

	assert(!!ddep);
	indirect_count = getpnn(ddep, dde_count);
	buf_len = getp32(ddep, ddebc);
	fprintf(stderr, "%s dde %p dde_count %d, ddebc 0x%x\n", msg, ddep,
		indirect_count, buf_len);

	if (indirect_count == 0) {
		/* direct dde */
		buf_len = getp32(ddep, ddebc);
		buf_addr = getp64(ddep, ddead);
		fprintf(stderr, "  direct dde: ddebc 0x%x ddead %p %p\n",
			buf_len, (void *)buf_addr, (void *)buf_addr + buf_len);
		return;
	}

	/* indirect dde */
	if (indirect_count > MAX_DDE_COUNT) {
		fprintf(stderr, "  error MAX_DDE_COUNT\n");
		return;
	}

	/* first address of the list */
	dde_list = (nx_dde_t *)getp64(ddep, ddead);

	for (i = 0; i < indirect_count; i++) {
		buf_len = get32(dde_list[i], ddebc);
		buf_addr = get64(dde_list[i], ddead);
		fprintf(stderr, " indirect dde: ddebc 0x%x ddead %p %p\n",
			buf_len, (void *)buf_addr, (void *)buf_addr + buf_len);
	}
	return;
}


/*zero out csb and init src and dst*/
void nx_init_csb(nx_gzip_crb_cpb_t *cmdp, void *src, void *dst, int srclen,
		 int dstlen, nx_dde_t *ddl, void *overflow)
{
	/* status, output byte count in tpbc */
	put64(cmdp->crb, csb_address, 0);
	put64(cmdp->crb, csb_address,
	      (uint64_t)&cmdp->crb.csb & csb_address_mask);

	/* source direct dde */
	clear_dde(cmdp->crb.source_dde);
	putnn(cmdp->crb.source_dde, dde_count, 0);
	put32(cmdp->crb.source_dde, ddebc, srclen);
	put64(cmdp->crb.source_dde, ddead, (uint64_t)src);

	if (ddl != NULL) {
		/*switch to indirect dde & append a dummy overflow buffer*/
		clearp_dde(ddl);
		nx_append_dde(ddl, (uint64_t)dst, dstlen);
		nx_append_dde(ddl, (uint64_t)overflow, OVERFLOW_BUFFER_SIZE);
		cmdp->crb.target_dde = *ddl;
		/*source is always direct -- we don't need to touch*/
		/*we touch target dde because of the overflow buffer we supply*/
		nx_touch_pages_dde(ddl, dstlen + OVERFLOW_BUFFER_SIZE, pagesize,
				   1);
	} else {
		/* target direct dde */
		clear_dde(cmdp->crb.target_dde);
		putnn(cmdp->crb.target_dde, dde_count, 0);
		put32(cmdp->crb.target_dde, ddebc, dstlen);
		put64(cmdp->crb.target_dde, ddead, (uint64_t)dst);
	}
}

/******************/
/*Library functions*/
/******************/
/*open device*/
p9_simple_handle_t *p9open()
{
	int cpu_id = 0;
	syscall(SYS_getcpu, &cpu_id, NULL, NULL);
	int chipId = findChip(cpu_id);
	p9_simple_handle_t *nx_device = nx_devices[chipId];
	__sync_fetch_and_add(&(nx_devices[chipId]->open_count), 1);
	return nx_device;
};

/*compress*/
int p9deflate(p9_simple_handle_t *handle, void *src, void *dst, int srclen,
	      int dstlen, char *fname, int flag)
{
	nx_gzip_crb_cpb_t cmd;
	nx_gzip_crb_cpb_t *cmdp = &cmd; // TODO fixit
	int cc;
	/*clear crb and csb*/
	memset(&cmdp->crb, 0, sizeof(cmdp->crb));
	memset((void *)&cmdp->crb.csb, 0, sizeof(cmdp->crb.csb));

	char *dstp = dst;
	int header = 0;
	int trailer = 0;
	/*add gzip/zlib headers*/
	if (flag == GZIP_WRAPPER) {
		dstp[0] = (char)31;   // gzip magic number 1
		dstp[1] = (char)139;  // gzip magic number 2
		dstp[2] = (char)8;    // compression method, 8 is deflate
		dstp[3] = (char)0;    // flags
		intcopy(0, dstp + 4); // last modification time
		dstp[8] = (char)0;    // extra compression flags
		dstp[9] = (char)8;    // TODO operating system
		put32(cmdp->cpb, in_crc, INIT_CRC);
		put32(cmdp->cpb, out_crc, INIT_CRC);
		header = 10;
		trailer = 8;
	} else if (flag == ZLIB_WRAPPER) {
		dstp[0] = (char)120; // zlib magic number 1
		dstp[1] = (char)0;   // TODO how to add this?
		put32(cmdp->cpb, in_adler, INIT_ADLER);
		put32(cmdp->cpb, out_adler, INIT_ADLER);
		header = 2;
		trailer = 4;
	} else if (flag == NO_WRAPPER) {
		// NOOP
	} else {
		return -1;
	}

	/*set command type*/
	put32(cmdp->crb, gzip_fc, 0);
	putnn(cmdp->crb, gzip_fc, GZIP_FC_COMPRESS_FHT);

	/*set source/destination and sizes*/
	nx_init_csb(cmdp, src, dstp + header, srclen, dstlen - header - trailer,
		    NULL, NULL);
	/*touch pages before submitting job*/
	nx_touch_pages(src, srclen, pagesize, 0);
	nx_touch_pages(dstp, dstlen, pagesize, 1);

	/*run the job */
	cc = nxu_run_job(cmdp, handle->vas_handle);

	if (!cc) {
		cc = getnn(cmdp->crb.csb, csb_cc);
	}

	if (cc != ERR_NX_OK && cc != ERR_NX_TPBC_GT_SPBC
	    && cc != ERR_NX_AT_FAULT) {
		fprintf(stderr, "nx deflate error: cc= %d\n", cc);
		return -1;
	}

	if (cc == ERR_NX_AT_FAULT) {
		fprintf(stderr, "nx deflate error page fault: cc= %d\n", cc);
		// TODO maybe handle page faults
		return -1;
	}

	int comp = get32(cmdp->crb.csb, tpbc);

	/*add gzip/zlib trailers*/
	int checksum;
	if (flag == GZIP_WRAPPER) {
		/*gzip trailer, crc32 and compressed data size*/
		checksum = cmdp->cpb.out_crc;
		intcopy(checksum, dstp + header + comp);
		intcopy(srclen, dstp + header + comp + 4);
	} else if (flag == ZLIB_WRAPPER) {
		/*zlib trailer, adler32 only*/
		checksum = cmdp->cpb.out_adler;
		intcopy(checksum, dstp + header + comp);
	}
	return comp + header + trailer;
};

/*decompress deflate buffer*/
int p9inflate(p9_simple_handle_t *handle, void *src, void *dst, int srclen,
	      int dstlen, int flag)
{
	nx_gzip_crb_cpb_t cmd;
	nx_gzip_crb_cpb_t *cmdp = &cmd; // TODO fixit
	int cc;
	int header = 0;
	int trailer = 0;
	char *srcp = src;
	/*advance the src ptr depending on the wrapper. gzip/zlib*/
	/*TODO correct way to do this would be scanning the header for possible
	 * optional fields*/
	if (flag == GZIP_WRAPPER) {
		header = 10;
		trailer = 8;
		put32(cmdp->cpb, in_crc, INIT_CRC);
		put32(cmdp->cpb, out_crc, INIT_CRC);
	} else if (flag == ZLIB_WRAPPER) {
		header = 2;
		trailer = 4;
		put32(cmdp->cpb, in_adler, INIT_ADLER);
		put32(cmdp->cpb, out_adler, INIT_ADLER);
	} else if (flag == NO_WRAPPER) {
		// NOOP
	} else {
		return -1;
	}

	/*clear crb and csb*/
	memset(&cmdp->crb, 0, sizeof(cmdp->crb));
	memset((void *)&cmdp->crb.csb, 0, sizeof(cmdp->crb.csb));

	/*set command type*/
	cmdp->crb.gzip_fc = 0;
	putnn(cmdp->crb, gzip_fc, GZIP_FC_DECOMPRESS);

	/*allocate a ddl on stack for a dummy buffer*/
	nx_dde_t ddl[3];

	/*set source/destination and sizes*/
	nx_init_csb(cmdp, src + header, dst, srclen - header - trailer, dstlen,
		    ddl, nx_overflow_buffer);

#ifdef P9DBG
	/*print dde*/
	nx_print_dde(ddl, "inflate test");
#endif
	/*touch all pages before submitting job*/
	nx_touch_pages(src, srclen, pagesize, 0);
	nx_touch_pages(dst, dstlen, pagesize, 1);
	/*run the job*/
	cc = nxu_run_job(cmdp, handle->vas_handle);

	if (!cc) {
		cc = getnn(cmdp->crb.csb, csb_cc);
	}

	if (cc != ERR_NX_OK) {
		fprintf(stderr, "nx inflate error: cc= %d\n", cc);
		return -1;
	}

	int uncompressed = get32(cmdp->crb.csb, tpbc);

	int checksum;
	int tail_checksum;
	if (flag == GZIP_WRAPPER) {
		checksum = cmdp->cpb.out_crc;
		tail_checksum = *(int *)(srcp + srclen - trailer);
		int tail_size = *(int *)(srcp + srclen - trailer + 4);
		if (checksum != tail_checksum || uncompressed != tail_size) {
			fprintf(stderr,
				"GZIP crc32 or size mismatch! [tail crc32 =%d computed=%d] [tail size =%d computed=%d]\n",
				tail_checksum, checksum, tail_size,
				uncompressed);
			return -1;
		}
	} else if (flag == ZLIB_WRAPPER) {
		checksum = cmdp->cpb.out_adler;
		if (checksum != tail_checksum) {
			fprintf(stderr,
				"ZLIB adler32 or size mismatch! [tail adler32 =%d computed=%d]\n",
				tail_checksum, checksum);
			return -1;
		}
	}
	return uncompressed;
};

/*close the compressor*/
int p9close(p9_simple_handle_t *handle)
{
	__sync_fetch_and_add(&(handle->open_count), -1);
	return 0;
};

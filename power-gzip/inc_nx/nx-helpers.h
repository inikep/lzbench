#include <sys/time.h>
#include <asm/byteorder.h>
#include <stdint.h>
#include <stdbool.h>
#include "crb.h"

#define cpu_to_be32		__cpu_to_be32
#define cpu_to_be64		__cpu_to_be64
#define be32_to_cpu		__be32_to_cpu
#define be64_to_cpu		__be64_to_cpu

/*
 * Several helpers/macros below were copied from the Linux kernel tree
 * (kernel.h, nx-842.h, nx-ftw.h, asm-compat.h etc)
 */

/* from kernel.h */
#define IS_ALIGNED(x, a)	(((x) & ((typeof(x))(a) - 1)) == 0)
#define __round_mask(x, y)	((__typeof__(x))((y)-1))
#define round_up(x, y)		((((x)-1) | __round_mask(x, y))+1)
#define round_down(x, y)	((x) & ~__round_mask(x, y))

#define min_t(t, x, y)	((x) < (y) ? (x) : (y))
/*
 * Get/Set bit fields. (from nx-842.h)
 */
#define GET_FIELD(m, v)         (((v) & (m)) >> MASK_LSH(m))
#define MASK_LSH(m)             (__builtin_ffsl(m) - 1)
#define SET_FIELD(m, v, val)    \
		(((v) & ~(m)) | ((((typeof(v))(val)) << MASK_LSH(m)) & (m)))

/* From asm-compat.h */
#define __stringify_in_c(...)	#__VA_ARGS__
#define stringify_in_c(...)	__stringify_in_c(__VA_ARGS__) " "

#define	pr_debug
#define	pr_debug_ratelimited	printf
#define	pr_err			printf
#define	pr_err_ratelimited	printf

#define WARN_ON_ONCE(x)		do {if (x) \
				printf("WARNING: %s:%d\n", __func__, __LINE__)\
				} while (0)

extern void dump_buffer(char *msg, char *buf, int len);
extern void *alloc_aligned_mem(int len, int align, char *msg);
extern void get_payload(char *buf, int len);
extern void time_add(struct timeval *in, int seconds, struct timeval *out);

extern bool time_after(struct timeval *a, struct timeval *b);
extern long time_delta(struct timeval *a, struct timeval *b);
extern void dump_dde(struct data_descriptor_entry *dde, char *msg);
extern void copy_paste_crb_data(struct coprocessor_request_block *crb);

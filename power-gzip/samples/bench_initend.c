/* Benchmark {de,in}flateInit and {de,in}flateEnd */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <zlib.h>
#include <sys/platform/ppc.h>

static inline int timebase_diff (uint64_t t1, uint64_t t2) {
	return t2 >= t1 ? t2 - t1 : (0xFFFFFFFFFFFFFFF-t1) + t2;
}

static inline double timebase_average_us(uint64_t sum, int count) {
        double avg = (double)sum / count;

        return avg * 1000000 / __ppc_get_timebase_freq();
}

int main () {
	uint64_t deflateInit_sum = 0, deflateEnd_sum = 0;
	uint64_t inflateInit_sum = 0, inflateEnd_sum = 0;
	uint64_t start;
	z_stream strm = {0};

	int const rounds = 100000;

	for (int i = 0; i < rounds; i++) {
		start = __ppc_get_timebase();
		(void) deflateInit(&strm, Z_DEFAULT_COMPRESSION);
		deflateInit_sum += timebase_diff(start, __ppc_get_timebase());

		start = __ppc_get_timebase();
		(void) deflateEnd(&strm);
		deflateEnd_sum += timebase_diff(start, __ppc_get_timebase());
	}

	memset(&strm, 0, sizeof(z_stream));

	for (int i = 0; i < rounds; i++) {
		start = __ppc_get_timebase();
		(void) inflateInit(&strm);
		inflateInit_sum += timebase_diff(start, __ppc_get_timebase());

		start = __ppc_get_timebase();
		(void) inflateEnd(&strm);
		inflateEnd_sum += timebase_diff(start, __ppc_get_timebase());
	}

	printf("Avg. deflateInit (us),Avg. deflateEnd (us),Avg. inflateInit (us),Avg. inflateEnd (us)\n");
	printf("%.3f,%.3f,%.3f,%.3f\n",
		timebase_average_us(deflateInit_sum, rounds),
		timebase_average_us(deflateEnd_sum, rounds),
		timebase_average_us(inflateInit_sum, rounds),
		timebase_average_us(inflateEnd_sum, rounds));

	return 0;
}

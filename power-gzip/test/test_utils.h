#ifndef TEST_UTILS_H_
#define TEST_UTILS_H_

#include <sys/time.h>

#define DATA_MAX_LEN (128*1024*1024) // 128M

#define MIN(x, y) (((x)<(y) && ((x)!=0))?(x):(y))
#define MAX(x, y) ((x)>(y)?(x):(y))

struct f_interval {
  /** Time at the start of initialization.  */
  struct timeval init_start;
  /** Time at the end of initialization.  */
  struct timeval init_end;
  /** Time at the start of the main part of the function.  */
  struct timeval start;
  /** Time at the end of the main part of the function.  */
  struct timeval end;
};

extern Byte ran_data[DATA_MAX_LEN];
extern void generate_random_data(int len);
extern Byte* generate_allocated_random_data(unsigned int len);
extern void generate_all_data(int len, char digit);
extern int compare_data(Byte* src, Byte* dest, int len);

void zcheck_internal(int retval, int expected, char* file, int line);
#define zcheck(val,exp) zcheck_internal((val),(exp),__FILE__,__LINE__)

int _test_nx_deflate(Byte* src, unsigned int src_len, Byte* compr,
		     unsigned int* compr_len, int step,
		     struct f_interval * time);
int _test_deflate(Byte* src, unsigned int src_len, Byte* compr,
		  unsigned int* compr_len, int step,
		  struct f_interval * time);
int _test_inflate(Byte* compr, unsigned int comprLen, Byte* uncompr,
		  unsigned int uncomprLen, Byte* src, unsigned int src_len,
		  int step,
		  struct f_interval * time);
int _test_nx_inflate(Byte* compr, unsigned int comprLen, Byte* uncompr,
		     unsigned int uncomprLen, Byte* src, unsigned int src_len,
		     int step, int flush,
		     struct f_interval * time);
#endif /* TEST_UTILS_H_ */

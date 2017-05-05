#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>
uint64_t NakaCompress(char* ret, char* src, unsigned int srcSize);
uint64_t NakaDecompress (char* ret, char* src, uint64_t srcSize);
#ifdef __cplusplus
}
#endif

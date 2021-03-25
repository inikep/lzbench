#ifndef __CPUID1_H__
#define __CPUID1_H__

//#include <cpuid.h>
#include <stdint.h>

enum cpuid_requests {
  CPUID_GETVENDORSTRING,
  CPUID_GETFEATURES,
  CPUID_GETTLB,
  CPUID_GETSERIAL,

  CPUID_EXTENDED = 0x80000000,
  CPUID_FEATURES,
  CPUID_BRANDSTRING,
  CPUID_BRANDSTRINGMORE,
  CPUID_BRANDSTRINGEND,
};

/** issue a single request to CPUID. Fits 'intel features', for instance
 *  note that even if only "eax" and "edx" are of interest, other registers
 *  will be modified by the operation, so we need to tell the compiler about it.
 */
inline void cpuid2(int code, uint32_t *a, uint32_t *d)
    {
    asm volatile("cpuid":"=a"(*a),"=d"(*d):"a"(code):"ecx","ebx");
    }

/** issue a complete request, storing general registers output as a string
 */
inline int cpuid_string(int code, uint32_t where[4])
    {
    asm volatile("cpuid":"=a"(*where),"=b"(*(where+1)),
               "=c"(*(where+2)),"=d"(*(where+3)):"a"(code));
    return (int)where[0];
    }

#endif // #ifndef __CPUID1_H__

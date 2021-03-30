#ifndef CPUID1_H
#define CPUID1_H

#ifndef __GNUC__
#define asm __asm__
#endif

#if (defined(__i386__) || defined(__x86_64__))
#include <cpuid.h>
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


inline int cpuid_string(int code, uint32_t where[4])
{
    //asm volatile("cpuid":"=a"(*where),"=b"(*(where+1)),
    //           "=c"(*(where+2)),"=d"(*(where+3)):"a"(code));
    __cpuid(code, (*where), *(where+1), *(where+2), *(where+3));
    return (int)where[0];
}
#endif /* #if (defined(__i386__) || defined(__x86_64__)) */
#endif /* #ifndef CPUID1_H */


#ifndef _CSC_PROFILE_H_
#define _CSC_PROFILE_H_
#include <csc_common.h>


#ifdef _HAVE_PROFILER_
void PEncodeLiteral(uint32_t c);
void PEncodeRepMatch(uint32_t len, uint32_t idx);
void PEncodeMatch(uint32_t len, uint32_t dist);
void PEncode1BMatch();
void PWriteLog();
#else
#define PEncodeLiteral(c)
#define PEncodeRepMatch(len, idx) 
#define PEncodeMatch(len, dist)
#define PEncode1BMatch()
#define PWriteLog()
#endif

#endif


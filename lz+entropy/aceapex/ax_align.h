#pragma once
#include <stdint.h>
#include <string.h>
/* Alignment-safe loads/stores (zstd MEM_read32 style). Compilers fold these
   memcpy calls into single mov/ldr where unaligned access is legal; on
   strict-alignment targets (32-bit ARM) they emit safe byte sequences
   instead of faulting LDM/LDRD. */
static inline uint16_t AX_read16(const void* p){uint16_t v;memcpy(&v,p,sizeof v);return v;}
static inline uint32_t AX_read32(const void* p){uint32_t v;memcpy(&v,p,sizeof v);return v;}
static inline uint64_t AX_read64(const void* p){uint64_t v;memcpy(&v,p,sizeof v);return v;}
static inline void AX_write16(void* p,uint16_t v){memcpy(p,&v,sizeof v);}
static inline void AX_write32(void* p,uint32_t v){memcpy(p,&v,sizeof v);}
static inline void AX_write64(void* p,uint64_t v){memcpy(p,&v,sizeof v);}

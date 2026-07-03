/**
 * Copyright (C) 2026, Advanced Micro Devices. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef DISPATCH_CPU_FEATURES_H
#define DISPATCH_CPU_FEATURES_H

#include <stdint.h>

#if !defined(_M_IX86) && !defined(_M_X64) && !defined(__i386__) && !defined(__x86_64__)
    #error "cpu_features only supports x86/x86_64 architectures."
#endif

/**
 * Compiler Detection for Cross-Platform Compatibility
 * 
 * Identifies the compiler at compile-time to enable appropriate
 * intrinsic and assembly implementations. Clang is checked before GCC
 * since Clang also defines __GNUC__.
 * 
 * Supported: MSVC, GCC, Clang
 */
#if defined(_MSC_VER)
    #define COMPILER_MSVC 1
    #define COMPILER_NAME "MSVC"
#elif defined(__clang__)
    #define COMPILER_CLANG 1
    #define COMPILER_NAME "Clang"
#elif defined(__GNUC__)
    #define COMPILER_GCC 1
    #define COMPILER_NAME "GCC"
#else
    #error "Unsupported compiler. Requires MSVC, GCC, or Clang."
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * CPUID Result Structure
 * 
 * Encapsulates the register values returned by CPUID instruction execution.
 * Contains the final state of EAX, EBX, ECX, and EDX registers.
 */
typedef struct {
    uint32_t eax;  /**< EAX register value */
    uint32_t ebx;  /**< EBX register value */
    uint32_t ecx;  /**< ECX register value */
    uint32_t edx;  /**< EDX register value */
} CpuidResult;

/**
 * Execute CPUID Instruction
 * 
 * @param leaf CPUID leaf index (e.g., 0, 1, 7)
 * @param subleaf CPUID subleaf index for extended queries
 * @return CPUID result containing EAX, EBX, ECX, EDX register values
 */
CpuidResult CPU_Cpuid(uint32_t leaf, uint32_t subleaf);

/**
 * Initialize CPU Feature Detection Caches
 * 
 * Populates internal CPUID caches for leaves 0, 1, and 7 to eliminate
 * redundant CPUID instruction executions. Must be called once before
 * any CPU feature detection. Safe to call multiple times (only executes once).
 */
void CPU_InitializeCpuidCache(void);

/**
 * Check CPUID Instruction Support
 * 
 * @return 1 if CPUID is supported (always true on x86-64), 0 otherwise
 */
int CPU_IsCpuidSupported(void);

/**
 * Get Maximum Supported CPUID Leaf
 * 
 * @return Highest valid CPUID leaf index for this processor
 */
uint32_t CPU_GetMaxCpuidLeaf(void);

/**
 * CPU Feature Detection Functions (CPU capability bits only)
 *
 * These APIs report CPUID-advertised hardware support. They do not imply
 * operating-system context management support (XSAVE/XCR0) for execution.
 */
int CPU_HasSSE2(void);
int CPU_HasAVX(void);
int CPU_HasAVX2(void);
int CPU_HasFMA(void);
int CPU_HasBMI1(void);
int CPU_HasBMI2(void);
int CPU_HasLZCNT(void);
int CPU_HasPCLMUL(void);
int CPU_HasVAES(void);
int CPU_HasGFNI(void);
int CPU_HasVPCLMULQDQ(void);
int CPU_HasAVXVNNI(void);
int CPU_HasAVX512F(void);
int CPU_HasAVX512DQ(void);
int CPU_HasAVX512IFMA(void);
int CPU_HasAVX512CD(void);
int CPU_HasAVX512BW(void);
int CPU_HasAVX512VL(void);
int CPU_HasAVX512VBMI(void);
int CPU_HasAVX512VBMI2(void);
int CPU_HasAVX512VNNI(void);
int CPU_HasAVX512BITALG(void);
int CPU_HasAVX512VPOPCNTDQ(void);
int CPU_HasAVX512VP2INTERSECT(void);
int CPU_HasAVX512BF16(void);
int CPU_HasAVX512FP16(void);

/**
 * CPU Usability Detection Functions (CPU+OS safe-to-execute)
 *
 * These APIs combine CPUID hardware capability with OS XSAVE/XCR0 state
 * support checks and are intended for runtime dispatch decisions.
 */
int CPU_CanUseAVX(void);
int CPU_CanUseAVX2(void);
int CPU_CanUseFMA(void);
int CPU_CanUseVAES(void);
int CPU_CanUseVPCLMULQDQ(void);
int CPU_CanUseAVXVNNI(void);
int CPU_CanUseAVX512F(void);
int CPU_CanUseAVX512DQ(void);
int CPU_CanUseAVX512IFMA(void);
int CPU_CanUseAVX512CD(void);
int CPU_CanUseAVX512BW(void);
int CPU_CanUseAVX512VL(void);
int CPU_CanUseAVX512VBMI(void);
int CPU_CanUseAVX512VBMI2(void);
int CPU_CanUseAVX512VNNI(void);
int CPU_CanUseAVX512BITALG(void);
int CPU_CanUseAVX512VPOPCNTDQ(void);
int CPU_CanUseAVX512VP2INTERSECT(void);
int CPU_CanUseAVX512BF16(void);
int CPU_CanUseAVX512FP16(void);

/**
 * Get CPU Vendor String
 * 
 * Retrieves the CPU vendor identification string from CPUID leaf 0.
 * Typical values include "GenuineIntel" or "AuthenticAMD".
 * 
 * @param vendor Output buffer (must accommodate 13 characters including null terminator)
 */
void CPU_GetVendorString(char vendor[13]);

/**
 * Get XCR0 Register Value
 * 
 * Reads the XCR0 control register to determine OS XSAVE state support.
 * XCR0 bits indicate which extended register states the OS will save/restore.
 * 
 * @return XCR0 register value, or 0 if XGETBV instruction is not supported
 */
uint64_t CPU_GetXCR0(void);

/**
 * Check XGETBV Instruction Support
 * 
 * Determines if the XGETBV instruction is available (OSXSAVE capability).
 * XGETBV is required to safely read XCR0.
 * 
 * @return 1 if XGETBV is supported, 0 otherwise
 */
int CPU_IsXGETBVSupported(void);

/**
 * Check OS Support for AVX State
 * 
 * Verifies that the operating system supports saving and restoring AVX/AVX2
 * extended register state (XMM and YMM registers) through XSAVE/XRSTOR.
 * Check XCR0 bits 1 and 2 (XMM and YMM state).
 * 
 * @return 1 if OS supports AVX state, 0 otherwise
 */
int CPU_HasAVXSupport(void);

/**
 * Check OS Support for AVX-512 State
 * 
 * Verifies that the operating system supports saving and restoring AVX-512
 * extended register state (Opmask, ZMM_Hi256, Hi16_ZMM registers) through XSAVE/XRSTOR.
 * Checks XCR0 bits 1, 2, 5, 6, and 7 (XMM, YMM, Opmask, and ZMM states).
 * 
 * @return 1 if OS supports AVX-512 state, 0 otherwise
 */
int CPU_HasAVX512Support(void);

#ifdef __cplusplus
}
#endif

#endif /* DISPATCH_CPU_FEATURES_H */

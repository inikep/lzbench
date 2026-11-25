// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

// Safe kernels for the entropy format: x86-64, SysV ABI, GAS syntax, ELF.
// Not assembled on its own: x86_64_avx2.S / x86_64_sse2.S define the KERNEL_* names (and
// KERNEL_AVX2) and include this file.
//
// The AVX2 build also uses BMI1, BMI2 and LZCNT (andn, shrx, lzcnt). The SSE2 build uses nothing
// past the x86-64 baseline.
//
// ---------------------------------------------------------------------------------------------
// uint64_t KERNEL_COMMANDS(io* p, uint64_t variant)       all commands of ONE block
//   variant bit 0: SUB block (literals are added to the bytes r0 back)
//           bit 1: check every match source against dst
//   returns 0  reject (a match source before dst, or out >= out_end with commands left)
//           1  every command ran; lit, out and r0 are written back
//
// io (8-byte fields):
//    0 cmd       first command byte        8 cmd_end
//   16 off       int32[], negated distances, one per N command, in order
//   24 lit       literal cursor           32 out
//   40 out_end   the block's end          48 dst
//   56 r0        negated repeat distance  64 T   u8[256][2] = {lit, lit + len}
//
// Requires of the caller:
//   - the number of cmd bytes >= FIRST_NEW equals the number of valid entries at off, and one
//     more entry is readable (every command loads one);
//   - [lit, lit + (out_end - out) + 16) is readable;
//   - out_end + 47 bytes are writable;
//   - -r0 >= 32 and out + r0 >= dst; every entry of off is <= -32;
//   - variant bit 1 is set unless every entry of off is >= dst - out;
//   - out_end >= 4 * 47 as an address, and addresses are < 2^63 (signed compares).
//
// A command reads 1 cmd byte, 16 literal bytes (lit += ll <= 15) and one offset, writes
// [out, out + 16) and [out + ll, out + ll + 32), and advances out by ll + ml in [1, 47]. So a
// group of 4 needs no per-command check when, at its head, 4 cmd bytes are left and
// out <= out_end - 3 * 47: each of its commands then starts at or before out_end. Past that,
// single commands run while out < out_end.
//
// Registers: rdi io, rax out, rdx lit, rcx cmd index (negative, up to 0), r8 cmd_end, r9 off,
//            r10 offset index, r11 r0, r12 tmp, r13 cmd byte, r14 dst, r15 T, rbx lit + len,
//            rbp lit, then the match destination, rsi out_end - 141.
//
// ---------------------------------------------------------------------------------------------
// void KERNEL_HUFFMAN(io* p)                  whole rounds of 60 symbols
//   Runs while out > out_stop and p[10] <= lim and p[11] <= lim; the caller has made sure that
//   this keeps every 8-byte load inside the payload and every store inside the output.
//
// io (8-byte fields): 0 out (descending), 8 out_stop, 16 table (u16[2048]), 24 lim,
//                     32 p[12], 128 bits[12]
//
// bits = (load64(p) | 1 << 63) >> used. A symbol: e = table[bits & 2047] (length | symbol << 8),
// bits >>= length, and ONE 2-byte store of e at out - 2 - index: the length byte falls on the
// next symbol's slot, which is written afterwards. After 5 symbols per bit string, lzcnt(bits)
// = used + consumed, so p += lz >> 3 and bits = (load64(p) | 1 << 63) >> (lz & 7).
//
// Registers: 12 bit buffers (rax rbx rdx rsi r8 r9 r10 r11 r12 r13 r15 rbp), rdi out, rcx tmp,
//            r14 = ~2047 (AVX2; the table is copied to the stack so that rsp addresses it) or
//            the table (SSE2).
// Stack:     0..88 p[12], 96 out_stop, 104 marker, 112 io, 120 caller's rsp, 128 lim,
//            192.. table copy (AVX2).
//
// ---------------------------------------------------------------------------------------------
// void KERNEL_DELTA(uint8_t* b, uint8_t* e, uint64_t stride)
//   x[i] += x[i - stride] for i in [b, e), ascending. Requires 1 <= stride <= 32 and the 32
//   bytes before b to be final output. Writes nothing outside [b, e).
//
// ---------------------------------------------------------------------------------------------
// int64_t KERNEL_MERGE(io* p)
//   Offset planes + width flags -> int32 negated distances -(v + 32). Returns a lower bound of
//   the smallest one written (exact up to 255, from the slack entries of the last group).
//
// io (8-byte fields):
//    0 o1    8 o2l   16 o2h   24 o3l   32 o3m   40 o3h   48 f1   56 f2
//   64 off_cnt       72 wide_cnt (2- and 3-byte offsets)
//   80 tmp[3]        AVX2: wide_cnt + 8 bytes each; SSE2: 2 * wide_cnt + 32 bytes at tmp[0]
//  104 out           off_cnt + 16 entries
//  112 E             AVX2: u8[256][8] expand masks, then u8[256] popcounts
//  120 o3_cnt        3-byte offsets
//
// Requires: f1 has wide_cnt set bits, f2 has o3_cnt, none past their last flag; 16 readable
// bytes after every plane.
//
// AVX2: a flag byte at a time, the planes are spread to their lanes with pshufb.
// SSE2: in place in `out`, back to front. The 3-byte offsets are put at its end, the 2-byte ones
// are woven in before them, then the 1-byte ones before those: each step writes entry j after
// reading entries >= j only. A flag is shifted into CF and selects with cmov.
// Both: a block without 3-byte offsets skips their pass, one with 1-byte offsets only is
// just widened.

#ifndef FIRST_NEW
#define FIRST_NEW 131
#endif

#if defined(KERNEL_AVX2)
#define VZU vzeroupper
#define MOVU vmovdqu
#define MOVA vmovdqa
#define PADDB(a, b) vpaddb a, b, b
#define POR(a, b) vpor a, b, b
#define PSLLDQ(n, a) vpslldq $(n), a, a
#define PSRLDQ(n, a) vpsrldq $(n), a, a
#else
#define VZU
#define MOVU movdqu
#define MOVA movdqa
#define PADDB(a, b) paddb a, b
#define POR(a, b) por a, b
#define PSLLDQ(n, a) pslldq $(n), a
#define PSRLDQ(n, a) psrldq $(n), a
#endif

// ---------------------------------------------------------------------------------------------
// commands

// one command at cmd index rcx + j
#define SRC(base) (base,%r11)
#define SRC16(base) 16(base,%r11)
.macro CMD j, sub, chk
    movzbl  \j(%r8,%rcx), %r13d
    movzbl  (%r15,%r13,2), %ebp             // lit
    movzbl  1(%r15,%r13,2), %ebx            // lit + len
    MOVU    (%rdx), %xmm0                   // always 16 literal bytes
  .if \sub
#if defined(KERNEL_AVX2)
    vpaddb  SRC(%rax), %xmm0, %xmm0
#else
    movdqu  SRC(%rax), %xmm3
    paddb   %xmm3, %xmm0
#endif
  .endif
    MOVU    %xmm0, (%rax)
    add     %rbp, %rdx
    cmp     $FIRST_NEW, %r13d               // CF: repeat
    movslq  (%r9,%r10,4), %r12              // always load an offset
    cmovae  %r12, %r11                      // new distance, or keep the repeat one
    sbb     $-1, %r10                       // consume the offset if new
    add     %rax, %rbp                      // match destination
  .if \chk
    lea     SRC(%rbp), %r12
    cmp     %r14, %r12
    jl      .Lcmd_fail                      // match source < dst (signed: no wrap)
  .endif
#if defined(KERNEL_AVX2)
    vmovdqu SRC(%rbp), %ymm1
    vmovdqu %ymm1, (%rbp)
#else
    movdqu  SRC(%rbp), %xmm1
    movdqu  SRC16(%rbp), %xmm2
    movdqu  %xmm1, (%rbp)
    movdqu  %xmm2, 16(%rbp)
#endif
    add     %rbx, %rax
.endm

.macro CMD_LOOP sub, chk
    jmp     2f
    .p2align 5
1:
    CMD     0, \sub, \chk
    CMD     1, \sub, \chk
    CMD     2, \sub, \chk
    CMD     3, \sub, \chk
    add     $4, %rcx
2:
    cmp     $-4, %rcx
    jg      3f                              // fewer than 4 cmd bytes left
    cmp     %rsi, %rax
    jbe     1b                              // out <= out_end - 141
3:
    test    %rcx, %rcx
    jz      .Lcmd_done
    cmp     40(%rdi), %rax
    jae     .Lcmd_fail                      // out >= out_end with commands left
    CMD     0, \sub, \chk
    add     $1, %rcx
    jmp     3b
.endm

    .text
    .globl  KERNEL_COMMANDS
    .hidden KERNEL_COMMANDS
    .type   KERNEL_COMMANDS, @function
    .p2align 6
KERNEL_COMMANDS:
    push    %rbx
    push    %rbp
    push    %r12
    push    %r13
    push    %r14
    push    %r15
    mov     %esi, %r12d                     // variant
    mov     0(%rdi), %rcx
    mov     8(%rdi), %r8
    mov     16(%rdi), %r9
    mov     24(%rdi), %rdx
    mov     32(%rdi), %rax
    mov     40(%rdi), %rsi
    sub     $141, %rsi                      // out_end - 3 * 47
    mov     48(%rdi), %r14
    mov     56(%rdi), %r11
    mov     64(%rdi), %r15
    xor     %r10d, %r10d
    sub     %r8, %rcx                       // -(number of commands)
    test    $2, %r12d
    jnz     .Lcmd_chk
    test    $1, %r12d
    jnz     .Lcmd_sub
    CMD_LOOP 0, 0
.Lcmd_sub:
    CMD_LOOP 1, 0
.Lcmd_chk:
    test    $1, %r12d
    jnz     .Lcmd_chk_sub
    CMD_LOOP 0, 1
.Lcmd_chk_sub:
    CMD_LOOP 1, 1

.Lcmd_done:
    mov     %rdx, 24(%rdi)
    mov     %rax, 32(%rdi)
    mov     %r11, 56(%rdi)
    mov     $1, %eax
    jmp     .Lcmd_out
.Lcmd_fail:
    xor     %eax, %eax
.Lcmd_out:
    VZU
    pop     %r15
    pop     %r14
    pop     %r13
    pop     %r12
    pop     %rbp
    pop     %rbx
    ret
    .size   KERNEL_COMMANDS, .-KERNEL_COMMANDS

// ---------------------------------------------------------------------------------------------
// Huffman

#if defined(KERNEL_AVX2)
#define HUF_TABLE 192
#define HUF_FRAME (HUF_TABLE + 4096 + 64)
#else
#define HUF_FRAME 192
#endif

// symbol number idx of the round, from the bit buffer r
.macro HUF_SYM r, idx
#if defined(KERNEL_AVX2)
    andn    \r, %r14, %rcx
    movzwl  HUF_TABLE(%rsp,%rcx,2), %ecx
    shrx    %rcx, \r, \r
#else
    mov     \r, %rcx
    and     $2047, %ecx
    movzwl  (%r14,%rcx,2), %ecx
    shr     %cl, \r
#endif
    movw    %cx, -(\idx + 2)(%rdi)
.endm

// refill the bit buffer r, whose pointer is at slot(%rsp)
.macro HUF_REFILL r, slot
#if defined(KERNEL_AVX2)
    lzcnt   \r, %rcx
#else
    xor     %ecx, %ecx                      // bsr merges into its destination
    bsr     \r, %rcx                        // never 0: the marker
    xor     $63, %ecx
#endif
    mov     %rcx, \r
    shr     $3, \r
    add     \slot(%rsp), \r
    and     $7, %ecx
    mov     \r, \slot(%rsp)
    mov     (\r), \r
    or      104(%rsp), \r
#if defined(KERNEL_AVX2)
    shrx    %rcx, \r, \r
#else
    shr     %cl, \r
#endif
.endm

    .globl  KERNEL_HUFFMAN
    .hidden KERNEL_HUFFMAN
    .type   KERNEL_HUFFMAN, @function
    .p2align 6
KERNEL_HUFFMAN:
    push    %rbx
    push    %rbp
    push    %r12
    push    %r13
    push    %r14
    push    %r15
    mov     %rsp, %rcx
    sub     $HUF_FRAME, %rsp
    and     $-64, %rsp
    mov     %rcx, 120(%rsp)
    mov     %rdi, 112(%rsp)
#if defined(KERNEL_AVX2)
    mov     16(%rdi), %rsi
    lea     HUF_TABLE(%rsp), %rdi
    mov     $4096, %ecx
    rep movsb
    mov     112(%rsp), %rdi
    mov     $-2048, %r14
#else
    mov     16(%rdi), %r14
#endif
  .irp k, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    mov     (32 + 8 * \k)(%rdi), %rcx
    mov     %rcx, (8 * \k)(%rsp)
  .endr
    mov     8(%rdi), %rcx
    mov     %rcx, 96(%rsp)
    mov     24(%rdi), %rcx
    mov     %rcx, 128(%rsp)
    mov     $0x8000000000000000, %rcx
    mov     %rcx, 104(%rsp)
    mov     %rdi, %rcx
    mov     128(%rcx), %rax
    mov     136(%rcx), %rbx
    mov     144(%rcx), %rdx
    mov     152(%rcx), %rsi
    mov     160(%rcx), %r8
    mov     168(%rcx), %r9
    mov     176(%rcx), %r10
    mov     184(%rcx), %r11
    mov     192(%rcx), %r12
    mov     200(%rcx), %r13
    mov     208(%rcx), %r15
    mov     216(%rcx), %rbp
    mov     0(%rcx), %rdi
    cmp     96(%rsp), %rdi
    jbe     2f
    .p2align 6
1:
  .irp row, 0, 1, 2, 3, 4
    HUF_SYM %rax, (12 * \row + 0)
    HUF_SYM %rbx, (12 * \row + 1)
    HUF_SYM %rdx, (12 * \row + 2)
    HUF_SYM %rsi, (12 * \row + 3)
    HUF_SYM %r8, (12 * \row + 4)
    HUF_SYM %r9, (12 * \row + 5)
    HUF_SYM %r10, (12 * \row + 6)
    HUF_SYM %r11, (12 * \row + 7)
    HUF_SYM %r12, (12 * \row + 8)
    HUF_SYM %r13, (12 * \row + 9)
    HUF_SYM %r15, (12 * \row + 10)
    HUF_SYM %rbp, (12 * \row + 11)
  .endr
    HUF_REFILL %rax, 0
    HUF_REFILL %rbx, 8
    HUF_REFILL %rdx, 16
    HUF_REFILL %rsi, 24
    HUF_REFILL %r8, 32
    HUF_REFILL %r9, 40
    HUF_REFILL %r10, 48
    HUF_REFILL %r11, 56
    HUF_REFILL %r12, 64
    HUF_REFILL %r13, 72
    HUF_REFILL %r15, 80
    HUF_REFILL %rbp, 88
    sub     $60, %rdi
    mov     128(%rsp), %rcx
    cmp     80(%rsp), %rcx
    jb      2f                              // p[10] > lim
    cmp     88(%rsp), %rcx
    jb      2f                              // p[11] > lim
    cmp     96(%rsp), %rdi
    ja      1b
2:
    mov     112(%rsp), %rcx
    mov     %rdi, 0(%rcx)
    mov     %rax, 128(%rcx)
    mov     %rbx, 136(%rcx)
    mov     %rdx, 144(%rcx)
    mov     %rsi, 152(%rcx)
    mov     %r8, 160(%rcx)
    mov     %r9, 168(%rcx)
    mov     %r10, 176(%rcx)
    mov     %r11, 184(%rcx)
    mov     %r12, 192(%rcx)
    mov     %r13, 200(%rcx)
    mov     %r15, 208(%rcx)
    mov     %rbp, 216(%rcx)
  .irp k, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    mov     (8 * \k)(%rsp), %rax
    mov     %rax, (32 + 8 * \k)(%rcx)
  .endr
    mov     120(%rsp), %rsp
    pop     %r15
    pop     %r14
    pop     %r13
    pop     %r12
    pop     %rbp
    pop     %rbx
    ret
    .size   KERNEL_HUFFMAN, .-KERNEL_HUFFMAN

// ---------------------------------------------------------------------------------------------
// DELTA restore
//
// Registers: rdi cursor, rsi e, r8 e - 16, xmm0 the 16 bytes at the cursor, xmm1 the 16 restored
// bytes before it, xmm2 the 16 before those, xmm3 / xmm4 tmp.
// One loop per stride (the byte shifts take immediates only):
//   stride < 16   prefix-sum the 16 bytes at that stride with shifts (off the carried chain),
//                 then add the last `stride` restored bytes, repeated across the vector
//   16..32        add the 16 restored bytes that start `stride` back

.macro DELTA_STRIDE s
    .p2align 4
.Ldelta_\s:
    cmp     %r8, %rdi
    ja      .Ldelta_tail
1:
    MOVU    (%rdi), %xmm0
  .if \s < 16
    MOVA    %xmm0, %xmm3
    PSLLDQ(\s, %xmm3)
    PADDB(%xmm3, %xmm0)
   .if 2 * \s < 16
    MOVA    %xmm0, %xmm3
    PSLLDQ(2 * \s, %xmm3)
    PADDB(%xmm3, %xmm0)
   .endif
   .if 4 * \s < 16
    MOVA    %xmm0, %xmm3
    PSLLDQ(4 * \s, %xmm3)
    PADDB(%xmm3, %xmm0)
   .endif
   .if 8 * \s < 16
    MOVA    %xmm0, %xmm3
    PSLLDQ(8 * \s, %xmm3)
    PADDB(%xmm3, %xmm0)
   .endif
#if defined(KERNEL_AVX2)
    vpshufb .Ldelta_rep_\s(%rip), %xmm1, %xmm3
#else
    MOVA    %xmm1, %xmm3
    PSRLDQ(16 - \s, %xmm3)
    MOVA    %xmm3, %xmm4
    PSLLDQ(\s, %xmm4)
    POR(%xmm4, %xmm3)
   .if 2 * \s < 16
    MOVA    %xmm3, %xmm4
    PSLLDQ(2 * \s, %xmm4)
    POR(%xmm4, %xmm3)
   .endif
   .if 4 * \s < 16
    MOVA    %xmm3, %xmm4
    PSLLDQ(4 * \s, %xmm4)
    POR(%xmm4, %xmm3)
   .endif
   .if 8 * \s < 16
    MOVA    %xmm3, %xmm4
    PSLLDQ(8 * \s, %xmm4)
    POR(%xmm4, %xmm3)
   .endif
#endif
    PADDB(%xmm3, %xmm0)
  .elseif \s == 16
    PADDB(%xmm1, %xmm0)
  .elseif \s < 32
#if defined(KERNEL_AVX2)
    vpalignr $(32 - \s), %xmm2, %xmm1, %xmm3
#else
    MOVA    %xmm2, %xmm3
    PSRLDQ(32 - \s, %xmm3)
    MOVA    %xmm1, %xmm4
    PSLLDQ(\s - 16, %xmm4)
    POR(%xmm4, %xmm3)
#endif
    PADDB(%xmm3, %xmm0)
  .else
    PADDB(%xmm2, %xmm0)
  .endif
    MOVU    %xmm0, (%rdi)
  .if \s > 16
    MOVA    %xmm1, %xmm2
  .endif
    MOVA    %xmm0, %xmm1
    add     $16, %rdi
    cmp     %r8, %rdi
    jbe     1b
    jmp     .Ldelta_tail
.endm

    .globl  KERNEL_DELTA
    .hidden KERNEL_DELTA
    .type   KERNEL_DELTA, @function
    .p2align 6
KERNEL_DELTA:
    lea     .Ldelta_table(%rip), %rax
    movslq  -4(%rax,%rdx,4), %rcx
    add     %rax, %rcx
    MOVU    -16(%rdi), %xmm1
    MOVU    -32(%rdi), %xmm2
    lea     -16(%rsi), %r8
    jmp     *%rcx

  .irp s, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    DELTA_STRIDE \s
  .endr
  .irp s, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
    DELTA_STRIDE \s
  .endr

    // fewer than 16 bytes left
.Ldelta_tail:
    cmp     %rsi, %rdi
    jae     2f
    neg     %rdx
1:
    movzbl  (%rdi,%rdx), %eax
    add     %al, (%rdi)
    add     $1, %rdi
    cmp     %rsi, %rdi
    jb      1b
2:
    ret
    .size   KERNEL_DELTA, .-KERNEL_DELTA

    .section .rodata
    .p2align 4
.Ldelta_table:
  .irp s, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    .long   .Ldelta_\s - .Ldelta_table
  .endr
  .irp s, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
    .long   .Ldelta_\s - .Ldelta_table
  .endr
#if defined(KERNEL_AVX2)
    // lane j = the restored byte at 16 - s + j % s of the previous vector
    .p2align 4
  .irp s, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
.Ldelta_rep_\s:
   .irp j, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    .byte   16 - \s + (\j % \s)
   .endr
  .endr
#endif
    .text

// ---------------------------------------------------------------------------------------------
// offset merge

#if defined(KERNEL_AVX2)
// Registers: rdi io, r12 E, rax flag index (8 per step), rbx flag byte, rcx its popcount,
//            first pass:  rsi o2l, rdx o2h, r8 o3l, r9 o3m, r10 o3h, r11 2-byte cursor,
//                         r13 3-byte cursor, r14 f2, r15 wide_cnt
//            second pass: rsi o1, r11 1-byte cursor, r13 wide cursor, r14 f1, r15 off_cnt,
//                         rdx out, ymm5 = -32, ymm6 = running minimum
// 8 offsets a step: the 1-byte plane (rsi) and `planes - 1` byte planes of the wide offsets
// (rbp, r8, r9) -> int32 at rdx
.macro MERGE_WIDEN planes
    .p2align 4
3:
    mov     %rax, %rcx
    shr     $3, %rcx
    movzbl  (%r14,%rcx), %ebx
    movzbl  2048(%r12,%rbx), %ecx
    vmovq   (%r12,%rbx,8), %xmm1            // lanes of the wide offsets
    xor     $255, %ebx
    vmovq   (%r12,%rbx,8), %xmm0            // lanes of the 1-byte offsets
    vmovq   (%rsi,%r11), %xmm2
    vpshufb %xmm0, %xmm2, %xmm2
    vmovq   (%rbp,%r13), %xmm3
    vpshufb %xmm1, %xmm3, %xmm3
    vpor    %xmm3, %xmm2, %xmm2
    vmovq   (%r8,%r13), %xmm3
    vpshufb %xmm1, %xmm3, %xmm3
  .if \planes == 3
    vmovq   (%r9,%r13), %xmm4
    vpshufb %xmm1, %xmm4, %xmm4
  .endif
    vpmovzxbd %xmm2, %ymm2
    vpmovzxbd %xmm3, %ymm3
    vpslld  $8, %ymm3, %ymm3
    vpor    %ymm3, %ymm2, %ymm2
  .if \planes == 3
    vpmovzxbd %xmm4, %ymm4
    vpslld  $16, %ymm4, %ymm4
    vpor    %ymm4, %ymm2, %ymm2
  .endif
    vpsubd  %ymm2, %ymm5, %ymm2             // -32 - v
    vmovdqu %ymm2, (%rdx,%rax,4)
    vpminsd %ymm2, %ymm6, %ymm6
    add     %rcx, %r13
    add     $8, %r11
    sub     %rcx, %r11
    add     $8, %rax
    cmp     %r15, %rax
    jb      3b
    jmp     4f
.endm

    .globl  KERNEL_MERGE
    .hidden KERNEL_MERGE
    .type   KERNEL_MERGE, @function
    .p2align 6
KERNEL_MERGE:
    push    %rbx
    push    %rbp
    push    %r12
    push    %r13
    push    %r14
    push    %r15
    mov     112(%rdi), %r12
    vpcmpeqd %ymm5, %ymm5, %ymm5
    vpslld  $5, %ymm5, %ymm5                // -32
    vmovdqa %ymm5, %ymm6

    // without 3-byte offsets the wide planes are the 2-byte ones as they lie
    cmpq    $0, 120(%rdi)
    jne     .Lmerge_planes
    mov     8(%rdi), %rbp
    mov     16(%rdi), %r8
    xor     %r9d, %r9d
    jmp     .Lmerge_widen
.Lmerge_planes:
    // 2- and 3-byte offsets -> three byte planes in flag order
    mov     8(%rdi), %rsi
    mov     16(%rdi), %rdx
    mov     24(%rdi), %r8
    mov     32(%rdi), %r9
    mov     40(%rdi), %r10
    mov     56(%rdi), %r14
    mov     72(%rdi), %r15
    mov     80(%rdi), %rbp
    xor     %eax, %eax
    xor     %r11d, %r11d
    xor     %r13d, %r13d
    test    %r15, %r15
    jz      2f
    .p2align 4
1:
    mov     %rax, %rcx
    shr     $3, %rcx
    movzbl  (%r14,%rcx), %ebx
    movzbl  2048(%r12,%rbx), %ecx
    vmovq   (%r12,%rbx,8), %xmm1            // lanes of the 3-byte offsets
    xor     $255, %ebx
    vmovq   (%r12,%rbx,8), %xmm0            // lanes of the 2-byte offsets
    vmovq   (%rsi,%r11), %xmm2
    vpshufb %xmm0, %xmm2, %xmm2
    vmovq   (%r8,%r13), %xmm3
    vpshufb %xmm1, %xmm3, %xmm3
    vpor    %xmm3, %xmm2, %xmm2
    vmovq   %xmm2, (%rbp,%rax)
    vmovq   (%rdx,%r11), %xmm2
    vpshufb %xmm0, %xmm2, %xmm2
    vmovq   (%r9,%r13), %xmm3
    vpshufb %xmm1, %xmm3, %xmm3
    vpor    %xmm3, %xmm2, %xmm2
    mov     88(%rdi), %rbx
    vmovq   %xmm2, (%rbx,%rax)
    vmovq   (%r10,%r13), %xmm3
    vpshufb %xmm1, %xmm3, %xmm3
    mov     96(%rdi), %rbx
    vmovq   %xmm3, (%rbx,%rax)
    add     %rcx, %r13
    add     $8, %r11
    sub     %rcx, %r11
    add     $8, %rax
    cmp     %r15, %rax
    jb      1b
2:

    // 1-byte offsets and the planes above -> int32
    mov     88(%rdi), %r8
    mov     96(%rdi), %r9
.Lmerge_widen:
    mov     0(%rdi), %rsi
    mov     48(%rdi), %r14
    mov     64(%rdi), %r15
    mov     104(%rdi), %rdx
    xor     %eax, %eax
    xor     %r11d, %r11d
    xor     %r13d, %r13d
    test    %r15, %r15
    jz      4f
    test    %r9, %r9
    jz      .Lmerge_widen_2
    MERGE_WIDEN 3
.Lmerge_widen_2:
    cmpq    $0, 72(%rdi)
    je      .Lmerge_widen_1
    MERGE_WIDEN 2
    // 1-byte offsets only
    .p2align 4
.Lmerge_widen_1:
    vpmovzxbd (%rsi,%rax), %ymm2
    vpsubd  %ymm2, %ymm5, %ymm2
    vmovdqu %ymm2, (%rdx,%rax,4)
    vpminsd %ymm2, %ymm6, %ymm6
    add     $8, %rax
    cmp     %r15, %rax
    jb      .Lmerge_widen_1
4:
    vextracti128 $1, %ymm6, %xmm0
    vpminsd %xmm0, %xmm6, %xmm6
    vpshufd $0x4e, %xmm6, %xmm0
    vpminsd %xmm0, %xmm6, %xmm6
    vpshufd $0xb1, %xmm6, %xmm0
    vpminsd %xmm0, %xmm6, %xmm6
    vmovd   %xmm6, %eax
    movslq  %eax, %rax
    vzeroupper
    pop     %r15
    pop     %r14
    pop     %r13
    pop     %r12
    pop     %rbp
    pop     %rbx
    ret
    .size   KERNEL_MERGE, .-KERNEL_MERGE
#else
// one weave: out[j] = flag j ? wide[taken++] : narrow[j - taken]
// Registers: r9 flags, r10 flag bytes left, rsi narrow (8 entries per step), r11 -taken,
//            r8 wide, r13 taken, rdx out, ebx flag byte, eax entry, r12 -flag
.macro WEAVE load, size
    xor     %r13d, %r13d
    xor     %r11d, %r11d
    test    %r10, %r10
    jz      2f
    .p2align 4
1:
    movzbl  (%r9), %ebx
    add     $1, %r9
  .irp k, 0, 1, 2, 3, 4, 5, 6, 7
    shr     $1, %ebx                        // CF = flag
    \load   (\k * \size)(%rsi,%r11,\size), %eax
    cmovc   (%r8,%r13,4), %eax
    sbb     %r12, %r12
    mov     %eax, (4 * \k)(%rdx)
    sub     %r12, %r13
    add     %r12, %r11
  .endr
    add     $(8 * \size), %rsi
    add     $32, %rdx
    sub     $1, %r10
    jnz     1b
2:
.endm

// 16 entries a step: `planes` byte planes (rsi, r8, r10) -> u32 at r9, rcx entries. xmm7 = 0.
.macro UNPACK planes
    xor     %eax, %eax
    test    %rcx, %rcx
    jz      2f
    .p2align 4
1:
    movdqu  (%rsi,%rax), %xmm0
  .if \planes >= 2
    movdqu  (%r8,%rax), %xmm1
  .else
    pxor    %xmm1, %xmm1
  .endif
  .if \planes == 3
    movdqu  (%r10,%rax), %xmm2
  .else
    pxor    %xmm2, %xmm2
  .endif
    movdqa  %xmm0, %xmm3
    punpcklbw %xmm1, %xmm0
    punpckhbw %xmm1, %xmm3
    movdqa  %xmm2, %xmm4
    punpcklbw %xmm7, %xmm2
    punpckhbw %xmm7, %xmm4
    movdqa  %xmm0, %xmm5
    punpcklwd %xmm2, %xmm0
    punpckhwd %xmm2, %xmm5
    movdqa  %xmm3, %xmm6
    punpcklwd %xmm4, %xmm3
    punpckhwd %xmm4, %xmm6
    movdqu  %xmm0, (%r9,%rax,4)
    movdqu  %xmm5, 16(%r9,%rax,4)
    movdqu  %xmm3, 32(%r9,%rax,4)
    movdqu  %xmm6, 48(%r9,%rax,4)
    add     $16, %rax
    cmp     %rcx, %rax
    jb      1b
2:
.endm

    .globl  KERNEL_MERGE
    .hidden KERNEL_MERGE
    .type   KERNEL_MERGE, @function
    .p2align 6
KERNEL_MERGE:
    push    %rbx
    push    %r12
    push    %r13

    pxor    %xmm7, %xmm7
    mov     64(%rdi), %rax
    mov     120(%rdi), %rcx
    mov     104(%rdi), %rdx
    test    %rcx, %rcx
    jnz     .Lmerge_three
    // no 3-byte offsets: the wide ones are the 2-byte ones, in order
    mov     72(%rdi), %rcx
    test    %rcx, %rcx
    jz      .Lmerge_narrow
    sub     %rcx, %rax
    lea     (%rdx,%rax,4), %r9
    mov     8(%rdi), %rsi
    mov     16(%rdi), %r8
    UNPACK  2
    mov     %r9, %rcx
    jmp     .Lmerge_all
    // 1-byte offsets only
.Lmerge_narrow:
    mov     %rdx, %r9
    mov     0(%rdi), %rsi
    mov     %rax, %rcx
    UNPACK  1
    jmp     .Lmerge_negate

    // 3-byte offsets -> the end of out
.Lmerge_three:
    sub     %rcx, %rax
    lea     (%rdx,%rax,4), %r9
    mov     24(%rdi), %rsi
    mov     32(%rdi), %r8
    mov     40(%rdi), %r10
    UNPACK  3

    // 2-byte offsets -> u16 at tmp
    mov     72(%rdi), %rcx
    sub     120(%rdi), %rcx
    mov     8(%rdi), %rsi
    mov     16(%rdi), %r8
    mov     80(%rdi), %r10
    xor     %eax, %eax
    test    %rcx, %rcx
    jz      2f
    .p2align 4
1:
    movdqu  (%rsi,%rax), %xmm0
    movdqu  (%r8,%rax), %xmm1
    movdqa  %xmm0, %xmm3
    punpcklbw %xmm1, %xmm0
    punpckhbw %xmm1, %xmm3
    movdqu  %xmm0, (%r10,%rax,2)
    movdqu  %xmm3, 16(%r10,%rax,2)
    add     $16, %rax
    cmp     %rcx, %rax
    jb      1b
2:

    // wide offsets: 2-byte ones woven with the 3-byte ones
    mov     %r9, %r8
    mov     56(%rdi), %r9
    mov     %r10, %rsi
    mov     72(%rdi), %r10
    add     $7, %r10
    shr     $3, %r10
    mov     64(%rdi), %rax
    sub     72(%rdi), %rax
    mov     104(%rdi), %rdx
    lea     (%rdx,%rax,4), %rdx
    mov     %rdx, %rcx                      // the wide offsets
    WEAVE   movzwl, 2

    // all offsets: 1-byte ones woven with the wide ones
.Lmerge_all:
    mov     %rcx, %r8
    mov     48(%rdi), %r9
    mov     0(%rdi), %rsi
    mov     64(%rdi), %r10
    add     $7, %r10
    shr     $3, %r10
    mov     104(%rdi), %rdx
    WEAVE   movzbl, 1

    // v -> -(v + 32), and the largest v
.Lmerge_negate:
    mov     104(%rdi), %rdx
    mov     64(%rdi), %rcx
    pcmpeqd %xmm5, %xmm5
    pslld   $5, %xmm5                       // -32
    pxor    %xmm6, %xmm6
    xor     %eax, %eax
    test    %rcx, %rcx
    jz      2f
    .p2align 4
1:
    movdqu  (%rdx,%rax,4), %xmm0
    movdqa  %xmm0, %xmm1
    pcmpgtd %xmm6, %xmm1
    movdqa  %xmm0, %xmm2
    pand    %xmm1, %xmm2
    pandn   %xmm6, %xmm1
    por     %xmm2, %xmm1
    movdqa  %xmm1, %xmm6
    movdqa  %xmm5, %xmm3
    psubd   %xmm0, %xmm3
    movdqu  %xmm3, (%rdx,%rax,4)
    add     $4, %rax
    cmp     %rcx, %rax
    jb      1b
2:
    movd    %xmm6, %eax
  .irp k, 1, 2, 3
    pshufd  $\k, %xmm6, %xmm0
    movd    %xmm0, %ecx
    cmp     %ecx, %eax
    cmovb   %ecx, %eax
  .endr
    add     $32, %rax
    neg     %rax
    pop     %r13
    pop     %r12
    pop     %rbx
    ret
    .size   KERNEL_MERGE, .-KERNEL_MERGE
#endif

    // keep the stack non-executable
    .section .note.GNU-stack,"",@progbits

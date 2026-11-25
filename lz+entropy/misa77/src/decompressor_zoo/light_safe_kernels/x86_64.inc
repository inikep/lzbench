// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

// Safe sequence loop for the light format: x86-64, SysV ABI, GAS syntax, ELF.
// Not assembled on its own: x86_64_avx2.S / x86_64_sse2.S define KERNEL_NAME (and KERNEL_AVX2)
// and include this file.
//
// uint8_t* KERNEL_NAME(const uint8_t* src, uint8_t* dst, const uint8_t* literals,
//                      uint8_t* out_limit)
//   src        start of the stream (the first sequence is at src + 16)
//   dst        start of the output buffer
//   literals   one past the literal region (start of the raw suffix)
//   out_limit  dst + (n - suffix_cnt), >= dst
//   returns the final `out` once control >= literals, or null on a rejected stream.
//   The caller must then require out == out_limit.
//
// Derived on entry:
//   prefix_end  dst + min(max_dis, out_limit - dst)
//   red         out_limit - RED_SLACK (dst if the output is no longer than that)
//   plim        min(prefix_end, red); dst when dst < 2^17 (out - dis must not wrap)
//
// A group is 4 sequences decoded with one output-bound compare. Inside a group a sequence
// advances `out` by <= 66 and writes <= 64 bytes past its start, so a group that starts below
// `red` writes below out_limit + 32: inside the raw suffix's 32 bytes. RED_SLACK = 66 * 4 - 35.
//
// Paths, chosen by .Ldispatch on entry and after every group / long literal run:
//   .Lpf    out < plim: groups; per sequence the termination test and a distance test
//   .Lfast  prefix_end <= out < red: groups; per sequence only the termination test
//   .Lchk   everything else: one fully checked sequence
//   .Lext   extended literal run with a first extra byte > 25 (or from .Lchk): fully checked
// A run of <= 32 literals (one extra byte <= 25) stays inside its group.
//
// Registers: r8 control, r9 literals, rdi out, r10 red, r11 out_limit, rsi prefix_end,
//            r15 plim, r12 dst, r13 src, rax match code, rcx dis, rdx lit, r14 temp.

#define MAX_DIS   65568
#define RED_SLACK 229

#if defined(KERNEL_AVX2)
#define LD16(m)        vmovdqu m, %xmm0
#define ST16(m)        vmovdqu %xmm0, m
#define CP32B(b, i, d) vmovdqu (b,i), %ymm1 ; vmovdqu %ymm1, (d,i)
#define VZU            vzeroupper
#else
#define LD16(m)        movdqu m, %xmm0
#define ST16(m)        movdqu %xmm0, m
#define CP32B(b, i, d) movdqu (b,i), %xmm1 ; movdqu 16(b,i), %xmm2 ; movdqu %xmm1, (d,i) ; movdqu %xmm2, 16(d,i)
#define VZU
#endif

// 32-byte copy between two plain memory operands
.macro CP32 src, dst, dst16, src16
#if defined(KERNEL_AVX2)
    vmovdqu \src, %ymm1
    vmovdqu %ymm1, \dst
#else
    movdqu  \src, %xmm1
    movdqu  \src16, %xmm2
    movdqu  %xmm1, \dst
    movdqu  %xmm2, \dst16
#endif
.endm

// read the sequence at control: eax = token, ecx = dis - 33, edx = 3-bit lit field; control += 3.
// Branches to \ext if the lit field is 7, else eax = match code (token & 31).
.macro SEQHEAD ext
    movzbl  (%r8), %eax
    movzwl  1(%r8), %ecx
    add     $3, %r8
    mov     %eax, %edx
    shr     $5, %edx
    cmp     $0xdf, %eax
    ja      \ext
    and     $31, %eax
.endm

// one sequence inside a group. dischk = 1 on the .Lpf path.
.macro GSEQ dischk
    SEQHEAD 8f
    sub     %rdx, %r9
    LD16((%r9))
    ST16((%rdi))
7:
    add     %rdx, %rdi
    neg     %rcx
  .if \dischk
    lea     -33(%rdi,%rcx), %r14
    cmp     %r12, %r14
    jb      .Lfail
    CP32    "(%r14)", "(%rdi)", "16(%rdi)", "16(%r14)"
  .else
    CP32    "-33(%rdi,%rcx)", "(%rdi)", "16(%rdi)", "-17(%rdi,%rcx)"
  .endif
    lea     3(%rdi,%rax), %rdi
    cmp     %r9, %r8
    jae     .Ldone
    // out of line: lit field == 7 inside a group
    .subsection 1
8:
    movzbl  (%r8), %r14d
    cmp     $25, %r14d
    ja      .Lext                   // long run: control not yet advanced past the extra byte
    inc     %r8
    lea     7(%r14), %edx           // lit = 7 + extra <= 32
    mov     %r9, %r14
    sub     %r13, %r14
    cmp     %rdx, %r14
    jb      .Lfail                  // lit > literals - src
    and     $31, %eax
    sub     %rdx, %r9
    CP32    "(%r9)", "(%rdi)", "16(%rdi)", "16(%r9)"
    jmp     7b
    .subsection 0
.endm

    .text
    .globl  KERNEL_NAME
    .hidden KERNEL_NAME
    .type   KERNEL_NAME, @function
    .p2align 6
KERNEL_NAME:
    push    %r12
    push    %r13
    push    %r14
    push    %r15
    mov     %rdi, %r13              // src
    mov     %rsi, %r12              // dst
    lea     16(%rdi), %r8           // control
    mov     %rdx, %r9               // literals
    mov     %rcx, %r11              // out_limit
    mov     %rcx, %rax
    sub     %rsi, %rax              // out_limit - dst
    mov     $MAX_DIS, %edx
    cmp     %rdx, %rax
    cmovb   %rax, %rdx
    lea     (%r12,%rdx), %rsi       // prefix_end
    lea     -RED_SLACK(%r11), %r10  // red
    cmp     $RED_SLACK, %rax
    cmovbe  %r12, %r10
    mov     %rsi, %r15
    cmp     %r10, %rsi
    cmova   %r10, %r15              // plim
    cmp     $0x20000, %r12
    cmovb   %r12, %r15
    mov     %r12, %rdi              // out
    jmp     .Ldispatch

    .p2align 6
.Lfast:
    .rept 4
    GSEQ 0
    .endr
    cmp     %r10, %rdi
    jb      .Lfast
    jmp     .Ldispatch

    .p2align 6
.Lpf:
    .rept 4
    GSEQ 1
    .endr
    cmp     %r15, %rdi
    jb      .Lpf

.Ldispatch:
    cmp     %r9, %r8
    jae     .Ldone
    cmp     %r10, %rdi
    jae     .Lchk                   // tail
    cmp     %rsi, %rdi
    jae     .Lfast
    cmp     %r15, %rdi
    jb      .Lpf
    // fall through: inside the prefix with .Lpf unusable

.Lchk:
    SEQHEAD .Lext
    lea     4(%rdi,%rdx), %r14
    cmp     %r11, %r14
    ja      .Lfail                  // out + lit + min_match_len > out_limit
    sub     %rdx, %r9
    LD16((%r9))
    ST16((%rdi))
    add     %rdx, %rdi
    add     $33, %ecx
    mov     %rdi, %r14
    sub     %r12, %r14
    cmp     %rcx, %r14
    jb      .Lfail                  // dis > out - dst
    neg     %rcx
    CP32    "(%rdi,%rcx)", "(%rdi)", "16(%rdi)", "16(%rdi,%rcx)"
    lea     3(%rdi,%rax), %rdi
    jmp     .Ldispatch

    // extended literal run. eax = token (>= 0xe0), ecx = dis - 33, edx = 7,
    // control = the first extra byte
.Lext:
    and     $31, %eax
    movzbl  (%r8), %r14d
    inc     %r8
    add     %r14, %rdx
    cmp     $255, %r14d
    jne     2f
1:
    movzbl  (%r8), %r14d
    cmp     $255, %r14d
    jne     3f
    add     $255, %rdx
    inc     %r8
    cmp     %r9, %r8
    jae     .Lfail
    jmp     1b
3:
    add     %r14, %rdx
    inc     %r8
2:
    mov     %r9, %r14
    sub     %r13, %r14
    cmp     %rdx, %r14
    jb      .Lfail                  // lit > literals - src
    lea     4(%rdi,%rdx), %r14
    cmp     %r11, %r14
    ja      .Lfail                  // out + lit + min_match_len > out_limit (lit < src_size)
    sub     %rdx, %r9
    xor     %r14d, %r14d
4:
    CP32B(%r9, %r14, %rdi)
    add     $32, %r14
    cmp     %rdx, %r14
    jb      4b
    add     %rdx, %rdi
    add     $33, %ecx
    mov     %rdi, %r14
    sub     %r12, %r14
    cmp     %rcx, %r14
    jb      .Lfail                  // dis > out - dst
    neg     %rcx
    CP32    "(%rdi,%rcx)", "(%rdi)", "16(%rdi)", "16(%rdi,%rcx)"
    lea     3(%rdi,%rax), %rdi
    jmp     .Ldispatch

.Lfail:
    xor     %edi, %edi
.Ldone:
    mov     %rdi, %rax
    VZU
    pop     %r15
    pop     %r14
    pop     %r13
    pop     %r12
    ret
    .size   KERNEL_NAME, .-KERNEL_NAME

    // keep the stack non-executable
    .section .note.GNU-stack,"",@progbits

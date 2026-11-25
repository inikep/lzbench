// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

// Safe token loop for the heavy format: x86-64, SysV ABI, GAS syntax, ELF.
// Not assembled on its own: x86_64_avx2.S / x86_64_sse2.S define KERNEL_NAME (and KERNEL_AVX2)
// and include this file.
//
// uint64_t KERNEL_NAME(io* p)      decodes tokens of ONE block
//   returns 0  reject
//           1  cp == cp_end (the block's tokens are done)
//           2  the caller must decode one token with its fully checked step, then call again
//              (no distance yet, or too close to the end of src / out for unconditional copies)
//   Before returning 1 or 2 it checks lp <= lp_end and op <= op_end, and writes cp, op, lp, out
//   and vneg back (not on reject).
//
// io (8-byte fields, in this order):
//    0 cp        cmd cursor               8 cp_end
//   16 op        offset cursor           24 op_end
//   32 lp        literal cursor          40 lp_end
//   48 src_end   one past the last readable source byte
//   56 out                               64 out_lim  (dst + n - literal_suffix)
//   72 dst                               80 chk_end  (dst + max_dis of the codebook)
//   88 T         kernel tables           96 vneg     (min_dis - dis; min_dis = no distance yet)
//  104 scratch[5]  kernel-private
//
// T (one per codebook):
//   T_PACK      u64[256]  mask | big << 31 | ob << 32 | lit << 40 | (lit + len) << 48
//                         (mask = the offset bytes' mask, ob = offset byte count; small codes)
//   T_FULL_LIT  u32[256]  literal count, any code
//   T_FULL_LEN  u32[256]  match length, any code
//
// Requires of both codebooks: a small code has lit <= 16, len <= 32, ob <= 3, mask != 0 iff
// ob != 0; a big code has no offset bytes and is literal-only or match-only with a
// multiple-of-32 length. Requires src and dst >= 64 KiB (the derived limits must not wrap) and
// user-space pointers < 2^47 (the distance compare is signed).
//
// A small token reads 1 cmd byte, 4 offset bytes at op (op += ob <= 3) and 16 literal bytes at lp
// (lp += lit <= 16), writes [out, out + 48) and advances out by <= 48. So a group of 8 small
// tokens needs no per-token check when, at its head,
//   cp <= cp_end - 8, out <= out_lim - 384, lp <= src_end - 128, op <= src_end - 32.
// A big token is fully checked, then a new group starts at the next cmd byte. When a group head
// fails, single tokens run under the 1-token limits; when those fail, return 2 (1 at cp_end).
// Distances: while out < chk_end (.Lpf) every match source is checked against dst; from chk_end
// on (.Lfast) no match can start before dst.
//
// Registers: rbp io, rax cmd byte, rcx/rsi/r11/r12 tmp, rdx vneg, rbx cp, rdi out, r8 op, r9 lp,
//            r10 T, r13 dst, r14 out_lim - 384, r15 src_end - 128.
// scratch:   104 src_end - 32, 112 out_lim - 48, 120 src_end - 16, 128 src_end - 4,
//            136 cp_end - 8.

#ifndef T_PACK
#define T_PACK     0
#define T_FULL_LIT 2048
#define T_FULL_LEN 3072
#endif

#define CPLIM 136(%rbp)

#if defined(KERNEL_AVX2)
#define VZU vzeroupper
.macro LIT16 src, dst
    vmovdqu \src, %xmm0
    vmovdqu %xmm0, \dst
.endm
.macro CP32 src, dst, src16, dst16
    vmovdqu \src, %ymm1
    vmovdqu %ymm1, \dst
.endm
#else
#define VZU
.macro LIT16 src, dst
    movdqu  \src, %xmm0
    movdqu  %xmm0, \dst
.endm
.macro CP32 src, dst, src16, dst16
    movdqu  \src, %xmm1
    movdqu  \src16, %xmm2
    movdqu  %xmm1, \dst
    movdqu  %xmm2, \dst16
.endm
#endif

// one unchecked small token at cmd (rbx + j). chk = 1: match source checked against dst
.macro TOK j, chk
    movzbl  \j(%rbx), %eax
    mov     T_PACK(%r10,%rax,8), %rcx
    mov     (%r8), %r11d                    // always load 4 offset bytes
    and     %ecx, %r11d
    neg     %r11
    test    %ecx, %ecx                      // SF: big code, ZF: no offset bytes
    js      9f
    cmovnz  %r11, %rdx                      // new distance, or keep the repeat one
    shr     $32, %rcx
    movzbl  %cl, %r11d                      // ob
    add     %r11, %r8
    movzbl  %ch, %esi                       // lit
    shr     $16, %ecx                       // lit + len
    LIT16   "(%r9)", "(%rdi)"
    add     %rsi, %r9
    lea     (%rdi,%rsi), %r11               // match destination
  .if \chk
    lea     -33(%r11,%rdx), %rsi            // match source = out + lit - dis
    cmp     %r13, %rsi
    jl      .Lfail                          // dis > out + lit - dst (signed: no wrap)
    CP32    "(%rsi)", "(%r11)", "16(%rsi)", "16(%r11)"
  .else
    CP32    "-33(%r11,%rdx)", "(%r11)", "-17(%r11,%rdx)", "16(%r11)"
  .endif
    add     %rcx, %rdi
    .subsection 1
9:
    lea     \j+1(%rbx), %rbx                // cmd pointer after this token
    jmp     .Lbig
    .subsection 0
.endm

.macro GROUP chk
    TOK 0, \chk
    TOK 1, \chk
    TOK 2, \chk
    TOK 3, \chk
    TOK 4, \chk
    TOK 5, \chk
    TOK 6, \chk
    TOK 7, \chk
    add     $8, %rbx
.endm

// group-head checks: falls through when a group may run
.macro HEAD
    cmp     CPLIM, %rbx
    ja      .Lsingle                        // fewer than 8 cmd bytes left
    cmp     %r14, %rdi
    ja      .Lsingle                        // out > out_lim - 384
    cmp     %r15, %r9
    ja      .Lsingle                        // lp > src_end - 128
    cmp     104(%rbp), %r8
    ja      .Lsingle                        // op > src_end - 32
.endm

    .text
    .globl  KERNEL_NAME
    .hidden KERNEL_NAME
    .type   KERNEL_NAME, @function
    .p2align 6
KERNEL_NAME:
    push    %rbx
    push    %rbp
    push    %r12
    push    %r13
    push    %r14
    push    %r15
    mov     %rdi, %rbp
    mov     0(%rbp), %rbx
    mov     16(%rbp), %r8
    mov     32(%rbp), %r9
    mov     56(%rbp), %rdi
    mov     72(%rbp), %r13
    mov     88(%rbp), %r10
    mov     96(%rbp), %rdx
    // derived limits
    mov     8(%rbp), %rsi
    sub     $8, %rsi
    mov     %rsi, 136(%rbp)                 // cp_end - 8
    mov     64(%rbp), %r14
    lea     -48(%r14), %rax
    mov     %rax, 112(%rbp)                 // out_lim - 48
    sub     $384, %r14                      // out_lim - 384
    mov     48(%rbp), %r15
    lea     -16(%r15), %rax
    mov     %rax, 120(%rbp)                 // src_end - 16
    lea     -4(%r15), %rax
    mov     %rax, 128(%rbp)                 // src_end - 4
    lea     -32(%r15), %rax
    mov     %rax, 104(%rbp)                 // src_end - 32
    sub     $128, %r15                      // src_end - 128
    test    %rdx, %rdx
    jg      .Lret2                          // no distance yet

    // loop selection: on entry and after every big token
.Lhead:
    cmp     80(%rbp), %rdi
    jb      .Lpf_head

    // groups without distance checks
.Lfast_head:
    HEAD
    .p2align 5
.Lfast:
    GROUP 0
    cmp     CPLIM, %rbx
    ja      .Lsingle
    cmp     %r14, %rdi
    ja      .Lsingle
    cmp     %r15, %r9
    ja      .Lsingle
    cmp     104(%rbp), %r8
    jbe     .Lfast
    jmp     .Lsingle

    // groups with out < chk_end: every match source checked
.Lpf_head:
    HEAD
    .p2align 5
.Lpf:
    GROUP 1
    cmp     80(%rbp), %rdi
    jae     .Lfast_head
    cmp     CPLIM, %rbx
    ja      .Lsingle
    cmp     %r14, %rdi
    ja      .Lsingle
    cmp     %r15, %r9
    ja      .Lsingle
    cmp     104(%rbp), %r8
    jbe     .Lpf
    jmp     .Lsingle

    // single tokens under the 1-token limits, always distance-checked
.Lsingle:
    cmp     8(%rbp), %rbx
    jae     .Lret1                          // cp == cp_end
    cmp     112(%rbp), %rdi
    ja      .Lret2                          // out > out_lim - 48
    cmp     120(%rbp), %r9
    ja      .Lret2                          // lp > src_end - 16
    cmp     128(%rbp), %r8
    ja      .Lret2                          // op > src_end - 4
    TOK 0, 1
    inc     %rbx
    jmp     .Lsingle

    // big token: eax = cmd byte, rbx = cmd pointer after it. vneg and op are unchanged.
    // The head limits give out <= out_lim and lp <= src_end - 16, so the subtractions below
    // do not wrap.
.Lbig:
    mov     T_FULL_LIT(%r10,%rax,4), %ecx   // lit
    mov     T_FULL_LEN(%r10,%rax,4), %r12d  // len
    lea     (%rcx,%r12), %r11
    mov     64(%rbp), %rax
    sub     %rdi, %rax
    cmp     %rax, %r11
    ja      .Lfail                          // lit + len > out_lim - out
    mov     48(%rbp), %r12
    sub     %r9, %r12
    cmp     %r12, %rcx
    ja      .Lfail                          // lit > src_end - lp
    // lit + len <= 128: 4 unconditional 32-byte copies from one source (lp, or out - dis).
    // Over-writes stay below out_lim + 32, over-reads below src_end or out + 128 - 33.
    cmp     $128, %r11
    ja      .Lbigloop
    cmp     $96, %rax
    jb      .Lbigloop                       // out_lim - out < 96
    cmp     $128, %r12
    jb      .Lbigloop                       // src_end - lp < 128
    lea     -33(%rdi,%rdx), %rax
    test    %rcx, %rcx
    cmovnz  %r9, %rax
    CP32    "(%rax)", "(%rdi)", "16(%rax)", "16(%rdi)"
    CP32    "32(%rax)", "32(%rdi)", "48(%rax)", "48(%rdi)"
    CP32    "64(%rax)", "64(%rdi)", "80(%rax)", "80(%rdi)"
    CP32    "96(%rax)", "96(%rdi)", "112(%rax)", "112(%rdi)"
    add     %rcx, %r9
    add     %r11, %rdi
    jmp     .Lhead
.Lbigloop:
    mov     %r11, %r12
    sub     %rcx, %r12                      // len
    test    %rcx, %rcx
    jz      2f
1:
    CP32    "(%r9)", "(%rdi)", "16(%r9)", "16(%rdi)"
    add     $32, %r9
    add     $32, %rdi
    sub     $32, %rcx
    jnz     1b
2:
    test    %r12, %r12
    jz      .Lhead
3:
    CP32    "-33(%rdi,%rdx)", "(%rdi)", "-17(%rdi,%rdx)", "16(%rdi)"
    add     $32, %rdi
    sub     $32, %r12
    jnz     3b
    jmp     .Lhead

.Lret1:
    mov     $1, %eax
    jmp     .Lsave
.Lret2:
    mov     $2, %eax
.Lsave:
    cmp     40(%rbp), %r9
    ja      .Lfail                          // lp > lp_end
    cmp     24(%rbp), %r8
    ja      .Lfail                          // op > op_end
    mov     %rbx, 0(%rbp)
    mov     %r8, 16(%rbp)
    mov     %r9, 32(%rbp)
    mov     %rdi, 56(%rbp)
    mov     %rdx, 96(%rbp)
    jmp     .Lout
.Lfail:
    xor     %eax, %eax
.Lout:
    VZU
    pop     %r15
    pop     %r14
    pop     %r13
    pop     %r12
    pop     %rbp
    pop     %rbx
    ret
    .size   KERNEL_NAME, .-KERNEL_NAME

    // keep the stack non-executable
    .section .note.GNU-stack,"",@progbits

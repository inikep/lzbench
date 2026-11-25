// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

// Safe token loop for the heavy format: AArch64 (AAPCS64), GAS syntax, ELF and Mach-O.
// Not assembled on its own: aarch64_neon.S defines KERNEL_NAME and includes this file.
//
// uint64_t KERNEL_NAME(io* p)      decodes tokens of ONE block
//   returns 0  reject
//           1  cp == cp_end (the block's tokens are done)
//           2  the caller must decode one token with its fully checked step, then call again
//              (no distance yet, or too close to the end of src / out for unconditional copies)
//   Before returning 1 or 2 it checks lp <= lp_end and op <= op_end, and writes cp, op, lp, out
//   and vneg back (not on reject).
//
// io (8-byte fields, in this order; the same struct as the x86-64 kernels):
//    0 cp        cmd cursor               8 cp_end
//   16 op        offset cursor           24 op_end
//   32 lp        literal cursor          40 lp_end
//   48 src_end   one past the last readable source byte
//   56 out                               64 out_lim  (dst + n - literal_suffix)
//   72 dst                               80 chk_end  (dst + max_dis of the codebook)
//   88 T         kernel tables           96 vneg     (min_dis - dis; min_dis = no distance yet)
//  104 scratch[5]  unused here
//
// T (one per codebook):
//   T_PACK      u64[256]  mask | big << 31 | ob << 32 | lit << 40 | (lit + len) << 48
//                         (mask = the offset bytes' mask, ob = offset byte count; small codes)
//   T_FULL_LIT  u32[256]  literal count, any code
//   T_FULL_LEN  u32[256]  match length, any code
//
// Requires of both codebooks: a small code has lit <= 16, len <= 32, ob <= 3, mask != 0 iff
// ob != 0; a big code has no offset bytes and is literal-only or match-only with a
// multiple-of-32 length. Requires src and dst >= 64 KiB (the derived limits must not wrap).
//
// A small token reads 1 cmd byte, 4 offset bytes at op (op += ob <= 3) and 16 literal bytes at lp
// (lp += lit <= 16), writes [out, out + 48) and advances out by <= 48. So a group of K = 8
// small tokens needs no per-token check when, at its head,
//   cp <= cp_end - K, out <= out_lim - 48 K, lp <= src_end - 16 K, op <= src_end - 4 K.
// A big token is fully checked, then a new group starts at the next cmd byte. When a group head
// fails, single tokens run under the 1-token limits; when those fail, return 2 (1 at cp_end).
// Distances: while out < chk_end (pf) every match source is checked against dst; from chk_end
// on (fast) no match can start before dst. The check is the sign of (match source - dst): exact
// for any pair of pointers into one buffer, tagged or not.
//
// The repeat distance lives in x6 as 32 - dis = ~(dis - 33), so a new one is a csinv of the
// masked offset bytes and the 32-byte source sits at a multiple-of-16 ldp offset.
//
// Registers: x0 io, x1 cp, x2 op, x3 lp, x4 out, x5 T, x6 32 - dis, x7 dst + 32,
//            x8 cp_end - K, x17 out_lim - 48 K, x10 src_end - 16 K, x14 src_end - 4 K,
//            x9 table entry, x11/x12/x13 tmp, x15 match destination, x16 match source + 32.
// Only caller-saved registers, no stack. x18 is never used (reserved on Apple).

#define T_FULL_LIT 2048
#define T_FULL_LEN 3072
#define BIG_BIT 31

// group head limits, K = 8
#define K_CP  8
#define K_OUT 384
#define K_LP  128
#define K_OP  32

// Mach-O: C symbols carry a leading underscore, assembler-local labels start with L (ELF: .L),
// and there is no .type / .size / .hidden / .note.GNU-stack.
#if defined(__APPLE__)
#define L(x)    L##x
#define SYM2(x) _##x
#else
#define L(x)    .L##x
#define SYM2(x) x
#endif
#define SYM(x) SYM2(x)

// 32-byte copy from s - 32 + o to d + o. Never overlapping: dis >= 33, or s is in the source.
.macro CP32 s, d, o
    ldp     q1, q2, [\s, #\o - 32]
    stp     q1, q2, [\d, #\o]
.endm

// one unchecked small token at cmd (x1 + j). p = path name, chk = 1: match source checked
// against dst. A big code leaves through BSTUB.
.macro TOK p, j, chk
    ldrb    w9, [x1, #\j]
    ldr     x9, [x5, x9, lsl #3]
    ldr     w11, [x2]                       // always load 4 offset bytes
    ldr     q0, [x3]
    tbnz    x9, #BIG_BIT, L(b_\p\()_\j)
    str     q0, [x4]
    and     w11, w11, w9
    ubfx    x12, x9, #32, #8                // ob
    ubfx    x13, x9, #40, #8                // lit
    cmp     w9, #0                          // Z: no offset bytes
    csinv   x6, x6, x11, eq                 // new distance, or keep the repeat one
    add     x2, x2, x12
    add     x3, x3, x13
    add     x15, x4, x13                    // match destination
    add     x16, x15, x6                    // match source + 32 = out + lit - dis + 32
  .if \chk
    cmp     x16, x7
    b.mi    L(fail)                         // dis > out + lit - dst
  .endif
    CP32    x16, x15, 0
    add     x4, x4, x9, lsr #48
.endm

// out of line: a big code at cmd (x1 + j)
.macro BSTUB p, j
L(b_\p\()_\j):
    add     x1, x1, #\j + 1                 // cmd pointer after this token
    b       L(big)
.endm

.macro GROUP p, chk
    TOK     \p, 0, \chk
    TOK     \p, 1, \chk
    TOK     \p, 2, \chk
    TOK     \p, 3, \chk
    TOK     \p, 4, \chk
    TOK     \p, 5, \chk
    TOK     \p, 6, \chk
    TOK     \p, 7, \chk
    add     x1, x1, #K_CP
.endm

.macro STUBS p
    BSTUB   \p, 0
    BSTUB   \p, 1
    BSTUB   \p, 2
    BSTUB   \p, 3
    BSTUB   \p, 4
    BSTUB   \p, 5
    BSTUB   \p, 6
    BSTUB   \p, 7
.endm

// group-head checks: falls through when a group may run
.macro HEAD
    cmp     x1, x8
    b.hi    L(single)                       // fewer than K cmd bytes left
    cmp     x4, x17
    b.hi    L(single)                       // out > out_lim - 48 K
    cmp     x3, x10
    b.hi    L(single)                       // lp > src_end - 16 K
    cmp     x2, x14
    b.hi    L(single)                       // op > src_end - 4 K
.endm

    .text
    .globl  SYM(KERNEL_NAME)
#if defined(__APPLE__)
    .private_extern SYM(KERNEL_NAME)
#else
    .hidden SYM(KERNEL_NAME)
    .type   SYM(KERNEL_NAME), %function
#endif
    .p2align 6
SYM(KERNEL_NAME):
    hint    #34                             // bti c
    ldp     x1, x11, [x0]                   // cp, cp_end
    ldr     x2, [x0, #16]
    ldr     x3, [x0, #32]
    ldr     x12, [x0, #48]                  // src_end
    ldp     x4, x13, [x0, #56]              // out, out_lim
    ldr     x7, [x0, #72]
    ldr     x5, [x0, #88]
    ldr     x6, [x0, #96]
    add     x7, x7, #32
    // derived limits
    sub     x8, x11, #K_CP
    sub     x17, x13, #K_OUT
    sub     x10, x12, #K_LP
    sub     x14, x12, #K_OP
    subs    x6, x6, #1                      // 32 - dis
    b.pl    L(ret2)                         // no distance yet

    // loop selection: on entry and after every big token
L(head):
    ldr     x16, [x0, #80]
    cmp     x4, x16
    b.lo    L(pf_head)

    // groups without distance checks
L(fast_head):
    HEAD
    .p2align 5
L(fast):
    GROUP   fast, 0
    cmp     x1, x8
    b.hi    L(single)
    cmp     x4, x17
    b.hi    L(single)
    cmp     x3, x10
    b.hi    L(single)
    cmp     x2, x14
    b.ls    L(fast)
    b       L(single)
    STUBS   fast

    // groups with out < chk_end: every match source checked
L(pf_head):
    HEAD
    .p2align 5
L(pf):
    GROUP   pf, 1
    ldr     x16, [x0, #80]
    cmp     x4, x16
    b.hs    L(fast_head)
    cmp     x1, x8
    b.hi    L(single)
    cmp     x4, x17
    b.hi    L(single)
    cmp     x3, x10
    b.hi    L(single)
    cmp     x2, x14
    b.ls    L(pf)

    // single tokens under the 1-token limits, always distance-checked
L(single):
    add     x16, x8, #K_CP
    cmp     x1, x16
    b.hs    L(ret1)                         // cp == cp_end
    add     x16, x17, #K_OUT - 48
    cmp     x4, x16
    b.hi    L(ret2)                         // out > out_lim - 48
    add     x16, x10, #K_LP - 16
    cmp     x3, x16
    b.hi    L(ret2)                         // lp > src_end - 16
    add     x16, x14, #K_OP - 4
    cmp     x2, x16
    b.hi    L(ret2)                         // op > src_end - 4
    TOK     sg, 0, 1
    add     x1, x1, #1
    b       L(single)
    BSTUB   sg, 0
    STUBS   pf

    // big token: x1 = cmd pointer after it. The distance and op are unchanged.
    // The head limits give out <= out_lim and lp <= src_end - 16, so the subtractions below
    // do not wrap.
L(big):
    ldurb   w9, [x1, #-1]
    add     x16, x5, #T_FULL_LIT
    ldr     w12, [x16, x9, lsl #2]          // lit
    add     x16, x5, #T_FULL_LEN
    ldr     w13, [x16, x9, lsl #2]          // len
    add     x11, x12, x13
    add     x15, x17, #K_OUT
    sub     x15, x15, x4
    cmp     x11, x15
    b.hi    L(fail)                         // lit + len > out_lim - out
    add     x16, x10, #K_LP
    sub     x16, x16, x3
    cmp     x12, x16
    b.hi    L(fail)                         // lit > src_end - lp
    // lit + len <= 128: 4 unconditional 32-byte copies from one source (lp, or out - dis).
    // Over-writes stay below out_lim + 32, over-reads below src_end or out + 128 - 33.
    cmp     x11, #128
    b.hi    L(bigloop)
    cmp     x15, #96
    b.lo    L(bigloop)                      // out_lim - out < 96
    cmp     x16, #128
    b.lo    L(bigloop)                      // src_end - lp < 128
    add     x9, x4, x6                      // out - dis + 32
    add     x16, x3, #32
    cmp     x12, #0
    csel    x9, x16, x9, ne
    add     x3, x3, x12
    add     x13, x4, x11                    // CP32 may use x11 / x12
    // each chunk stored before the next is loaded: a match may read what it just wrote
    CP32    x9, x4, 0
    CP32    x9, x4, 32
    CP32    x9, x4, 64
    CP32    x9, x4, 96
    mov     x4, x13
    b       L(head)
L(bigloop):
    cbz     x12, 2f
1:
    ldp     q1, q2, [x3], #32
    stp     q1, q2, [x4], #32
    subs    x12, x12, #32
    b.ne    1b
2:
    cbz     x13, L(head)
    add     x9, x4, x6
3:
    ldp     q1, q2, [x9, #-32]
    add     x9, x9, #32
    stp     q1, q2, [x4], #32
    subs    x13, x13, #32
    b.ne    3b
    b       L(head)

L(ret1):
    mov     x9, #1
    b       L(save)
L(ret2):
    mov     x9, #2
L(save):
    ldr     x16, [x0, #40]
    cmp     x3, x16
    b.hi    L(fail)                         // lp > lp_end
    ldr     x16, [x0, #24]
    cmp     x2, x16
    b.hi    L(fail)                         // op > op_end
    add     x6, x6, #1                      // back to min_dis - dis
    str     x1, [x0]
    str     x2, [x0, #16]
    str     x3, [x0, #32]
    str     x4, [x0, #56]
    str     x6, [x0, #96]
    mov     x0, x9
    ret
L(fail):
    mov     x0, #0
    ret
#if !defined(__APPLE__)
    .size   SYM(KERNEL_NAME), .-SYM(KERNEL_NAME)

    // keep the stack non-executable
    .section .note.GNU-stack,"",%progbits
#endif

// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

// Safe sequence loop for the light format: AArch64 (AAPCS64), GAS syntax, ELF and Mach-O.
// Not assembled on its own: aarch64_neon.S defines KERNEL_NAME and includes this file.
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
//   plim        min(prefix_end, red)
//
// A group is 4 sequences decoded with one output-bound compare. Inside a group a
// sequence advances `out` by <= 66 and writes <= 64 bytes past its start, so a group that starts
// below `red` writes below out_limit + 32: inside the raw suffix's 32 bytes.
// RED_SLACK = 66 * 4 - 35.
//
// Paths, chosen by `dispatch` on entry and after every group / long literal run:
//   pf    out < plim: groups; per sequence the termination test and a distance test
//   fast  prefix_end <= out < red: groups; per sequence only the termination test
//   chk   everything else: one fully checked sequence
//   ext   extended literal run with a first extra byte > 25 (or from chk): fully checked
// A run of <= 32 literals (one extra byte <= 25) stays inside its group.
//
// The distance test is the sign of (match source - dst): exact for any pair of pointers into
// one buffer, tagged or not, so no path needs a minimum address.
//
// Registers: x0 src, x1 dst, x2 literals, x3 out_limit, x4 out, x5 control, x6 red,
//            x7 prefix_end, x8 plim, x17 dst + 33, w9 token, w10 dis - 33, x12 lit,
//            x13 match code, x15 match destination, x16 match source + 33, x11/x14 temp.
// Only caller-saved registers, no stack. x18 is never used (reserved on Apple).

#define RED_SLACK 229

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

// 32-byte match copy. x16 = source + 33, x15 = destination; dis >= 33, so they never overlap.
.macro MATCH32
    ldur    q1, [x16, #-33]
    ldur    q2, [x16, #-17]
    stp     q1, q2, [x15]
.endm

// one sequence inside a group. p = path name, j = index in the group, dischk = 1 on the pf path.
// The lit field == 7 case is out of line (GSTUB), and rejoins at r_p_j.
.macro GSEQ p, j, dischk
    ldrb    w9, [x5], #3
    ldurh   w10, [x5, #-2]
    cmp     w9, #0xdf
    b.hi    L(g_\p\()_\j)
    lsr     w12, w9, #5
    and     w13, w9, #31
    sub     x2, x2, x12
    ldr     q0, [x2]
    str     q0, [x4]
    add     w14, w12, w13
    add     w14, w14, #3
    add     x15, x4, x12
L(r_\p\()_\j):
    sub     x16, x15, x10
  .if \dischk
    cmp     x16, x17
    b.mi    L(fail)                         // match source < dst
  .endif
    MATCH32
    add     x4, x4, x14
    cmp     x5, x2
    b.hs    L(done)
.endm

// out of line: lit field == 7 inside a group. control is past the 3 sequence bytes.
.macro GSTUB p, j
L(g_\p\()_\j):
    ldrb    w11, [x5]
    cmp     w11, #25
    b.hi    L(ext)                          // long run: control still at the extra byte
    add     x5, x5, #1
    add     w12, w11, #7                    // lit = 7 + extra <= 32
    sub     x11, x2, x0
    cmp     x11, x12
    b.lo    L(fail)                         // lit > literals - src
    and     w13, w9, #31
    sub     x2, x2, x12
    ldp     q1, q2, [x2]
    stp     q1, q2, [x4]
    add     w14, w12, w13
    add     w14, w14, #3
    add     x15, x4, x12
    b       L(r_\p\()_\j)
.endm

.macro GROUP p, dischk
    GSEQ    \p, 0, \dischk
    GSEQ    \p, 1, \dischk
    GSEQ    \p, 2, \dischk
    GSEQ    \p, 3, \dischk
.endm

.macro STUBS p
    GSTUB   \p, 0
    GSTUB   \p, 1
    GSTUB   \p, 2
    GSTUB   \p, 3
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
    add     x5, x0, #16                     // control
    mov     x4, x1                          // out
    sub     x9, x3, x1                      // out_limit - dst
    mov     x10, #0x10000
    add     x10, x10, #32                   // max_dis
    cmp     x9, x10
    csel    x10, x9, x10, lo
    add     x7, x1, x10                     // prefix_end
    sub     x6, x3, #RED_SLACK
    cmp     x9, #RED_SLACK
    csel    x6, x1, x6, ls                  // red
    cmp     x7, x6
    csel    x8, x6, x7, hi                  // plim
    add     x17, x1, #33
    b       L(dispatch)

    .p2align 6
L(fast):
    GROUP   fast, 0
    cmp     x4, x6
    b.lo    L(fast)
    b       L(dispatch)
    STUBS   fast

    .p2align 6
L(pf):
    GROUP   pf, 1
    cmp     x4, x8
    b.lo    L(pf)

L(dispatch):
    cmp     x5, x2
    b.hs    L(done)
    cmp     x4, x6
    b.hs    L(chk)                          // tail
    cmp     x4, x7
    b.hs    L(fast)
    cmp     x4, x8
    b.lo    L(pf)
    // fall through: inside the prefix with pf unusable

L(chk):
    ldrb    w9, [x5], #3
    ldurh   w10, [x5, #-2]
    cmp     w9, #0xdf
    b.hi    L(ext)
    lsr     w12, w9, #5
    and     w13, w9, #31
    add     x11, x4, x12
    add     x11, x11, #4
    cmp     x11, x3
    b.hi    L(fail)                         // out + lit + min_match_len > out_limit
    sub     x2, x2, x12
    ldr     q0, [x2]
    str     q0, [x4]
    add     x4, x4, x12
    // checked match: x4 = match destination, w10 = dis - 33, w13 = match code
L(chkmatch):
    add     x10, x10, #33
    sub     x11, x4, x1
    cmp     x11, x10
    b.lo    L(fail)                         // dis > out - dst
    sub     x16, x4, x10
    ldp     q1, q2, [x16]
    stp     q1, q2, [x4]
    add     x4, x4, x13
    add     x4, x4, #3
    b       L(dispatch)

    // extended literal run. w9 = token (>= 0xe0), w10 = dis - 33, control = the first extra byte
L(ext):
    and     w13, w9, #31
    ldrb    w11, [x5], #1
    add     x12, x11, #7
    cmp     w11, #255
    b.ne    2f
1:
    ldrb    w11, [x5]
    cmp     w11, #255
    b.ne    3f
    add     x12, x12, #255
    add     x5, x5, #1
    cmp     x5, x2
    b.hs    L(fail)
    b       1b
3:
    add     x12, x12, x11
    add     x5, x5, #1
2:
    sub     x11, x2, x0
    cmp     x11, x12
    b.lo    L(fail)                         // lit > literals - src
    add     x11, x4, x12
    add     x11, x11, #4
    cmp     x11, x3
    b.hi    L(fail)                         // out + lit + min_match_len > out_limit (lit < src_size)
    sub     x2, x2, x12
    mov     x11, x2
    mov     x14, x4
    add     x4, x4, x12
4:
    ldp     q1, q2, [x11], #32
    stp     q1, q2, [x14], #32
    cmp     x14, x4
    b.lo    4b
    b       L(chkmatch)
    STUBS   pf

L(fail):
    mov     x4, #0
L(done):
    mov     x0, x4
    ret
#if !defined(__APPLE__)
    .size   SYM(KERNEL_NAME), .-SYM(KERNEL_NAME)

    // keep the stack non-executable
    .section .note.GNU-stack,"",%progbits
#endif

// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

// Safe kernels for the entropy format: AArch64 (AAPCS64), GAS syntax, ELF and Mach-O.
// Not assembled on its own: aarch64_neon.S defines the KERNEL_* names and includes this file.
//
// ARMv8.0-A + AdvSIMD only. x18 is never used (reserved on Apple), x29 is never written.
// Everything, tables included, is in the text section and position independent (no relocations).
// The contracts are those of x86_64.inc; differences are marked (ARM).
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
//   - out_end >= 4 * 47 and cmd_end >= 4 as addresses.
//   (ARM) Pointers may carry a tag in the top byte: out, out_end and dst are only compared with
//   each other (the distance check is the sign of a difference), cmd only with cmd_end.
//
// A command reads 1 cmd byte, 16 literal bytes (lit += ll <= 15) and one offset, writes
// [out, out + 16) and [out + ll, out + ll + 32), and advances out by ll + ml in [1, 47]. So a
// group of 4 needs no per-command check when, at its head, 4 cmd bytes are left and
// out <= out_end - 3 * 47: each of its commands then starts at or before out_end. Past that,
// single commands run while out < out_end.
//
// Four copies of the loop: plain / SUB x unchecked / checked, chosen once per block.
//
// Registers: x0 io, x1 cmd cursor, x2 off, x3 lit, x4 out, x5 T, x6 r0, x7 dst, x8 cmd_end,
//            x9 cmd byte, x10 T entry (lit | (lit + len) << 8), x11 offset, x12 out_end,
//            x13 offset index, x14 out_end - 141, x15 match destination, x16 match source,
//            x17 cmd_end - 4. No stack.
//
// ---------------------------------------------------------------------------------------------
// void KERNEL_HUFFMAN(io* p)                  whole rounds of 60 symbols
//   Runs while out > out_stop and p[10] <= lim and p[11] <= lim; the caller has made sure that
//   this keeps every 8-byte load inside the payload and every store inside the output.
//   (ARM) All three are tested before the first round too.
//
// io (8-byte fields): 0 out (descending), 8 out_stop, 16 table (u16[2048]), 24 lim,
//                     32 p[12], 128 bits[12]
//
// bits = (load64(p) | 1 << 63) >> used. A symbol: e = table[bits & 2047] (length | symbol << 8),
// bits >>= length, and ONE 2-byte store of e at out - 2 - index: the length byte falls on the
// next symbol's slot, which is written afterwards (a round also writes the byte at out - 61).
// After 5 symbols per bit string, clz(bits) = used + consumed, so p += lz >> 3 and
// bits = (load64(p) | 1 << 63) >> (lz & 7).
//
// Registers: bits[0..11] x0-x11, p[0..11] x12-x17 x19-x24, x25 out, x26 table, x27 lim,
//            x28 out_stop, x30 tmp.
// Stack:     96 bytes: x19-x28, x30, io.
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
//   80 tmp[3]        wide_cnt + 8 bytes each
//  104 out           off_cnt + 16 entries (written: off_cnt rounded up to 8)
//  112 E             u8[256][8] expand masks, then u8[256] popcounts
//  120 o3_cnt        3-byte offsets
//
// Requires: f1 has wide_cnt set bits, f2 has o3_cnt, none past their last flag; 16 readable
// bytes after every plane.
//
// The AVX2 method: a flag byte at a time, the planes are spread to their lanes with tbl (an
// index of 0x80 selects zero). A block without 3-byte offsets skips their pass, one with 1-byte
// offsets only is just widened.

#ifndef FIRST_NEW
#define FIRST_NEW 131
#endif

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

.macro FUNC name
    .globl  \name
#if defined(__APPLE__)
    .private_extern \name
#else
    .hidden \name
    .type   \name, %function
#endif
    .p2align 6
\name:
    hint    #34                             // bti c
.endm

.macro ENDFUNC name
#if !defined(__APPLE__)
    .size   \name, .-\name
#endif
.endm

    .text

// ---------------------------------------------------------------------------------------------
// commands

// one command at cmd (x1 + j)
.macro CMD j, sub, chk
    ldrb    w9, [x1, #\j]
    ldr     q0, [x3]                        // always 16 literal bytes
    ldrsw   x11, [x2, x13, lsl #2]          // always load an offset
    ldrh    w10, [x5, x9, lsl #1]
  .if \sub
    ldr     q3, [x4, x6]
    add     v0.16b, v0.16b, v3.16b
  .endif
    str     q0, [x4]
    cmp     w9, #FIRST_NEW
    csel    x6, x11, x6, hs                 // new distance, or keep the repeat one
    cinc    x13, x13, hs                    // consume the offset if new
    add     x3, x3, w10, uxtb
    add     x15, x4, w10, uxtb              // match destination
    add     x16, x15, x6
  .if \chk
    cmp     x16, x7
    b.mi    L(cmd_fail)                     // match source < dst (sign of the difference)
  .endif
    ldp     q1, q2, [x16]
    stp     q1, q2, [x15]
    add     x4, x4, x10, lsr #8
.endm

.macro CMD_LOOP sub, chk
    b       2f
    .p2align 5
1:
    CMD     0, \sub, \chk
    CMD     1, \sub, \chk
    CMD     2, \sub, \chk
    CMD     3, \sub, \chk
    add     x1, x1, #4
2:
    cmp     x1, x17
    b.hi    3f                              // fewer than 4 cmd bytes left
    cmp     x4, x14
    b.ls    1b                              // out <= out_end - 141
3:
    cmp     x1, x8
    b.hs    L(cmd_done)
    cmp     x4, x12
    b.hs    L(cmd_fail)                     // out >= out_end with commands left
    CMD     0, \sub, \chk
    add     x1, x1, #1
    b       3b
.endm

    FUNC    SYM(KERNEL_COMMANDS)
    mov     x9, x1                          // variant
    ldp     x1, x8, [x0]
    ldp     x2, x3, [x0, #16]
    ldp     x4, x12, [x0, #32]
    ldp     x7, x6, [x0, #48]
    ldr     x5, [x0, #64]
    mov     x13, #0
    sub     x17, x8, #4
    sub     x14, x12, #141                  // out_end - 3 * 47
    tbnz    x9, #1, L(cmd_chk)
    tbnz    x9, #0, L(cmd_sub)
    CMD_LOOP 0, 0
L(cmd_sub):
    CMD_LOOP 1, 0
L(cmd_chk):
    tbnz    x9, #0, L(cmd_chk_sub)
    CMD_LOOP 0, 1
L(cmd_chk_sub):
    CMD_LOOP 1, 1

L(cmd_done):
    str     x3, [x0, #24]
    str     x4, [x0, #32]
    str     x6, [x0, #56]
    mov     x0, #1
    ret
L(cmd_fail):
    mov     x0, #0
    ret
    ENDFUNC SYM(KERNEL_COMMANDS)

// ---------------------------------------------------------------------------------------------
// Huffman

// symbol number idx of the round, from the bit buffer b
.macro HUF_SYM b, idx
    and     x30, \b, #2047
    ldrh    w30, [x26, x30, lsl #1]
    lsr     \b, \b, x30                     // by the low 6 bits: the length
    sturh   w30, [x25, #-(\idx + 2)]
.endm

// refill the bit buffer b, whose pointer is p
.macro HUF_REFILL b, p
    clz     x30, \b                         // never 64: the marker
    add     \p, \p, x30, lsr #3
    ldr     \b, [\p]
    and     x30, x30, #7
    orr     \b, \b, #0x8000000000000000
    lsr     \b, \b, x30
.endm

    FUNC    SYM(KERNEL_HUFFMAN)
    sub     sp, sp, #96
    stp     x19, x20, [sp]
    stp     x21, x22, [sp, #16]
    stp     x23, x24, [sp, #32]
    stp     x25, x26, [sp, #48]
    stp     x27, x28, [sp, #64]
    stp     x30, x0, [sp, #80]
    mov     x30, x0
    ldp     x25, x28, [x30]
    ldp     x26, x27, [x30, #16]
    ldp     x12, x13, [x30, #32]
    ldp     x14, x15, [x30, #48]
    ldp     x16, x17, [x30, #64]
    ldp     x19, x20, [x30, #80]
    ldp     x21, x22, [x30, #96]
    ldp     x23, x24, [x30, #112]
    ldp     x0, x1, [x30, #128]
    ldp     x2, x3, [x30, #144]
    ldp     x4, x5, [x30, #160]
    ldp     x6, x7, [x30, #176]
    ldp     x8, x9, [x30, #192]
    ldp     x10, x11, [x30, #208]
    b       2f
    .p2align 6
1:
  .irp row, 0, 1, 2, 3, 4
    HUF_SYM x0, (12 * \row + 0)
    HUF_SYM x1, (12 * \row + 1)
    HUF_SYM x2, (12 * \row + 2)
    HUF_SYM x3, (12 * \row + 3)
    HUF_SYM x4, (12 * \row + 4)
    HUF_SYM x5, (12 * \row + 5)
    HUF_SYM x6, (12 * \row + 6)
    HUF_SYM x7, (12 * \row + 7)
    HUF_SYM x8, (12 * \row + 8)
    HUF_SYM x9, (12 * \row + 9)
    HUF_SYM x10, (12 * \row + 10)
    HUF_SYM x11, (12 * \row + 11)
  .endr
    HUF_REFILL x0, x12
    HUF_REFILL x1, x13
    HUF_REFILL x2, x14
    HUF_REFILL x3, x15
    HUF_REFILL x4, x16
    HUF_REFILL x5, x17
    HUF_REFILL x6, x19
    HUF_REFILL x7, x20
    HUF_REFILL x8, x21
    HUF_REFILL x9, x22
    HUF_REFILL x10, x23
    HUF_REFILL x11, x24
    sub     x25, x25, #60
2:
    cmp     x23, x27
    b.hi    3f                              // p[10] > lim
    cmp     x24, x27
    b.hi    3f                              // p[11] > lim
    cmp     x25, x28
    b.hi    1b                              // out > out_stop
3:
    ldr     x30, [sp, #88]
    str     x25, [x30]
    stp     x12, x13, [x30, #32]
    stp     x14, x15, [x30, #48]
    stp     x16, x17, [x30, #64]
    stp     x19, x20, [x30, #80]
    stp     x21, x22, [x30, #96]
    stp     x23, x24, [x30, #112]
    stp     x0, x1, [x30, #128]
    stp     x2, x3, [x30, #144]
    stp     x4, x5, [x30, #160]
    stp     x6, x7, [x30, #176]
    stp     x8, x9, [x30, #192]
    stp     x10, x11, [x30, #208]
    ldp     x19, x20, [sp]
    ldp     x21, x22, [sp, #16]
    ldp     x23, x24, [sp, #32]
    ldp     x25, x26, [sp, #48]
    ldp     x27, x28, [sp, #64]
    ldr     x30, [sp, #80]
    add     sp, sp, #96
    ret
    ENDFUNC SYM(KERNEL_HUFFMAN)

// ---------------------------------------------------------------------------------------------
// DELTA restore
//
// Registers: x0 cursor, x1 e, x2 stride, x3 e - 16, v0 the 16 bytes at the cursor, v1 the 16
// restored bytes before it, v2 the 16 before those, v3 tmp, v6 tbl indices, v7 zero.
// One loop per stride (ext takes immediates only):
//   stride < 16   prefix-sum the 16 bytes at that stride with byte shifts (off the carried
//                 chain), then add the last `stride` restored bytes, repeated across the vector
//   16..32        add the 16 restored bytes that start `stride` back

.macro DELTA_STRIDE s
L(delta_\s):
    hint    #36                             // bti j
  .if \s < 16
    ldr     q6, L(delta_rep_\s)
  .endif
    cmp     x0, x3
    b.hi    L(delta_tail)
1:
    ldr     q0, [x0]
  .if \s < 16
    ext     v3.16b, v7.16b, v0.16b, #(16 - \s)
    add     v0.16b, v0.16b, v3.16b
   .if 2 * \s < 16
    ext     v3.16b, v7.16b, v0.16b, #(16 - 2 * \s)
    add     v0.16b, v0.16b, v3.16b
   .endif
   .if 4 * \s < 16
    ext     v3.16b, v7.16b, v0.16b, #(16 - 4 * \s)
    add     v0.16b, v0.16b, v3.16b
   .endif
   .if 8 * \s < 16
    ext     v3.16b, v7.16b, v0.16b, #(16 - 8 * \s)
    add     v0.16b, v0.16b, v3.16b
   .endif
    tbl     v3.16b, {v1.16b}, v6.16b
    add     v1.16b, v0.16b, v3.16b
  .elseif \s == 16
    add     v1.16b, v0.16b, v1.16b
  .elseif \s < 32
    ext     v3.16b, v2.16b, v1.16b, #(32 - \s)
    mov     v2.16b, v1.16b
    add     v1.16b, v0.16b, v3.16b
  .else
    add     v0.16b, v0.16b, v2.16b
    mov     v2.16b, v1.16b
    mov     v1.16b, v0.16b
  .endif
    str     q1, [x0], #16
    cmp     x0, x3
    b.ls    1b
    b       L(delta_tail)
.endm

    FUNC    SYM(KERNEL_DELTA)
    adr     x4, L(delta_table)
    sub     x5, x2, #1
    ldrsw   x5, [x4, x5, lsl #2]
    ldp     q2, q1, [x0, #-32]
    movi    v7.16b, #0
    sub     x3, x1, #16
    add     x4, x4, x5
    br      x4

  .irp s, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    DELTA_STRIDE \s
  .endr
  .irp s, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
    DELTA_STRIDE \s
  .endr

    // fewer than 16 bytes left
L(delta_tail):
    cmp     x0, x1
    b.hs    2f
    sub     x4, x0, x2
1:
    ldrb    w5, [x4], #1
    ldrb    w6, [x0]
    add     w6, w6, w5
    strb    w6, [x0], #1
    cmp     x0, x1
    b.lo    1b
2:
    ret

    .p2align 2
L(delta_table):
  .irp s, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    .long   L(delta_\s) - L(delta_table)
  .endr
  .irp s, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
    .long   L(delta_\s) - L(delta_table)
  .endr
    // lane j = the restored byte at 16 - s + j % s of the previous vector
    .p2align 4
  .irp s, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
L(delta_rep_\s):
   .irp j, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    .byte   16 - \s + (\j % \s)
   .endr
  .endr
    ENDFUNC SYM(KERNEL_DELTA)

// ---------------------------------------------------------------------------------------------
// offset merge
//
// Registers: x0 io, x13 E, x17 its popcounts, x14 flag byte, x15 its popcount, x10 entry index
//            (8 per step),
//            first pass:  x1 o2l, x2 o2h, x3 o3l, x4 o3m, x5 o3h, x6 f2, x7-x9 tmp[0..2],
//                         x11 2-byte cursor, x12 3-byte cursor, x16 wide_cnt
//            second pass: x1 o1, x2-x4 the wide offsets' byte planes, x6 f1, x7 out,
//                         x11 1-byte cursor, x12 wide cursor, x16 off_cnt,
//                         v5 = -32, v6 = running minimum, v7 = 0

// 8 offsets a step: the 1-byte plane (x1) and `planes - 1` byte planes of the wide offsets
// (x2, x3, x4) -> int32 at x7
.macro MERGE_WIDEN planes
    .p2align 4
3:
  .if \planes == 1
    ldr     d2, [x1, x10]
    zip1    v2.16b, v2.16b, v7.16b
    zip2    v3.8h, v2.8h, v7.8h
    zip1    v2.8h, v2.8h, v7.8h
  .else
    ldrb    w14, [x6], #1
    ldr     d1, [x13, x14, lsl #3]          // lanes of the wide offsets
    ldrb    w15, [x17, x14]
    eor     w14, w14, #255
    ldr     d0, [x13, x14, lsl #3]          // lanes of the 1-byte offsets
    ldr     d2, [x1, x11]
    ldr     d3, [x2, x12]
    ldr     d4, [x3, x12]
    tbl     v2.16b, {v2.16b}, v0.16b
    tbl     v3.16b, {v3.16b}, v1.16b
    tbl     v4.16b, {v4.16b}, v1.16b
    orr     v2.16b, v2.16b, v3.16b
    zip1    v2.16b, v2.16b, v4.16b          // 8 x (low | mid << 8)
   .if \planes == 3
    ldr     d4, [x4, x12]
    tbl     v4.16b, {v4.16b}, v1.16b
    zip1    v4.16b, v4.16b, v7.16b          // 8 x high
    zip2    v3.8h, v2.8h, v4.8h
    zip1    v2.8h, v2.8h, v4.8h
   .else
    zip2    v3.8h, v2.8h, v7.8h
    zip1    v2.8h, v2.8h, v7.8h
   .endif
    add     x12, x12, x15
    add     x11, x11, #8
    sub     x11, x11, x15
  .endif
    sub     v2.4s, v5.4s, v2.4s             // -32 - v
    sub     v3.4s, v5.4s, v3.4s
    stp     q2, q3, [x7], #32
    smin    v6.4s, v6.4s, v2.4s
    smin    v6.4s, v6.4s, v3.4s
    add     x10, x10, #8
    cmp     x10, x16
    b.lo    3b
    b       L(merge_done)
.endm

    FUNC    SYM(KERNEL_MERGE)
    ldr     x13, [x0, #112]
    add     x17, x13, #2048
    mvni    v5.4s, #31                      // -32
    mov     v6.16b, v5.16b
    movi    v7.16b, #0

    // without 3-byte offsets the wide planes are the 2-byte ones as they lie
    ldr     x9, [x0, #120]
    cbnz    x9, L(merge_planes)
    ldp     x2, x3, [x0, #8]
    mov     x4, #0
    b       L(merge_widen)
L(merge_planes):
    // 2- and 3-byte offsets -> three byte planes in flag order
    ldp     x1, x2, [x0, #8]
    ldp     x3, x4, [x0, #24]
    ldr     x5, [x0, #40]
    ldr     x6, [x0, #56]
    ldr     x16, [x0, #72]
    ldp     x7, x8, [x0, #80]
    ldr     x9, [x0, #96]
    mov     x10, #0
    mov     x11, #0
    mov     x12, #0
    cbz     x16, 2f
    .p2align 4
1:
    ldrb    w14, [x6], #1
    ldr     d1, [x13, x14, lsl #3]          // lanes of the 3-byte offsets
    ldrb    w15, [x17, x14]
    eor     w14, w14, #255
    ldr     d0, [x13, x14, lsl #3]          // lanes of the 2-byte offsets
    ldr     d2, [x1, x11]
    ldr     d3, [x3, x12]
    tbl     v2.16b, {v2.16b}, v0.16b
    tbl     v3.16b, {v3.16b}, v1.16b
    orr     v2.16b, v2.16b, v3.16b
    str     d2, [x7, x10]
    ldr     d2, [x2, x11]
    ldr     d3, [x4, x12]
    tbl     v2.16b, {v2.16b}, v0.16b
    tbl     v3.16b, {v3.16b}, v1.16b
    orr     v2.16b, v2.16b, v3.16b
    str     d2, [x8, x10]
    ldr     d3, [x5, x12]
    tbl     v3.16b, {v3.16b}, v1.16b
    str     d3, [x9, x10]
    add     x12, x12, x15
    add     x11, x11, #8
    sub     x11, x11, x15
    add     x10, x10, #8
    cmp     x10, x16
    b.lo    1b
2:
    ldp     x2, x3, [x0, #80]
    ldr     x4, [x0, #96]

    // 1-byte offsets and the planes above -> int32
L(merge_widen):
    ldr     x1, [x0]
    ldr     x6, [x0, #48]
    ldr     x16, [x0, #64]
    ldr     x7, [x0, #104]
    mov     x10, #0
    mov     x11, #0
    mov     x12, #0
    cbz     x16, L(merge_done)
    cbz     x4, L(merge_widen_2)
    MERGE_WIDEN 3
L(merge_widen_2):
    ldr     x9, [x0, #72]
    cbz     x9, L(merge_widen_1)
    MERGE_WIDEN 2
    // 1-byte offsets only
L(merge_widen_1):
    MERGE_WIDEN 1
L(merge_done):
    sminv   s0, v6.4s
    fmov    w0, s0
    sxtw    x0, w0
    ret
    ENDFUNC SYM(KERNEL_MERGE)

#if !defined(__APPLE__)
    // keep the stack non-executable
    .section .note.GNU-stack,"",%progbits
#endif

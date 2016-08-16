/*
 * Copyright (C) 2013-2015 Willy Tarreau <w@1wt.eu>
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#ifndef _WIN32
    #include <sys/mman.h>
    #include <sys/user.h>
#endif
#include "slz.h"

/* First, RFC1951-specific declarations and extracts from the RFC.
 *
 * RFC1951 - deflate stream format


             * Data elements are packed into bytes in order of
               increasing bit number within the byte, i.e., starting
               with the least-significant bit of the byte.
             * Data elements other than Huffman codes are packed
               starting with the least-significant bit of the data
               element.
             * Huffman codes are packed starting with the most-
               significant bit of the code.

      3.2.3. Details of block format

         Each block of compressed data begins with 3 header bits
         containing the following data:

            first bit       BFINAL
            next 2 bits     BTYPE

         Note that the header bits do not necessarily begin on a byte
         boundary, since a block does not necessarily occupy an integral
         number of bytes.

         BFINAL is set if and only if this is the last block of the data
         set.

         BTYPE specifies how the data are compressed, as follows:

            00 - no compression
            01 - compressed with fixed Huffman codes
            10 - compressed with dynamic Huffman codes
            11 - reserved (error)

      3.2.4. Non-compressed blocks (BTYPE=00)

         Any bits of input up to the next byte boundary are ignored.
         The rest of the block consists of the following information:

              0   1   2   3   4...
            +---+---+---+---+================================+
            |  LEN  | NLEN  |... LEN bytes of literal data...|
            +---+---+---+---+================================+

         LEN is the number of data bytes in the block.  NLEN is the
         one's complement of LEN.

      3.2.5. Compressed blocks (length and distance codes)

         As noted above, encoded data blocks in the "deflate" format
         consist of sequences of symbols drawn from three conceptually
         distinct alphabets: either literal bytes, from the alphabet of
         byte values (0..255), or <length, backward distance> pairs,
         where the length is drawn from (3..258) and the distance is
         drawn from (1..32,768).  In fact, the literal and length
         alphabets are merged into a single alphabet (0..285), where
         values 0..255 represent literal bytes, the value 256 indicates
         end-of-block, and values 257..285 represent length codes
         (possibly in conjunction with extra bits following the symbol
         code) as follows:

Length encoding :
                Extra               Extra               Extra
            Code Bits Length(s) Code Bits Lengths   Code Bits Length(s)
            ---- ---- ------     ---- ---- -------   ---- ---- -------
             257   0     3       267   1   15,16     277   4   67-82
             258   0     4       268   1   17,18     278   4   83-98
             259   0     5       269   2   19-22     279   4   99-114
             260   0     6       270   2   23-26     280   4  115-130
             261   0     7       271   2   27-30     281   5  131-162
             262   0     8       272   2   31-34     282   5  163-194
             263   0     9       273   3   35-42     283   5  195-226
             264   0    10       274   3   43-50     284   5  227-257
             265   1  11,12      275   3   51-58     285   0    258
             266   1  13,14      276   3   59-66

Distance encoding :
                  Extra           Extra               Extra
             Code Bits Dist  Code Bits   Dist     Code Bits Distance
             ---- ---- ----  ---- ----  ------    ---- ---- --------
               0   0    1     10   4     33-48    20    9   1025-1536
               1   0    2     11   4     49-64    21    9   1537-2048
               2   0    3     12   5     65-96    22   10   2049-3072
               3   0    4     13   5     97-128   23   10   3073-4096
               4   1   5,6    14   6    129-192   24   11   4097-6144
               5   1   7,8    15   6    193-256   25   11   6145-8192
               6   2   9-12   16   7    257-384   26   12  8193-12288
               7   2  13-16   17   7    385-512   27   12 12289-16384
               8   3  17-24   18   8    513-768   28   13 16385-24576
               9   3  25-32   19   8   769-1024   29   13 24577-32768

      3.2.6. Compression with fixed Huffman codes (BTYPE=01)

         The Huffman codes for the two alphabets are fixed, and are not
         represented explicitly in the data.  The Huffman code lengths
         for the literal/length alphabet are:

                   Lit Value    Bits        Codes
                   ---------    ----        -----
                     0 - 143     8          00110000 through
                                            10111111
                   144 - 255     9          110010000 through
                                            111111111
                   256 - 279     7          0000000 through
                                            0010111
                   280 - 287     8          11000000 through
                                            11000111

         The code lengths are sufficient to generate the actual codes,
         as described above; we show the codes in the table for added
         clarity.  Literal/length values 286-287 will never actually
         occur in the compressed data, but participate in the code
         construction.

         Distance codes 0-31 are represented by (fixed-length) 5-bit
         codes, with possible additional bits as shown in the table
         shown in Paragraph 3.2.5, above.  Note that distance codes 30-
         31 will never actually occur in the compressed data.

*/

/* Length table for lengths 3 to 258, generated by mklen.sh
 * The entries contain :
 *   code - 257 = 0..28 in bits 0..4
 *   bits       = 0..5  in bits 5..7
 *   value      = 0..31 in bits 8..12
 */
static const uint16_t len_code[259] = {
	0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0002, 0x0003, 0x0004, //   0
	0x0005, 0x0006, 0x0007, 0x0028, 0x0128, 0x0029, 0x0129, 0x002a, //   8
	0x012a, 0x002b, 0x012b, 0x004c, 0x014c, 0x024c, 0x034c, 0x004d, //  16
	0x014d, 0x024d, 0x034d, 0x004e, 0x014e, 0x024e, 0x034e, 0x004f, //  24
	0x014f, 0x024f, 0x034f, 0x0070, 0x0170, 0x0270, 0x0370, 0x0470, //  32
	0x0570, 0x0670, 0x0770, 0x0071, 0x0171, 0x0271, 0x0371, 0x0471, //  40
	0x0571, 0x0671, 0x0771, 0x0072, 0x0172, 0x0272, 0x0372, 0x0472, //  48
	0x0572, 0x0672, 0x0772, 0x0073, 0x0173, 0x0273, 0x0373, 0x0473, //  56
	0x0573, 0x0673, 0x0773, 0x0094, 0x0194, 0x0294, 0x0394, 0x0494, //  64
	0x0594, 0x0694, 0x0794, 0x0894, 0x0994, 0x0a94, 0x0b94, 0x0c94, //  72
	0x0d94, 0x0e94, 0x0f94, 0x0095, 0x0195, 0x0295, 0x0395, 0x0495, //  80
	0x0595, 0x0695, 0x0795, 0x0895, 0x0995, 0x0a95, 0x0b95, 0x0c95, //  88
	0x0d95, 0x0e95, 0x0f95, 0x0096, 0x0196, 0x0296, 0x0396, 0x0496, //  96
	0x0596, 0x0696, 0x0796, 0x0896, 0x0996, 0x0a96, 0x0b96, 0x0c96, // 104
	0x0d96, 0x0e96, 0x0f96, 0x0097, 0x0197, 0x0297, 0x0397, 0x0497, // 112
	0x0597, 0x0697, 0x0797, 0x0897, 0x0997, 0x0a97, 0x0b97, 0x0c97, // 120
	0x0d97, 0x0e97, 0x0f97, 0x00b8, 0x01b8, 0x02b8, 0x03b8, 0x04b8, // 128
	0x05b8, 0x06b8, 0x07b8, 0x08b8, 0x09b8, 0x0ab8, 0x0bb8, 0x0cb8, // 136
	0x0db8, 0x0eb8, 0x0fb8, 0x10b8, 0x11b8, 0x12b8, 0x13b8, 0x14b8, // 144
	0x15b8, 0x16b8, 0x17b8, 0x18b8, 0x19b8, 0x1ab8, 0x1bb8, 0x1cb8, // 152
	0x1db8, 0x1eb8, 0x1fb8, 0x00b9, 0x01b9, 0x02b9, 0x03b9, 0x04b9, // 160
	0x05b9, 0x06b9, 0x07b9, 0x08b9, 0x09b9, 0x0ab9, 0x0bb9, 0x0cb9, // 168
	0x0db9, 0x0eb9, 0x0fb9, 0x10b9, 0x11b9, 0x12b9, 0x13b9, 0x14b9, // 176
	0x15b9, 0x16b9, 0x17b9, 0x18b9, 0x19b9, 0x1ab9, 0x1bb9, 0x1cb9, // 184
	0x1db9, 0x1eb9, 0x1fb9, 0x00ba, 0x01ba, 0x02ba, 0x03ba, 0x04ba, // 192
	0x05ba, 0x06ba, 0x07ba, 0x08ba, 0x09ba, 0x0aba, 0x0bba, 0x0cba, // 200
	0x0dba, 0x0eba, 0x0fba, 0x10ba, 0x11ba, 0x12ba, 0x13ba, 0x14ba, // 208
	0x15ba, 0x16ba, 0x17ba, 0x18ba, 0x19ba, 0x1aba, 0x1bba, 0x1cba, // 216
	0x1dba, 0x1eba, 0x1fba, 0x00bb, 0x01bb, 0x02bb, 0x03bb, 0x04bb, // 224
	0x05bb, 0x06bb, 0x07bb, 0x08bb, 0x09bb, 0x0abb, 0x0bbb, 0x0cbb, // 232
	0x0dbb, 0x0ebb, 0x0fbb, 0x10bb, 0x11bb, 0x12bb, 0x13bb, 0x14bb, // 240
	0x15bb, 0x16bb, 0x17bb, 0x18bb, 0x19bb, 0x1abb, 0x1bbb, 0x1cbb, // 248
	0x1dbb, 0x1ebb, 0x001c					        // 256
};

/* Distance codes are stored on 5 bits reversed. The RFC doesn't state that
 * they are reversed, but it's the only way it works.
 */
static const uint8_t dist_codes[32] = {
	0, 16, 8, 24, 4, 20, 12, 28,
	2, 18, 10, 26, 6, 22, 14, 30,
	1, 17, 9, 25, 5, 21, 13, 29,
	3, 19, 11, 27, 7, 23, 15, 31
};

/* Fixed Huffman table as per RFC1951.
 *
 *       Lit Value    Bits        Codes
 *       ---------    ----        -----
 *         0 - 143     8          00110000 through  10111111
 *       144 - 255     9         110010000 through 111111111
 *       256 - 279     7           0000000 through   0010111
 *       280 - 287     8          11000000 through  11000111
 *
 * The codes are encoded in reverse, the high bit of the code appears encoded
 * as bit 0. The table is built by mkhuff.sh. The 16 bits are encoded this way :
 *  - bits 0..3  : bits
 *  - bits 4..12 : code
 */
static const uint16_t fixed_huff[288] = {
	0x00c8, 0x08c8, 0x04c8, 0x0cc8, 0x02c8, 0x0ac8, 0x06c8, 0x0ec8, //   0
	0x01c8, 0x09c8, 0x05c8, 0x0dc8, 0x03c8, 0x0bc8, 0x07c8, 0x0fc8, //   8
	0x0028, 0x0828, 0x0428, 0x0c28, 0x0228, 0x0a28, 0x0628, 0x0e28, //  16
	0x0128, 0x0928, 0x0528, 0x0d28, 0x0328, 0x0b28, 0x0728, 0x0f28, //  24
	0x00a8, 0x08a8, 0x04a8, 0x0ca8, 0x02a8, 0x0aa8, 0x06a8, 0x0ea8, //  32
	0x01a8, 0x09a8, 0x05a8, 0x0da8, 0x03a8, 0x0ba8, 0x07a8, 0x0fa8, //  40
	0x0068, 0x0868, 0x0468, 0x0c68, 0x0268, 0x0a68, 0x0668, 0x0e68, //  48
	0x0168, 0x0968, 0x0568, 0x0d68, 0x0368, 0x0b68, 0x0768, 0x0f68, //  56
	0x00e8, 0x08e8, 0x04e8, 0x0ce8, 0x02e8, 0x0ae8, 0x06e8, 0x0ee8, //  64
	0x01e8, 0x09e8, 0x05e8, 0x0de8, 0x03e8, 0x0be8, 0x07e8, 0x0fe8, //  72
	0x0018, 0x0818, 0x0418, 0x0c18, 0x0218, 0x0a18, 0x0618, 0x0e18, //  80
	0x0118, 0x0918, 0x0518, 0x0d18, 0x0318, 0x0b18, 0x0718, 0x0f18, //  88
	0x0098, 0x0898, 0x0498, 0x0c98, 0x0298, 0x0a98, 0x0698, 0x0e98, //  96
	0x0198, 0x0998, 0x0598, 0x0d98, 0x0398, 0x0b98, 0x0798, 0x0f98, // 104
	0x0058, 0x0858, 0x0458, 0x0c58, 0x0258, 0x0a58, 0x0658, 0x0e58, // 112
	0x0158, 0x0958, 0x0558, 0x0d58, 0x0358, 0x0b58, 0x0758, 0x0f58, // 120
	0x00d8, 0x08d8, 0x04d8, 0x0cd8, 0x02d8, 0x0ad8, 0x06d8, 0x0ed8, // 128
	0x01d8, 0x09d8, 0x05d8, 0x0dd8, 0x03d8, 0x0bd8, 0x07d8, 0x0fd8, // 136
	0x0139, 0x1139, 0x0939, 0x1939, 0x0539, 0x1539, 0x0d39, 0x1d39, // 144
	0x0339, 0x1339, 0x0b39, 0x1b39, 0x0739, 0x1739, 0x0f39, 0x1f39, // 152
	0x00b9, 0x10b9, 0x08b9, 0x18b9, 0x04b9, 0x14b9, 0x0cb9, 0x1cb9, // 160
	0x02b9, 0x12b9, 0x0ab9, 0x1ab9, 0x06b9, 0x16b9, 0x0eb9, 0x1eb9, // 168
	0x01b9, 0x11b9, 0x09b9, 0x19b9, 0x05b9, 0x15b9, 0x0db9, 0x1db9, // 176
	0x03b9, 0x13b9, 0x0bb9, 0x1bb9, 0x07b9, 0x17b9, 0x0fb9, 0x1fb9, // 184
	0x0079, 0x1079, 0x0879, 0x1879, 0x0479, 0x1479, 0x0c79, 0x1c79, // 192
	0x0279, 0x1279, 0x0a79, 0x1a79, 0x0679, 0x1679, 0x0e79, 0x1e79, // 200
	0x0179, 0x1179, 0x0979, 0x1979, 0x0579, 0x1579, 0x0d79, 0x1d79, // 208
	0x0379, 0x1379, 0x0b79, 0x1b79, 0x0779, 0x1779, 0x0f79, 0x1f79, // 216
	0x00f9, 0x10f9, 0x08f9, 0x18f9, 0x04f9, 0x14f9, 0x0cf9, 0x1cf9, // 224
	0x02f9, 0x12f9, 0x0af9, 0x1af9, 0x06f9, 0x16f9, 0x0ef9, 0x1ef9, // 232
	0x01f9, 0x11f9, 0x09f9, 0x19f9, 0x05f9, 0x15f9, 0x0df9, 0x1df9, // 240
	0x03f9, 0x13f9, 0x0bf9, 0x1bf9, 0x07f9, 0x17f9, 0x0ff9, 0x1ff9, // 248
	0x0007, 0x0407, 0x0207, 0x0607, 0x0107, 0x0507, 0x0307, 0x0707, // 256
	0x0087, 0x0487, 0x0287, 0x0687, 0x0187, 0x0587, 0x0387, 0x0787, // 264
	0x0047, 0x0447, 0x0247, 0x0647, 0x0147, 0x0547, 0x0347, 0x0747, // 272
	0x0038, 0x0838, 0x0438, 0x0c38, 0x0238, 0x0a38, 0x0638, 0x0e38  // 280
};

/* length from 3 to 258 converted to bit strings for use with fixed huffman
 * coding. It was built by tools/dump_len.c. The format is the following :
 *   - bits 0..15  = code
 *   - bits 16..19 = #bits
 */
static const uint32_t len_fh[259] = {
	0x000000,  0x000000,  0x000000,  0x070040,   /* 0-3 */
	0x070020,  0x070060,  0x070010,  0x070050,   /* 4-7 */
	0x070030,  0x070070,  0x070008,  0x080048,   /* 8-11 */
	0x0800c8,  0x080028,  0x0800a8,  0x080068,   /* 12-15 */
	0x0800e8,  0x080018,  0x080098,  0x090058,   /* 16-19 */
	0x0900d8,  0x090158,  0x0901d8,  0x090038,   /* 20-23 */
	0x0900b8,  0x090138,  0x0901b8,  0x090078,   /* 24-27 */
	0x0900f8,  0x090178,  0x0901f8,  0x090004,   /* 28-31 */
	0x090084,  0x090104,  0x090184,  0x0a0044,   /* 32-35 */
	0x0a00c4,  0x0a0144,  0x0a01c4,  0x0a0244,   /* 36-39 */
	0x0a02c4,  0x0a0344,  0x0a03c4,  0x0a0024,   /* 40-43 */
	0x0a00a4,  0x0a0124,  0x0a01a4,  0x0a0224,   /* 44-47 */
	0x0a02a4,  0x0a0324,  0x0a03a4,  0x0a0064,   /* 48-51 */
	0x0a00e4,  0x0a0164,  0x0a01e4,  0x0a0264,   /* 52-55 */
	0x0a02e4,  0x0a0364,  0x0a03e4,  0x0a0014,   /* 56-59 */
	0x0a0094,  0x0a0114,  0x0a0194,  0x0a0214,   /* 60-63 */
	0x0a0294,  0x0a0314,  0x0a0394,  0x0b0054,   /* 64-67 */
	0x0b00d4,  0x0b0154,  0x0b01d4,  0x0b0254,   /* 68-71 */
	0x0b02d4,  0x0b0354,  0x0b03d4,  0x0b0454,   /* 72-75 */
	0x0b04d4,  0x0b0554,  0x0b05d4,  0x0b0654,   /* 76-79 */
	0x0b06d4,  0x0b0754,  0x0b07d4,  0x0b0034,   /* 80-83 */
	0x0b00b4,  0x0b0134,  0x0b01b4,  0x0b0234,   /* 84-87 */
	0x0b02b4,  0x0b0334,  0x0b03b4,  0x0b0434,   /* 88-91 */
	0x0b04b4,  0x0b0534,  0x0b05b4,  0x0b0634,   /* 92-95 */
	0x0b06b4,  0x0b0734,  0x0b07b4,  0x0b0074,   /* 96-99 */
	0x0b00f4,  0x0b0174,  0x0b01f4,  0x0b0274,   /* 100-103 */
	0x0b02f4,  0x0b0374,  0x0b03f4,  0x0b0474,   /* 104-107 */
	0x0b04f4,  0x0b0574,  0x0b05f4,  0x0b0674,   /* 108-111 */
	0x0b06f4,  0x0b0774,  0x0b07f4,  0x0c0003,   /* 112-115 */
	0x0c0103,  0x0c0203,  0x0c0303,  0x0c0403,   /* 116-119 */
	0x0c0503,  0x0c0603,  0x0c0703,  0x0c0803,   /* 120-123 */
	0x0c0903,  0x0c0a03,  0x0c0b03,  0x0c0c03,   /* 124-127 */
	0x0c0d03,  0x0c0e03,  0x0c0f03,  0x0d0083,   /* 128-131 */
	0x0d0183,  0x0d0283,  0x0d0383,  0x0d0483,   /* 132-135 */
	0x0d0583,  0x0d0683,  0x0d0783,  0x0d0883,   /* 136-139 */
	0x0d0983,  0x0d0a83,  0x0d0b83,  0x0d0c83,   /* 140-143 */
	0x0d0d83,  0x0d0e83,  0x0d0f83,  0x0d1083,   /* 144-147 */
	0x0d1183,  0x0d1283,  0x0d1383,  0x0d1483,   /* 148-151 */
	0x0d1583,  0x0d1683,  0x0d1783,  0x0d1883,   /* 152-155 */
	0x0d1983,  0x0d1a83,  0x0d1b83,  0x0d1c83,   /* 156-159 */
	0x0d1d83,  0x0d1e83,  0x0d1f83,  0x0d0043,   /* 160-163 */
	0x0d0143,  0x0d0243,  0x0d0343,  0x0d0443,   /* 164-167 */
	0x0d0543,  0x0d0643,  0x0d0743,  0x0d0843,   /* 168-171 */
	0x0d0943,  0x0d0a43,  0x0d0b43,  0x0d0c43,   /* 172-175 */
	0x0d0d43,  0x0d0e43,  0x0d0f43,  0x0d1043,   /* 176-179 */
	0x0d1143,  0x0d1243,  0x0d1343,  0x0d1443,   /* 180-183 */
	0x0d1543,  0x0d1643,  0x0d1743,  0x0d1843,   /* 184-187 */
	0x0d1943,  0x0d1a43,  0x0d1b43,  0x0d1c43,   /* 188-191 */
	0x0d1d43,  0x0d1e43,  0x0d1f43,  0x0d00c3,   /* 192-195 */
	0x0d01c3,  0x0d02c3,  0x0d03c3,  0x0d04c3,   /* 196-199 */
	0x0d05c3,  0x0d06c3,  0x0d07c3,  0x0d08c3,   /* 200-203 */
	0x0d09c3,  0x0d0ac3,  0x0d0bc3,  0x0d0cc3,   /* 204-207 */
	0x0d0dc3,  0x0d0ec3,  0x0d0fc3,  0x0d10c3,   /* 208-211 */
	0x0d11c3,  0x0d12c3,  0x0d13c3,  0x0d14c3,   /* 212-215 */
	0x0d15c3,  0x0d16c3,  0x0d17c3,  0x0d18c3,   /* 216-219 */
	0x0d19c3,  0x0d1ac3,  0x0d1bc3,  0x0d1cc3,   /* 220-223 */
	0x0d1dc3,  0x0d1ec3,  0x0d1fc3,  0x0d0023,   /* 224-227 */
	0x0d0123,  0x0d0223,  0x0d0323,  0x0d0423,   /* 228-231 */
	0x0d0523,  0x0d0623,  0x0d0723,  0x0d0823,   /* 232-235 */
	0x0d0923,  0x0d0a23,  0x0d0b23,  0x0d0c23,   /* 236-239 */
	0x0d0d23,  0x0d0e23,  0x0d0f23,  0x0d1023,   /* 240-243 */
	0x0d1123,  0x0d1223,  0x0d1323,  0x0d1423,   /* 244-247 */
	0x0d1523,  0x0d1623,  0x0d1723,  0x0d1823,   /* 248-251 */
	0x0d1923,  0x0d1a23,  0x0d1b23,  0x0d1c23,   /* 252-255 */
	0x0d1d23,  0x0d1e23,  0x0800a3               /* 256-258 */
};

/* Table of *inverted* CRC32 for each 8-bit quantity based on the position of
 * the byte being read relative to the last byte. Eg: [0] means we're on the
 * last byte, [1] on the previous one etc. These values have 8 inverted bits
 * at each position so that when processing 32-bit little endian quantities,
 * the CRC already appears inverted in each individual byte and doesn't need
 * to be inverted again in the loop.
 */
static uint32_t crc32_fast[4][256];
static uint32_t fh_dist_table[32768];

/* back references, built in a way that is optimal for 32/64 bits */
union ref {
	struct {
		uint32_t pos;
		uint32_t word;
	} by32;
	uint64_t by64;
};

/* Returns code for lengths 1 to 32768. The bit size for the next value can be
 * found this way :
 *
 *	bits = code >> 1;
 *	if (bits)
 *		bits--;
 *
 */
static uint32_t dist_to_code(uint32_t l)
{
	uint32_t code;

	code = 0;
	switch (l) {
	case 24577 ... 32768: code++;
	case 16385 ... 24576: code++;
	case 12289 ... 16384: code++;
	case 8193 ... 12288: code++;
	case 6145 ... 8192: code++;
	case 4097 ... 6144: code++;
	case 3073 ... 4096: code++;
	case 2049 ... 3072: code++;
	case 1537 ... 2048: code++;
	case 1025 ... 1536: code++;
	case 769 ... 1024: code++;
	case 513 ... 768: code++;
	case 385 ... 512: code++;
	case 257 ... 384: code++;
	case 193 ... 256: code++;
	case 129 ... 192: code++;
	case 97 ... 128: code++;
	case 65 ... 96: code++;
	case 49 ... 64: code++;
	case 33 ... 48: code++;
	case 25 ... 32: code++;
	case 17 ... 24: code++;
	case 13 ... 16: code++;
	case 9 ... 12: code++;
	case 7 ... 8: code++;
	case 5 ... 6: code++;
	case 4: code++;
	case 3: code++;
	case 2: code++;
	}

	return code;
}

/* enqueue code x of <xbits> bits (LSB aligned, at most 16) and copy complete
 * bytes into out buf. X must not contain non-zero bits above xbits. Prefer
 * enqueue8() when xbits is known for being 8 or less.
 */
static void enqueue16(struct slz_stream *strm, uint32_t x, uint32_t xbits)
{
	uint32_t queue = strm->queue + (x << strm->qbits);
	uint32_t qbits = strm->qbits + xbits;

	if (__builtin_expect(qbits < 16, 1)) {
		if (qbits >= 8) {
			/* usual case */
			qbits -= 8;
			*strm->outbuf++ = queue;
			queue >>= 8;
		}
		strm->qbits = qbits;
		strm->queue = queue;
		return;
	}
	/* case where we queue large codes after small ones, eg: 7 then 9 */

#ifndef UNALIGNED_LE_OK
	strm->outbuf[0] = queue;
	strm->outbuf[1] = queue >> 8;
#else
	*(uint16_t *)strm->outbuf = queue;
#endif
	strm->outbuf += 2;
	queue >>= 16;
	qbits -= 16;
	strm->qbits = qbits;
	strm->queue = queue;
}

/* enqueue code x of <xbits> bits (at most 8) and copy complete bytes into
 * out buf. X must not contain non-zero bits above xbits.
 */
static inline void enqueue8(struct slz_stream *strm, uint32_t x, uint32_t xbits)
{
	uint32_t queue = strm->queue + (x << strm->qbits);
	uint32_t qbits = strm->qbits + xbits;

	if (__builtin_expect((signed)(qbits - 8) >= 0, 1)) {
		qbits -= 8;
		*strm->outbuf++ = queue;
		queue >>= 8;
	}

	strm->qbits = qbits;
	strm->queue = queue;
}

/* align to next byte */
static inline void flush_bits(struct slz_stream *strm)
{
	if (strm->qbits) {
		*strm->outbuf++ = strm->queue;
		strm->queue = 0;
		strm->qbits = 0;
	}
}

/* only valid if buffer is already aligned */
static inline void copy_8b(struct slz_stream *strm, uint32_t x)
{
	*strm->outbuf++ = x;
}

/* only valid if buffer is already aligned */
static inline void copy_16b(struct slz_stream *strm, uint32_t x)
{
	strm->outbuf[0] = x;
	strm->outbuf[1] = x >> 8;
	strm->outbuf += 2;
}

/* only valid if buffer is already aligned */
static inline void copy_32b(struct slz_stream *strm, uint32_t x)
{
	strm->outbuf[0] = x;
	strm->outbuf[1] = x >> 8;
	strm->outbuf[2] = x >> 16;
	strm->outbuf[3] = x >> 24;
	strm->outbuf += 4;
}

static inline void send_huff(struct slz_stream *strm, uint32_t code)
{
	uint32_t bits;

	code = fixed_huff[code];
	bits = code & 15;
	code >>= 4;
	enqueue16(strm, code, bits);
}

static inline void send_eob(struct slz_stream *strm)
{
	send_huff(strm, 256); // cf rfc1951: 256 = EOB
}

/* copies at most <len> litterals from <buf>, returns the amount of data
 * copied. <more> indicates that there are data past buf + <len>. It must not
 * be called with len <= 0.
 */
static unsigned int copy_lit(struct slz_stream *strm, const void *buf, int len, int more)
{
	if (len > 65535) {
		len = 65535;
		more = 1;
	}

	if (strm->state != SLZ_ST_EOB)
		send_eob(strm);

	strm->state = more ? SLZ_ST_EOB : SLZ_ST_DONE;

	enqueue8(strm, !more, 3); // BFINAL = !more ; BTYPE = 00
	flush_bits(strm);
	copy_16b(strm, len);  // len
	copy_16b(strm, ~len); // nlen
	memcpy(strm->outbuf, buf, len);
	strm->outbuf += len;
	return len;
}

/* copies at most <len> litterals from <buf>, returns the amount of data
 * copied. <more> indicates that there are data past buf + <len>. It must not
 * be called with len <= 0.
 */
static unsigned int copy_lit_huff(struct slz_stream *strm, const unsigned char *buf, int len, int more)
{
	uint32_t pos;

	/* This ugly construct limits the mount of tests and optimizes for the
	 * most common case (more > 0).
	 */
	if (strm->state == SLZ_ST_EOB) {
	eob:
		strm->state = more ? SLZ_ST_FIXED : SLZ_ST_LAST;
		enqueue8(strm, 2 + !more, 3); // BFINAL = !more ; BTYPE = 01
	}
	else if (!more) {
		send_eob(strm);
		goto eob;
	}

	pos = 0;
	while (pos < len) {
		send_huff(strm, buf[pos++]);
	}
	return len;
}

/* format:
 * bit0..31  = word
 * bit32..63 = last position in buffer of similar content
 */

/* This hash provides good average results on HTML contents, and is among the
 * few which provide almost optimal results on various different pages.
 */
static inline uint32_t slz_hash(uint32_t a)
{
	return ((a << 19) + (a << 6) - a) >> (32 - HASH_BITS);
}

/* This function compares buffers <a> and <b> and reads 32 or 64 bits at a time
 * during the approach. It makes us of unaligned little endian memory accesses
 * on capable architectures. <max> is the maximum number of bytes that can be
 * read, so both <a> and <b> must have at least <max> bytes ahead. <max> may
 * safely be null or negative if that simplifies computations in the caller.
 */
static inline long memmatch(const unsigned char *a, const unsigned char *b, long max)
{
	long len = 0;

#ifdef UNALIGNED_LE_OK
	unsigned long xor;

	while (1) {
		if (len + 2 * sizeof(long) > max) {
			while (len < max) {
				if (a[len] != b[len])
					break;
				len++;
			}
			return len;
		}

		xor = *(long *)&a[len] ^ *(long *)&b[len];
		if (xor)
			break;
		len += sizeof(long);

		xor = *(long *)&a[len] ^ *(long *)&b[len];
		if (xor)
			break;
		len += sizeof(long);
	}

	if (sizeof(long) > 4 && !(xor & 0xffffffff)) {
		/* This code is optimized out on 32-bit archs, but we still
		 * need to shift in two passes to avoid a warning. It is
		 * properly optimized out as a single shift.
		 */
		xor >>= 16; xor >>= 16;
		if (xor & 0xffff) {
			if (xor & 0xff)
				return len + 4;
			return len + 5;
		}
		if (xor & 0xffffff)
			return len + 6;
		return len + 7;
	}

	if (xor & 0xffff) {
		if (xor & 0xff)
			return len;
		return len + 1;
	}
	if (xor & 0xffffff)
		return len + 2;
	return len + 3;

#else // UNALIGNED_LE_OK
	/* This is the generic version for big endian or unaligned-incompatible
	 * architectures.
	 */
	while (len < max) {
		if (a[len] != b[len])
			break;
		len++;
	}
	return len;

#endif
}

/* sets <count> BYTES to -32769 in <refs> so that any uninitialized entry will
 * verify (pos-last-1 >= 32768) and be ignored. <count> must be a multiple of
 * 128 bytes and <refs> must be at least one count in length. It's supposed to
 * be applied to 64-bit aligned data exclusively, which makes it slightly
 * faster than the regular memset() since no alignment check is performed.
 */
void reset_refs(union ref *refs, long count)
{
	/* avoid a shift/mask by casting to void* */
	union ref *end = (void *)refs + count;

	do {
		refs[ 0].by64 = -32769;
		refs[ 1].by64 = -32769;
		refs[ 2].by64 = -32769;
		refs[ 3].by64 = -32769;
		refs[ 4].by64 = -32769;
		refs[ 5].by64 = -32769;
		refs[ 6].by64 = -32769;
		refs[ 7].by64 = -32769;
		refs[ 8].by64 = -32769;
		refs[ 9].by64 = -32769;
		refs[10].by64 = -32769;
		refs[11].by64 = -32769;
		refs[12].by64 = -32769;
		refs[13].by64 = -32769;
		refs[14].by64 = -32769;
		refs[15].by64 = -32769;
		refs += 16;
	} while (refs < end);
}

/* Compresses <ilen> bytes from <in> into <out> according to RFC1951. The
 * output result may be up to 5 bytes larger than the input, to which 2 extra
 * bytes may be added to send the last chunk due to BFINAL+EOB encoding (10
 * bits) when <more> is not set. The caller is responsible for ensuring there
 * is enough room in the output buffer for this. The amount of output bytes is
 * returned, and no CRC is computed.
 */
long slz_rfc1951_encode(struct slz_stream *strm, unsigned char *out, const unsigned char *in, long ilen, int more)
{
	long rem = ilen;
	unsigned long pos = 0;
	unsigned long last;
	uint32_t word = 0;
	long mlen;
	uint32_t h;
	uint64_t ent;

	uint32_t len;
	uint32_t plit = 0;
	uint32_t bit9 = 0;
	uint32_t dist, code;
	union ref refs[1 << HASH_BITS];

	if (!strm->level) {
		/* force to send as literals (eg to preserve CPU) */
		strm->outbuf = out;
		plit = pos = ilen;
		bit9 = 52; /* force literal dump */
		goto final_lit_dump;
	}

	reset_refs(refs, sizeof(refs));

	strm->outbuf = out;

#ifndef UNALIGNED_FASTER
	word = ((unsigned char)in[pos] << 8) + ((unsigned char)in[pos + 1] << 16) + ((unsigned char)in[pos + 2] << 24);
#endif
	while (rem >= 4) {
#ifndef UNALIGNED_FASTER
		word = ((unsigned char)in[pos + 3] << 24) + (word >> 8);
#else
		word = *(uint32_t *)&in[pos];
#endif
		h = slz_hash(word);
		__asm volatile ("" ::); // prevent gcc from trying to be smart with the prefetch

		if (sizeof(long) >= 8) {
			ent = refs[h].by64;
			last = (uint32_t)ent;
			ent >>= 32;
			refs[h].by64 = ((uint64_t)pos) + ((uint64_t)word << 32);
		} else {
			ent  = refs[h].by32.word;
			last = refs[h].by32.pos;
			refs[h].by32.pos = pos;
			refs[h].by32.word = word;
		}

#if FIND_OPTIMAL_MATCH
		/* Experimental code to see what could be saved with an ideal
		 * longest match lookup algorithm. This one is very slow but
		 * scans the whole window. In short, here are the savings :
		 *   file        orig     fast(ratio)  optimal(ratio)
		 *  README       5185    3419 (65.9%)    3165 (61.0%)  -7.5%
		 *  index.html  76799   35662 (46.4%)   29875 (38.9%) -16.3%
		 *  rfc1952.c   29383   13442 (45.7%)   11793 (40.1%) -12.3%
		 *
		 * Thus the savings to expect for large files is at best 16%.
		 *
		 * A non-colliding hash gives 33025 instead of 35662 (-7.4%),
		 * and keeping the last two entries gives 31724 (-11.0%).
		 */
		unsigned long scan;
		int saved = 0;
		int bestpos = 0;
		int bestlen = 0;
		int firstlen = 0;
		int max_lookup = 2; // 0 = no limit

		for (scan = pos - 1; scan < pos && (unsigned long)(pos - scan - 1) < 32768; scan--) {
			if (*(uint32_t *)(in + scan) != word)
				continue;

			len = memmatch(in + pos, in + scan, rem);
			if (!bestlen)
				firstlen = len;

			if (len > bestlen) {
				bestlen = len;
				bestpos = scan;
			}
			if (!--max_lookup)
				break;
		}
		if (bestlen) {
			//printf("pos=%d last=%d bestpos=%d word=%08x ent=%08x len=%d\n",
			//       (int)pos, (int)last, (int)bestpos, (int)word, (int)ent, bestlen);
			last = bestpos;
			ent  = word;
			saved += bestlen - firstlen;
		}
		//fprintf(stderr, "first=%d best=%d saved_total=%d\n", firstlen, bestlen, saved);
#endif

		if ((uint32_t)ent != word) {
		send_as_lit:
			rem--;
			plit++;
			bit9 += ((unsigned char)word >= 144);
			pos++;
			continue;
		}

		/* We reject pos = last and pos > last+32768 */
		if ((unsigned long)(pos - last - 1) >= 32768)
			goto send_as_lit;

		/* Note: cannot encode a length larger than 258 bytes */
		mlen = memmatch(in + pos + 4, in + last + 4, (rem > 258 ? 258 : rem) - 4) + 4;

		/* found a matching entry */

		if (bit9 >= 52 && mlen < 6)
			goto send_as_lit;

		/* compute the output code, its size and the length's size in
		 * bits to know if the reference is cheaper than literals.
		 */
		code = len_fh[mlen];

		/* direct mapping of dist->huffman code */
		dist = fh_dist_table[pos - last - 1];

		/* if encoding the dist+length is more expensive than sending
		 * the equivalent as bytes, lets keep the literals.
		 */
		if ((dist & 0x1f) + (code >> 16) + 8 >= 8 * mlen + bit9)
			goto send_as_lit;

		/* first, copy pending literals */
		while (plit) {
			/* Huffman encoding requires 9 bits for octets 144..255, so this
			 * is a waste of space for binary data. Switching between Huffman
			 * and no-comp then huffman consumes 52 bits (7 for EOB + 3 for
			 * block type + 7 for alignment + 32 for LEN+NLEN + 3 for next
			 * block. Only use plain literals if there are more than 52 bits
			 * to save then.
			 */
			if (bit9 >= 52)
				len = copy_lit(strm, in + pos - plit, plit, 1);
			else
				len = copy_lit_huff(strm, in + pos - plit, plit, 1);

			plit -= len;
		}

		/* use mode 01 - fixed huffman */
		if (strm->state == SLZ_ST_EOB) {
			strm->state = SLZ_ST_FIXED;
			enqueue8(strm, 0x02, 3); // BTYPE = 01, BFINAL = 0
		}

		/* copy the length first */
		enqueue16(strm, code & 0xFFFF, code >> 16);

		/* in fixed huffman mode, dist is fixed 5 bits */
		enqueue16(strm, dist >> 5, dist & 0x1f);
		bit9 = 0;
		rem -= mlen;
		pos += mlen;

#ifndef UNALIGNED_FASTER
#ifdef UNALIGNED_LE_OK
		word = *(uint32_t *)&in[pos - 1];
#else
		word = ((unsigned char)in[pos] << 8) + ((unsigned char)in[pos + 1] << 16) + ((unsigned char)in[pos + 2] << 24);
#endif
#endif
	}

	if (__builtin_expect(rem, 0)) {
		/* we're reading the 1..3 last bytes */
		plit += rem;
		do {
			bit9 += ((unsigned char)in[pos++] >= 144);
		} while (--rem);
	}

 final_lit_dump:
	/* now copy remaining literals or mark the end */
	while (plit) {
		if (bit9 >= 52)
			len = copy_lit(strm, in + pos - plit, plit, more);
		else
			len = copy_lit_huff(strm, in + pos - plit, plit, more);

		plit -= len;
	}

	strm->ilen += ilen;
	return strm->outbuf - out;
}

/* Initializes stream <strm> for use with raw deflate (rfc1951). The CRC is
 * unused but set to zero. The compression level passed in <level> is set. This
 * value can only be 0 (no compression) or 1 (compression) and other values
 * will lead to unpredictable behaviour. The function always returns 0.
 */
int slz_rfc1951_init(struct slz_stream *strm, int level)
{
	strm->state = SLZ_ST_EOB; // no header
	strm->level = level;
	strm->format = SLZ_FMT_DEFLATE;
	strm->crc32 = 0;
	strm->ilen  = 0;
	strm->qbits = 0;
	strm->queue = 0;
	return 0;
}

/* Flushes any pending for stream <strm> into buffer <buf>, then sends BTYPE=1
 * and BFINAL=1 if needed. The stream ends in SLZ_ST_DONE. It returns the number
 * of bytes emitted. The trailer consists in flushing the possibly pending bits
 * from the queue (up to 7 bits), then possibly EOB (7 bits), then 3 bits, EOB,
 * a rounding to the next byte, which amounts to a total of 4 bytes max, that
 * the caller must ensure are available before calling the function.
 */
int slz_rfc1951_finish(struct slz_stream *strm, unsigned char *buf)
{
	strm->outbuf = buf;

	if (strm->state == SLZ_ST_FIXED || strm->state == SLZ_ST_LAST) {
		strm->state = (strm->state == SLZ_ST_LAST) ? SLZ_ST_DONE : SLZ_ST_EOB;
		send_eob(strm);
	}

	if (strm->state != SLZ_ST_DONE) {
		/* send BTYPE=1, BFINAL=1 */
		enqueue8(strm, 3, 3);
		send_eob(strm);
		strm->state = SLZ_ST_DONE;
	}

	flush_bits(strm);
	return strm->outbuf - buf;
}

/* not thread-safe, must be called exactly once */
void slz_prepare_dist_table()
{
	uint32_t dist;
	uint32_t code;
	uint32_t bits;

	for (dist = 0; dist < sizeof(fh_dist_table) / sizeof(*fh_dist_table); dist++) {
		code = dist_to_code(dist + 1);
		bits = code >> 1;
		if (bits)
			bits--;

		code = dist_codes[code];
		code += (dist & ((1 << bits) - 1)) << 5;
		fh_dist_table[dist] = (code << 5) + bits + 5;
	}
}

/* Now RFC1952-specific declarations and extracts from RFC.
 * From RFC1952 about the GZIP file format :

A gzip file consists of a series of "members" ...

2.3. Member format

      Each member has the following structure:

         +---+---+---+---+---+---+---+---+---+---+
         |ID1|ID2|CM |FLG|     MTIME     |XFL|OS | (more-->)
         +---+---+---+---+---+---+---+---+---+---+

      (if FLG.FEXTRA set)

         +---+---+=================================+
         | XLEN  |...XLEN bytes of "extra field"...| (more-->)
         +---+---+=================================+

      (if FLG.FNAME set)

         +=========================================+
         |...original file name, zero-terminated...| (more-->)
         +=========================================+

      (if FLG.FCOMMENT set)

         +===================================+
         |...file comment, zero-terminated...| (more-->)
         +===================================+

      (if FLG.FHCRC set)

         +---+---+
         | CRC16 |
         +---+---+

         +=======================+
         |...compressed blocks...| (more-->)
         +=======================+

           0   1   2   3   4   5   6   7
         +---+---+---+---+---+---+---+---+
         |     CRC32     |     ISIZE     |
         +---+---+---+---+---+---+---+---+


2.3.1. Member header and trailer

         ID1 (IDentification 1)
         ID2 (IDentification 2)
            These have the fixed values ID1 = 31 (0x1f, \037), ID2 = 139
            (0x8b, \213), to identify the file as being in gzip format.

         CM (Compression Method)
            This identifies the compression method used in the file.  CM
            = 0-7 are reserved.  CM = 8 denotes the "deflate"
            compression method, which is the one customarily used by
            gzip and which is documented elsewhere.

         FLG (FLaGs)
            This flag byte is divided into individual bits as follows:

               bit 0   FTEXT
               bit 1   FHCRC
               bit 2   FEXTRA
               bit 3   FNAME
               bit 4   FCOMMENT
               bit 5   reserved
               bit 6   reserved
               bit 7   reserved

            Reserved FLG bits must be zero.

         MTIME (Modification TIME)
            This gives the most recent modification time of the original
            file being compressed.  The time is in Unix format, i.e.,
            seconds since 00:00:00 GMT, Jan.  1, 1970.  (Note that this
            may cause problems for MS-DOS and other systems that use
            local rather than Universal time.)  If the compressed data
            did not come from a file, MTIME is set to the time at which
            compression started.  MTIME = 0 means no time stamp is
            available.

         XFL (eXtra FLags)
            These flags are available for use by specific compression
            methods.  The "deflate" method (CM = 8) sets these flags as
            follows:

               XFL = 2 - compressor used maximum compression,
                         slowest algorithm
               XFL = 4 - compressor used fastest algorithm

         OS (Operating System)
            This identifies the type of file system on which compression
            took place.  This may be useful in determining end-of-line
            convention for text files.  The currently defined values are
            as follows:

                 0 - FAT filesystem (MS-DOS, OS/2, NT/Win32)
                 1 - Amiga
                 2 - VMS (or OpenVMS)
                 3 - Unix
                 4 - VM/CMS
                 5 - Atari TOS
                 6 - HPFS filesystem (OS/2, NT)
                 7 - Macintosh
                 8 - Z-System
                 9 - CP/M
                10 - TOPS-20
                11 - NTFS filesystem (NT)
                12 - QDOS
                13 - Acorn RISCOS
               255 - unknown

 ==> A file compressed using "gzip -1" on Unix-like systems can be :

        1F 8B 08 00  00 00 00 00  04 03
        <deflate-compressed stream>
        crc32 size32
*/

static const unsigned char gzip_hdr[] = { 0x1F, 0x8B,   // ID1, ID2
                                          0x08, 0x00,   // Deflate, flags (none)
                                          0x00, 0x00, 0x00, 0x00, // mtime: none
                                          0x04, 0x03 }; // fastest comp, OS=Unix

/* Make the table for a fast CRC.
 * Not thread-safe, must be called exactly once.
 */
void slz_make_crc_table(void)
{
	uint32_t c;
	int n, k;

	for (n = 0; n < 256; n++) {
		c = (uint32_t) n ^ 255;
		for (k = 0; k < 8; k++) {
			if (c & 1) {
				c = 0xedb88320 ^ (c >> 1);
			} else {
				c = c >> 1;
			}
		}
		crc32_fast[0][n] = c ^ 0xff000000;
	}

	/* Note: here we *do not* have to invert the bits corresponding to the
	 * byte position, because [0] already has the 8 highest bits inverted,
	 * and these bits are shifted by 8 at the end of the operation, which
	 * results in having the next 8 bits shifted in turn. That's why we
	 * have the xor in the index used just after a computation.
	 */
	for (n = 0; n < 256; n++) {
		crc32_fast[1][n] = 0xff000000 ^ crc32_fast[0][(0xff000000 ^ crc32_fast[0][n] ^ 0xff) & 0xff] ^ (crc32_fast[0][n] >> 8);
		crc32_fast[2][n] = 0xff000000 ^ crc32_fast[0][(0x00ff0000 ^ crc32_fast[1][n] ^ 0xff) & 0xff] ^ (crc32_fast[1][n] >> 8);
		crc32_fast[3][n] = 0xff000000 ^ crc32_fast[0][(0x0000ff00 ^ crc32_fast[2][n] ^ 0xff) & 0xff] ^ (crc32_fast[2][n] >> 8);
	}
}

static inline uint32_t crc32_char(uint32_t crc, uint8_t x)
{
	return crc32_fast[0][(crc ^ x) & 0xff] ^ (crc >> 8);
}

static inline uint32_t crc32_uint32(uint32_t data)
{
	data = crc32_fast[3][(data >>  0) & 0xff] ^
	       crc32_fast[2][(data >>  8) & 0xff] ^
	       crc32_fast[1][(data >> 16) & 0xff] ^
	       crc32_fast[0][(data >> 24) & 0xff];
	return data;
}

/* Modified version originally from RFC1952, working with non-inverting CRCs */
uint32_t slz_crc32_by1(uint32_t crc, const unsigned char *buf, int len)
{
	int n;

	for (n = 0; n < len; n++)
		crc = crc32_char(crc, buf[n]);
	return crc;
}

/* This version computes the crc32 of <buf> over <len> bytes, doing most of it
 * in 32-bit chunks.
 */
uint32_t slz_crc32_by4(uint32_t crc, const unsigned char *buf, int len)
{
	const unsigned char *end = buf + len;

	while (buf <= end - 16) {
#ifdef UNALIGNED_LE_OK
		crc ^= *(uint32_t *)buf;
		crc = crc32_fast[3][(crc >>  0) & 0xff] ^
		      crc32_fast[2][(crc >>  8) & 0xff] ^
		      crc32_fast[1][(crc >> 16) & 0xff] ^
		      crc32_fast[0][(crc >> 24) & 0xff];

		crc ^= *(uint32_t *)(buf + 4);
		crc = crc32_fast[3][(crc >>  0) & 0xff] ^
		      crc32_fast[2][(crc >>  8) & 0xff] ^
		      crc32_fast[1][(crc >> 16) & 0xff] ^
		      crc32_fast[0][(crc >> 24) & 0xff];

		crc ^= *(uint32_t *)(buf + 8);
		crc = crc32_fast[3][(crc >>  0) & 0xff] ^
		      crc32_fast[2][(crc >>  8) & 0xff] ^
		      crc32_fast[1][(crc >> 16) & 0xff] ^
		      crc32_fast[0][(crc >> 24) & 0xff];

		crc ^= *(uint32_t *)(buf + 12);
		crc = crc32_fast[3][(crc >>  0) & 0xff] ^
		      crc32_fast[2][(crc >>  8) & 0xff] ^
		      crc32_fast[1][(crc >> 16) & 0xff] ^
		      crc32_fast[0][(crc >> 24) & 0xff];
#else
		crc = crc32_fast[3][(buf[0] ^ (crc >>  0)) & 0xff] ^
		      crc32_fast[2][(buf[1] ^ (crc >>  8)) & 0xff] ^
		      crc32_fast[1][(buf[2] ^ (crc >> 16)) & 0xff] ^
		      crc32_fast[0][(buf[3] ^ (crc >> 24)) & 0xff];

		crc = crc32_fast[3][(buf[4] ^ (crc >>  0)) & 0xff] ^
		      crc32_fast[2][(buf[5] ^ (crc >>  8)) & 0xff] ^
		      crc32_fast[1][(buf[6] ^ (crc >> 16)) & 0xff] ^
		      crc32_fast[0][(buf[7] ^ (crc >> 24)) & 0xff];

		crc = crc32_fast[3][(buf[8] ^ (crc >>  0)) & 0xff] ^
		      crc32_fast[2][(buf[9] ^ (crc >>  8)) & 0xff] ^
		      crc32_fast[1][(buf[10] ^ (crc >> 16)) & 0xff] ^
		      crc32_fast[0][(buf[11] ^ (crc >> 24)) & 0xff];

		crc = crc32_fast[3][(buf[12] ^ (crc >>  0)) & 0xff] ^
		      crc32_fast[2][(buf[13] ^ (crc >>  8)) & 0xff] ^
		      crc32_fast[1][(buf[14] ^ (crc >> 16)) & 0xff] ^
		      crc32_fast[0][(buf[15] ^ (crc >> 24)) & 0xff];
#endif
		buf += 16;
	}

	while (buf <= end - 4) {
#ifdef UNALIGNED_LE_OK
		crc ^= *(uint32_t *)buf;
		crc = crc32_fast[3][(crc >>  0) & 0xff] ^
		      crc32_fast[2][(crc >>  8) & 0xff] ^
		      crc32_fast[1][(crc >> 16) & 0xff] ^
		      crc32_fast[0][(crc >> 24) & 0xff];
#else
		crc = crc32_fast[3][(buf[0] ^ (crc >>  0)) & 0xff] ^
		      crc32_fast[2][(buf[1] ^ (crc >>  8)) & 0xff] ^
		      crc32_fast[1][(buf[2] ^ (crc >> 16)) & 0xff] ^
		      crc32_fast[0][(buf[3] ^ (crc >> 24)) & 0xff];
#endif
		buf += 4;
	}

	while (buf < end)
		crc = crc32_fast[0][(crc ^ *buf++) & 0xff] ^ (crc >> 8);
	return crc;
}

/* uses the most suitable crc32 function to update crc on <buf, len> */
static inline uint32_t update_crc(uint32_t crc, const void *buf, int len)
{
	return slz_crc32_by4(crc, buf, len);
}

/* Sends the gzip header for stream <strm> into buffer <buf>. When it's done,
 * the stream state is updated to SLZ_ST_EOB. It returns the number of bytes
 * emitted which is always 10. The caller is responsible for ensuring there's
 * always enough room in the buffer.
 */
int slz_rfc1952_send_header(struct slz_stream *strm, unsigned char *buf)
{
	memcpy(buf, gzip_hdr, sizeof(gzip_hdr));
	strm->state = SLZ_ST_EOB;
	return sizeof(gzip_hdr);
}

/* Encodes the block according to rfc1952. This means that the CRC of the input
 * block is computed according to the CRC32 algorithm. If the header was never
 * sent, it may be sent first. The number of output bytes is returned.
 */
long slz_rfc1952_encode(struct slz_stream *strm, unsigned char *out, const unsigned char *in, long ilen, int more)
{
	long ret = 0;

	if (__builtin_expect(strm->state == SLZ_ST_INIT, 0))
		ret += slz_rfc1952_send_header(strm, out);

	strm->crc32 = update_crc(strm->crc32, in, ilen);
	ret += slz_rfc1951_encode(strm, out + ret, in, ilen, more);
	return ret;
}

/* Initializes stream <strm> for use with the gzip format (rfc1952). The
 * compression level passed in <level> is set. This value can only be 0 (no
 * compression) or 1 (compression) and other values will lead to unpredictable
 * behaviour. The function always returns 0.
 */
int slz_rfc1952_init(struct slz_stream *strm, int level)
{
	strm->state  = SLZ_ST_INIT;
	strm->level  = level;
	strm->format = SLZ_FMT_GZIP;
	strm->crc32  = 0;
	strm->ilen   = 0;
	strm->qbits  = 0;
	strm->queue  = 0;
	return 0;
}

/* Flushes pending bits and sends the gzip trailer for stream <strm> into
 * buffer <buf>. When it's done, the stream state is updated to SLZ_ST_END. It
 * returns the number of bytes emitted. The trailer consists in flushing the
 * possibly pending bits from the queue (up to 24 bits), rounding to the next
 * byte, then 4 bytes for the CRC and another 4 bytes for the input length.
 * That may abount to 4+4+4 = 12 bytes, that the caller must ensure are
 * available before calling the function. Note that if the initial header was
 * never sent, it will be sent first as well (10 extra bytes).
 */
int slz_rfc1952_finish(struct slz_stream *strm, unsigned char *buf)
{
	strm->outbuf = buf;

	if (__builtin_expect(strm->state == SLZ_ST_INIT, 0))
		strm->outbuf += slz_rfc1952_send_header(strm, strm->outbuf);

	slz_rfc1951_finish(strm, strm->outbuf);
	copy_32b(strm, strm->crc32);
	copy_32b(strm, strm->ilen);
	strm->state = SLZ_ST_END;

	return strm->outbuf - buf;
}


/* RFC1950-specific stuff. This is for the Zlib stream format.
 * From RFC1950 (zlib) :
 *

   2.2. Data format

      A zlib stream has the following structure:

           0   1
         +---+---+
         |CMF|FLG|   (more-->)
         +---+---+


      (if FLG.FDICT set)

           0   1   2   3
         +---+---+---+---+
         |     DICTID    |   (more-->)
         +---+---+---+---+

         +=====================+---+---+---+---+
         |...compressed data...|    ADLER32    |
         +=====================+---+---+---+---+

      Any data which may appear after ADLER32 are not part of the zlib
      stream.

      CMF (Compression Method and flags)
         This byte is divided into a 4-bit compression method and a 4-
         bit information field depending on the compression method.

            bits 0 to 3  CM     Compression method
            bits 4 to 7  CINFO  Compression info

      CM (Compression method)
         This identifies the compression method used in the file. CM = 8
         denotes the "deflate" compression method with a window size up
         to 32K.  This is the method used by gzip and PNG (see
         references [1] and [2] in Chapter 3, below, for the reference
         documents).  CM = 15 is reserved.  It might be used in a future
         version of this specification to indicate the presence of an
         extra field before the compressed data.

      CINFO (Compression info)
         For CM = 8, CINFO is the base-2 logarithm of the LZ77 window
         size, minus eight (CINFO=7 indicates a 32K window size). Values
         of CINFO above 7 are not allowed in this version of the
         specification.  CINFO is not defined in this specification for
         CM not equal to 8.

      FLG (FLaGs)
         This flag byte is divided as follows:

            bits 0 to 4  FCHECK  (check bits for CMF and FLG)
            bit  5       FDICT   (preset dictionary)
            bits 6 to 7  FLEVEL  (compression level)

         The FCHECK value must be such that CMF and FLG, when viewed as
         a 16-bit unsigned integer stored in MSB order (CMF*256 + FLG),
         is a multiple of 31.


      FDICT (Preset dictionary)
         If FDICT is set, a DICT dictionary identifier is present
         immediately after the FLG byte. The dictionary is a sequence of
         bytes which are initially fed to the compressor without
         producing any compressed output. DICT is the Adler-32 checksum
         of this sequence of bytes (see the definition of ADLER32
         below).  The decompressor can use this identifier to determine
         which dictionary has been used by the compressor.

      FLEVEL (Compression level)
         These flags are available for use by specific compression
         methods.  The "deflate" method (CM = 8) sets these flags as
         follows:

            0 - compressor used fastest algorithm
            1 - compressor used fast algorithm
            2 - compressor used default algorithm
            3 - compressor used maximum compression, slowest algorithm

         The information in FLEVEL is not needed for decompression; it
         is there to indicate if recompression might be worthwhile.

      compressed data
         For compression method 8, the compressed data is stored in the
         deflate compressed data format as described in the document
         "DEFLATE Compressed Data Format Specification" by L. Peter
         Deutsch. (See reference [3] in Chapter 3, below)

         Other compressed data formats are not specified in this version
         of the zlib specification.

      ADLER32 (Adler-32 checksum)
         This contains a checksum value of the uncompressed data
         (excluding any dictionary data) computed according to Adler-32
         algorithm. This algorithm is a 32-bit extension and improvement
         of the Fletcher algorithm, used in the ITU-T X.224 / ISO 8073
         standard. See references [4] and [5] in Chapter 3, below)

         Adler-32 is composed of two sums accumulated per byte: s1 is
         the sum of all bytes, s2 is the sum of all s1 values. Both sums
         are done modulo 65521. s1 is initialized to 1, s2 to zero.  The
         Adler-32 checksum is stored as s2*65536 + s1 in most-
         significant-byte first (network) order.

  ==> The stream can start with only 2 bytes :
        - CM  = 0x78 : CMINFO=7 (32kB window),  CM=8 (deflate)
        - FLG = 0x01 : FLEVEL = 0 (fastest), FDICT=0 (no dict), FCHECK=1 so
          that 0x7801 is a multiple of 31 (30721 = 991 * 31).

  ==> and it ends with only 4 bytes, the Adler-32 checksum in big-endian format.

 */

static const unsigned char zlib_hdr[] = { 0x78, 0x01 };   // 32k win, deflate, chk=1


/* Original version from RFC1950, verified and works OK */
uint32_t slz_adler32_by1(uint32_t crc, const unsigned char *buf, int len)
{
	uint32_t s1 = crc & 0xffff;
	uint32_t s2 = (crc >> 16) & 0xffff;
	int n;

	for (n = 0; n < len; n++) {
		s1 = (s1 + buf[n]) % 65521;
		s2 = (s2 + s1)     % 65521;
	}
	return (s2 << 16) + s1;
}

/* Computes the adler32 sum on <buf> for <len> bytes. It avoids the expensive
 * modulus by retrofitting the number of bytes missed between 65521 and 65536
 * which is easy to count : For every sum above 65536, the modulus is offset
 * by (65536-65521) = 15. So for any value, we can count the accumulated extra
 * values by dividing the sum by 65536 and multiplying this value by
 * (65536-65521). That's easier with a drawing with boxes and marbles. It gives
 * this :
 *          x % 65521 = (x % 65536) + (x / 65536) * (65536 - 65521)
 *                    = (x & 0xffff) + (x >> 16) * 15.
 */
uint32_t slz_adler32_block(uint32_t crc, const unsigned char *buf, long len)
{
	long s1 = crc & 0xffff;
	long s2 = (crc >> 16);
	long blk;
	long n;

	do {
		blk = len;
		/* ensure we never overflow s2 (limit is about 2^((32-8)/2) */
		if (blk > (1U << 12))
			blk = 1U << 12;
		len -= blk;

		for (n = 0; n < blk; n++) {
			s1 = (s1 + buf[n]);
			s2 = (s2 + s1);
		}

		/* Largest value here is 2^12 * 255 = 1044480 < 2^20. We can
		 * still overflow once, but not twice because the right hand
		 * size is 225 max, so the total is 65761. However we also
		 * have to take care of the values between 65521 and 65536.
		 */
		s1 = (s1 & 0xffff) + 15 * (s1 >> 16);
		if (s1 > 65521)
			s1 -= 65521;

		/* For s2, the largest value is estimated to 2^32-1 for
		 * simplicity, so the right hand side is about 15*65535
		 * = 983025. We can overflow twice at most.
		 */
		s2 = (s2 & 0xffff) + 15 * (s2 >> 16);
		s2 = (s2 & 0xffff) + 15 * (s2 >> 16);
		if (s2 > 65521)
			s2 -= 65521;

		buf += blk;
	} while (len);
	return (s2 << 16) + s1;
}

/* Sends the zlib header for stream <strm> into buffer <buf>. When it's done,
 * the stream state is updated to SLZ_ST_EOB. It returns the number of bytes
 * emitted which is always 2. The caller is responsible for ensuring there's
 * always enough room in the buffer.
 */
int slz_rfc1950_send_header(struct slz_stream *strm, unsigned char *buf)
{
	memcpy(buf, zlib_hdr, sizeof(zlib_hdr));
	strm->state = SLZ_ST_EOB;
	return sizeof(zlib_hdr);
}

/* Encodes the block according to rfc1950. This means that the CRC of the input
 * block is computed according to the ADLER32 algorithm. If the header was never
 * sent, it may be sent first. The number of output bytes is returned.
 */
long slz_rfc1950_encode(struct slz_stream *strm, unsigned char *out, const unsigned char *in, long ilen, int more)
{
	long ret = 0;

	if (__builtin_expect(strm->state == SLZ_ST_INIT, 0))
		ret += slz_rfc1950_send_header(strm, out);

	strm->crc32 = slz_adler32_block(strm->crc32, in, ilen);
	ret += slz_rfc1951_encode(strm, out + ret, in, ilen, more);
	return ret;
}

/* Initializes stream <strm> for use with the zlib format (rfc1952). The
 * compression level passed in <level> is set. This value can only be 0 (no
 * compression) or 1 (compression) and other values will lead to unpredictable
 * behaviour. The function always returns 0.
 */
int slz_rfc1950_init(struct slz_stream *strm, int level)
{
	strm->state  = SLZ_ST_INIT;
	strm->level  = level;
	strm->format = SLZ_FMT_ZLIB;
	strm->crc32  = 1; // rfc1950/zlib starts with initial crc=1
	strm->ilen   = 0;
	strm->qbits  = 0;
	strm->queue  = 0;
	return 0;
}

/* Flushes pending bits and sends the gzip trailer for stream <strm> into
 * buffer <buf>. When it's done, the stream state is updated to SLZ_ST_END. It
 * returns the number of bytes emitted. The trailer consists in flushing the
 * possibly pending bits from the queue (up to 24 bits), rounding to the next
 * byte, then 4 bytes for the CRC. That may abount to 4+4 = 8 bytes, that the
 * caller must ensure are available before calling the function. Note that if
 * the initial header was never sent, it will be sent first as well (2 extra
 * bytes).
 */
int slz_rfc1950_finish(struct slz_stream *strm, unsigned char *buf)
{
	strm->outbuf = buf;

	if (__builtin_expect(strm->state == SLZ_ST_INIT, 0))
		strm->outbuf += slz_rfc1952_send_header(strm, strm->outbuf);

	slz_rfc1951_finish(strm, strm->outbuf);
	copy_8b(strm, (strm->crc32 >> 24) & 0xff);
	copy_8b(strm, (strm->crc32 >> 16) & 0xff);
	copy_8b(strm, (strm->crc32 >>  8) & 0xff);
	copy_8b(strm, (strm->crc32 >>  0) & 0xff);
	strm->state = SLZ_ST_END;
	return strm->outbuf - buf;
}

__attribute__((constructor))
static void __slz_compute_tables(void)
{
	slz_make_crc_table();
	slz_prepare_dist_table();
}


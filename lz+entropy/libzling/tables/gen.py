#!/usr/bin/env python

# table auto-generator for zling.
# author: Zhang Li <zhangli10@baidu.com>

import math

kBucketItemSize = 4096

matchidx_blen = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7] + [8] * 1024
matchidx_code = []
matchidx_bits = []
matchidx_base = []

while len(matchidx_code) < kBucketItemSize:
    for bits in range(2 ** matchidx_blen[len(matchidx_base)]):
        matchidx_code.append(len(matchidx_base))
    matchidx_base.append(len(matchidx_code) - 2 ** matchidx_blen[len(matchidx_base)])

f_blen = open("table_matchidx_blen.inc", "w")
f_base = open("table_matchidx_base.inc", "w")
f_code = open("table_matchidx_code.inc", "w")

for i in range(0, matchidx_base.__len__()):
    f_blen.write("%4u," % matchidx_blen[i] + "\n\x20" [int(i % 16 != 15)])
    f_base.write("%4u," % matchidx_base[i] + "\n\x20" [int(i % 16 != 15)])

for i in range(0, matchidx_code.__len__()):
    f_code.write("%4u," % matchidx_code[i] + "\n\x20" [int(i % 16 != 15)])

f_mtfinit = open("table_mtfinit.inc", "w")
f_mtfinit.write("""  // auto-generate from enwik8
         32, 101, 116,  97, 105, 111, 110, 114, 115, 108, 104, 100,  99, 117,  93,  91,
        109, 112, 103, 102,  10, 121,  98,  39, 119,  46,  44, 118,  59,  38, 124,  47,
         49, 107,  61,  48,  67,  65,  58,  45,  84,  83,  60,  62,  50, 113,  73,  57,
         42, 120,  41,  40,  66,  77,  80,  69,  68,  53,  51,  72,  70,  56,  52,  71,
         82,  54,  76,  55,  78,  87, 122, 125, 123,  79, 106,  85,  74,  75, 208,  95,
        195,  35,  86, 215,  90,  34,  89, 209, 128, 224, 184, 131,  92, 227,  37,  33,
        176, 169, 206, 226, 130,  63,  88,  81, 161, 153,  43, 129, 188, 179, 216, 164,
        181, 189, 148, 190, 173, 187, 186, 229, 225, 167, 217, 177, 178, 168, 149, 185,
        197, 144, 147, 196, 207, 194, 180, 156, 132, 170, 166, 136, 182, 191,   9, 230,
        141, 160, 175,  36, 152, 140, 165, 145,  94, 133, 163, 183, 171, 157, 137, 174,
        134, 135, 236, 151, 231, 155, 201, 158, 138, 143, 150, 162, 159, 139, 172, 154,
        126, 232, 235, 146, 233, 228, 202, 203, 142, 214, 237, 204, 219, 234, 213,  96,
        218, 199,  64, 210, 239, 198, 211, 205, 212, 240, 222, 220, 200,   0,   1,   2,
          3,   4,   5,   6,   7,   8,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,
         21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31, 127, 192, 193, 221, 223,
        238, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
""")

f_mtfnext = open("table_mtfnext.inc", "w")
for i in range(0, 256):
    if i < 128:
        f_mtfnext.write("%4u," % int(i * 0.95) + "\n\x20" [int(i % 16 != 15)])
    else:
        f_mtfnext.write("%4u," % int(i * 0.55) + "\n\x20" [int(i % 16 != 15)])

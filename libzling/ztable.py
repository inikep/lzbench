#!/usr/bin/env python

# table auto-generator for zling.
# author: Zhang Li <zhangli10@baidu.com>

kBucketItemSize = 4096

matchidx_blen = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7] + [8] * 1024
matchidx_code = []
matchidx_bits = []
matchidx_base = []

while len(matchidx_code) < kBucketItemSize:
    for bits in range(2 ** matchidx_blen[len(matchidx_base)]):
        matchidx_code.append(len(matchidx_base))
    matchidx_base.append(len(matchidx_code) - 2 ** matchidx_blen[len(matchidx_base)])

f_blen = open("ztable_matchidx_blen.inc", "w")
f_base = open("ztable_matchidx_base.inc", "w")
f_code = open("ztable_matchidx_code.inc", "w")

for i in range(0, matchidx_base.__len__()):
    f_blen.write("%4u," % matchidx_blen[i] + "\n\x20" [int(i % 16 != 15)])
    f_base.write("%4u," % matchidx_base[i] + "\n\x20" [int(i % 16 != 15)])

for i in range(0, matchidx_code.__len__()):
    f_code.write("%4u," % matchidx_code[i] + "\n\x20" [int(i % 16 != 15)])

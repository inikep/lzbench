# huffman-zlib: zlib's Z_HUFFMAN_ONLY strategy (Huffman coding without LZ77), in
# entropybench; it calls zlib's objects (mk/zlib.mk)
ENTROPY_CODECS += HUFFMAN_ZLIB
HUFFMAN_ZLIB_NEEDS := ZLIB

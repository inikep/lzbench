#ifndef TOR_TEST_H
#define TOR_TEST_H

uint32_t tor_decompress(uint8_t* inbuf, uint32_t inlen, uint8_t* outbuf, uint32_t outlen);
uint32_t tor_compress(uint8_t method, uint8_t* inbuf, uint32_t inlen, uint8_t* outbuf, uint32_t outlen);

#endif



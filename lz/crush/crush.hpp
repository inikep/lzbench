namespace crush
{
	uint32_t compress(int level, uint8_t* buf, int size, uint8_t* outbuf);
	uint32_t decompress(uint8_t* inbuf, uint8_t* outbuf, int outsize);
}

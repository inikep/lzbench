#ifndef _CSC_ANALYZER_H_
#define _CSC_ANALYZER_H_

#include <csc_typedef.h>


class Analyzer
{
public:
	void Init();
	//~Analyzer();
	uint32_t Analyze(uint8_t* src, uint32_t size, uint32_t *bpb);
	uint32_t AnalyzeHeader(uint8_t *src, uint32_t size);//,uint32_t *typeArg1,uint32_t *typeArg2,uint32_t *typeArg3);
    uint32_t GetDltBpb(uint8_t *src, uint32_t size, uint32_t chn);

private:
	uint32_t logTable[(MinBlockSize >> 4) + 1];
	int32_t get_channel_idx(uint8_t *src, uint32_t size);
};


#endif


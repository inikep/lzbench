#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <csc_analyzer.h>

#define AYZDEBUG


void Analyzer::Init()
{
    for(uint32_t i = 0; i< (MinBlockSize >> 4); i++)
        logTable[i] = (double)100 * log((double)i * 16 + 8)/ log((double)2);

    logTable[MinBlockSize >> 4] = (double)100*log((double)(MinBlockSize))/log((double)2);
    //fLog=fopen("r:\\dataLog.txt","w");
}

uint32_t Analyzer::AnalyzeHeader(uint8_t *src, uint32_t size) //,uint32_t *typeArg1,uint32_t *typeArg2,uint32_t *typeArg3)
{
    
    /*
	if (size<128)
		return DT_NONE;

	if (src[0]=='B' && src[1]=='M')
	{
		uint32_t width=*(uint32_t*)(src+0x12);
		uint32_t colorbits=*(uint32_t*)(src+0x1C);
		uint32_t compressed=*(uint32_t*)(src+0x1E);

#ifdef AYZDEBUG
		printf(" ===== Detect BM: width:%u colorbits:%u compressed:%u\n",width,colorbits,compressed);
#endif
		if (width<16384 && 
			(colorbits==8||
			colorbits==16||
			colorbits==24||
			colorbits==32) &&
			compressed==0 &&
			*(uint16_t*)(src+0x1A)==1
			)
		{
			*typeArg1=width;
			*typeArg2=colorbits;
			return DT_RGB;
		}
	}

	if (src[0]==0 && src[1]==0 &&src[3]==0)
	{
		if ( (src[2]==2 || src[2]==10)
			&& (src[16]==24 || src[16]==32) )
		{
			uint32_t width=*(uint16_t*)(src+0x0C);

#ifdef AYZDEBUG
			printf(" ===== Detect TGA: width:%u colorbits:%u\n",width,src[16]);
#endif
			*typeArg1=width;
			*typeArg2=src[16];
			return DT_RGB;
		}
	}


	if (src[0]=='P' && src[1]=='6' &&src[2]==0x0a)
	{
		uint8_t *pTmp=src+3;
		bool inComent=false;
		while(*pTmp==0 || *pTmp=='#' || inComent)
		{
			if (*pTmp==0x0a) inComent=false;
			else if (*pTmp=='#') inComent=true;
			pTmp++;
			if (pTmp-src>127)
				break;
		}

		uint32_t width;
		sscanf((char*)pTmp,"%d",&width);

#ifdef AYZDEBUG
		printf(" ===== Detect PPM - P6: width:%u\n",width);
#endif
			*typeArg1=width;
			*typeArg2=24;
			return DT_RGB;
	}

//	if (src[0]=='R' && src[1]=='I' && src[2]=='F' && src[3]=='F' &&
//		src[8]=='W' && src[9]=='A' && src[10]=='V' && src[11]=='E')
//	{
//#ifdef AYZDEBUG
//		printf(" ===== Detect Riff Wav \n");
//#endif
//		return DT_AUDIO;
//	}
	if (src[0]==80 && src[2]==10 && (src[1]==53||src[1]==54))
	{
		P5623
	}

	//if (*(tufx)==80  && *(tufx+1)==54 && *(tufx+2)==10 ){PPMread();Decont=1;chn=1;deltarange=1;}
	//if (*(tufx)==80  && *(tufx+1)==53 && *(tufx+2)==10 ){PPMread();Decont=1;chn=1;deltarange=1;}
	//if (Peekl(tufx+8)==1163280727)  {Decont=1;processed=1;chn=Peekw(tufx+22);deltarange=(Peekw(tufx+34)>>3);}//
	//if (Peekl(tufx)==1297239878 && Peekl(tufx+8)==1179011393 )  {Decont=1;chn=Peekb(tufx+21);deltarange=Peekb(tufx+27)>>3;} //AIFF AUDIO

	//if (*(tufx)==80  && *(tufx+1)==54  && *(tufx+2)==10 ){ CTX1=3;CTX2=6;processed=0;chn=1;deltarange=Deltac;ROLZ=0;RZCLEVEL=2;PPMread();} //PPM

	*.wav -----   if (Peekl(tufx)==1179011410)       { processed=1;ROLZ=0;RZCLEVEL=2;chn=1;deltarange=1;} //RIFF WAVE
	if (Peekl(tufx)==1297239878        && Peekl(tufx+8)==1179011393 )            {processed=Deltac;ROLZ=0;RZCLEVEL=2;chn=Peekb(tufx+21);CTX1=chn*(Peekb(tufx+27)>>3);CTX2=CTX1*2;deltarange=Peekb(tufx+27)>>3;} //AIFF AUDIO
	if (Peekl(tufx+8)==1163280727)                                              { processed=Deltac;ROLZ=0;RZCLEVEL=2;chn=Peekw(tufx+22);CTX1=chn;CTX2=CTX1*2;deltarange=(Peekw(tufx+34)>>3);}// wave //
	//if (*(tufx)==0   && *(tufx+1)==0   && *(tufx+2)==2  && *(tufx+3)==0 ) { CTX1=3;CTX2=6;processed=Deltac;chn=1;deltarange=1;ROLZ=0;RZCLEVEL=2;} // TARGA image //
	//if (*(tufx)==0   && *(tufx+1)==0   && *(tufx+2)==10 && *(tufx+3)==0)  { CTX1=3;CTX2=6;processed=Deltac;chn=1;deltarange=1;ROLZ=0;RZCLEVEL=2;} // TARGA image RLE//
	//if (*(tufx)==80  && *(tufx+1)==53  && *(tufx+2)==10 )                { processed=Deltac;chn=1;deltarange=1;ROLZ=1;RZCLEVEL=2;PPMread();} // PGM image //
*/
	return DT_NONE;
}

uint32_t lastType=0;

int32_t Analyzer::get_channel_idx(uint8_t *src,uint32_t size)
{
    uint32_t sameDist[DLT_CHANNEL_MAX]={0},succValue[DLT_CHANNEL_MAX]={0};
    uint32_t maxSame,maxSucc,minSame,minSucc,bestChnNum;

    for(uint32_t i = 0; i + 16 < size; i++) {
        sameDist[0] += (src[i]==src[i+1]);
        sameDist[1] += (src[i]==src[i+2]);
        sameDist[2] += (src[i]==src[i+3]);
        sameDist[3] += (src[i]==src[i+4]);
        sameDist[4] += (src[i]==src[i+8]);
        succValue[0] += abs((signed)src[i]-(signed)src[i+1]);
        succValue[1] += abs((signed)src[i]-(signed)src[i+2]);
        succValue[2] += abs((signed)src[i]-(signed)src[i+3]);
        succValue[3] += abs((signed)src[i]-(signed)src[i+4]);
        succValue[4] += abs((signed)src[i]-(signed)src[i+8]);
    }

    maxSame=minSame = sameDist[0];
    maxSucc=minSucc = succValue[0];
    bestChnNum = 0;

    for (uint32_t i = 0;i < DLT_CHANNEL_MAX; i++) {
        if (sameDist[i] < minSame) minSame=sameDist[i];
        if (sameDist[i] > maxSame) maxSame=sameDist[i];
        if (succValue[i] > maxSucc) maxSucc=succValue[i];
        if (succValue[i] < minSucc) {
            minSucc = succValue[i];
            bestChnNum = i;
        }
    }


    if ( ((maxSucc > succValue[bestChnNum] * 4) || (maxSucc > succValue[bestChnNum] + 40 * size)) 
        && (sameDist[bestChnNum] > minSame * 3)
        //&& (entropy>700*size || diffNum>245) 
        && (sameDist[0] < 0.3 * size))
    {
        //printf("delta:%d %d %d %d %d %dr:%d \n",succValue[0],succValue[1],succValue[2],succValue[3],succValue[4],sameDist[5],bestChnNum);
        return bestChnNum;
    }
    return -1;
}

uint32_t Analyzer::GetDltBpb(uint8_t *src, uint32_t size, uint32_t chn)
{
    uint32_t freq[256] = {0};
    uint8_t prev = 0;
    uint32_t bpb = 0;
    for(uint32_t i = 0; i < chn; i++)
        for(uint32_t j = i; j < size; j += chn) {
            freq[uint8_t(src[j] - prev)]++;
            prev = src[j];
        }

	bpb = size * logTable[size>>4];
	for(uint32_t i = 0; i < 256; i++) 
		bpb -= freq[i] * logTable[freq[i] >> 4];
    bpb /= size;
    return bpb;
}

uint32_t Analyzer::Analyze(uint8_t *src, uint32_t size, uint32_t *bpb)
{
    uint32_t avgFreq,freq[256]={0};
    uint32_t freq0x80[2]={0};
    uint32_t entropy,alpha_num,diffNum;

    if (size>MinBlockSize)
        size=MinBlockSize;

    if (size<512)
        return DT_SKIP;

    for(uint32_t i = 0; i < size; i++)
        freq[src[i]]++;

    diffNum = 0;
    entropy = size * logTable[size>>4];

    for(uint32_t i = 0; i < 256; i++) {
        entropy -= freq[i] * logTable[freq[i] >> 4];
        diffNum += (freq[i] > 0);
        freq0x80[i >> 7] += freq[i];
    }
    *bpb = entropy / size;
    avgFreq = size >> 8;

    alpha_num = 0;
    for(uint32_t i='a';i<='z';i++)
        alpha_num += freq[i];

    if (freq0x80[1] < (size >> 3) && (freq[' '] + freq['\n'] + freq[':'] + freq['.'] + freq['/'] > (size >> 4)) 
        && (freq['a'] + freq['e'] + freq['t'] > (size >> 4)) 
        && entropy > 300 * size 
        && alpha_num > (size / 3))
        return DT_ENGTXT;

    if (freq[0x8b] > avgFreq && freq[0x00] > avgFreq * 2 && freq[0xE8] > 6)
        return DT_EXE;

    if (entropy > (log((double)diffNum - 2) / log((double)2) - 0.6) * 100.0 * size  && diffNum < 16 && diffNum >= 6)
        return DT_ENTROPY;

    if (entropy < 400 * size && diffNum < 200)
        return DT_NORMAL;

    int32_t dltIdx = get_channel_idx(src,size);
    if (dltIdx != -1)
        return DT_DLT + dltIdx;

    if (entropy > 795 * size)
        return DT_BAD;
    else if (entropy > 780 * size)
        return DT_FAST; 

    return DT_NORMAL;//DT_NORMAL;
}



#include <stdlib.h>
#include <string.h>
#include <csc_filters.h>
#include <stdio.h>

const uint32_t wordNum=123;

uint8_t wordList[wordNum][8]=
{
    "",
    "ac","ad","ai","al","am",
    "an","ar","as","at","ea",
    "ec","ed","ee","el","en",
    "er","es","et","id","ie",
    "ig","il","in","io","is",
    "it","of","ol","on","oo",
    "or","os","ou","ow","ul",
    "un","ur","us","ba","be",
    "ca","ce","co","ch","de",
    "di","ge","gh","ha","he",
    "hi","ho","ra","re","ri",
    "ro","rs","la","le","li",
    "lo","ld","ll","ly","se",
    "si","so","sh","ss","st",
    "ma","me","mi","ne","nc",
    "nd","ng","nt","pa","pe",
    "ta","te","ti","to","th",
    "tr","wa","ve",
    "all","and","but","dow",
    "for","had","hav","her",
    "him","his","man","mor",
    "not","now","one","out",
    "she","the","was","wer",
    "whi","whe","wit","you",
    "any","are",
    "that","said","with","have",
    "this","from","were","tion",
    //==================
    /*
    "the",   "th",   "he",   "in",   "er",   "an",   
    "on",   "re",   "tion",   "and",   "ion",   "or",   
    "ti",   "at",   "te",   "en",   "es",   "ing",   
    "al",   "is",   "ar",   "nd",   "st",   "quot",   
    "tio",   "ed",   "nt",   "ent",   "it",   "of",   
    "le",   "ri",   "to",   "ng",   "atio",   "io",   
    "ic",   "me",   "as",   "ter",   "ati",   "se",   
    "co",   "ra",   "de",   "ve",   "ro",   "quo",   
    "uot",   "ate",   "ot",   "li",   "la",   "om",   
    "ha",   "ne",   "ce",   "ea",   "for",   "si",   
    "ta",   "ma",   "ou",   "hi",   "ment",   "ll",   
    "am",   "el",   "con",   "ca",   "ch",   "her",   
    "ver",   "all",   "us",   "ers",   "ther",   "ns",   
    "com",   "na",   "tr",   "qu",   "that",   "ist",   
    "ge",   "di",   "ur",   "with",   "ons",   "ted",   
    "be",   "ec",   "men",   "ere",   "res",   "ia",   
    "ni",   "ica",   "ol",   "ac",   "il",   "est",   
    "ct",   "rs",   "nce",   "amp",   "mp",   "sion",   
    "pe",   "et",   "id",   "nc",   "sta",   "ie",   
    "age",   "his",   "tha",   "rt",   "fo",   "tor",   
    "ly",   "ive",   
    "the",   "and",   "ing",   "tion",   "that",   "with",   
    "hat",   "ion",   "her",   "ther",   "atio",   "ent",   
    "for",   "ith",   "ight",   "tio",   "tha",   "his",   
    "thou",   "all",   "here",   "ter",   "ere",   "ough",   
    "wit",   "ver",   "res",   "ment",   "ati",   "ght",   
    "igh",   "from",   "thi",   "ate",   "ess",   "ould",   
    "hou",   "this",   "nce",   "not",   "ers",   "heir",   
    "ear",   "our",   "ting",   "thin",   "est",   "rea",   
    "sion",   "ave",   "tho",   "hing",   "hich",   "pro",   
    "ect",   "thei",   "ill",   "ence",   "ound",   "they",   
    "con",   "ted",   "one",   "com",   "form",   "what",   
    "ring",   "ions",   "ever",   "you",   "comp",   "pres",   
    "reat",   "will",   "whic",   "are",   "eave",   "ugh",   
    "int",   "eve",   "hin",   "ning",   "ons",   "ore",   
    "have",   "wor",   "work",   "ive",   "said",   "men",   
    "rom",   "ain",   "houg",   "ine",   "othe",   "nte",   
    "aven",   "hen",   "oun",   "eat",   "lect",   "oug",   
    "able",   "age",   "nter",   "ture",   "text",   "per",   
    "rese",   "ove",   "use",   "ven",   "them",   "she",   
    "ble",   "ught",   "more",   "ous",   "und",   "out",   
    "nder",   "ese",
    */
};



void Filters::MakeWordTree()
{
    uint32_t i,j;
    uint32_t treePos;
    uint8_t symbolIndex=0x82;

    nodeMum=1;
    memset(wordTree,0,sizeof(wordTree));
    for (i=1;i<wordNum;i++) {	
        treePos=0;
        for(j=0;wordList[i][j]!=0;j++) {
            uint32_t idx=wordList[i][j]-'a';
            if (wordTree[treePos].next[idx]) {
                treePos=wordTree[treePos].next[idx];
            } else {
                wordTree[treePos].next[idx]=nodeMum;
                treePos=nodeMum;
                nodeMum++;
            }
        }
        wordIndex[symbolIndex]=i;
        wordTree[treePos].symbol=symbolIndex++;
    }
    maxSymbol=symbolIndex;
}


void Filters::Init(ISzAlloc *alloc)
{
    alloc_ = alloc;
    m_fltSwapSize = 0;
    MakeWordTree();
}



void Filters::Destroy()
{
    if (m_fltSwapSize > 0) {
        alloc_->Free(alloc_, m_fltSwapBuf);
    }
    m_fltSwapBuf = 0;
}


void Filters::Forward_Delta(uint8_t *src,uint32_t size,uint32_t chnNum)
{
	uint32_t dstPos,i,j;
    uint8_t prevByte = 0;

	if (size<512)
		return;

	if (m_fltSwapSize < size) {
		if (m_fltSwapSize > 0) {
            alloc_->Free(alloc_, m_fltSwapBuf);
        }
        m_fltSwapBuf = (uint8_t*)alloc_->Alloc(alloc_, size);
		m_fltSwapSize = size;
	}

	memcpy(m_fltSwapBuf, src, size);

	dstPos = 0;

	for (i=0;i<chnNum;i++) {
		for(j=i;j<size;j+=chnNum)
		{
			src[dstPos++]=m_fltSwapBuf[j]-prevByte;
			prevByte=m_fltSwapBuf[j];
            /*
			src[dstPos++] = m_fltSwapBuf[j] - (2 * prev1 - prev2 + prev1);
            prev2 = prev1;
            prev1 = m_fltSwapBuf[j];
            */
		}
    }
}


/*
void Filters::Forward_RGB(uint8_t *src,uint32_t size,uint32_t width,uint32_t colorBits)
{
	if (size<512)
		return;

	uint32_t totalTest=0;

	int channelNum=colorBits/8;

	if (m_fltSwapSize<size+(width+1)*channelNum) //no need boundry check
	{
		if (m_fltSwapSize>0)
			free(m_fltSwapBuf);

		m_fltSwapBuf=(uint8_t*)malloc(size+(width+1)*channelNum);
		m_fltSwapSize=size;
	}

	memset(m_fltSwapBuf,0,(width+1)*channelNum);
	memcpy(m_fltSwapBuf+(width+1)*channelNum,src,size);
	uint8_t *newSrc=m_fltSwapBuf+(width+1)*channelNum;
	uint32_t dstPos=0;


	for (int i=0;i<size-channelNum;i+=channelNum)
	{
		uint8_t G=newSrc[i+1];
		newSrc[i]-=G/4;
		newSrc[i+2]-=G*3/4;
	}

	for(uint32_t i=0;i<channelNum;i++)
	{
		uint32_t vLeft,vUpper,vUpperLeft;
		int vPredict;
		uint32_t pa,pb,pc;
		for(uint32_t j=i;j<size;j+=channelNum)
		{
			vLeft=newSrc[j-channelNum];
			vUpper=newSrc[j-channelNum*width];
			vUpperLeft=newSrc[j-channelNum*(width+1)];
			vPredict=((int)vLeft+vUpper-vUpperLeft);
			if (vPredict>255)
				vPredict=255;
			if (vPredict<0)
				vPredict=0;
			src[dstPos++]=vPredict-newSrc[j];

			totalTest+=abs(newSrc[j]-vPredict);
		}
	}

	printf("size:%d --- %f\n",size,(float)totalTest/size);


}

void Filters::Inverse_RGB(uint8_t *src,uint32_t size,uint32_t width,uint32_t colorBits)
{
	if (size<512)
		return;

	if (m_fltSwapSize<size) {
		if (m_fltSwapSize>0)
			free(m_fltSwapBuf);
        
		m_fltSwapBuf=(uint8_t*)malloc(size);
		m_fltSwapSize=size;
	}

	memcpy(m_fltSwapBuf,src,size);

	int channelNum=colorBits/8;

}



void Filters::Forward_Audio(uint8_t *src,uint32_t size,uint32_t width,uint32_t colorBits)
{
}

void Filters::Inverse_Audio(uint8_t *src,uint32_t size,uint32_t width,uint32_t colorBits)
{
}

*/

uint32_t Filters::Foward_Dict(uint8_t *src,uint32_t size)
{
    if (size < 16384) {
        return 0;
    }

    if (m_fltSwapSize < size) {
        if (m_fltSwapSize > 0) {
            alloc_->Free(alloc_, m_fltSwapBuf);
        }

        m_fltSwapBuf = (uint8_t*)alloc_->Alloc(alloc_, size);
        m_fltSwapSize = size;
    }

    uint8_t *dst = m_fltSwapBuf;
    uint32_t i, j, treePos = 0;
    uint32_t dstSize = 0;
    uint32_t idx;


    for(i=0;i<size-5;) {
        if (dstSize>m_fltSwapSize-16) {
            return 0;
        }

        if (src[i]>='a'&& src[i]<='z') {
            uint32_t matchSymbol = 0,longestWord = 0;
            treePos = 0;
            for(j = 0;;) {
                idx = src[i+j]-'a';
                if (idx > 25 || wordTree[treePos].next[idx] == 0) {
                    break;
                }

                treePos = wordTree[treePos].next[idx];
                j++;
                if (wordTree[treePos].symbol) {
                    matchSymbol = wordTree[treePos].symbol;
                    longestWord = j;
                }
            }

            if (matchSymbol) {
                dst[dstSize++] = matchSymbol;
                i+=longestWord;
                continue;
            }
            dst[dstSize++] = src[i];
            i++;
        } else {
            if (src[i] >= 0x82) {
                dst[dstSize++] = 254;
                dst[dstSize++] = src[i];
            } else {
                dst[dstSize++] = src[i];
            }
            treePos = 0;
            i++;
        }

    }

    for (;i<size;i++) {
        if (src[i]>=0x82) {
            dst[dstSize++]=254;
            dst[dstSize++]=src[i];
        } else {
            dst[dstSize++]=src[i];
        }
    }

    if (dstSize > size * 0.82) {
        return 0;
    }

    memset(dst + dstSize, 0x20, size - dstSize);
    memcpy(src, dst, size);
    return 1;
}

void Filters::Inverse_Dict(uint8_t *src,uint32_t size)
{

    if (m_fltSwapSize<size) {
        if (m_fltSwapSize > 0) {
            alloc_->Free(alloc_, m_fltSwapBuf);
        }

        m_fltSwapBuf = (uint8_t*)alloc_->Alloc(alloc_, size);
        m_fltSwapSize = size;
	}

    uint8_t *dst=m_fltSwapBuf;
    uint32_t i=0,j;
    uint32_t dstPos=0,idx;

    while(dstPos < size) {
        if (src[i] >= 0x82 && src[i] < maxSymbol) {
            idx = wordIndex[src[i]];
            for(j = 0; wordList[idx][j] && dstPos < size; j++) {
                dst[dstPos++] = wordList[idx][j];
            }
        } else if (src[i] == 254 && (i + 1 < size && src[i+1] >= 0x82)) { 
            i++;
            dst[dstPos++] = src[i];
        } else {
            dst[dstPos++] = src[i];
        }
        i++;
    }
    memcpy(src, dst, size);
}


void Filters::Inverse_Delta(uint8_t *src,uint32_t size,uint32_t chnNum)
{
    uint32_t dstPos,i,j,prevByte;

    if (size<512) 
        return;

    if (m_fltSwapSize<size) {
        if (m_fltSwapSize>0) {
            alloc_->Free(alloc_, m_fltSwapBuf);
        }

        m_fltSwapBuf = (uint8_t*)alloc_->Alloc(alloc_, size);
        m_fltSwapSize=size;
    }

    memcpy(m_fltSwapBuf,src,size);

    dstPos = 0;
    prevByte = 0;
    for (i = 0; i < chnNum; i++) {
        for(j = i; j < size;j += chnNum)
        {
            src[j] = m_fltSwapBuf[dstPos++] + prevByte;
            prevByte = src[j];
        }
    }
}


//void Filters::Forward_Audio4(uint8_t *src,uint32_t size)
//{
//
//	uint32_t dstPos,i,j,prevByte;
//	uint32_t chnNum=4;
//
//	if (m_fltSwapSize<size)
//	{
//		if (m_fltSwapSize>0)
//		{
//			SAFEFREE(m_fltSwapBuf);
//		}
//		m_fltSwapBuf=(uint8_t*)malloc(size);
//		m_fltSwapSize=size;
//	}
//
//	memcpy(m_fltSwapBuf,src,size);
//
//	dstPos=0;
//	prevByte=0;
//	for (i=0;i<chnNum;i++)
//		for(j=i;j<size;j+=chnNum)
//		{
//			src[dstPos++]=m_fltSwapBuf[j]-prevByte;
//			prevByte=m_fltSwapBuf[j];
//		}
//		/*uint8_t *SrcData;
//		uint8_t *DestData;
//		int Channels=4;
//
//		if (m_fltSwapSize<size)
//		{
//		if (m_fltSwapSize>0)
//		{
//		SAFEFREE(m_fltSwapBuf);
//		}
//		m_fltSwapBuf=(uint8_t*)malloc(size);
//		m_fltSwapSize=size;
//		}
//
//		memcpy(m_fltSwapBuf,src,size);
//
//		SrcData=m_fltSwapBuf;
//		DestData=src;
//
//		for (int CurChannel=0;CurChannel<Channels;CurChannel++)
//		{
//		unsigned int PrevByte=0,PrevDelta=0,Dif[7];
//		int D1=0,D2=0,D3;
//		int K1=0,K2=0,K3=0;
//		memset(Dif,0,sizeof(Dif));
//
//		for (int I=CurChannel,ByteCount=0;I<size;I+=Channels,ByteCount++)
//		{
//		D3=D2;
//		D2=PrevDelta-D1;
//		D1=PrevDelta;
//
//		unsigned int Predicted=8*PrevByte+K1*D1+K2*D2+K3*D3;
//		Predicted=(Predicted>>3) & 0xff;
//
//
//		unsigned int CurByte=SrcData[I];
//
//		PrevDelta=(signed char)(CurByte-PrevByte);
//		*DestData++=Predicted-CurByte;
//		PrevByte=CurByte;
//
//		int D=((signed char)(Predicted-CurByte))<<3;
//
//		Dif[0]+=abs(D);
//		Dif[1]+=abs(D-D1);
//		Dif[2]+=abs(D+D1);
//		Dif[3]+=abs(D-D2);
//		Dif[4]+=abs(D+D2);
//		Dif[5]+=abs(D-D3);
//		Dif[6]+=abs(D+D3);
//
//		if ((ByteCount & 0x1f)==0)
//		{
//		unsigned int MinDif=Dif[0],NumMinDif=0;
//		Dif[0]=0;
//		for (int J=1;J<sizeof(Dif)/sizeof(Dif[0]);J++)
//		{
//		if (Dif[J]<MinDif)
//		{
//		MinDif=Dif[J];
//		NumMinDif=J;
//		}
//		Dif[J]=0;
//		}
//		switch(NumMinDif)
//		{
//		case 1: if (K1>=-16) K1--; break;
//		case 2: if (K1 < 16) K1++; break;
//		case 3: if (K2>=-16) K2--; break;
//		case 4: if (K2 < 16) K2++; break;
//		case 5: if (K3>=-16) K3--; break;
//		case 6: if (K3 < 16) K3++; break;
//		}
//		}
//		}
//		}*/
//}



void Filters::E89init( void ) 
{
	cs = 0xFF;
	x0 = x1 = 0;
	i  = 0;
	k  = 5;
}

int32_t Filters::E89cache_byte( int32_t c ) 
{
	int32_t d = cs&0x80 ? -1 : (uint8_t)(x1);
	x1>>=8;
	x1|=(x0<<24);
	x0>>=8;
	x0|=(c <<24);
	cs<<=1; i++;
	return d;
}

uint32_t Filters::E89xswap( uint32_t x ) 
{
	x<<=7;
	return (x>>24)|((uint8_t)(x>>16)<<8)|((uint8_t)(x>>8)<<16)|((uint8_t)(x)<<(24-7));
}

uint32_t Filters::E89yswap( uint32_t x ) 
{
	x = ((uint8_t)(x>>24)<<7)|((uint8_t)(x>>16)<<8)|((uint8_t)(x>>8)<<16)|(x<<24);
	return x>>7;
}


int32_t Filters::E89forward( int32_t c ) 
{
	uint32_t x;
	if( i>=k ) {
		if( (x1&0xFE000000)==0xE8000000 ) {
			k = i+4;
			x= x0 - 0xFF000000;
			if( x<0x02000000 ) {
				x = (x+i) & 0x01FFFFFF;
				x = E89xswap(x);
				x0 = x + 0xFF000000;
			}
		}
	} 
	return E89cache_byte(c);
}

int32_t Filters::E89inverse( int32_t c ) 
{
	uint32_t x;
	if( i>=k ) {
		if( (x1&0xFE000000)==0xE8000000 ) {
			k = i+4;
			x = x0 - 0xFF000000;
			if( x<0x02000000 ) {
				x = E89yswap(x);
				x = (x-i) & 0x01FFFFFF;
				x0 = x + 0xFF000000;
			}
		}
	}
	return E89cache_byte(c);
}

int32_t Filters::E89flush() 
{
	int32_t d;
	if( cs!=0xFF ) {
		while( cs&0x80 ) E89cache_byte(0),++cs;
		d = E89cache_byte(0); ++cs;
		return d;
	} else {
		E89init();
		return -1;
	}
}


void Filters::Forward_E89(uint8_t* src, uint32_t size) 
{
    uint32_t i, j;
    int32_t c;
	E89init();
	for(i=0,j=0; i<size; i++ ) {
		c = E89forward( src[i] );
		if( c>=0 ) src[j++]=c;
	}
	while( (c=E89flush())>=0 ) src[j++]=c;
}

void Filters::Inverse_E89(uint8_t* src, uint32_t size) 
{
    uint32_t i, j;
    int32_t c;
	E89init();
	for(i=0,j=0; i<size; i++ ) {
		c = E89inverse( src[i] );
		if( c>=0 ) src[j++] = c;
	}
	while( (c=E89flush())>=0 ) src[j++]=c;
}

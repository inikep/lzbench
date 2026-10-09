/*
 * WZIP - Huffman encoding of literal blocks and sequence streams
 * Copyright (c) 2018-present, Yingquan (Cody) Wu.
 * SPDX-License-Identifier: BSD-2-Clause (see LICENSE)
 */

/*MSB first packing of a multi-bit symbol. Herein we assume symBits is less than 32.
 sym: Symbol to be packed
 symBits: number of bits representing symbol
 The function packs MSB first so that Huffman Codes can retain their prefix property
*/

#include <string.h>     /* memcpy, memset */
#include <stdio.h>      /* fprintf (debug) */
#include <math.h>       /* log2 (Log2_Price) */

#include "BitStream_Huffman.h"
#include "Memry.h"

/* by frequency, then by symbol: qsort need not be stable, and ties left to it would give equally frequent symbols
   different lengths on different platforms (glibc's qsort keeps their order, others do not) */
int struct_cmp(const void* a, const void* b)
{
	const Huffman_Str* ia = (const Huffman_Str*)a;
	const Huffman_Str* ib = (const Huffman_Str*)b;
	if (ia->freq != ib->freq) return ia->freq < ib->freq ? -1 : 1;
	return (int)ia->lit - (int)ib->lit;
}

/* 2^(63 + j/256) rounded, j = 0..255: the thresholds of Log2_Price */
static const Uint64 Pow2_Frac256[256] = {
	0x8000000000000000ULL, 0x8058D7D2D5E5F6B1ULL, 0x80B1ED4FD999AB6CULL, 0x810B40A1D81406D4ULL,
	0x8164D1F3BC030773ULL, 0x81BEA1708DDE6056ULL, 0x8218AF4373FC25ECULL, 0x8272FB97B2A5894CULL,
	0x82CD8698AC2BA1D7ULL, 0x83285071E0FC4547ULL, 0x8383594EEFB6EE37ULL, 0x83DEA15B9541B132ULL,
	0x843A28C3ACDE4046ULL, 0x8495EFB3303EFD30ULL, 0x84F1F656379C1A29ULL, 0x854E3CD8F9C8C95DULL,
	0x85AAC367CC487B15ULL, 0x86078A2F23642A9FULL, 0x8664915B923FBA04ULL, 0x86C1D919CAEF5C88ULL,
	0x871F61969E8D1010ULL, 0x877D2AFEFD4E256CULL, 0x87DB357FF698D792ULL, 0x88398146B919F1D4ULL,
	0x88980E8092DA8527ULL, 0x88F6DD5AF155AC6BULL, 0x8955EE03618E5FDDULL, 0x89B540A7902557A4ULL,
	0x8A14D575496EFD9AULL, 0x8A74AC9A79896E47ULL, 0x8AD4C6452C728924ULL, 0x8B3522A38E1E1032ULL,
	0x8B95C1E3EA8BD6E7ULL, 0x8BF6A434ADDE0085ULL, 0x8C57C9C4646F4DDEULL, 0x8CB932C1BAE97A95ULL,
	0x8D1ADF5B7E5BA9E6ULL, 0x8D7CCFC09C50E2F8ULL, 0x8DDF042022E69CD6ULL, 0x8E417CA940E35A01ULL,
	0x8EA4398B45CD53C0ULL, 0x8F073AF5A2013520ULL, 0x8F6A8117E6C8E5C4ULL, 0x8FCE0C21C6726481ULL,
	0x9031DC431466B1DCULL, 0x9095F1ABC540CA6BULL, 0x90FA4C8BEEE4B12BULL, 0x915EED13C89689D3ULL,
	0x91C3D373AB11C336ULL, 0x9228FFDC10A051ADULL, 0x928E727D9531F9ACULL, 0x92F42B88F673AA7CULL,
	0x935A2B2F13E6E92CULL, 0x93C071A0EEF94BC1ULL, 0x9426FF0FAB1C04B6ULL, 0x948DD3AC8DDB7ED3ULL,
	0x94F4EFA8FEF70961ULL, 0x955C5336887894D5ULL, 0x95C3FE86D6CC7FEFULL, 0x962BF1CBB8D97560ULL,
	0x96942D3720185A00ULL, 0x96FCB0FB20AC4BA3ULL, 0x97657D49F17AB08EULL, 0x97CE9255EC4357ABULL,
	0x9837F0518DB8A96FULL, 0x98A1976F7597E996ULL, 0x990B87E266C189AAULL, 0x9975C1DD47518C77ULL,
	0x99E0459320B7FA65ULL, 0x9A4B13371FD166CAULL, 0x9AB62AFC94FF864AULL, 0x9B218D16F441D63DULL,
	0x9B8D39B9D54E5539ULL, 0x9BF93118F3AA4CC1ULL, 0x9C6573682EC32C2DULL, 0x9CD200DB8A0774CBULL,
	0x9D3ED9A72CFFB751ULL, 0x9DABFDFF6367A2AAULL, 0x9E196E189D472420ULL, 0x9E872A276F0B98FFULL,
	0x9EF5326091A111AEULL, 0x9F6386F8E28BA651ULL, 0x9FD228256400DD06ULL, 0xA041161B3D0121BEULL,
	0xA0B0510FB9714FC2ULL, 0xA11FD9384A344CF7ULL, 0xA18FAECA8544B6E4ULL, 0xA1FFD1FC25CEA188ULL,
	0xA27043030C496819ULL, 0xA2E102153E918F9EULL, 0xA3520F68E802BB93ULL, 0xA3C36B345991B47CULL,
	0xA43515AE09E6809EULL, 0xA4A70F0C95768EC5ULL, 0xA5195786BE9EF339ULL, 0xA58BEF536DBEB6EEULL,
	0xA5FED6A9B15138EAULL, 0xA6720DC0BE08A20CULL, 0xA6E594CFEEE86B1EULL, 0xA7596C0EC55FF55BULL,
	0xA7CD93B4E965356AULL, 0xA8420BFA298F70D1ULL, 0xA8B6D5167B320E09ULL, 0xA92BEF41FA77771BULL,
	0xA9A15AB4EA7C0EF8ULL, 0xAA1717A7B5693979ULL, 0xAA8D2652EC907629ULL, 0xAB0386EF48868DE1ULL,
	0xAB7A39B5A93ED337ULL, 0xABF13EDF162675E9ULL, 0xAC6896A4BE3FE929ULL, 0xACE0413FF83E5D04ULL,
	0xAD583EEA42A14AC6ULL, 0xADD08FDD43D01491ULL, 0xAE493452CA35B80EULL, 0xAEC22C84CC5C9465ULL,
	0xAF3B78AD690A4375ULL, 0xAFB51906E75B8661ULL, 0xB02F0DCBB6E04584ULL, 0xB0A957366FB7A3C9ULL,
	0xB123F581D2AC2590ULL, 0xB19EE8E8C94FEB09ULL, 0xB21A31A66618FE3BULL, 0xB295CFF5E47DB4A4ULL,
	0xB311C412A9112489ULL, 0xB38E0E38419FAE18ULL, 0xB40AAEA2654B9841ULL, 0xB487A58CF4A9C180ULL,
	0xB504F333F9DE6484ULL, 0xB58297D3A8B9F0D2ULL, 0xB60093A85ED5F76CULL, 0xB67EE6EEA3B22B8FULL,
	0xB6FD91E328D17791ULL, 0xB77C94C2C9D725E9ULL, 0xB7FBEFCA8CA41E7CULL, 0xB87BA337A1743834ULL,
	0xB8FBAF4762FB9EE9ULL, 0xB97C143756844DBFULL, 0xB9FCD2452C0B9DEBULL, 0xBA7DE9AEBE5FEA09ULL,
	0xBAFF5AB2133E45FBULL, 0xBB81258D5B704B6FULL, 0xBC034A7EF2E9FB0DULL, 0xBC85C9C560E7B269ULL,
	0xBD08A39F580C36BFULL, 0xBD8BD84BB67ED483ULL, 0xBE0F6809860993E2ULL, 0xBE935317FC378238ULL,
	0xBF1799B67A731083ULL, 0xBF9C3C248E2486F8ULL, 0xC0213AA1F0D08DB0ULL, 0xC0A6956E8836CA8DULL,
	0xC12C4CCA66709456ULL, 0xC1B260F5CA0FBB33ULL, 0xC238D2311E3D6673ULL, 0xC2BFA0BCFAD907C9ULL,
	0xC346CCDA24976407ULL, 0xC3CE56C98D21B15DULL, 0xC4563ECC5334CB33ULL, 0xC4DE8523C2C07BAAULL,
	0xC5672A115506DADDULL, 0xC5F02DD6B0BBC3D9ULL, 0xC67990B5AA245F79ULL, 0xC70352F04336C51EULL,
	0xC78D74C8ABB9B15DULL, 0xC817F681416452B2ULL, 0xC8A2D85C8FFE2C45ULL, 0xC92E1A9D517F0ECCULL,
	0xC9B9BD866E2F27A3ULL, 0xCA45C15AFCC72624ULL, 0xCAD2265E4290774EULL, 0xCB5EECD3B38597C9ULL,
	0xCBEC14FEF2727C5DULL, 0xCC799F23D11510E5ULL, 0xCD078B86503DCDD2ULL, 0xCD95DA6A9FF06445ULL,
	0xCE248C151F8480E4ULL, 0xCEB3A0CA5DC6A55DULL, 0xCF4318CF191918C1ULL, 0xCFD2F4683F94EEB5ULL,
	0xD06333DAEF2B2595ULL, 0xD0F3D76C75C5DB8DULL, 0xD184DF6251699AC6ULL, 0xD2164C023056BCABULL,
	0xD2A81D91F12AE45AULL, 0xD33A5457A3029054ULL, 0xD3CCF099859AC379ULL, 0xD45FF29E0972C561ULL,
	0xD4F35AABCFEDFA1FULL, 0xD5872909AB75D18AULL, 0xD61B5DFE9F9BCE07ULL, 0xD6AFF9D1E13BA2FEULL,
	0xD744FCCAD69D6AF4ULL, 0xD7DA67311797F56AULL, 0xD870394C6DB32C84ULL, 0xD9067364D44A929CULL,
	0xD99D15C278AFD7B6ULL, 0xDA3420ADBA4D8704ULL, 0xDACB946F2AC9CC72ULL, 0xDB63714F8E295255ULL,
	0xDBFBB797DAF23755ULL, 0xDC9467913A4F1C92ULL, 0xDD2D818508324C20ULL, 0xDDC705BCD378F7F0ULL,
	0xDE60F4825E0E9124ULL, 0xDEFB4E1F9D1037F2ULL, 0xDF9612DEB8F04420ULL, 0xE031430A0D99E627ULL,
	0xE0CCDEEC2A94E111ULL, 0xE168E6CFD3295D23ULL, 0xE2055AFFFE83D369ULL, 0xE2A23BC7D7D91226ULL,
	0xE33F8972BE8A5A51ULL, 0xE3DD444C46499619ULL, 0xE47B6CA0373DA88DULL, 0xE51A02BA8E26D681ULL,
	0xE5B906E77C8348A8ULL, 0xE658797368B3A717ULL, 0xE6F85AAAEE1FCE22ULL, 0xE798AADADD5B9CBFULL,
	0xE8396A503C4BDC68ULL, 0xE8DA9958464B42ABULL, 0xE97C38406C4F8C57ULL, 0xEA1E4756550EB27BULL,
	0xEAC0C6E7DD24392FULL, 0xEB63B74317369840ULL, 0xEC0718B64C1CBDDCULL, 0xECAAEB8FFB03AB41ULL,
	0xED4F301ED9942B84ULL, 0xEDF3E6B1D418A491ULL, 0xEE990F980DA3025BULL, 0xEF3EAB20E032BC6BULL,
	0xEFE4B99BDCDAF5CBULL, 0xF08B3B58CBE8B76AULL, 0xF13230A7AD094509ULL, 0xF1D999D8B7708CC1ULL,
	0xF281773C59FFB13AULL, 0xF329C9233B6BAE9CULL, 0xF3D28FDE3A641A5BULL, 0xF47BCBBE6DB9FDDFULL,
	0xF5257D152486CC2CULL, 0xF5CFA433E6537290ULL, 0xF67A416C733F846EULL, 0xF7255510C4288239ULL,
	0xF7D0DF730AD13BB9ULL, 0xF87CE0E5B2094D9CULL, 0xF92959BB5DD4BA74ULL, 0xF9D64A46EB939F35ULL,
	0xFA83B2DB722A033AULL, 0xFB3193CC4227C3F4ULL, 0xFBDFED6CE5F09C49ULL, 0xFC8EC01121E447BBULL,
	0xFD3E0C0CF486C175ULL, 0xFDEDD1B496A89F35ULL, 0xFE9E115C7B8F884CULL, 0xFF4ECB59511EC8A5ULL,
};

/* the high and low halves of a * b */
static void Mul_64x64(const Uint64 a, const Uint64 b, Uint64* const hi, Uint64* const lo)
{
	const Uint64 a0 = (Uint32)a, a1 = a >> 32, b0 = (Uint32)b, b1 = b >> 32;
	const Uint64 p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0;
	const Uint64 mid = (p00 >> 32) + (Uint32)p01 + (Uint32)p10;
	*lo = mid << 32 | (Uint32)p00;
	*hi = a1 * b1 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

/* whether total / d >= 2^(p / 256): total 2^63 against d 2^(p / 256) 2^63, in 128 bits */
static int Ratio_Reaches(const Uint64 total, const Uint64 d, const int p)
{
	Uint64 hi, lo;
	Mul_64x64(Pow2_Frac256[p & 255], d << (p >> 8), &hi, &lo);
	return (total >> 1) > hi || ((total >> 1) == hi && total << 63 >= lo);
}

/* floor(256 log2(total / d)), 1 <= d <= total < 2^40: a price in 1/256 bit. The estimate from log2 is corrected in
   integers, so that every platform gets the same prices: libraries' log2 and floating-point evaluation differ, and
   prices near a step would otherwise round apart and change an encoder's output */
int Log2_Price(const Uint64 total, const Uint64 d)
{
	int p = (int)(256 * log2((double)total / (double)d));
	if (p < 0) p = 0;
	while (p > 0 && !Ratio_Reaches(total, d, p)) p--;
	while (Ratio_Reaches(total, d, p + 1)) p++;
	return p;
}

/* Compute Huffman minimum reduancy and the associated bit lengths */
void Calculate_Minimum_Redundancy(Huffman_Str* A, int nLits)
{
	int root, leaf, next, avbl, used, dpth;

	A[0].freq += A[1].freq; root = 0; leaf = 2;
	for (next = 1; next < nLits - 1; next++)
	{
		if (leaf >= nLits || A[root].freq < A[leaf].freq) { A[next].freq = A[root].freq; A[root++].freq = next; }
		else A[next].freq = A[leaf++].freq;

		if (leaf >= nLits || (root < next && A[root].freq < A[leaf].freq)) { A[next].freq = A[next].freq + A[root].freq; A[root++].freq = next; }
		else A[next].freq = (A[next].freq + A[leaf++].freq);
	}
	A[nLits - 2].freq = 0; for (next = nLits - 3; next >= 0; next--) A[next].freq = A[A[next].freq].freq + 1;
	avbl = 1; used = dpth = 0; root = nLits - 2; next = nLits - 1;
	while (avbl > 0)
	{
		while (root >= 0 && A[root].freq == (Uint32)dpth) { used++; root--; }
		while (avbl > used) { A[next--].freq = dpth; avbl--; }
		avbl = 2 * used; dpth++; used = 0;
	}
}

static int Enforce_Max_Weight(Huffman_Str* sortHufStr, const int nEffLits, int capLitBits)
{
	assert(nEffLits > 0);
	int maxLitBits = sortHufStr[0].nbits;  /* Note sortHufStr is under descreasing order of nbits */
	//return maxLitBits;

	if (maxLitBits <= capLitBits ) {
		if ( maxLitBits <=9 || sortHufStr[2].nbits == maxLitBits) return maxLitBits;
		else capLitBits = maxLitBits - 1;    /* With small cost, reduce max bit length by 1 to simplify decompression */
	}

	int totalCost = 0;
	const int baseCost = 1 << (maxLitBits - capLitBits);
	int n = 0;

	while (sortHufStr[n].nbits > capLitBits) {
		totalCost += baseCost - (1 << (maxLitBits - sortHufStr[n].nbits));
		sortHufStr[n].nbits = (Uint16)capLitBits;
		n++;
	}  /* n stops at huffNode[n].nbBits <= maxNbBits */
	while (sortHufStr[n].nbits == capLitBits) n++;   /* n end at index of largest symbol using < maxNbBits */

	/* renormalize totalCost */
	totalCost >>= (maxLitBits - capLitBits);  /* note : totalCost is necessarily a multiple of baseCost */

	/* repay normalized cost */
	Uint32 const noSymbol = 0xF0F0F0F0;
	Uint32 rankFirst[MAX_HufWeight + 2];

	/* Get pos of last (smallest) symbol per rank */
	memset(rankFirst, 0xF0, sizeof(rankFirst));
	Uint32 currentNbBits = capLitBits;
	int pos;
	for (pos = n; pos < nEffLits; pos++) {
		if (sortHufStr[pos].nbits >= currentNbBits) continue;
		currentNbBits = sortHufStr[pos].nbits;   /* < capLitBits */
		rankFirst[capLitBits - currentNbBits] = (Uint32)pos;      /* first position of the kind */
	} 

	while (totalCost > 0) {
		int nBitsToDecrease = High_Bit32((Uint32)totalCost) + 1;
		for (; nBitsToDecrease > 1; nBitsToDecrease--) {
			Uint32 const highPos = rankFirst[nBitsToDecrease];
			Uint32 const lowPos = rankFirst[nBitsToDecrease - 1];
			if (highPos == noSymbol) continue;
			if (lowPos == noSymbol) break;
			{   Uint32 const highTotal = sortHufStr[highPos].freq;
			Uint32 const lowTotal = 2 * sortHufStr[lowPos].freq;
			if (highTotal <= lowTotal) break;
			}
		}

		/* only triggered when no more rank 1 symbol left => find closest one (note : there is necessarily at least one !) */
		while ((nBitsToDecrease <= MAX_HufWeight) && (rankFirst[nBitsToDecrease] == noSymbol))
			nBitsToDecrease++;
		totalCost -= 1 << (nBitsToDecrease - 1);
		if (rankFirst[nBitsToDecrease - 1] == noSymbol)
			rankFirst[nBitsToDecrease - 1] = rankFirst[nBitsToDecrease];   /* this rank is no longer empty */
		sortHufStr[rankFirst[nBitsToDecrease]].nbits++;
		if ( (int)rankFirst[nBitsToDecrease] == nEffLits - 1)    /* special case, reached largest symbol */
			rankFirst[nBitsToDecrease] = noSymbol;
		else {
			rankFirst[nBitsToDecrease]++;
			if (sortHufStr[rankFirst[nBitsToDecrease]].nbits != capLitBits - nBitsToDecrease)
				rankFirst[nBitsToDecrease] = noSymbol;   /* this rank is now empty */
		}
	}   /* while (totalCost > 0) */

	if (totalCost < 0) {  /* Sometimes, cost correction overshoot */
		if (rankFirst[1] == noSymbol) {  /* special case : no rank 1 symbol (using capLitBits-1); let's create one from largest rank 0 (using capLitBits) */
			while (sortHufStr[n].nbits == capLitBits) n++;
			sortHufStr[n - 1].nbits--;
			assert(n < nEffLits);
			rankFirst[1] = (Uint32)(n - 1);
			totalCost++;
		}
		while (totalCost < 0) {
			rankFirst[1]--;
			sortHufStr[rankFirst[1]].nbits--;
			totalCost++;
		}
	}    

	return capLitBits;
}
	

/*Construct canonical Huffman codes based on the given bit widths
  This function is the code construction for Huffman Tree and length enforcement
  litHuf->nbits: The huffman bit width representation of literal
  litHuf->size: Total number of literals to be constructed
  litHuf->capBits: Maximum length of Huffman Tree to be enforced (i.e. no longer than 7 bits)
  litHuf->code: Huffman code tree
  If we do not enforce Maximum huffman bit width we may need around 10 bytes for header
  Enforcing Header also simplifies tree creation hardware since tree is limited to 7 bits

  After Replacing value Set current position to zero then compare with following
  position to see if it is the natural position
  if not compare with next value until natural position
  is found, then insert value ( first compare 7 with 5, shift 5 by 1 position
  then compare with 6 and shift by 1 position, finally compare with 8, then
  insert in current position for natural order

  Bits     1    2    3    4    5    7    7    7    7
  Map      4    3    0    1    2    5    7    6    8

  Bits     1    2    3    4    5    7    7    7    7
  Map      4    3    0    1    2    5    6    7    8
  Ovrflw   64 + 32 + 16 + 8  + 4  + 1  + 1  + 1  + 1 = 128

  Code Construction
  Map represents Address into Table storing the HuffCode
  Bits     1    2    3    4       5        7       7        7        7
  Map      4    3    0    1       2        5       6        7        8
 Code     0    10   110  1110    11110    1111100 1111101  1111110  1111111*/

/*It constructs Huffman tree subject to a maximum tree depth
  It returns the number of total compressed bytes through Huffman encoding (excluding Huffman tree)
 This is the wrapper function to call the previous functions to generate the Huffman Tree */
int Build_Huffman_Table(Huffman_Str* hufStr, const Uint32 hufCodeSize, const Uint32 hufCodeCapBits, HufCode_Str *litHuf)
{
	Uint16 lit;
	int i, n, nEffLits;      /* number of effective literals */
	Uint32 nbits, minIdx;
	Huffman_Str sortHufStr[1024];                /* alphabets up to 1024 symbols (WZIP_L's joint symbol) */
	memset(litHuf, 0, hufCodeSize * sizeof(HufCode_Str));

	nEffLits = 0;                    /* eliminate zero entries */
	for (i = 0; i < (int)hufCodeSize; i++) {
		if (hufStr[i].freq) {
			sortHufStr[nEffLits].freq = hufStr[i].freq;
			sortHufStr[nEffLits++].lit = (Uint16)i;
		}
	}
	switch (nEffLits) {
	case 0: 
		return 0;
	case 1: 
		litHuf[sortHufStr[0].lit].nbits = 1;
		litHuf[sortHufStr[0].lit].code = 0;
		return 1;
	case 2: 
		litHuf[sortHufStr[0].lit].nbits = 1;
		litHuf[sortHufStr[0].lit].code = 0;
		litHuf[sortHufStr[1].lit].nbits = 1;
		litHuf[sortHufStr[1].lit].code = 1;
		return 1;
	}

	qsort(sortHufStr, nEffLits, sizeof(Huffman_Str), struct_cmp);     /* sort the effective literals in increasing order of frequency */
	Calculate_Minimum_Redundancy(sortHufStr, nEffLits);
	for (i = 0; i < nEffLits; i++) {
		sortHufStr[i].nbits = (Uint8)sortHufStr[i].freq;
		sortHufStr[i].freq = hufStr[sortHufStr[i].lit].freq;    /* recover the original frequency */
	}
	
	Enforce_Max_Weight(sortHufStr, nEffLits, hufCodeCapBits);

	/* Sequential Bubble sorting for literals among equal nbits */
	for (i = nEffLits - 1; i > 0; i--) {  
		nbits = sortHufStr[i].nbits;
		minIdx = i;
		for (n = i - 1; n >= 0 && sortHufStr[n].nbits == nbits; n--)  
			if (sortHufStr[n].lit < sortHufStr[minIdx].lit) {
				minIdx = n;
			}
		lit = sortHufStr[minIdx].lit;
		sortHufStr[minIdx].lit = sortHufStr[i].lit;
		sortHufStr[i].lit = lit;
	}

	memset(litHuf, 0, hufCodeSize*sizeof(HufCode_Str));
	Uint32 hufCode = 0;
	Uint64 totHufBits = 0;
	for (i = nEffLits-1; i>0; i--) {
		assert(sortHufStr[i-1].nbits >= sortHufStr[i].nbits);
		litHuf[sortHufStr[i].lit].code = (Uint16)hufCode;
		litHuf[sortHufStr[i].lit].nbits = sortHufStr[i].nbits;
		totHufBits += sortHufStr[i].nbits * sortHufStr[i].freq;
		hufCode = (hufCode + 1) << (sortHufStr[i-1].nbits - sortHufStr[i].nbits);
	}
	litHuf[sortHufStr[0].lit].code = (Uint16)hufCode;
	litHuf[sortHufStr[0].lit].nbits = sortHufStr[0].nbits;
	totHufBits += sortHufStr[0].nbits * sortHufStr[0].freq;

	return (int)( (totHufBits+7)>>3 );
}


/* For dynamic Huffman encoding, the head of sequential Huffman bit widths must be packed in front of the Huffman coded data,
   so that cononical Huffman tree can be reconstructed at the decompressor
   litHuf->nbits: The huffman bit width representation of literal
   litHuf->size: Total number of literals to be constructed
   litHuf->capBits: Maximum length of Huffman Tree to be enforced
   */

void Write_Huffman_Header(Bit_Stream* bitStr, const Uint32 hufCodeSize, const Uint32 hufCodeCapBits, HufCode_Str* hufStr)
{
	Uint32 i, repLen, nBits;
	const Uint32 hufLenBits = N_Bits(hufCodeCapBits);
	register Bit_Stream bitStream = *bitStr;

	BITStream_Write(bitStream, hufStr[0].nbits, hufLenBits);

	for (i = 1; i < hufCodeSize; i++) {
		nBits = hufStr[i].nbits;
		BITStream_Write(bitStream, nBits, hufLenBits);
		if (hufStr[i - 1].nbits == nBits) {
			for (repLen = 0; i < hufCodeSize - 1 && nBits == hufStr[i + 1].nbits; i++)
				repLen++;
			if (repLen < 3) {
				BITStream_Write(bitStream, repLen, 2);
			}
			else {
				BITStream_Write(bitStream, 3, 2);
				repLen -= 3;
				if (repLen < 15) {
					BITStream_Write(bitStream, repLen, 4);
				}
				else {
					BITStream_Write(bitStream, 15, 4);
					repLen -= 15;
					BITStream_Write(bitStream, repLen, 8);
				}
			}
		}
		BITStream_Write_Flush(bitStream);
	}
	*bitStr = bitStream;
}

int Count_Huffman_Weight_Frequency(HufCode_Str* litHuf, const Uint32 hufCodeSize, Huffman_Str* hufWtHufStr, Uint8* hufWtSeq)
{
	Uint32 i, nBits, repZero, repLen;
	const Uint32 repZeroSym = MAX_HufWeight + 2;
	const Uint32 repLenSym = MAX_HufWeight + 1;
	Uint8* hufWtSeqPtr = hufWtSeq;
	if (litHuf[0].nbits) {
		*hufWtSeqPtr++ = (Uint8)litHuf[0].nbits;
		hufWtHufStr[litHuf[0].nbits].freq++;
		i = 1;
	}
	else { i = 0; }

	while(i < hufCodeSize) {
		nBits = litHuf[i].nbits;
		if (0 == nBits) {
			for (repZero = 0; i < hufCodeSize && 0 == litHuf[i].nbits && repZero < 256; i++)    /* longer runs: in chunks */
				repZero++;
			if (repZero == 1) {
				*hufWtSeqPtr++ = 0;
				hufWtHufStr[0].freq++;
			}
			else {
				*hufWtSeqPtr++ = (Uint8)repZeroSym;
				*hufWtSeqPtr++ = (Uint8)(repZero-1); /* this allow repZero=256 to represented by a byte */
				hufWtHufStr[repZeroSym].freq++;
			}
		}
		else if (nBits == litHuf[i - 1].nbits) {
			for (repLen = 1; i < hufCodeSize - 1 && nBits == litHuf[i + 1].nbits && repLen < 255; i++)    /* in chunks */
				repLen++;
			i++;
			if (repLen == 1) {
				*hufWtSeqPtr++ = (Uint8)nBits;
				hufWtHufStr[nBits].freq++;
			}
			else {
				*hufWtSeqPtr++ = (Uint8)repLenSym;
				*hufWtSeqPtr++ = (Uint8)repLen;
				hufWtHufStr[repLenSym].freq++;
			}
		}
		else {
			*hufWtSeqPtr++ = (Uint8)nBits;
			hufWtHufStr[nBits].freq++;
			i++;
		}
	}

	return (Uint32)(hufWtSeqPtr - hufWtSeq);
}

/* the bits Write_Huffman_Header_byHuffman writes for the weight sequence hufWtSeq[0, seqSize) */
Uint32 Huffman_Header_Bits(const HufCode_Str* hufLenHufStr, const Uint8* hufWtSeq, int seqSize)
{
	const Uint32 repZeroSym = MAX_HufWeight + 2;
	const Uint32 repLenSym = MAX_HufWeight + 1;
	const Uint8* p = hufWtSeq;
	const Uint8* const end = hufWtSeq + seqSize;
	Uint32 bits = 0;
	while (p < end) {
		const Uint32 nBits = *p++;
		bits += hufLenHufStr[nBits].nbits;
		if (repZeroSym == nBits) {
			const Uint32 repZero = 1 + *p++;
			bits += 3 + (repZero >= 9 ? 6 + (repZero - 9 >= 63 ? 8 : 0) : 0);
		}
		else if (repLenSym == nBits) {
			const Uint32 repLen = *p++ - 2u;
			bits += 2 + (repLen >= 3 ? 4 + (repLen - 3 >= 15 ? 8 : 0) : 0);
		}
	}
	return bits;
}

/*Write Huffman header which is also Huffman coded.
  It contains two special elements, one is the number of repeated zeros (at least 2).
  The other is the number of repeated non-zero elements (at least 2). */
 void Write_Huffman_Header_byHuffman(Bit_Stream* bitStr, HufCode_Str *hufLenHufStr, Uint8* hufWtSeq, int seqSize)
{
	Uint32 nBits, repZero, repLen;
	const Uint32 repZeroSym = MAX_HufWeight + 2;
	const Uint32 repLenSym = MAX_HufWeight + 1;
	Uint8* hufWtSeqPtr = hufWtSeq;
	const Uint8* hufWtSeqEnd = hufWtSeq + seqSize;
	register Bit_Stream bitStream = *bitStr;
	
	while( hufWtSeqPtr<hufWtSeqEnd ) {
		/*fprintf(stderr, "hufWtSeq[%d] = %d\n", (int)(hufWtSeqPtr - hufWtSeq), *hufWtSeqPtr);
		if (195 == (int)(hufWtSeqPtr - hufWtSeq)) {
			nBits += 0;
		}*/

		nBits = *hufWtSeqPtr++;
		if( nBits<repLenSym ) {
			BITStream_Write(bitStream, hufLenHufStr[nBits].code, hufLenHufStr[nBits].nbits);
		}	
		else if (repZeroSym == nBits) {
			BITStream_Write(bitStream, hufLenHufStr[repZeroSym].code, hufLenHufStr[repZeroSym].nbits);
			repZero = 1 + *hufWtSeqPtr++;
			if (repZero < 9) {
				BITStream_Write(bitStream, repZero - 2, 3);
			}
			else {
				BITStream_Write(bitStream, 7, 3);
				repZero -= 9;
				if (repZero < 63) {
					BITStream_Write(bitStream, repZero, 6);
				}
				else {
					BITStream_Write(bitStream, 63, 6);
					repZero -= 63;
					BITStream_Write(bitStream, repZero, 8);
				}
			}
		}
		else {   /*if (repLenSym == nBits) */
			repLen = *hufWtSeqPtr++;
			repLen -= 2;
			BITStream_Write(bitStream, hufLenHufStr[repLenSym].code, hufLenHufStr[repLenSym].nbits);
			if (repLen < 3) {
				BITStream_Write(bitStream, repLen, 2);
			}
			else {
				BITStream_Write(bitStream, 3, 2);
				repLen -= 3;
				if (repLen >= 15) {
					BITStream_Write(bitStream, 15, 4);
					repLen -= 15;
					BITStream_Write(bitStream, repLen, 8);
				}
				else BITStream_Write(bitStream, repLen, 4);
			}
		}		
		BITStream_Write_Flush(bitStream);
	}
	*bitStr = bitStream;
}

ForceInlineTemplate Uint32 Huffman_Compress1X_Body(const void* srcStart, Uint32 srcSize, void* dest, HufCode_Str* litHuf)
{
	Uint8* srcPtr = (Uint8*)srcStart;
	Bit_Stream bitStream = { 0, 0, (Uint8*)dest };
	Uint8* srcEnd;
	Uint32 srcModSize;

	srcModSize = srcSize & ~3;
	srcEnd = srcPtr + srcModSize;
	while (srcPtr < srcEnd) {
		BITStream_Write(bitStream, litHuf[*srcPtr].code, litHuf[*srcPtr].nbits);   srcPtr++;
		BITStream_Write(bitStream, litHuf[*srcPtr].code, litHuf[*srcPtr].nbits);   srcPtr++;
		BITStream_Write(bitStream, litHuf[*srcPtr].code, litHuf[*srcPtr].nbits);   srcPtr++;
		BITStream_Write(bitStream, litHuf[*srcPtr].code, litHuf[*srcPtr].nbits);   srcPtr++;
		BITStream_Write_Flush(bitStream);
	}

	switch (srcSize & 3) {
	case 3:
		BITStream_Write(bitStream, litHuf[*srcPtr].code, litHuf[*srcPtr].nbits);   srcPtr++;
	case 2:
		BITStream_Write(bitStream, litHuf[*srcPtr].code, litHuf[*srcPtr].nbits);   srcPtr++;
	case 1:
		BITStream_Write(bitStream, litHuf[*srcPtr].code, litHuf[*srcPtr].nbits);
	default:
		BITStream_Write_FlushEnd(bitStream);
	}

	return (Uint32)(bitStream.streamPtr - (Uint8*)dest);
}

Uint32 Huffman_Compress1X_Kernel(const void* srcStart, Uint32 srcSize, void* dest, HufCode_Str* litHuf)
{
	return Huffman_Compress1X_Body(srcStart, srcSize, dest, litHuf);
}


/* It assumes the source literals may be arranged in either direction, 1 being normal starting, -1 being reverse */
Uint32  Huffman_Compress4X_Kernel(const void* srcStart, Uint32 srcSize, void* dest, HufCode_Str* litHuf)
{
	Uint8* srcPtr = (Uint8*)srcStart;
	Uint8* destPtr = (Uint8*)dest + 6;                      /* the first 6 bytes are reserved to record 3 segment starting positions */

	int srcSegSize = (srcSize>>4) <<2 ;                     /* divide into 4 segments wherein the first three divides 4 */
	Uint32 resSegSize[4];

	resSegSize[0] = Huffman_Compress1X_Kernel(srcPtr,                  srcSegSize, destPtr, litHuf);
	destPtr += resSegSize[0];

	resSegSize[1] = Huffman_Compress1X_Kernel(srcPtr + srcSegSize,      srcSegSize, destPtr, litHuf);
	destPtr += resSegSize[1];
	resSegSize[1] += resSegSize[0];           /* accumulation */

	resSegSize[2] = Huffman_Compress1X_Kernel(srcPtr + 2* srcSegSize,  srcSegSize,  destPtr, litHuf);
	destPtr += resSegSize[2];
	resSegSize[2] += resSegSize[1];

	resSegSize[3] = Huffman_Compress1X_Kernel(srcPtr + 3* srcSegSize,  srcSize-3*srcSegSize, destPtr, litHuf);
	destPtr += resSegSize[3];
	resSegSize[3] += resSegSize[2];

	/* Record segment starting locations */
	destPtr = (Uint8*)dest;
	MemWriteLE2(destPtr, (Uint16)resSegSize[0]);    
	MemWriteLE2(destPtr+2, (Uint16)resSegSize[1]);
	MemWriteLE2(destPtr+4, (Uint16)resSegSize[2]);

	return resSegSize[3] + 6;
}

/* bits of the symbols counted in h under code; all ones if code lacks one of them */
Uint64 Huffman_Code_Bits(const Huffman_Str* h, const HufCode_Str* code, const Uint32 n)
{
	Uint64 bits = 0;
	for (Uint32 k = 0; k < n; k++)
		if (h[k].freq) {
			if (0 == code[k].nbits) return ((Uint64)-1);
			bits += (Uint64)h[k].freq * code[k].nbits;
		}
	return bits;
}

/* Codes a block of literals: stored (type 0), with a code of its own whose lengths the block carries (type 1), or
   with the code of the last type-1 block of the stream, prev (type 2), whichever is smallest; a type-1 block becomes
   prev. litHuf: the block's literal counts. prev may be NULL (no type 2). */
Uint32 Huffman_Compress_Block_Rep(const void* srcStart, Uint32 srcSize, void* dest, Huffman_Str* litHuf, Uint32 nLits,
	Uint32 litHufCapBits, Huffman_Prev* prev)
{
	HufCode_Str litHufCode[MAX_HufSize];
	Huffman_Str hufHufStr[MAX_HufWeight + 3] = { 0 };
	HufCode_Str hufHufCode[MAX_HufWeight + 3];
	Uint8 hufWtSeq[MAX_HufSize];
	Uint8 header[HUF_HeaderBound];
	Uint8* destPtr = (Uint8*)dest;
	Uint32 comprSize;

	const Uint32 estSize = (Uint32)Build_Huffman_Table(litHuf, nLits, litHufCapBits, litHufCode);    /* coded size in bytes, no header */
	const Uint64 prevBits = prev && prev->valid ? Huffman_Code_Bits(litHuf, prev->code, nLits) : ((Uint64)-1);
	Uint32 headerSize = 0;
	if (estSize < srcSize && srcSize >= 64) {            /* the header of a code of its own */
		const Uint32 seqSize = Count_Huffman_Weight_Frequency(litHufCode, nLits, hufHufStr, hufWtSeq);
		Build_Huffman_Table(hufHufStr, MAX_HufWeight + 3, MAX_HufHufWt, hufHufCode);
		Bit_Stream bitStream = { 0, 0, header };
		Write_Huffman_Header(&bitStream, MAX_HufWeight + 3, MAX_HufHufWt, hufHufCode);
		Write_Huffman_Header_byHuffman(&bitStream, hufHufCode, hufWtSeq, seqSize);
		BITStream_Write_FlushEnd(bitStream);
		headerSize = (Uint32)(bitStream.streamPtr - header);
	}
	/* sizes past the type byte and body size: the previous code's, a new one's; the previous code also where a
	   new one would not pay for its header (as on a short last block) */
	const Uint64 repSize = prevBits == ((Uint64)-1) ? ((Uint64)-1) : (prevBits + 7) / 8;
	const Uint64 newSize = headerSize ? (Uint64)headerSize + estSize : ((Uint64)-1);
	const int useRep = repSize <= newSize;
	if ((useRep ? repSize : newSize) >= srcSize) {       /* incompressible scenario */
		*destPtr++ = 0;
		MemWildCopy( destPtr, srcStart, destPtr + srcSize );
		return srcSize + 1;
	}

	/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Huffman Compression ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
	*destPtr++ = useRep ? 2 : 1;
	destPtr += 2;                                        /* the body's size */
	if (!useRep) {
		memcpy(destPtr, header, headerSize);
		destPtr += headerSize;
	}
	const HufCode_Str* const code = useRep ? prev->code : litHufCode;
	if (srcSize >= MinStream4XSize) {
		comprSize = Huffman_Compress4X_Kernel(srcStart, srcSize, destPtr, (HufCode_Str*)code);
	}
	else {
		comprSize = Huffman_Compress1X_Kernel(srcStart, srcSize, destPtr, (HufCode_Str*)code);
	}

	if (comprSize >= 99 * srcSize / 100) {              /* nearly incompressible (99% or more): store; in integers, alike everywhere */
		destPtr = (Uint8*)dest;
		*destPtr++ = 0;
		MemWildCopy(destPtr, srcStart, destPtr + srcSize);
		return srcSize + 1;
	}

	assert(comprSize < (1u << 16));                      /* below 0.99 of a block of at most 64K literals */
	MemWriteLE2((Uint8*)dest+1, (Uint16)comprSize);            /* Record the compressed literal size, excluding Huffman header */
	destPtr += comprSize;                              /* it is used to determine the decompressor type  */
	if (!useRep && prev) {
		memcpy(prev->code, litHufCode, nLits * sizeof(HufCode_Str));
		prev->valid = 1;
	}

	return (Uint32)(destPtr - (Uint8*)dest);            /* The second part is the size of the Huffman header */
}

Uint32  Huffman_Compress_Block(const void* srcStart, Uint32 srcSize, void* dest, Huffman_Str* litHuf, Uint32 nLits, Uint32 litHufCapBits)
{
	return Huffman_Compress_Block_Rep(srcStart, srcSize, dest, litHuf, nLits, litHufCapBits, NULL);
}

Uint32 Huffman_Compress(const void* srcStart, Uint32 srcSize, void* dest, Uint32 nLits, Uint32 litHufCapBits)
{
	Huffman_Str litHuf[MAX_HufSize];

	Uint32 i, n;
	Uint32 nFullBlocks = srcSize / HUF_BlockSize;
	Uint32 lastBlockSize = srcSize - HUF_BlockSize * nFullBlocks;
	Uint8* srcPtr = (Uint8*)srcStart;
	Uint8* destPtr = (Uint8*)dest;
	Uint32 blockComprSize = 0;

	for (n = 0; n < nFullBlocks; n++) {
		memset(litHuf, 0, MAX_HufSize * sizeof(Huffman_Str));
		for (i = 0; i < HUF_BlockSize; i++)
			litHuf[*srcPtr++].freq++;
		blockComprSize = Huffman_Compress_Block((Uint8*)srcStart + n * HUF_BlockSize, HUF_BlockSize, destPtr, litHuf, nLits, litHufCapBits);
		destPtr += blockComprSize;
	}
	if (lastBlockSize) {
		memset(litHuf, 0, MAX_HufSize * sizeof(Huffman_Str));
		for (i = 0; i < lastBlockSize; i++)
			litHuf[*srcPtr++].freq++;
		blockComprSize = Huffman_Compress_Block((Uint8*)srcStart + n * HUF_BlockSize, lastBlockSize, destPtr, litHuf, nLits, litHufCapBits);
		destPtr += blockComprSize;
	}

	return (Uint32)(destPtr - (Uint8*)dest);
}
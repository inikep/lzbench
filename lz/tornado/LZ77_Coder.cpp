// Code for byte/bit/huffman/arithmetic encoders for LZ77 output
//
// (c) Bulat Ziganshin
// (c) Joachim Henke
// This code is provided on the GPL license.
// If you need a commercial license to use the code, please write to Bulat.Ziganshin@gmail.com

// These len/dist codes are used to encode EOF and other special cases
#define IMPOSSIBLE_LEN  (INT_MAX/2)
#define IMPOSSIBLE_DIST (INT_MAX/2)

// LZ77 literal/match bytecoder *******************************************************************

struct LZ77_ByteCoder : OutputByteStream
{
    // Used to save literal/match flags, grouped by 16 values = 32 bits
    uint     flags;       // Flags word
    uint     flagbit;     // Current bit in flags word (flags are filled when flagbit==2^32)
    uint32   garbage;     // We store here first, garbage value of flags


    // Encoding statistics
    int chars, matches2, matches3, matches4;

    // Init and finish encoder
    LZ77_ByteCoder (int coder, CALLBACK_FUNC *callback, void *auxdata, UINT chunk, UINT pad);
    void finish();

    void before_shift(BYTE *p) {}   // Called prior to shifting buffer contents to the beginning
    void after_shift (BYTE *p) {}   // The same, after shift

    // Writes match/literal into output. Returns 0 - literal encoded, 1 - match encoded
    int encode (int len, byte *current, byte *match, const int MINLEN)
    {
        // Save final value of flags word into reserved place of outbuf when all 32 bits of flag are filled
        if ((flagbit<<=2) == 0) {   // 1u<<32 for 64-bit systems
            debug (printf (" flags %x\n", flags));
            // Flags are filled now, save the old value and start a new one
            setvalue32(get_anchor(), flags);
            flags=0, flagbit=1;
            set_anchor(output), output+=4;
        }

        if (len<MINLEN) {
            stat_only (chars++);
            put8 (*current);
            return 0;
        }
        uint dist = current - match;
        if (len<MINLEN+16 && dist<(1<<12)) {
            stat_only (matches2++);
            put16 (((len-MINLEN)<<12) + dist);
            flags += flagbit;    // Mark this position as short match
        } else if (len<MINLEN+64 && dist<(1<<18)) {
            stat_only (matches3++);
            put24 (((len-MINLEN)<<18) + dist);
            flags += flagbit*2;  // Mark this position as medium-length match
        } else {
            stat_only (matches4++);
            len -= MINLEN;
            if (dist >= (1<<24))   put8 (255), put8  (dist>>24);
            if (len>=254)          put8 (254), put24 (len>>8), len%=256;
            put32 (len + (dist<<8));
            flags += flagbit*3;  // Mark this position as long match
        }
        return 1;
    }

    static const bool support_tables = FALSE;
    // Send info about diffed table. type=1..4 and len is number of table elements
    void encode_table (int type, int len)
    {
        CHECK (FREEARC_ERRCODE_INTERNAL,  FALSE, (s,"Fatal error: encode_table() isn't implemented in this coder"));
    }

    // Bits required to encode single literal/match
    PRICE lit_price   (UINT c)            {return                         10;}
    PRICE match_price (int len, int dist) {return len<16 && dist<(1<<12)? 18:
                                                  len<64 && dist<(1<<18)? 26:
                                                                          34 + (dist >= (1<<24)? 16 : 0) + (len>=254? 32 : 0);}

    static const bool support_repdist = FALSE;
    PRICE repdist_price(int,int)  {return 0;}
};

LZ77_ByteCoder::LZ77_ByteCoder (int coder, CALLBACK_FUNC *callback, void *auxdata, UINT chunk, UINT pad)
    : OutputByteStream (callback, auxdata, chunk, pad)
{
    chars = matches2 = matches3 = matches4 = 0;
    // Start a flags business
    flags   = 0;
    flagbit = 0;
    set_anchor (&garbage);
}

void LZ77_ByteCoder::finish()
{
#ifdef STAT
    printf ("\rLiterals %d, matches %d = %d + %d + %d                   \n",
        chars/1000, (matches2+matches3+matches4)/1000, matches2/1000, matches3/1000, matches4/1000);
#endif
    setvalue32(get_anchor(), flags);
    OutputByteStream::finish();
}


// And that's the decoder
struct LZ77_ByteDecoder : InputByteStream
{
    // Used to save literal/match flags, grouped by 16 values = 32 bits
    uint     flags;       // Flags word
    uint     flagpos;

    // Init decoder
    LZ77_ByteDecoder (CALLBACK_FUNC *callback, void *auxdata, UINT bufsize) : InputByteStream(callback, auxdata, bufsize)  {flagpos=1;}

    // Decode next element and return true if it's a literal
    uint is_literal (void)
    {
        if (--flagpos)   flags>>=2;
        else flagpos=16, flags = get32(), debug (printf (" flags %x\n", flags));
        return (flags&3) == 0;
    }
    uint dist;  // Temporary storage for decoded distance
    // Decode literal
    uint getchar (void)
    {
        return getc();
    }
    // Decode length (should be called before getdist!)
    uint getlen (const uint MINLEN)
    {
        uint x, len;
        switch (flags&3) {
        case 1:  x = get16(); len = x>>12; dist = x%(1<<12); break;
        case 2:  x = get24(); len = x>>18; dist = x%(1<<18); break;
        case 3:  len = get8();
                 if (len==255)  dist=get8()<<24, len=get8();  else dist = 0;
                 if (len==254)  len=get24()<<8,  len+=get8();
                 dist += get24();
                 break;
        }
        return MINLEN + len;
    }
    // Decode distance
    uint getdist (void)
    {
        return dist;
    }
};


// Variable-length data coder *********************************************************************

// We support up to a 256 codes, which encodes values
// up to a 2^30 (using encoding one can find in the DistanceCoder)
#define MAX_CODE 256
#define VLE_SIZE (1024+16384+1)
struct VLE
{
    uchar xcode       [VLE_SIZE];     // Code for each (preprocessed) value
    uint  xextra_bits [MAX_CODE];     // Amount of extra bits for each code
    uint  xbase_value [MAX_CODE];     // Base (first) value for each code

    VLE (uint _extra_bits[], uint extra_bits_size);

    uint code (uint value)
    {
        return xcode[value];
    }
    uint extra_bits (uint code)
    {
        return xextra_bits[code];
    }
    uint base_value (uint code)
    {
        return xbase_value[code];
    }
};

// Inits array used for the fast value->code mapping.
// Each entry in extra_bits[] corresponds to exactly one code
// and tells us how many additional bits are used with this code.
// So, extra_bits_size == number of codes used
VLE::VLE (uint _extra_bits[], uint extra_bits_size)
{
    // Initialize the mappings value -> code and code -> base value
    uint value = 0;
    for (uint code = 0; code < extra_bits_size; code++) {
        xextra_bits[code] = _extra_bits[code];
        xbase_value[code] = value;
        for (uint n = 0; n < (1<<xextra_bits[code]); n++) {
            if (value>=VLE_SIZE)  break;
            xcode[value++] = (uchar)code;
        }
    }
}


// Extra bits for each length code (for bitcoder and aricoder, accordingly)
uint extra_lbits [8]  = {0,0,0,1,2,4,8,30};
uint extra_lbits2[16] = {0,0,0,0,0,0,0,1,1,2,2,3,3,4,8,30};
//uint extra_lbits2[32] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,5,6,7,8,30};

// Variable-length encoder for LZ match lengths
template <unsigned ELEMENTS>
struct LengthCoder : VLE
{
    LengthCoder (uint _extra_bits[])  :  VLE (_extra_bits, ELEMENTS) {};

    uint code (uint length)
    {
        return length>600?  ELEMENTS-1  :  VLE::code(length);
    }
};

LengthCoder <elements(extra_lbits)>   lc  (extra_lbits);
LengthCoder <elements(extra_lbits2)>  lc2 (extra_lbits2);


// Extra bits for each distance code
//uint extra_dbits[16] = {6,6,7,8,9,10,11,12,13,14,15,16,17,19,22,30};
uint extra_dbits[32] = {4,4,5,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,16,17,18,19,21,23,30};
//uint extra_dbits2[63] = {0,3,3,3,3,3,3,4,4,4,4,4,5,5,5,5,6,6,6,6,8,8,8,8,8,8,9,9,9,9,10,10,10,10,11,11,
//                         11,11,12,12,12,12,13,13,13,13,14,14,14,14,16,16,16,17,17,18,18,19,20,21,22,23,30};
//uint extra_dbits[] = {0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,16,16,17,17,18,18,19,19,20,20,21,21,22,22,23,23,24,24,25,25,26,26,27,27,30};

// Variable-length encoder for LZ match distances up to 1Gb
struct DistanceCoder : VLE
{
    DistanceCoder (uint _extra_bits[], uint extra_bits_size);

    uint code (uint distance)
    {
        return distance < 512     ? xcode[distance] :
               distance < 512*256 ? xcode[512+(distance>>8)]
                                  : xcode[1024+(distance>>16)];
    }
} dc (extra_dbits,  elements(extra_dbits));


// Distance coder has its own init routine which takes into account
// wide range of encoded values and therefore more complex
// code() routine having 3 branches
DistanceCoder::DistanceCoder (uint _extra_bits[], uint extra_bits_size) : VLE (0,0)
{
    /* Initialize the mapping dist (0..1G) -> dist code (0..15) */
    int dist = 0, code;
    for (code = 0; dist < 512; code++) {
        xextra_bits[code] = _extra_bits[code];
        xbase_value[code] = dist;
        for (uint n = 0; n < (1<<_extra_bits[code]); n++) {
            xcode[mymax(dist,0)] = (uchar)code;
            dist++;
        }
    }
    dist >>= 8; /* from now on, all distances are divided by 256 */
    for ( ; dist < 512; code++) {
        xextra_bits[code] = _extra_bits[code];
        xbase_value[code] = dist << 8;
        CHECK (FREEARC_ERRCODE_INTERNAL,  _extra_bits[code] >= 8,  (s,"Fatal error: DistanceCoder::_extra_bits[%d] = %d is lower than minimum allowed value 8", code, _extra_bits[code]));
        for (uint n = 0; n < (1<<(_extra_bits[code]-8)); n++) {
            xcode[512 + dist++] = (uchar)code;
        }
    }
    dist >>= 8; /* from now on, all distances are divided by 65536 */
    for ( ; code < extra_bits_size; code++) {  // distances up to 1G
        xextra_bits[code] = _extra_bits[code];
        xbase_value[code] = dist << 16;
        CHECK (FREEARC_ERRCODE_INTERNAL,  _extra_bits[code] >= 16,  (s,"Fatal error: DistanceCoder::_extra_bits[%d] = %d is lower than minimum allowed value 8", code, _extra_bits[code]));
        for (uint n = 0; n < (1<<(_extra_bits[code]-16)); n++) {
            if (1024+dist >= elements(xcode))  break;
            xcode[1024 + dist++] = (uchar)code;
        }
    }
}


// LZ77 literal/match bit-precise coder ***********************************************************

// It's the coder
struct LZ77_BitCoder : OutputBitStream
{
#ifdef STAT
    int chars, matches, lencnt[8], distcnt[32];  // Encoding statistics
#endif

    // Init and finish encoder
    LZ77_BitCoder (int coder, CALLBACK_FUNC *callback, void *auxdata, UINT chunk, UINT pad);
    void finish();

    void before_shift(BYTE *p) {}   // Called prior to shifting buffer contents to the beginning
    void after_shift (BYTE *p) {}   // The same, after shift

    // Writes match/literal into buffer. Returns 0 - literal encoded, 1 - match encoded
    int encode (int len, byte *current, byte *match, const int MINLEN)
    {
        // Encode a literal if match is too short
        if ((len-=MINLEN) < 0)  {
            stat_only (chars++);
            putbits (9, *current);
            return 0;
        }

        // It's a match
        stat_only (matches++);
        uint dist = current - match;

        // Find len code
        uint lcode = lc.code(len);
        uint lbits = lc.extra_bits(lcode);
        uint lbase = lc.base_value(lcode);
        stat_only (lencnt[lcode]++);

        // Find dist code
        uint dcode = dc.code(dist);
        uint dbits = dc.extra_bits(dcode);
        uint dbase = dc.base_value(dcode);
        stat_only (distcnt[dcode]++);

        // Send combined len/dist code and remaining bits
        putbits (9, 256 + (lcode<<5) + dcode);
        putlowerbits (lbits, len-lbase);
        putlowerbits (dbits, dist-dbase);
        return 1;
    }

    static const bool support_tables = FALSE;
    // Send info about diffed table. type=1..4 and len is number of table elements
    void encode_table (int type, int len)
    {
        CHECK (FREEARC_ERRCODE_INTERNAL,  FALSE,  (s,"Fatal error: encode_table() isn't implemented in this coder"));
    }

    // Bits required to encode single literal/match
    PRICE lit_price (UINT c)
    {
        return 9;
    }
    PRICE match_price (int len, int dist)
    {
        uint lcode = lc.code(len);
        uint lbits = lc.extra_bits(lcode);
        uint dcode = dc.code(dist);
        uint dbits = dc.extra_bits(dcode);
        return 9+lbits+dbits;
    }

    static const bool support_repdist = FALSE;
    PRICE repdist_price(int,int)  {return 0;}
};

LZ77_BitCoder::LZ77_BitCoder (int coder, CALLBACK_FUNC *callback, void *auxdata, UINT chunk, UINT pad)
    : OutputBitStream (callback, auxdata, chunk, pad)
{
#ifdef STAT
    iterate_var(i,8)   lencnt[i]  = 0;
    iterate_var(i,32)  distcnt[i] = 0;
    chars = matches = 0;
#endif
}

void LZ77_BitCoder::finish()
{
#ifdef STAT
    printf ("\rLiterals %d, matches %d. Length codes:", chars/1000, matches/1000);
    iterate_var(i,8)   printf (" %d", lencnt[i]/1000);
    printf ("\n");
    iterate_var(i,32)  printf (" %d", distcnt[i]/1000);
    printf ("\n");
#endif
    OutputBitStream::finish();
}



// And that's the decoder
struct LZ77_BitDecoder : InputBitStream
{
    // Init decoder
    LZ77_BitDecoder (CALLBACK_FUNC *callback, void *auxdata, UINT bufsize) : InputBitStream(callback, auxdata, bufsize) {};

    uint x;  // Temporary value used for storing first 9 bits of code

    // Decode next element and return true if it's a literal
    uint is_literal (void)
    {
        x = InputBitStream::getbits(9);
        return (x < 256);
    }
    // Decode literal
    uint getchar (void)
    {
        return x;
    }
    // Decode length (should be called before getdist!)
    uint getlen (const uint MINLEN)
    {
        uint lcode = (x>>5)-8;
        uint lbits = lc.extra_bits(lcode);
        uint lbase = lc.base_value(lcode);
        return MINLEN + lbase + InputBitStream::getbits(lbits);
    }
    // Decode distance
    uint getdist (void)
    {
        uint dcode = x & 31;
        uint dbits = dc.extra_bits(dcode);
        uint dbase = dc.base_value(dcode);
        return dbase + InputBitStream::getbits(dbits);
    }
};


// LZ77 literals/matches generic coder ************************************************************

// Amount of "repeat previous distance", length and distance codes
const int REPDIST_CODES = 4;
const int DIST_CODES = elements(extra_dbits)+REPDIST_CODES;
const int LEN_CODES  = elements(extra_lbits2);
// First code after regular ones - End Of Block
const int EOB_CODE = 256 + LEN_CODES*DIST_CODES;
// Another code - copy char at repdist0 distance
const int REPCHAR = EOB_CODE + 1;
// One more code - repeat both length & distance
const int REPBOTH = EOB_CODE + 2;
// Total amount of codes, including 7 spare ones
const int CODES   = EOB_CODE + 10;

// It's the coder
template <class Coder>
struct LZ77_Coder : public Coder
{
#ifdef STAT
    int chars, matches, rep0s, lencnt[LEN_CODES], distcnt[DIST_CODES]; // Encoding statistics
    int cnt[LEN_CODES][DIST_CODES];
#endif
    int prevdist3, prevdist2, prevdist1, prevdist0;  // last distances encoded so far

    // Init and finish encoder
    LZ77_Coder (int coder, CALLBACK_FUNC *callback, void *auxdata, UINT chunk, UINT pad);
    void finish();

    void before_shift(BYTE *p) {}                // Called prior to shifting buffer contents to the beginning
    void after_shift (BYTE *p) {prevdist0=-1;}   // The same, after shift. We need to invalidate prevdist0 in order to avoid adressing data shifted out when checking for REPCHAR

    // Writes match/literal into buffer. Returns 0 - literal encoded, 1 - match encoded
    int encode (int len, byte *current, byte *match, const int MINLEN)
    {
        // Encode a literal if match is too short
        if ((len-=MINLEN) < 0)  {
            if (*current == current[-prevdist0-1] && prevdist0>=0)
                 Coder::encode (REPCHAR),  stat_only (rep0s++), debug (printf (" REPCHAR: \n"));
            else Coder::encode (*current), stat_only (chars++);
            return 0;
        }

        // It's a match
        stat_only (matches++);
        encode_match (len, current-match-1);
        return 1;
    }
    // Writes match into buffer
    void encode_match (int len, int dist)
    {
        // Find dist code, checking first for "repeat previous distance" cases
        uint dcode, dbits, dbase, x, y;
             if (x=prevdist0, prevdist0=dist, dist==x)   dcode=0, dbits=0;
        else if (y=prevdist1, prevdist1=x,    dist==y)   dcode=1, dbits=0;
        else if (x=prevdist2, prevdist2=y,    dist==x)   dcode=2, dbits=0;
        else if (y=prevdist3, prevdist3=x,    dist==y)   dcode=3, dbits=0;
        else {
            dcode = dc.code(dist);
            dbits = dc.extra_bits(dcode);
            dbase = dc.base_value(dcode);
            dcode += REPDIST_CODES;
        }
        stat_only (distcnt[dcode]++);
        debug (dcode<REPDIST_CODES &&  printf (" REPDIST: %d\n", dcode));

        // Improve table encoding by using codes of lengths 101..104 to represent tables
        // Also invalidates prevdist0 because it should contain only real distances, otherwise check for REPCHAR may crash
        if (len>100) {
            if (len>IMPOSSIBLE_LEN) {
                debug (printf (" TABLE: %d*%d\n", len-IMPOSSIBLE_LEN, dist));
                prevdist0=-1;
                if (len<=IMPOSSIBLE_LEN+4)  len -= IMPOSSIBLE_LEN-100;
            } else {
                len+=4;
            }
        }

        // Find len code
        uint lcode = lc2.code(len);
        uint lbits = lc2.extra_bits(lcode);
        uint lbase = lc2.base_value(lcode);
        stat_only (lencnt[lcode]++);
        stat_only (cnt[lcode][dcode]++);

        // Send combined len/dist code and remaining bits
        Coder::encode (256 + dcode*elements(extra_lbits2) + lcode);
        Coder::putlowerbits (lbits, len-lbase);
        Coder::putlowerbits (dbits, dist-dbase);
    }

    static const bool support_repdist = TRUE;
    static const bool support_tables  = TRUE;
    // Send info about diffed table. type=1..4 and len is number of table elements
    void encode_table (int type, int len)
    {
        encode_match (IMPOSSIBLE_LEN+type, len-1);
    }

    // Bits required to encode single literal/match
    PRICE lit_price (UINT c)
    {
        return Coder::price(c,0);
    }
    PRICE match_price (int len, int dist)
    {
        uint dcode = dc.code(dist);
        uint dbits = dc.extra_bits(dcode);
        return price_len_dcode(len, dcode+REPDIST_CODES, dbits);
    }
    PRICE repdist_price (int len, int prevdist)
    {
        return price_len_dcode(len, prevdist, 0);
    }
    PRICE price_len_dcode (int len, uint dcode, uint dbits)
    {
        uint lcode = lc2.code(len);
        uint lbits = lc2.extra_bits(lcode);
        uint code = 256 + dcode*elements(extra_lbits2) + lcode;
        return Coder::price(code,lbits+dbits);
    }
};

template <class Coder>
LZ77_Coder<Coder> :: LZ77_Coder (int coder, CALLBACK_FUNC *callback, void *auxdata, UINT chunk, UINT pad) :
    Coder (callback, auxdata, chunk, pad, CODES)
{
#ifdef STAT
    chars = matches = rep0s = 0;
    iterate_var(i,LEN_CODES)       lencnt[i]  = 0;
    iterate_var(i,DIST_CODES)      distcnt[i] = 0;
    iterate_var(i,LEN_CODES)
        iterate_var(j,DIST_CODES)  cnt[i][j] = 0;
#endif
    prevdist3=prevdist2=prevdist1=prevdist0=-1;
}

template <class Coder>
void LZ77_Coder<Coder> :: finish()
{
#ifdef STAT
    printf ("\rLiterals %d, matches %d, rep0s %d\n Length codes:", chars/1000, matches/1000, rep0s/1000);
    iterate_var(i,LEN_CODES)         printf (" %d", lencnt[i]/1000);
    printf ("\n");
    iterate_var(i,DIST_CODES)        printf (" %d", distcnt[i]/1000);
    printf ("\n");
    iterate_var(i,LEN_CODES) {
        printf ("%d:", lc2.base_value(i)+2);
        iterate_var(j,DIST_CODES)    printf (" %d", cnt[i][j]/1000);
        printf ("\n");
    }
#endif
    Coder::finish();
}



// And that's the decoder
template <class Decoder>
struct LZ77_Decoder : Decoder
{
    // Init decoder
    LZ77_Decoder (CALLBACK_FUNC *callback, void *auxdata, UINT bufsize) : Decoder (callback, auxdata, bufsize, CODES)
    {
        iterate_var(i,REPDIST_CODES)  prevdists[i]=0;
        prevdist = prevdists+REPDIST_CODES;
    }

    // prevdists[] saves last REPDIST_CODES distances encoded so far. It has larger size to improve performance.
    // *prevdist points to the last distance saved in this buffer. When buffer overflows, we move its last REPDIST_CODES into beginning
    int prevdists[128], *prevdist;
    uint x;  // Temporary value used for storing code (char or len/dist slot of match)

    // Decode next element and return true if it's a literal
    uint is_literal (void)
    {
        x = Decoder::decode();
        return (x < 256);
    }
    // Decode literal
    uint getchar (void)
    {
        return x;
    }
    // Decode length (should be called before getdist!)
    uint getlen (const uint MINLEN)
    {
        if (x==REPCHAR)  return 1;
        uint lcode = x % elements(extra_lbits2);
        uint lbits = lc2.extra_bits(lcode);
        uint lbase = lc2.base_value(lcode);
        uint len   = lbase + Decoder::getbits(lbits);
        return len>100?  (len<=104? len-100+IMPOSSIBLE_LEN
                                  : len-4+MINLEN)
                      :             len+MINLEN;
    }
    // Decode distance
    uint getdist (void)
    {
        if (x==REPCHAR)  return prevdist[-1];
        int dcode = (x-256) / elements(extra_lbits2), dist;
        if ((dcode-=REPDIST_CODES) < 0) {
            switch (dcode) {
            case -4: return prevdist[-1];
            case -3: dist = prevdist[-2]; prevdist[-2]=prevdist[-1]; prevdist[-1]=dist; return dist;
            case -2: dist = prevdist[-3]; prevdist[-3]=prevdist[-2]; prevdist[-2]=prevdist[-1]; prevdist[-1]=dist; return dist;
            default: dist = prevdist[-4]; prevdist[-4]=prevdist[-3]; prevdist[-3]=prevdist[-2]; prevdist[-2]=prevdist[-1]; prevdist[-1]=dist; return dist;
            }
        }
        uint dbits = dc.extra_bits(dcode);
        uint dbase = dc.base_value(dcode);
        dist = dbase + Decoder::getbits(dbits) + 1;
            // Move last REPDIST_CODES-1 distances into beginning of prevdists[] if we reached end of this array
            if (prevdist==endof(prevdists)) {
                iterate_var(i,REPDIST_CODES-1)   prevdists[i] = endof(prevdists)[i-(REPDIST_CODES-1)];
                prevdist = prevdists+REPDIST_CODES-1;
            }
            // Save the new distance into prevdists[]
            *prevdist++ = dist;
        return dist;
    }
};


// Dynamic coder (selects at run-time between byte/bit/huffman/ari coders) ************************
struct LZ77_DynamicCoder
{
    bool support_repdist;
    bool support_tables;
    int coder;
    LZ77_ByteCoder                         coder1;
    LZ77_BitCoder                          coder2;
    LZ77_Coder<HuffmanEncoder<EOB_CODE> >  coder3;
    LZ77_Coder<ArithCoder<EOB_CODE> >      coder4;

    // Init and finish encoder
    LZ77_DynamicCoder (int _coder, CALLBACK_FUNC *callback, void *auxdata, UINT chunk, UINT pad);
    void finish();
    void before_shift(BYTE *p);   // Called prior to shifting buffer contents to the beginning
    void after_shift (BYTE *p);   // The same, after shift
    void encode_table (int type, int len);   // Send info about diffed table. type=1..4 and len is number of table elements
    int  error();
    void put8 (uint c);
    void put32 (uint c);
    void flush();
    uint64 outsize();             // Number of bytes already written to coder

    void set_context(int i)
    {
        switch (coder)
        {
        case 4: return coder4.set_context(i);
        case 3: return coder3.set_context(i);
        case 2: return coder2.set_context(i);
        default:return coder1.set_context(i);
        }
    }

    // Writes match/literal into output. Returns 0 - literal encoded, 1 - match encoded
    int encode (int len, byte *current, byte *match, const int MINLEN)
    {
        switch (coder)
        {
        case 4: return coder4.encode (len, current, match, MINLEN);
        case 3: return coder3.encode (len, current, match, MINLEN);
        case 2: return coder2.encode (len, current, match, MINLEN);
        default:return coder1.encode (len, current, match, MINLEN);
        }
    }

    // Number of bits required to encode the literal
    PRICE lit_price (UINT c)
    {
        switch (coder)
        {
        case 4: return coder4.lit_price(c);
        case 3: return coder3.lit_price(c);
        case 2: return coder2.lit_price(c);
        default:return coder1.lit_price(c);
        }
    }
    // Number of bits required to encode the match
    PRICE match_price (int len, int dist)
    {
        switch (coder)
        {
        case 4: return coder4.match_price(len,dist);
        case 3: return coder3.match_price(len,dist);
        case 2: return coder2.match_price(len,dist);
        default:return coder1.match_price(len,dist);
        }
    }
    // Number of bits required to encode match with the same distance as the one of a few last matches
    PRICE repdist_price (int len, int prevdist)
    {
        switch (coder)
        {
        case 4:  return coder4.repdist_price(len,prevdist);
        default: return coder3.repdist_price(len,prevdist);
        }
    }
};

LZ77_DynamicCoder::LZ77_DynamicCoder (int _coder, CALLBACK_FUNC *callback, void *auxdata, UINT chunk, UINT pad):
    coder1 (_coder, callback, auxdata, _coder==1? chunk : 0, pad),
    coder2 (_coder, callback, auxdata, _coder==2? chunk : 0, pad),
    coder3 (_coder, callback, auxdata, _coder==3? chunk : 0, pad),
    coder4 (_coder, callback, auxdata, _coder==4? chunk : 0, pad)
{
    coder = _coder;
    support_repdist = support_tables = (coder>=3);
}

void LZ77_DynamicCoder::finish()
{
    switch (coder)
    {
    case 4: return coder4.finish();
    case 3: return coder3.finish();
    case 2: return coder2.finish();
    default:return coder1.finish();
    }
}

// Called prior to shifting buffer contents to the beginning
void LZ77_DynamicCoder::before_shift(BYTE *p)
{
    switch (coder)
    {
    case 4: return coder4.before_shift(p);
    case 3: return coder3.before_shift(p);
    case 2: return coder2.before_shift(p);
    default:return coder1.before_shift(p);
    }
}

// Called after shifting buffer contents to the beginning
void LZ77_DynamicCoder::after_shift(BYTE *p)
{
    switch (coder)
    {
    case 4: return coder4.after_shift(p);
    case 3: return coder3.after_shift(p);
    case 2: return coder2.after_shift(p);
    default:return coder1.after_shift(p);
    }
}

void LZ77_DynamicCoder::encode_table (int type, int len)
{
    switch (coder)
    {
    case 4: return coder4.encode_table (type, len);
    case 3: return coder3.encode_table (type, len);
    case 2: return coder2.encode_table (type, len);
    default:return coder1.encode_table (type, len);
    }
}

int LZ77_DynamicCoder::error()
{
    switch (coder)
    {
    case 4: return coder4.error();
    case 3: return coder3.error();
    case 2: return coder2.error();
    default:return coder1.error();
    }
}

void LZ77_DynamicCoder::put8 (uint c)
{
    switch (coder)
    {
    case 4: return coder4.put8(c);
    case 3: return coder3.put8(c);
    case 2: return coder2.put8(c);
    default:return coder1.put8(c);
    }
}

void LZ77_DynamicCoder::put32 (uint c)
{
    switch (coder)
    {
    case 4: return coder4.put32(c);
    case 3: return coder3.put32(c);
    case 2: return coder2.put32(c);
    default:return coder1.put32(c);
    }
}

void LZ77_DynamicCoder::flush()
{
    switch (coder)
    {
    case 4: return coder4.flush();
    case 3: return coder3.flush();
    case 2: return coder2.flush();
    default:return coder1.flush();
    }
}

// Number of bytes already written to coder
uint64 LZ77_DynamicCoder::outsize()
{
    switch (coder)
    {
    case 4: return coder4.outsize();
    case 3: return coder3.outsize();
    case 2: return coder2.outsize();
    default:return coder1.outsize();
    }
}

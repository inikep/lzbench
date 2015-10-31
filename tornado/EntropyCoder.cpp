// Code for various streams and entropy codecs:
//   - in/out byte streams
//   - in/out bit streams
//   - fast & elegant huffman tree builder
//   - semi-adaptive huffman codec
//   - fast semi-adaptive arithmetic codec
// Here semi-adaptive means that statistics collected on previous symbols are
// used to encode new ones, but encoding tables are updated with some intervals
// (after each 10k-20k symbols encoded)
//
// (c) Bulat Ziganshin
// (c) Joachim Henke
// This code is provided on the GPL license.
// If you need a commercial license to use the code, please write to Bulat.Ziganshin@gmail.com

// Selects either encoder or decoder
enum CodecDirection {Encoder, Decoder};

// Masks out higher bits of x leaving only n lower ones
#define mask(x,n) ((x) & ((1<<(n))-1))


// Byte-aligned I/O streams ***********************************************************************

#define MAXELEM 8  /* size of maximum data element that can be read/written to byte stream */

struct OutputByteStream
{
    CALLBACK_FUNC *callback;  // Function that writes data to the outstream
    void          *auxdata;
    BYTE          *buf;       // Output buffer
    BYTE          *output;    // Current pos
    BYTE          *last_qwrite;  // Value of output pointer for last quasi-write call
    BYTE          *anchor;    // Marked position in output buffer. Shifted synchronous to buffer contents
    UINT          chunk;      // How many bytes should be written at least in each "write" call
    uint64        shifted_out;   // Bytes shifted out of buffer after saved by write callback
    int           errcode;

    // chunk - how many bytes should be written each time, at least
    // pad   - how many bytes may be put to buffer between flush()/putbuf() calls
    OutputByteStream (CALLBACK_FUNC *_callback, void *_auxdata, UINT _chunk, UINT pad)
    {
        callback = _callback;  auxdata = _auxdata;  chunk = _chunk;
        // Add 512 bytes for LZ77_ByteCoder (its LZSS scheme needs to overwrite old flag words) and 4096 for rounding written chunks
        last_qwrite = output = buf = (byte*) BigAlloc (chunk+pad+512+4096);
        errcode = buf==NULL?  FREEARC_ERRCODE_NOT_ENOUGH_MEMORY : FREEARC_OK;
        shifted_out = 0;
    }
    ~OutputByteStream()       {BigFree(buf);}
    // Returns error code if there was any problems in stream work
    int error()               {return errcode;}
    // Drop/use anchor which marks place in buffer
    void set_anchor(void *p)  {anchor=(BYTE*)p;}
    void*get_anchor()         {return anchor;};
    // Advance output pointer n bytes forward and flush buffer if required
    void advance (uint n)     {output += n;}
    // Flush buffer data to outstream
    void flush();
    // Finish working, flushing any pending data
    void finish (int n=-1);
    // Writes 8-64 bits to the buffer
    void put8   (uint c)      {          *output= c;  advance(1);}
    void put16  (uint c)      {setvalue16(output, c); advance(2);}
    void put24  (uint32 c)    {setvalue32(output, c); advance(3);}
    void put32  (uint32 c)    {setvalue32(output, c); advance(4);}
    void put64  (uint64 c)    {setvalue64(output, c); advance(8);}
    // Number of bytes already written to coder
    uint64 outsize()          {return shifted_out + (output-buf);}
    // Writes len bytes pointed by ptr
    void putbuf (void *ptr, uint len);
    // Set context for order-1 coders
    void set_context(int i) {}
};

void OutputByteStream::putbuf (void *ptr, uint len)
{
    if ((output-buf)+len > chunk)    flush();
    CHECK (FREEARC_ERRCODE_INTERNAL,  (output-buf)+len <= chunk,  (s,"Fatal error: putbuf %d in buffer of size %d", len, chunk));
    memcpy (output, ptr, len);
    output += len;
}

void OutputByteStream::finish (int n)
{
    if (errcode==FREEARC_OK) {
        errcode = callback ("write", buf, n==-1? output-buf : n, auxdata);
        if (errcode>0)  errcode = FREEARC_OK;
    }
}

void OutputByteStream::flush()
{
    QUASIWRITE(output-last_qwrite);
    if (output-buf>512) {
        UINT n = (output-buf-512) & ~4095;  // how much bytes we can flush now (don't flush last 512 bytes due to possible LZSS use and round down to 4096-byte chunk)
        if (n >= chunk) {                   // go if this value is larger than requested chunk size
           finish (n);
           memcpy (buf, buf+n, output-(buf+n));
           output -= n;                     // Shift write pointer
           anchor -= n;                     // Shift data placed in buffer anchor too
           shifted_out += n;
        }
    }
    last_qwrite=output;
}


struct InputByteStream
{
    CALLBACK_FUNC *callback;    // Function that reads data from the instream
    void          *auxdata;
    UINT          bufsize;
    byte          *buf;         // Buffer (MAXELEM additional bytes are allocated for get64, see below)
    byte          *input;       // Current reading pos in buf[]
    byte          *read_point;  // Fence before next read callback
//  byte          *dataend;     // End of real data in buf[]
    uint64        shifted_out;  // Bytes shifted out of buffer after saved by write callback
    int           errcode;      // Value returned by last "read" callback

    // In order to improve speed, we use MAXELEM bytes at the buffer beginning
    // to hold last bytes of previous data chunk read. This allows us
    // to implement get64 and likewise operations without worrying about
    // crossing read-chunk boundaries. Therefore, we fill buffer each time
    // starting from its MAXELEM position
    InputByteStream (CALLBACK_FUNC *_callback, void *_auxdata, UINT _bufsize)
    {
        callback   = _callback;  auxdata = _auxdata;
        bufsize    = compress_all_at_once? _bufsize+(_bufsize/4) : LARGE_BUFFER_SIZE;  // For all-at-once compression input buffer should be large enough to hold all compressed data
        buf        = (byte*) BigAlloc (MAXELEM+bufsize);
        errcode    = buf==NULL?  FREEARC_ERRCODE_NOT_ENOUGH_MEMORY : FREEARC_OK;
        input      = buf + MAXELEM;
        read_point = buf + bufsize;
        if (error()==FREEARC_OK)   errcode = callback ("read", buf+MAXELEM, bufsize, auxdata);
    }
    ~InputByteStream()  {BigFree(buf);}
    // Returns error code if there is any problem in stream work
    int error()  {return errcode<0? errcode : FREEARC_OK;}

    // Reads next data chunk if lookahead is not enough for get64() execution
    void fill()
    {
        // If input ptr points inside last MAXELEM bytes in buf,
        // then move these bytes to the buffer start and read next portion of data
        if (input >= read_point) {
           memcpy (buf, buf+bufsize, MAXELEM);
           if (error()==FREEARC_OK)   errcode = callback ("read", buf+MAXELEM, bufsize, auxdata);
           input       -= bufsize;
           shifted_out += bufsize;
        }
    }
    // Number of bytes already consumed by decoder
    uint64 insize()  {return (shifted_out + (input - buf)) - MAXELEM;}

    // Reads 8-64 bits from the buffer
    uint   getc   ()  {fill(); return *input++;}
    uint   get8   ()  {fill(); return *input++;}
    uint   get16  ()  {fill(); uint   n = value16(input); input+=2; return n;}
    uint32 get24  ()  {fill(); uint32 n = value24(input); input+=3; return n;}
    uint32 get32  ()  {fill(); uint32 n = value32(input); input+=4; return n;}
    uint64 get64  ()  {fill(); uint64 n = value64(input); input+=8; return n;}
};


// Bit-aligned I/O streams ***********************************************************************

// It's an output bit stream
struct OutputBitStream : OutputByteStream
{
#ifdef FREEARC_64BIT
    uint64 bitbuf;   // Bit buffer - written to outstream when filled
#else
    uint32 bitbuf;
#endif
    int    bitcount; // Count of lower bits already filled in current bitbuf

    // Init and finish bit stream
    OutputBitStream (CALLBACK_FUNC *callback, void *auxdata, UINT chunk, UINT pad);
    void finish();

    // Write n lower bits of x
    void putbits (int n, uint32 x)
    {
        bitbuf |=
#ifdef FREEARC_64BIT
                  (uint64)
#endif
                          x << bitcount;
        bitcount += n;
        if (bitcount >= CHAR_BIT * sizeof(bitbuf)) {
#ifdef FREEARC_64BIT
            put64 (bitbuf);
#else
            put32 (bitbuf);
#endif
            bitcount -= CHAR_BIT * sizeof(bitbuf);
            bitbuf = x >> (n-bitcount);
        }
    }

    // Write n lower bits of x
    void putlowerbits (int n, uint32 x)
    {
        putbits (n, mask(x,n));
    }
};

// Skip a few bytes at the start and end of buffer in order to align it like it aligned on disk
#define eat_at_start 0
#define eat_at_end   0

OutputBitStream::OutputBitStream (CALLBACK_FUNC *callback, void *auxdata, UINT chunk, UINT pad)
    : OutputByteStream (callback, auxdata, chunk, pad)
{
    bitbuf   = 0;
    bitcount = CHAR_BIT*eat_at_start;
}

void OutputBitStream::finish()
{
    while (bitcount > 0) {
        put8 (bitbuf);
        bitbuf  >>= 8;
        bitcount -= 8;
    }
    OutputByteStream::finish();
}


// And it's an input bit stream
struct InputBitStream : InputByteStream
{
    uint64  bitbuf;      // Bit buffer - filled by reading from InputByteStream
    int     bitcount;    // Count of higher bits not yet filled in current bitbuf

    // Initialize bit stream
    InputBitStream (CALLBACK_FUNC *callback, void *auxdata, UINT bufsize);

    // Ensure that bitbuf contains at least n valid bits (n<=32)
    uint needbits (int n)
    {
        if (bitcount<=32)
            bitbuf |= uint64(get32()) << bitcount, bitcount+=32;
        return mask(bitbuf,n);
    }
    // Throw out n used bits
    void dumpbits (int n)
    {
        bitbuf >>= n;
        bitcount -= n;
    }
    // Read next n bits of input
    uint getbits (int n)
    {
        uint x = needbits(n);
        dumpbits(n);
        return x;
    }
};

InputBitStream::InputBitStream (CALLBACK_FUNC *callback, void *auxdata, UINT bufsize)  :  InputByteStream (callback, auxdata, bufsize)
{
    bitbuf   = 0;
    bitcount = 0;
}


// Huffman tree ***********************************************************************************

#define MAXHUF    2048  /* maximum number of elements allowed in huffman tree */
#define FAST_BITS 11    /* symbols with shorter codes (<=FAST_BITS bits) are decoded much faster */

// This structure used for intermediate data when building huffman tree
struct Node {uint32 cnt, code; uint16 left, right; uint8 bits;};
// Simplified Node structure, saving counter in higher bits and index in lower ones (in order to allow stable sorting by counter!)
typedef uint Node0;
#define make_node(i,cnt)  ((cnt)*MAXHUF+(i))
#define index0(node)      ((node)%MAXHUF)
#define cnt0(node)        ((node)/MAXHUF)
// For stable sorting Nodes by counters
int __cdecl by_cnt_and_index (const void* a, const void* b)   {return *(const Node0*)a - *(const Node0*)b;}

// Huffman tree for dynamic encoding.
// If you want to work with block-wise static encoding, you need to
// make a few trivial changes marked by STATIC
struct HuffmanTree
{
    CodecDirection type;              // Determines whether this tree used for encoding or decoding
    int    n;                         // Number of symbols in huffman tree
    uint   counter[MAXHUF];           // Symbol counters used to build huffman tree
                                      // Built huffman tree:
    int    maxbits;                   //   Maximum number of bits in any code of current table
    int    maxbits_so_far;            //   Maximum maxbits encountered so far
    uint8  bits[MAXHUF];              //   Number of bits for each symbol
    uint32 code[MAXHUF];              //   Code (bit sequence) for each symbol
    int    fast_index[1<<FAST_BITS];  //   Direct decoding table for short codes (<=FAST_BITS)
    uint16 *index;                    //   Direct decoding table for long codes

    HuffmanTree (CodecDirection _type, int _n)    {init(_type, _n);}
    void   init (CodecDirection _type, int _n);
    HuffmanTree ()                                {}
    ~HuffmanTree()                                {free(index);}
    void Inc (int s, int i=1)                     {counter[s] += i;}
    void build_tree (int rescale_mode);

    // Decode value by code (todo: and return its bitsize)
    int Decode (UINT code)
    {
        // We first try to decode value using first FAST_BITS input bits
        // via small fast_index[] table. If this cannot produce single decoded
        // value (flagged by x<0) then we decode using maxbits input bits
        // via index[] table which guarantees single-meaning decoding
        int x = fast_index[mask(code,FAST_BITS)];
        return x>=0? x : index[code];
    }
};

void HuffmanTree::init (CodecDirection _type, int _n)
{
    CHECK (FREEARC_ERRCODE_INTERNAL,  _n<=MAXHUF,  (s,"Fatal error: HuffmanTree::n=%d is larger than maximum allowed value %d", _n, MAXHUF));
    if (_n==0) return;
    type  = _type;
    n     = _n;
    index = NULL;
    maxbits_so_far = 0;
    iterate_var(s,n) counter[s]=1;   // STATIC: counter[s]=0
    build_tree(0);                   // STATIC: remove this line
}

// It seems this is the only original implementation of "build huffman tree"
// procedure, all other compressors borrow zip's one :)
// The algorithm is obvious - instead of using heap for searching of nodes with
// smallest counters, we keep nodes in two separate lists - first includes
// "original" nodes - i.e. those correspoding to symbols, second list
// includes constructed intermediate nodes. We keep both lists sorted
// by counters ascending - first list is sorted by qsort, while second
// remains sorted due to its construction algorithm - each new node inserted
// in this list has higher count than all previous nodes. So, each new node
// may be constructed in one of only 3 ways - from two first original nodes,
// or two first combined nodes, or one node from each list, depending on
// comparison of their counters. It is the main part of algorithm.
// INT_MAX values used as fence at end of each list. Don't tell me that i'm genie :)
void HuffmanTree::build_tree (int rescale_mode)
{
    debug (printf ("=== BUILD_TREE rescale_mode=%d ===\n", rescale_mode));

    Node buf [2*MAXHUF+4];              // Intermediate data used to build huffman tree
    int *places = (int*) (buf+MAXHUF);  // Part of the buffer reused for better caching
    int b = 0;                          // This var will contain count of non-zero symbols

    // Nodes with small counters (<CHUNKS) sorted by counting, remaining nodes (<10%) sorted by STL sort()
    const int CHUNKS = 250;
    iterate_var(i,CHUNKS+1)
        places[i] = 0;
    iterate_var(i,n) {               // STATIC: add "if (counter[i])"
        if (counter[i] < CHUNKS)
            places[counter[i] + 1]++;
        b++;
    }
    iterate_var(i,CHUNKS)
        places[i + 1] += places[i];
    // code[] is reused here as a temporary buffer for better caching  (== Node0 rest[b - places[CHUNKS]])
    Node0 *rest = (Node0 *)code, *r = rest;
    iterate_var(i,n)                 // STATIC: add "if (counter[i])"
        if (counter[i] < CHUNKS) {
            int p = places[counter[i]]++;
            buf[p].cnt = counter[i];
            buf[p].left = i;
        } else
            *r++ = make_node(i, counter[i]);
    // Stable sorting of nodes by counter (== std::sort (rest, r))
    qsort (rest, r - rest, sizeof(*rest), by_cnt_and_index);
    iterate_var(i, r - rest) {
        int p = i + places[CHUNKS];
        buf[p].cnt = cnt0(rest[i]);
        buf[p].left = index0(rest[i]);
    }
    // Now buf[0..b-1] contains nodes stable-sorted by counter.

    iterate_var(i,b+4)  buf[b+i].cnt = INT_MAX;  // maximum possible int value used here as a fence
    int p1 = 0,    // Index of first remaining original node
        p2 = b+2,  // Index of first remaining combined node
        p3 = b+2;  // Index of next combined node to create

    // Cycle finished when all original nodes are used and
    // only one combined node remains - zero-length code will be assigned to it
    while (! (p1==b && p3-p2==1)) {
        // Join two nodes with smallest counters. We select among
        // buf[p1], buf[p1+1], buf[p2], buf[p2+1], knowing that
        // buf[p1].cnt < buf[p1+1].cnt  and  buf[p2].cnt < buf[p2+1].cnt
        if (buf[p1+1].cnt < buf[p2].cnt) {             // Smallest nodes are p1, p1+1
            buf[p3].cnt = buf[p1].cnt + buf[p1+1].cnt;
            buf[p3].left  = p1;
            buf[p3].right = p1+1;
            p1 += 2;
        } else if (buf[p1].cnt > buf[p2+1].cnt) {      // Smallest nodes are p2, p2+1
            buf[p3].cnt = buf[p2].cnt + buf[p2+1].cnt;
            buf[p3].left  = p2;
            buf[p3].right = p2+1;
            p2 += 2;
        } else {                                       // Otherwise smallest nodes are p1, p2
            buf[p3].cnt = buf[p1].cnt + buf[p2].cnt;
            buf[p3].left  = p1;
            buf[p3].right = p2;
            p1++, p2++;
        }
        p3++;
    }
    // Now huffman tree is built.

    // The last remaining node got zero-length code, all other nodes got codes
    // of their parents with additional bit '0' for left child and '1' for right one
    buf[p2].bits = 0;
    buf[p2].code = 0;
    for (int i=p2; i >= b+2; i--) {
        buf [buf[i].left].bits = buf[i].bits+1;
        buf [buf[i].left].code = buf[i].code;
        buf [buf[i].right].bits = buf[i].bits+1;
        buf [buf[i].right].code = buf[i].code+(1<<buf[i].bits);
    }
    // Move bits/code values to the arrays indexed by symbol
    if (type==Encoder) {
        iterate_var(i,b)  {int s=buf[i].left; bits[s]=buf[i].bits; code[s]=buf[i].code;}
    } else {
        // Use symbol with smallest frequency to determine maximum number of bits in any code
        maxbits = buf[0].bits;
        // 'index' table should contain at least 1<<maxbits elements
        if (maxbits > maxbits_so_far) {
            maxbits_so_far = maxbits;
            index = (uint16*) realloc (index, (1<<maxbits)*sizeof(*index));
        }
        iterate_var(i,b) {
            UINT s     = buf[i].left;
            UINT sbits = buf[i].bits;
            UINT scode = buf[i].code;
            bits[s] = sbits;
            // For decoder we fill index[] table used to decode symbols using first 'maxbits' of input
            // (it is always possible because maxbits is a maximum number of bits in any code)
            // We also fill fast_index[] table used to decode short codes
            if (sbits<=FAST_BITS) {
                iterate_var(j, 1<<(FAST_BITS-sbits))   fast_index [scode + (j<<sbits)] = s;
            } else {
                fast_index[mask(scode,FAST_BITS)] = -1;
                iterate_var(j, 1<<(maxbits-sbits))     index [scode + (j<<sbits)] = s;
            }
        }
    }
    // Prepare counters for next block. STATIC: counter[s]=0
#define rescale(FACTOR)  iterate_var(s,n)  counter[s] -= (counter[s]>1 && counter[s]<FACTOR)? 1 : counter[s]/FACTOR
    switch (rescale_mode) {
    case 0:  rescale(2); break;
    case 1:  rescale(3); break;
    case 2:  rescale(4); break;
    case 3:  rescale(6); break;
    case 4:  rescale(8); break;
    case 5:  rescale(10); break;
    case 6:  rescale(12); break;
    case 7:  rescale(16); break;
    }
}


// Semi-adaptive huffman coder ********************************************************************

#define HUFBLOCKSIZE 5000

// Encode symbols using huffman coder
template <int EOB>
struct HuffmanEncoder : OutputBitStream
{
    HuffmanTree  huf;           // Huffman tree used to encode symbols
    int          remainder;     // Count symbols remaining before huffman tree rebuild

    HuffmanEncoder (CALLBACK_FUNC *callback, void *auxdata, UINT chunk, UINT pad, int n)
        : OutputBitStream (callback,auxdata,chunk,pad), huf(Encoder,n) {remainder=HUFBLOCKSIZE/4;}

    // Encode symbol and count it in huffman tree
    void encode (UINT x)
    {
        if (--remainder==0)  {      // Rebuild huffman tree periodically
            const int rescale_mode = 3;  // Statistics update rate
            putbits (huf.bits[EOB], huf.code[EOB]);
            putbits (3, rescale_mode);
            huf.build_tree (rescale_mode);
            remainder = HUFBLOCKSIZE;
        }
        huf.Inc (x);
        putbits (huf.bits[x], huf.code[x]);
    }

    // Bits required to encode the single code
    PRICE price(UINT x, int extra_bits)  {return huf.bits[x] + extra_bits;}
};

// Decode symbols using huffman coder
template <int EOB>
struct HuffmanDecoder : InputBitStream
{
    HuffmanTree  huf;           // Huffman tree used to decode symbols

    HuffmanDecoder (CALLBACK_FUNC *callback, void *auxdata, UINT bufsize, int n)
        : InputBitStream (callback, auxdata, bufsize), huf(Decoder,n) {}

    // Decode symbol and count it in huffman tree
    UINT decode()
    {
        UINT x;
        while (1) {
            x = huf.Decode (needbits (huf.maxbits));
            dumpbits (huf.bits[x]);
            if (x != EOB) break;
            huf.build_tree (getbits(3));  // Rebuild huffman tree on EOB code
        }
        huf.Inc (x);
        return x;
    }
};


// Order-1 semi-adaptive huffman coder ************************************************************

// Encode symbols using many huffman coders
template <int ORDER1_CONTEXTS, int EOB>
struct HuffmanEncoderOrder1 : OutputBitStream
{
    HuffmanTree  *huf;                           // Huffman trees used to encode symbols
    int          remainder[ORDER1_CONTEXTS];     // Count symbols remaining before huffman tree rebuild

    HuffmanEncoderOrder1 (CALLBACK_FUNC *callback, void *auxdata, UINT chunk, UINT pad, int n)
        : OutputBitStream (callback,auxdata,chunk,pad)  {context=0; huf = new HuffmanTree[ORDER1_CONTEXTS]; for(int i=0;i<ORDER1_CONTEXTS;i++)  remainder[i]=HUFBLOCKSIZE/4, huf[i].init(Encoder,n);}

    // In order to unify order-0 and order-1 coder APIs, we provide API to specify context in separate call
    int context;
    void set_context(int i) {context=i;}
    void encode (UINT x)    {encode(context,x);}

    // Encode symbol and count it in huffman tree
    void encode (int i, UINT x)
    {
        if (--remainder[i]==0)  {      // Rebuild huffman tree periodically
            const int rescale_mode = 3;  // Statistics update rate
            putbits (huf[i].bits[EOB], huf[i].code[EOB]);
            putbits (3, rescale_mode);
            huf[i].build_tree (rescale_mode);
            remainder[i] = HUFBLOCKSIZE;
        }
        huf[i].Inc (x);
        putbits (huf[i].bits[x], huf[i].code[x]);
    }
};

// Decode symbols using many huffman coders
template <int ORDER1_CONTEXTS, int EOB>
struct HuffmanDecoderOrder1 : InputBitStream
{
    HuffmanTree  huf[ORDER1_CONTEXTS];           // Huffman trees used to decode symbols

    HuffmanDecoderOrder1 (CALLBACK_FUNC *callback, void *auxdata, UINT bufsize, int n)
        : InputBitStream (callback, auxdata, bufsize)  {iterate(ORDER1_CONTEXTS, huf[i].init(Decoder,n));}

    // Decode symbol and count it in huffman tree
    UINT decode(int i)
    {
        UINT x;
        while (1) {
            x = huf[i].Decode (needbits (huf[i].maxbits));
            dumpbits (huf[i].bits[x]);
            if (x != EOB) break;
            huf[i].build_tree (getbits(3));  // Rebuild huffman tree on EOB code
        }
        huf[i].Inc (x);
        return x;
    }
};


// Shindler's rangecoder **************************************************************************

class TRangeCoder : public OutputByteStream
{
private:
  int64 low;
  unsigned int range;
  unsigned int buffer;
  unsigned int help;

  inline void ShiftLow() {
    if ((low ^ 0xff000000) >= (1 << 24)) {
      unsigned int c = static_cast<unsigned int>(low >> 32);
      put8(buffer + c);
      c += 255;
      for (; help > 0; help--)
        put8(c);
      buffer = static_cast<unsigned int>(low) >> 24;
    }
    else
      help++;
    low = static_cast<unsigned int>(low) << 8;
  }

public:
  TRangeCoder (CALLBACK_FUNC *callback, void *auxdata, UINT chunk, UINT pad) :
    OutputByteStream (callback, auxdata, chunk, pad),
    low(0), range(0xffffffff), buffer(0), help(0) {}

  void Encode(unsigned int cum, unsigned int cnt, unsigned int bits) {
    low += (cum * (range >>= bits));
    range *= cnt;
    while (range < (1 << 24)) {
      range <<= 8;
      ShiftLow();
    }
  }

  // Finish working, flushing any pending data
  void finish()
  {
    for (int i = 0; i < 5; i++)
      ShiftLow();
    OutputByteStream::finish();
  }
};


class TRangeDecoder : public InputByteStream
{
private:
  unsigned int range;
  unsigned int buffer;

public:
  TRangeDecoder (CALLBACK_FUNC *callback, void *auxdata, UINT bufsize) : InputByteStream (callback, auxdata, bufsize), range(0xffffffff)
  {
    for (int i = 0; i < 5; i++)
      buffer = (buffer << 8) + getc();
  }

  unsigned int GetCount(unsigned int bits)
  {
    unsigned int count = buffer / (range >>= bits);
    if (count >= (1<<bits))
      fprintf(stderr, "data error\n"), exit(1);
    return (count);
  }

  void Update(unsigned int cum, unsigned int cnt)
  {
    buffer -= (cum * range);
    range *= cnt;
    while (range < (1 << 24)) {
      range <<= 8;
      buffer = (buffer << 8) + getc();
    }
  }
};


// Semi-adaptive arithmetic coder ******************************************************************
//   This coder updates counters after (de)coding each symbol,
//   but keeps to use previous encoding. After some amount of
//   symbols encoded it recalculates encoding using counters
//   gathered to this moment. So, each time it uses for encoding
//   stats of previous block. Actually, counters for new block
//   are started from 5/6 of previous counters, so algorithm
//   is more resistant to temporary statistic changes

#define NUM        2048             /* maximum number of symbols + 1 */
#define INDEXES    2048             /* amount of indexes used for fast searching in cum[] table */
#define RANGE      (1u<<RANGE_BITS) /* automagically, on each recalculation there are just RANGE symbols counted in livecnt[] */
#define RANGE_BITS 14               /* the higher this value, тем реже происходят обновления статистики, но она при этом точнее */

template <CodecDirection type>
struct TCounter
{
    UINT n, remainder;
    UINT cnt[NUM], cum[NUM], livecnt[NUM], index[INDEXES];
    PRICE prices[NUM];

    TCounter (unsigned _n = NUM);

    // Count one more occurence of symbol s
    // and recalculate encoding tables if enough symbols was counted since last recalculation
    // so that sum(livecnt[])==RANGE now
    void Inc (unsigned s)
    {
        livecnt[s]++;
        if (--remainder==0)  Rescale();
    }

    // Recalculate (de)coding tables according to last gathered stats
    // and prepare to collect stats on next block
    void Rescale();

    // Find symbol corresponding to code 'count'
    UINT Decode (UINT count)
    {
        // index[] gives us a quick hint and then we search for a first symbol
        // whose end of range is greater than or equal to 'count'
        UINT s = index [count/(RANGE/INDEXES)];
        while (cum[s]+cnt[s]-1 < count)  s++;
        debug (printf("symbol %x, count %x of %x-%x\n", s, count, cum[s], cum[s]+cnt[s]-1));
        return s;
    }
};

template <CodecDirection type>
TCounter<type> :: TCounter (unsigned _n)
{
    n = _n;
    // Initially, allot RANGE points equally to n symbols
    // (superfluous RANGE-RANGE/n*n points are assigned to first symbols)
    for (int s = 0; s < n; s++)
        livecnt[s] = RANGE/n + (s < RANGE-RANGE/n*n? 1 : 0);
    Rescale();
}

template <CodecDirection type>
void TCounter<type> :: Rescale ()
{
    UINT total = 0;
    remainder = RANGE;
    for (int s=0, ind=0; s < n; s++) {
        cnt[s]      = livecnt[s],
        cum[s]      = total,
        total      += cnt[s],
        livecnt[s] -= (livecnt[s]>1 && livecnt[s]<6)? 1 : livecnt[s]/6;
        remainder  -= livecnt[s];
        // While last element of this symbol range may be mapped to index[ind]
        // fill index[ind] with this symbol. Finally, each index[n]
        // will contain *first* possible symbol whose first N bits equal to n
        // (N = logb(INDEXES)), which is used for quick symbol decoding
        if (type==Decoder)
            while (cum[s]+cnt[s]-1 >= RANGE/INDEXES*ind)
                index[ind++] = s;
        // Calculate encoding price in bits
        if (type==Encoder)
            prices[s] = PRICE_SCALE( log2( double(RANGE) / cnt[s] ));
    }
    debug (printf("total %d\n",total));
}


// Semi-adaptive arithmetic coder ******************************************************************

template <int EOB>
struct ArithCoder : TRangeCoder
{
    TCounter<Encoder> c;   // Provides stats about symbol frequencies

    ArithCoder (CALLBACK_FUNC *callback, void *auxdata, UINT chunk, UINT pad, int n)
        : TRangeCoder (callback,auxdata,chunk,pad), c(n) {}

    // Encode symbol x using TCounter stats
    void encode (UINT x)
    {
        debug (printf("symbol %x, count %x-%x\n",x,c.cum[x], c.cum[x]+c.cnt[x]-1));
        Encode (c.cum[x], c.cnt[x], RANGE_BITS);
        c.Inc (x);
    }

    // Write n lower bits of x
    void putlowerbits (int n, UINT x)
    {
        debug (printf("putbits %d of %x\n",n,x));
        if (n<=24) {
            Encode (mask(x,n), 1, n);
            return;
        }
        Encode (mask(x,15), 1, 15);
        debug (printf("encoded %x, %x ==> %x\n",mask(x,15),mask(x>>15,n-15),x));
        x>>=15, n-=15;
        Encode (mask(x,n), 1, n);
    }

    // (Scaled) bits required to encode the single code
    PRICE price (UINT x, int extra_bits)  {return c.prices[x] + PRICE_BITS(extra_bits);}
};


template <int EOB>
struct ArithDecoder : TRangeDecoder
{
    TCounter<Decoder> c;   // Provides stats about symbol frequencies

    ArithDecoder (CALLBACK_FUNC *callback, void *auxdata, UINT bufsize, int elements)
        : TRangeDecoder (callback, auxdata, bufsize), c(elements) {}

    // Decode next symbol using TCounter stats
    UINT decode()
    {
        UINT count = GetCount (RANGE_BITS);
        UINT x = c.Decode (count);
        Update (c.cum[x], c.cnt[x]);
        c.Inc (x);
        return x;
    }

    // Read next n bits of input
    UINT getbits (int n)
    {
        debug (printf("getbits %d\n",n));
        if (n<=24) {
            UINT x = GetCount (n);
            Update (x, 1);
            return x;
        } else {
            UINT x1 = GetCount (15);
            Update (x1, 1);
            UINT x2 = GetCount (n-15);
            Update (x2, 1);
            debug (printf("decoded %x, %x ==> %x\n",x1,x2,(x2<<15) + x1));
            return (x2<<15) + x1;
        }
    }
};

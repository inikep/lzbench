// Tornado - fast LZ77 compression algorithm.
//
// (c) Bulat Ziganshin
// (c) Joachim Henke
// This code is provided on the GPL license.
// If you need a commercial license to use the code, please write to Bulat.Ziganshin@gmail.com

#include "Compression.h"

typedef UINT PRICE;
#define MAX_PRICE       UINT_MAX   /* used in #if */
#define PRICE_BITS(x)   ((x)*1024)
#define PRICE_SCALE(x)  (PRICE( round( PRICE_BITS(x) ) ) )
typedef uint32 DISTANCE;
const int MAX_MATCHES = 2048+256+16;  // maximum length of hash row plus a few low-order matches from auxiliary MFs

#include "MatchFinder.cpp"
#include "EntropyCoder.cpp"
#include "LZ77_Coder.cpp"
#include "DataTables.cpp"

// Compression method parameters
struct PackMethod
{
    int  number;            // Preset number
    int  encoding_method;   // Coder (0 - storing, 1 - bytecoder, 2 - bitcoder, 3 - huffman, 4 - arithmetic)
    bool find_tables;       // Enable searching for MM tables
    int  hash_row_width;    // Length of hash row
    uint hashsize;          // Hash size
    int  match_finder;      // Match finder type and number of hashed bytes
    uint buffer;            // Buffer (dictionary) size
    int  match_parser;      // Match parser (1 - greedy, 2 - lazy, 3 - flexible, 4 - optimal, 5 - even better)
    int  hash3;             // 2/3-byte hash presence and size
    int  shift;             // How much bytes to shift out/keep when window slides
    int  update_step;       // How much bytes are skipped in mf.update()
    uint auxhash_size;      // Auxiliary hash size
    int  auxhash_row_width; // Length of auxiliary hash row
    int  fast_bytes;        // Optimal parser: after a match/repmatch longer than or equal to fast_bytes, we skip everything until match end in order to avoid O(n^2) time behavior
};

extern "C" {
// Main compression and decompression routines
int tor_compress   (PackMethod &m, CALLBACK_FUNC *callback, void *auxdata);
int tor_decompress (CALLBACK_FUNC *callback, void *auxdata);
}

enum { STORING=0, BYTECODER=1, BITCODER=2, HUFCODER=3, ARICODER=4 };
enum { GREEDY=1, LAZY=2, OPTIMAL=4 };
enum { NON_CACHING_MF=0, CACHING_MF4_DUB=1, CYCLED_MF4_DUB=2
     ,  CYCLED_MF4= 4,  CYCLED_MF5= 5,  CYCLED_MF6= 6,  CYCLED_MF7= 7
     , CACHING_MF4=14, CACHING_MF5=15, CACHING_MF6=16, CACHING_MF7=17
     ,      BT_MF4=24,      BT_MF5=25,      BT_MF6=26,      BT_MF7=27 };

const int64 MIN_BUFFER_SIZE = 4*kb,  MAX_BUFFER_SIZE = 1*gb,
            MIN_HASH_SIZE   = 4*kb,  MAX_HASH_SIZE   = UINT_MAX,
            MAX_HASH_ROW_WIDTH = 64*kb,  MAX_UPDATE_STEP = 64*kb,
            MIN_FAST_BYTES = 1,  MAX_FAST_BYTES = 64*kb;

// Preconfigured compression modes
PackMethod std_Tornado_method[] =
    //                 tables row  hashsize  matchfinder       buffer   parser hash3 shift update     auxhash  fast_bytes
    { {  0, STORING,   false,   0,        0, NON_CACHING_MF,    1*mb,        0,  0,    0,  999,       0,    0,  128 }
    , {  1, BYTECODER, false,   1,    16*kb, NON_CACHING_MF,    1*mb,   GREEDY,  0,    0,  999,       0,    0,  128 }
    , {  2, BITCODER,  false,   1,    64*kb, NON_CACHING_MF,    2*mb,   GREEDY,  0,    0,  999,       0,    0,  128 }
    , {  3, HUFCODER,   true,   2,   128*kb, NON_CACHING_MF,    4*mb,   GREEDY,  0,    0,  999,       0,    0,  128 }
    , {  4, HUFCODER,   true,   2,     2*mb,    CACHING_MF4,    8*mb,   GREEDY,  0,    0,  999,       0,    0,  128 }
    , {  5, ARICODER,   true,   4,     8*mb,    CACHING_MF4,   16*mb,     LAZY,  1,    0,  999,       0,    0,  128 }
    , {  6, ARICODER,   true,   8,    32*mb,    CACHING_MF4,   64*mb,     LAZY,  1,    0,    4,       0,    0,  128 }
    , {  7, ARICODER,   true,  32,   128*mb,     CYCLED_MF5,  256*mb,     LAZY,  2,    0,    1,  128*kb,    4,  128 }
    , {  8, ARICODER,   true, 128,   512*mb,     CYCLED_MF5,    1*gb,     LAZY,  2,    0,    1,  128*kb,    4,  128 }
    , {  9, ARICODER,   true, 256,     2*gb,     CYCLED_MF5,    1*gb,     LAZY,  2,    0,    1,  512*kb,    4,  128 }
    , { 10, ARICODER,   true, 256,     2*gb,     CYCLED_MF7,    1*gb,     LAZY,  2,    0,    1,   16*mb,  128,  128 }
    , { 11, ARICODER,  false,   8,   256*mb,    CACHING_MF5,  128*mb,  OPTIMAL,  1,    0,    1,   64*kb,    1,   16 }
    , { 12, ARICODER,  false,  16,   256*mb,    CACHING_MF5,  128*mb,  OPTIMAL,  1,    0,    1,   64*kb,    1,   32 }
    , { 13, ARICODER,  false,  24,   384*mb,    CACHING_MF5,  128*mb,  OPTIMAL,  1,    0,    1,  128*kb,    1,   64 }
    , { 14, ARICODER,  false,  32,   512*mb,    CACHING_MF6,  128*mb,  OPTIMAL,  2,    0,    1,    2*mb,    2,   64 }
    , { 15, ARICODER,  false,  64,   512*mb,    CACHING_MF6,  128*mb,  OPTIMAL,  2,    0,    1,    2*mb,    4,  128 }
    , { 16, ARICODER,  false, 128,   128*mb,         BT_MF5,  128*mb,  OPTIMAL,  2,    0,    1,    4*mb,    8,  512 }
    };

// Default compression parameters are equivalent to option -5
const int default_Tornado_method = 5;

// If data table was not encountered in last table_dist bytes, don't check next table_shift bytes in order to make things faster
const int table_dist=256*1024, table_shift=128;

// Minimum lookahead for next match which compressor tries to guarantee.
// Also minimum amount of allocated space after end of buf (this allows to use things like p[11] without additional checks)
#define LOOKAHEAD 1200

// Maximum number of bytes used for hashing in any match finder.
// If this value will be smaller than real, we can hash bytes in buf that are not yet read
// Also it's number of bytes reserved after bufend in order to simplify p+N<=bufend checks
#define MAX_HASHED_BYTES 600


// Îêðóãëèì ðàçìåð õåøà ñ ó÷¸òîì hash_row_width
uint round_to_nearest_hashsize (LongMemSize hashsize, uint hash_row_width)
{return mymin (round_to_nearest_power_of(mymin(hashsize,2*gb-1) / hash_row_width, 2) * hash_row_width, 2*gb-1);}

// Dictionary size depending on memory available for dictionary+outbuf (opposite to tornado_compressor_outbuf_size)
uint tornado_compressor_calc_dict (uint mem)
{return compress_all_at_once?  mem/9*4
                            :  mem>2*LARGE_BUFFER_SIZE ? mem-LARGE_BUFFER_SIZE : mem/2;}

// Output buffer size for compressor (the worst case is bytecoder that adds 2 bits per byte on incompressible data)
uint tornado_compressor_outbuf_size (uint buffer, int bytes_to_compress = -1)
{return bytes_to_compress!=-1? bytes_to_compress+(bytes_to_compress/4)+512 :
        compress_all_at_once?  buffer+(buffer/4)+512 :
                               mymin (buffer+512, LARGE_BUFFER_SIZE);}

// Output buffer size for decompressor
uint tornado_decompressor_outbuf_size (uint buffer)
{return compress_all_at_once?  buffer+(buffer/8)+512 :
                               mymax (buffer, LARGE_BUFFER_SIZE);}


#ifndef FREEARC_DECOMPRESS_ONLY

// Returns true if compression parameters are invalid
bool is_tornado_method_valid (PackMethod &method)
{
    if (!(MIN_BUFFER_SIZE<=method.buffer          && method.buffer         <=MAX_BUFFER_SIZE))          return false;
    if (!(  MIN_HASH_SIZE<=method.hashsize        && method.hashsize-1     <=MAX_HASH_SIZE-1))          return false;
    if (!(              1<=method.hash_row_width  && method.hash_row_width <=MAX_HASH_ROW_WIDTH))       return false;
    if (!(              1<=method.update_step     && method.update_step    <=MAX_UPDATE_STEP))          return false;
    if (!(      BYTECODER<=method.encoding_method && method.encoding_method<=ARICODER))                 return false;
    if (!(              0<=method.hash3           && method.hash3          <=2))                        return false;
    if (!(method.match_parser==GREEDY || method.match_parser==LAZY  || method.match_parser==OPTIMAL))   return false;
    if (method.hash3==0 && method.match_parser==OPTIMAL)                                                return false;
    int mf = method.match_finder;
    if (!(mf==NON_CACHING_MF || (CYCLED_MF4<=mf && mf<=CYCLED_MF7) || (CACHING_MF4<=mf && mf<=CACHING_MF7) || (BT_MF4<=mf && mf<=BT_MF7)))   return false;
    if (mf%10 >= 5  &&  (method.auxhash_size==0 || method.auxhash_row_width==0))                        return false;
    if (method.find_tables  &&  method.match_parser==OPTIMAL)                                           return false;
    if (method.find_tables  &&  (BT_MF4<=mf && mf<=BT_MF7))                                             return false;
    return true;
}

// Check for data table with N byte elements at current pos
#define CHECK_FOR_DATA_TABLE(N)                                                                         \
{                                                                                                       \
    if (p[-1]==p[N-1]                                                                                   \
    &&  uint(p[  N-1] - p[2*N-1] + 4) <= 2*4                                                            \
    &&  uint(p[2*N-1] - p[3*N-1] + 4) <= 2*4                                                            \
    &&  !val32equ(p+2*N-4, p+N-4))                                                                      \
    {                                                                                                   \
        int type, items;                                                                                \
        if (check_for_data_table (N, type, items, p, bufend, table_end, buf, offset, last_checked)) {   \
            coder.encode_table (type, items);                                                           \
            /* If data table was diffed, we should invalidate match cached by lazy match finder */      \
            mf.invalidate_match();                                                                      \
            goto found;                                                                                 \
        }                                                                                               \
    }                                                                                                   \
}


// Read next datachunk into buffer, shifting old contents if required
template <class MatchFinder, class Coder>
int read_next_chunk (PackMethod &m, CALLBACK_FUNC *callback, void *auxdata, MatchFinder &mf, Coder &coder, byte *&p, byte *buf, BYTE *&bufend, BYTE *&table_end, BYTE *&last_found, BYTE *&read_point, int &bytes, int &chunk, uint64 &offset, byte *(&last_checked)[MAX_TABLE_ROW][MAX_TABLE_ROW])
{
    if (bytes==0 || compress_all_at_once)  return 0;     // All input data was successfully compressed
    // If we can't provide 256 byte lookahead then shift data toward buffer beginning,
    // freeing space at buffer end for the new data
    if (bufend-buf > m.buffer-LOOKAHEAD) {
        coder.before_shift(p);
        int shift;
        if (m.shift == -1) {
            shift = p-(buf+2);  // p should become buf+2 after this shift
            memcpy (buf, buf+shift, bufend-(buf+shift));
            mf.clear_hash (buf);
        } else {
            shift = m.shift>0? m.shift : bufend-buf+m.shift;
            memcpy (buf, buf+shift, bufend-(buf+shift));
            mf.shift (buf, shift);
        }
        p      -= shift;
        bufend -= shift;
        offset += shift;
        if (coder.support_tables && m.find_tables) {
            table_end  = table_end >buf+shift? table_end -shift : buf;
            last_found = last_found>buf+shift? last_found-shift : buf;
            iterate_var(i,MAX_TABLE_ROW)  iterate_var(j,MAX_TABLE_ROW)  last_checked[i][j] = buf;
        }
        mf.invalidate_match();  // invalidate match stored in lazy MF; otherwise it may fuck up the NEXT REPCHAR checking
        coder.after_shift(p);    // tell to the coder what shift occurs
        debug (printf ("==== SHIFT %08x: p=%08x ====\n", shift, p-buf));
    }
    bytes = callback ("read", bufend, mymin (chunk, buf+m.buffer-bufend), auxdata);
    debug (printf ("==== read %08x ====\n", bytes));
    if (bytes<0)  return bytes;    // Return errcode on error
    bufend += bytes;
    read_point = bytes==0? bufend:bufend-LOOKAHEAD;
    coder.flush();          // Sometimes data should be written to disk :)
    return p<bufend? 1 : 0; // Result code: 1 if we still have bytes to compress, 0 otherwise
}


// Compress one chunk of data
template <class MatchFinder, class Coder>
int tor_compress_chunk (PackMethod &m, CALLBACK_FUNC *callback, void *auxdata, byte *buf, int bytes_to_compress)
{
    // Read data in these chunks
    int chunk = compress_all_at_once? m.buffer : mymin (m.shift>0? m.shift:m.buffer, LARGE_BUFFER_SIZE);
    uint64 offset = 0;                        // Current offset of buf[] contents relative to file (increased with each shift() operation)
    int bytes = bytes_to_compress!=-1? bytes_to_compress : callback ("read", buf, chunk, auxdata);   // Number of bytes read by last "read" call
    if (bytes<0)  return bytes;               // Return errcode on error
    BYTE *bufend = buf + bytes;               // Current end of real data in buf[]
    BYTE *matchend = bufend - mymin (MAX_HASHED_BYTES, bufend-buf);   // Maximum pos where match may finish (less than bufend in order to simplify hash updating)
    BYTE *read_point = compress_all_at_once || bytes_to_compress!=-1? bufend : bufend-mymin(LOOKAHEAD,bytes); // Next point where next chunk of data should be read to buf
    BYTE *progress_point = buf,  *next_special_p = buf;  uint64 prev_insize = 0,  prev_outsize = 0;
    // Match finder will search strings similar to current one in previous data
    MatchFinder mf (buf, m.buffer, m.hashsize, m.hash_row_width, m.auxhash_size, m.auxhash_row_width);
    if (mf.error() != FREEARC_OK)  return mf.error();
    // Coder will encode LZ output into bits and put them to outstream
    // Data should be written in HUGE_BUFFER_SIZE chunks (at least) plus chunk*2 bytes should be allocated to ensure that no buffer overflow may occur (because we flush() data only after processing each 'chunk' input bytes)
    Coder coder (m.encoding_method, callback, auxdata, tornado_compressor_outbuf_size (m.buffer, bytes_to_compress), compress_all_at_once? 0:chunk*2);
    if (coder.error() != FREEARC_OK)  return coder.error();
    BYTE *table_end  = coder.support_tables && m.find_tables? buf : buf+m.buffer+LOOKAHEAD;    // The end of last data table processed
    BYTE *last_found = buf;                             // Last position where data table was found
    byte *last_checked[MAX_TABLE_ROW][MAX_TABLE_ROW];   // Last position where data table of size %1 with offset %2 was tried
    if(coder.support_tables)  {iterate_var(i,MAX_TABLE_ROW)  iterate_var(j,MAX_TABLE_ROW)  last_checked[i][j] = buf;}
    // Use first output bytes to store encoding_method, minlen and buffer size
    coder.put8 (m.encoding_method);
    coder.put8 (mf.min_length());
    coder.put32(m.buffer);
    // Encode first four bytes directly (at least 2 bytes should be saved directly in order to avoid problems with using p-2 in MatchFinder.update())
    BYTE *p; for (p=buf; p<buf+4; p++) {
        if (p>=bufend)  goto finished;
        coder.encode (0, p, buf, mf.min_length());
    }

    // ========================================================================
    // MAIN CYCLE: FIND AND ENCODE MATCHES UNTIL DATA END
    while (TRUE) {
        if (p >= next_special_p) {
            // Read next chunk of data if all data up to read_point was already processed
            if (p >= read_point) {
                if (bytes_to_compress!=-1)  goto finished;  // We shouldn't read/write any data!
                byte *p1=p;  // This trick allows to not take address of p and this buys us a bit better program optimization
                int res = read_next_chunk (m, callback, auxdata, mf, coder, p1, buf, bufend, table_end, last_found, read_point, bytes, chunk, offset, last_checked);
                progress_point=p=p1, matchend = bufend - mymin (MAX_HASHED_BYTES, bufend-buf);
                if (res==0)  goto finished;    // All input data were successfully compressed
                if (res<0)   return res;       // Error occurred while reading data
            }
            // Report to the caller samples of insize/outsize ratio
            if (p >= progress_point) {
                PROGRESS((offset+(p-buf)) - prev_insize, coder.outsize()-prev_outsize);
                prev_insize  = offset + (p-buf);
                prev_outsize = coder.outsize();
                progress_point += PROGRESS_CHUNK_SIZE;
            }
            next_special_p = mymin(read_point,progress_point);
        }

        // Check for data table that may be subtracted to improve compression
        if (coder.support_tables  &&  p > table_end) {
            if (mf.min_length() < 4)                      // increase speed by skipping this check in faster modes
              CHECK_FOR_DATA_TABLE (2);
            CHECK_FOR_DATA_TABLE (4);
            if (p-last_found > table_dist)  table_end = p + table_shift;
            goto not_found;
            found: last_found=table_end;
            not_found:;
        }

        // Find match length and position
        UINT len = mf.find_matchlen (p, matchend, 0);
        BYTE *q  = mf.get_matchptr();
        // Encode either match or literal
        coder.set_context(p[-1]);
        if (!coder.encode (len, p, q, mf.min_length())) {      // literal encoded
            print_literal (p-buf+offset, *p); p++;
        } else {                                               // match encoded
            // Update hash and skip matched data
            check_match (p, q, len);
            print_match (p-buf+offset, len, p-q);
            mf.update_hash (p, len, bufend, m.update_step);
            p += len;
        }
    }
    // END OF MAIN CYCLE
    // ========================================================================

finished:
    stat_only (printf("\nTables %d * %d = %d bytes\n", int(table_count), int(table_sumlen/mymax(table_count,1)), int(table_sumlen)));
    // Return mf/coder error code or mark data end and flush coder
    if (mf.error()    != FREEARC_OK)   return mf.error();
    if (coder.error() != FREEARC_OK)   return coder.error();
    coder.encode (IMPOSSIBLE_LEN, buf, buf-IMPOSSIBLE_DIST, mf.min_length());
    coder.finish();
    PROGRESS((offset+(p-buf)) - prev_insize, coder.outsize()-prev_outsize);
    return coder.error();
}

// tor_compress template parameterized by MatchFinder and Coder
template <class MatchFinder, class Coder>
int tor_compress0 (PackMethod &m, CALLBACK_FUNC *callback, void *auxdata, void *buf0, int bytes_to_compress)
{
    //SET_JMP_POINT( FREEARC_ERRCODE_GENERAL);
    // Make buffer at least 32kb long and round its size up to 4kb chunk
    m.buffer = bytes_to_compress==-1?  (mymax(m.buffer, 32*kb) + 4095) & ~4095  :  bytes_to_compress;
    // If hash is too large - make it smaller
    if (m.hashsize/8     > m.buffer)  m.hashsize     = 1<<lb(m.buffer*8);
    if (m.auxhash_size/8 > m.buffer)  m.auxhash_size = 1<<lb(m.buffer*8);
    // >0: shift data in these chunks, <0: how many old bytes should be kept when buf shifts,
    // -1: don't slide buffer, fill it with new data instead
    m.shift = m.shift?  m.shift  :  (m.hash_row_width>4? m.buffer/4   :
                                     m.hash_row_width>2? m.buffer/2   :
                                     m.hashsize>=512*kb? m.buffer/4*3 :
                                                         -1);
    // Allocate buffer for input data
    void *buf = buf0? buf0 : BigAlloc (m.buffer+LOOKAHEAD);       // use calloc() to make Valgrind happy :)  or can we just clear a few bytes after fread?
    if (!buf)  return FREEARC_ERRCODE_NOT_ENOUGH_MEMORY;

    // MAIN COMPRESSION FUNCTION
    int result = tor_compress_chunk<MatchFinder,Coder> (m, callback, auxdata, (byte*) buf, bytes_to_compress);

    if (!buf0)  BigFree(buf);
    return result;
}

#include "OptimalParsing.cpp"

template <class MatchFinder, class Coder>
int tor_compress4 (PackMethod &m, CALLBACK_FUNC *callback, void *auxdata, void *buf, int bytes_to_compress)
{
    switch (m.match_parser) {
    case GREEDY: return tor_compress0 <             MatchFinder,  Coder> (m, callback, auxdata, buf, bytes_to_compress);
    case LAZY:   return tor_compress0 <LazyMatching<MatchFinder>, Coder> (m, callback, auxdata, buf, bytes_to_compress);
    default:     return FREEARC_ERRCODE_INVALID_COMPRESSOR;
    }
}

template <class MatchFinder, class Coder>
int tor_compress3 (PackMethod &m, CALLBACK_FUNC *callback, void *auxdata, void *buf, int bytes_to_compress)
{
    switch (m.hash3) {
    case 0: return tor_compress4 <MatchFinder, Coder> (m, callback, auxdata, buf, bytes_to_compress);
    case 1: return tor_compress4 <Hash3<MatchFinder,14,10,FALSE>, Coder> (m, callback, auxdata, buf, bytes_to_compress);
    case 2: return tor_compress4 <Hash3<MatchFinder,16,12,TRUE >, Coder> (m, callback, auxdata, buf, bytes_to_compress);
    default:return FREEARC_ERRCODE_INVALID_COMPRESSOR;
    }
}

template <class MatchFinder, class Coder>
int tor_compress3o (PackMethod &m, CALLBACK_FUNC *callback, void *auxdata, void *buf, int bytes_to_compress)
{
    switch (m.hash3) {
    case 1: return tor_compress0_optimal <Hash3<MatchFinder,14,10,FALSE>, Coder> (m, callback, auxdata, buf, bytes_to_compress);
    case 2: return tor_compress0_optimal <Hash3<MatchFinder,16,12,TRUE >, Coder> (m, callback, auxdata, buf, bytes_to_compress);
    default:return FREEARC_ERRCODE_INVALID_COMPRESSOR;
    }
}

template <class MatchFinder>
int tor_compress2o (PackMethod &m, CALLBACK_FUNC *callback, void *auxdata, void *buf, int bytes_to_compress)
{
    switch (m.encoding_method) {
    case BYTECODER: // Byte-aligned encoding
                    return tor_compress3o <MatchFinder, LZ77_ByteCoder>                          (m, callback, auxdata, buf, bytes_to_compress);
    case BITCODER:  // Bit-precise encoding
                    return tor_compress3o <MatchFinder, LZ77_BitCoder>                           (m, callback, auxdata, buf, bytes_to_compress);
    case HUFCODER:  // Huffman encoding
                    return tor_compress3o <MatchFinder, LZ77_Coder <HuffmanEncoder<EOB_CODE> > > (m, callback, auxdata, buf, bytes_to_compress);
    case ARICODER:  // Arithmetic encoding
                    return tor_compress3o <MatchFinder, LZ77_Coder <ArithCoder<EOB_CODE> >     > (m, callback, auxdata, buf, bytes_to_compress);
    default:        return FREEARC_ERRCODE_INVALID_COMPRESSOR;
    }
}

template <class MatchFinder>
int tor_compress2 (PackMethod &m, CALLBACK_FUNC *callback, void *auxdata, void *buf, int bytes_to_compress)
{
    switch (m.encoding_method) {
    case STORING:   // Storing - go to any tor_compress2 call
    case BYTECODER: // Byte-aligned encoding
                    return tor_compress3 <MatchFinder, LZ77_ByteCoder>                          (m, callback, auxdata, buf, bytes_to_compress);
    case BITCODER:  // Bit-precise encoding
                    return tor_compress3 <MatchFinder, LZ77_BitCoder>                           (m, callback, auxdata, buf, bytes_to_compress);
    case HUFCODER:  // Huffman encoding
                    return tor_compress3 <MatchFinder, LZ77_Coder <HuffmanEncoder<EOB_CODE> > > (m, callback, auxdata, buf, bytes_to_compress);
    case ARICODER:  // Arithmetic encoding
                    return tor_compress3 <MatchFinder, LZ77_Coder <ArithCoder<EOB_CODE> >     > (m, callback, auxdata, buf, bytes_to_compress);
    default:        return FREEARC_ERRCODE_INVALID_COMPRESSOR;
    }
}

template <class MatchFinder>
int tor_compress2d (PackMethod &m, CALLBACK_FUNC *callback, void *auxdata, void *buf, int bytes_to_compress)
{
    return tor_compress3 <MatchFinder, LZ77_DynamicCoder> (m, callback, auxdata, buf, bytes_to_compress);
}

// Compress data using compression method m and callback for i/o
int tor_compress (PackMethod &m, CALLBACK_FUNC *callback, void *auxdata, void *buf, int bytes_to_compress)
{
    if (! is_tornado_method_valid(m))
        return FREEARC_ERRCODE_INVALID_COMPRESSOR;

// When FULL_COMPILE is defined, we compile (5*4+8)*3*2 + 9*4*2 = 222 variants of compressor
// Otherwise, we compile only variants actually used by the predefined -0..-16 modes
#ifdef FULL_COMPILE
    if (m.match_parser == GREEDY  ||  m.match_parser == LAZY)
    {
        switch (m.match_finder) {
        case NON_CACHING_MF: switch (m.hash_row_width) {
                             case 1:    return tor_compress2 <MatchFinder1>     (m, callback, auxdata, buf, bytes_to_compress);
                             case 2:    return tor_compress2 <MatchFinder2>     (m, callback, auxdata, buf, bytes_to_compress);
                             default:   return tor_compress2 <MatchFinderN<4> > (m, callback, auxdata, buf, bytes_to_compress);
                             }

        case CACHING_MF4:  return tor_compress2             <   CachingMatchFinder<4> >                                   (m, callback, auxdata, buf, bytes_to_compress);
        case CACHING_MF5:  return tor_compress2d <CombineMF <   CachingMatchFinder<5>,            ExactMatchFinder<4> > > (m, callback, auxdata, buf, bytes_to_compress);
        case CACHING_MF6:  return tor_compress2d <CombineMF <   CachingMatchFinder<6>,                MatchFinderN<4> > > (m, callback, auxdata, buf, bytes_to_compress);
        case CACHING_MF7:  return tor_compress2d <CombineMF <   CachingMatchFinder<7>,          CachingMatchFinder<4> > > (m, callback, auxdata, buf, bytes_to_compress);

        case BT_MF4:       return tor_compress2d            <BinaryTreeMatchFinder<4> >                                   (m, callback, auxdata, buf, bytes_to_compress);
        case BT_MF5:       return tor_compress2d <CombineMF <BinaryTreeMatchFinder<5>,            ExactMatchFinder<4> > > (m, callback, auxdata, buf, bytes_to_compress);

        case CYCLED_MF4:   if (m.hash_row_width > 256)  return FREEARC_ERRCODE_INVALID_COMPRESSOR;
                           return tor_compress2             <CycledCachingMatchFinder<4> >                                (m, callback, auxdata, buf, bytes_to_compress);
        case CYCLED_MF5:   if (m.hash_row_width > 256)  return FREEARC_ERRCODE_INVALID_COMPRESSOR;
                           return tor_compress2d <CombineMF <CycledCachingMatchFinder<5>, ExactMatchFinder<4> > >         (m, callback, auxdata, buf, bytes_to_compress);
        case CYCLED_MF6:   if (m.hash_row_width > 256)  return FREEARC_ERRCODE_INVALID_COMPRESSOR;
                           return tor_compress2d <CombineMF <CycledCachingMatchFinder<6>, CycledCachingMatchFinder<4> > > (m, callback, auxdata, buf, bytes_to_compress);
        case CYCLED_MF7:   if (m.hash_row_width > 256)  return FREEARC_ERRCODE_INVALID_COMPRESSOR;
                           return tor_compress2d <CombineMF <CycledCachingMatchFinder<7>, CycledCachingMatchFinder<4> > > (m, callback, auxdata, buf, bytes_to_compress);
        }
    }
    else if (m.match_parser == OPTIMAL)
    {
        switch (m.match_finder) {
        case NON_CACHING_MF:  return tor_compress2o                                       <         MatchFinderN<4> >   (m, callback, auxdata, buf, bytes_to_compress);
        case CACHING_MF4:     return tor_compress2o                                       <   CachingMatchFinder<4> >   (m, callback, auxdata, buf, bytes_to_compress);
        case CACHING_MF5:     return tor_compress2o <CombineMF <   CachingMatchFinder<5>,       ExactMatchFinder<4> > > (m, callback, auxdata, buf, bytes_to_compress);
        case CACHING_MF6:     return tor_compress2o <CombineMF <   CachingMatchFinder<6>,           MatchFinderN<4> > > (m, callback, auxdata, buf, bytes_to_compress);
        case CACHING_MF7:     return tor_compress2o <CombineMF <   CachingMatchFinder<7>,     CachingMatchFinder<4> > > (m, callback, auxdata, buf, bytes_to_compress);
        case BT_MF4:          return tor_compress2o                                       <BinaryTreeMatchFinder<4> >   (m, callback, auxdata, buf, bytes_to_compress);
        case BT_MF5:          return tor_compress2o <CombineMF <BinaryTreeMatchFinder<5>,       ExactMatchFinder<4> > > (m, callback, auxdata, buf, bytes_to_compress);
        case BT_MF6:          return tor_compress2o <CombineMF <BinaryTreeMatchFinder<6>,           MatchFinderN<4> > > (m, callback, auxdata, buf, bytes_to_compress);
        case BT_MF7:          return tor_compress2o <CombineMF <BinaryTreeMatchFinder<7>,     CachingMatchFinder<4> > > (m, callback, auxdata, buf, bytes_to_compress);
        }
    }
#else
    // -1..-5(-6)
    if ((m.encoding_method==BYTECODER && m.hash_row_width==1 && m.hash3==0 && m.match_finder==NON_CACHING_MF && m.match_parser==GREEDY) ||
        m.encoding_method==STORING ) {
        return tor_compress0 <MatchFinder1, LZ77_ByteCoder> (m, callback, auxdata, buf, bytes_to_compress);
    } else if (m.encoding_method==BITCODER && m.hash_row_width==1 && m.hash3==0 && m.match_finder==NON_CACHING_MF && m.match_parser==GREEDY ) {
        return tor_compress0 <MatchFinder1, LZ77_BitCoder > (m, callback, auxdata, buf, bytes_to_compress);
    } else if (m.encoding_method==HUFCODER && m.hash_row_width==2 && m.hash3==0 && m.match_finder==NON_CACHING_MF && m.match_parser==GREEDY ) {
        return tor_compress0 <MatchFinder2, LZ77_Coder< HuffmanEncoder<EOB_CODE> > > (m, callback, auxdata, buf, bytes_to_compress);
    } else if (m.encoding_method==HUFCODER && m.hash3==0 && m.match_finder==CACHING_MF4 && m.match_parser==GREEDY ) {
        return tor_compress0 <CachingMatchFinder<4>, LZ77_Coder< HuffmanEncoder<EOB_CODE> > > (m, callback, auxdata, buf, bytes_to_compress);
    } else if (m.encoding_method==ARICODER && m.hash3==1 && m.match_finder==CACHING_MF4 && m.match_parser==LAZY ) {
        return tor_compress0 <LazyMatching<Hash3<CachingMatchFinder<4>,14,10,FALSE> >, LZ77_Coder<ArithCoder<EOB_CODE> > > (m, callback, auxdata, buf, bytes_to_compress);
    // -5 -c3 - used for FreeArc -m4$compressed
    } else if (m.encoding_method==HUFCODER && m.hash3==1 && m.match_finder==CACHING_MF4 && m.match_parser==LAZY ) {
        return tor_compress0 <LazyMatching<Hash3<CachingMatchFinder<4>,14,10,FALSE> >, LZ77_Coder< HuffmanEncoder<EOB_CODE> > > (m, callback, auxdata, buf, bytes_to_compress);

    // -7..-9
    } else if (m.hash3==2 && m.match_finder==CYCLED_MF5 && m.match_parser==LAZY ) {
        return tor_compress0 <LazyMatching <Hash3 <CombineMF <CycledCachingMatchFinder<5>, ExactMatchFinder<4> >,16,12,TRUE> >,
                              LZ77_DynamicCoder > (m, callback, auxdata, buf, bytes_to_compress);
    // -10
    } else if (m.hash3==2 && m.match_finder==CYCLED_MF7 && m.match_parser==LAZY ) {
        return tor_compress0 <LazyMatching <Hash3 <CombineMF <CycledCachingMatchFinder<7>, CycledCachingMatchFinder<4> >,16,12,TRUE> >,
                              LZ77_DynamicCoder > (m, callback, auxdata, buf, bytes_to_compress);

    // -11..-13 -c4
    } else if (m.hash3==1 && m.match_finder==CACHING_MF5 && m.match_parser==OPTIMAL && m.encoding_method==ARICODER ) {
        return tor_compress0_optimal <CombineMF <CachingMatchFinder<5>, Hash3<ExactMatchFinder<4>,14,10,FALSE> > ,
                                      LZ77_Coder<ArithCoder<EOB_CODE> > > (m, callback, auxdata, buf, bytes_to_compress);
    // -14..-15 -c4
    } else if (m.hash3==2 && m.match_finder==CACHING_MF6 && m.match_parser==OPTIMAL && m.encoding_method==ARICODER ) {
        return tor_compress0_optimal <CombineMF <CachingMatchFinder<6>, Hash3<MatchFinderN<4>,16,12,TRUE> > ,
                                      LZ77_Coder<ArithCoder<EOB_CODE> > > (m, callback, auxdata, buf, bytes_to_compress);
    // -14..-15
    } else if (m.hash3==2 && m.match_finder==CACHING_MF6 && m.match_parser==OPTIMAL ) {
        return tor_compress0_optimal <CombineMF <CachingMatchFinder<6>, Hash3<MatchFinderN<4>,16,12,TRUE> > ,
                                      LZ77_DynamicCoder > (m, callback, auxdata, buf, bytes_to_compress);
    // -16
    } else if (m.hash3==2 && m.match_finder==BT_MF5 && m.match_parser==OPTIMAL ) {
        return tor_compress0_optimal <CombineMF <BinaryTreeMatchFinder<5>, Hash3<ExactMatchFinder<4>,16,12,TRUE> > ,
                                      LZ77_DynamicCoder > (m, callback, auxdata, buf, bytes_to_compress);
    }
#endif
    return FREEARC_ERRCODE_INVALID_COMPRESSOR;
}

#endif // FREEARC_DECOMPRESS_ONLY

// LZ77 decompressor ******************************************************************************

// If condition is true, write data to outstream
#define WRITE_DATA_IF(condition)                                                                  \
{                                                                                                 \
    if (condition) {                                                                              \
        if (decoder.error() != FREEARC_OK)  goto finished;                                        \
        tables.undiff_tables (write_start, output);                                               \
        debug (printf ("==== write %08x:%x ====\n", write_start-outbuf+offset, output-write_start)); \
        WRITE (write_start, output-write_start);                                                  \
        PROGRESS(decoder.insize()-prev_insize, output-write_start);                               \
        prev_insize = decoder.insize();                                                           \
        tables.diff_tables (write_start, output);                                                 \
        write_start = output;  /* next time we should start writing from this pos */              \
                                                                                                  \
        /* Check that we should shift the output pointer to start of buffer */                    \
        if (output >= outbuf + bufsize) {                                                         \
            offset_overflow |= (offset > (uint64(1) << 63));                                      \
            offset      += output-outbuf;                                                         \
            write_start -= output-outbuf;                                                         \
            write_end   -= output-outbuf;                                                         \
            tables.shift (output,outbuf);                                                         \
            output      -= output-outbuf;  /* output = outbuf; */                                 \
        }                                                                                         \
                                                                                                  \
        /* If we wrote data because write_end was reached (not because */                         \
        /* table list was filled), then set write_end into its next position */                   \
        if (write_start >= write_end) {                                                           \
            /* Set up next write chunk to HUGE_BUFFER_SIZE or until buffer end - whatever is smaller */ \
            write_end = write_start + mymin (outbuf+bufsize-write_start, HUGE_BUFFER_SIZE);       \
        }                                                                                         \
    }                                                                                             \
}


template <class Decoder>
int tor_decompress0 (CALLBACK_FUNC *callback, void *auxdata, int _bufsize, int minlen)
{
    //SET_JMP_POINT (FREEARC_ERRCODE_GENERAL);
    int errcode = FREEARC_OK;                             // Error code of last "write" call
    Decoder decoder (callback, auxdata, _bufsize);        // LZ77 decoder parses raw input bitstream and returns literals&matches
    if (decoder.error() != FREEARC_OK)  return decoder.error();
    uint bufsize = tornado_decompressor_outbuf_size (_bufsize);  // Size of output buffer
    BYTE *outbuf = (byte*) BigAlloc (bufsize+PAD_FOR_TABLES*2);  // Circular buffer for decompressed data
    if (!outbuf)  return FREEARC_ERRCODE_NOT_ENOUGH_MEMORY;
    outbuf += PAD_FOR_TABLES;       // We need at least PAD_FOR_TABLES bytes available before and after outbuf in order to simplify datatables undiffing
    BYTE *output      = outbuf;     // Current position in decompressed data buffer
    BYTE *write_start = outbuf;     // Data up to this point was already writen to outsream
    BYTE *write_end   = outbuf + mymin (bufsize, HUGE_BUFFER_SIZE); // Flush buffer when output pointer reaches this point
    if (compress_all_at_once)  write_end = outbuf + bufsize + 1;    // All data should be written after decompression finished
    uint64 offset = 0;                    // Current outfile position corresponding to beginning of outbuf
    int offset_overflow = 0;              // Flags that offset was overflowed so we can't use it for match checking
    uint64 prev_insize = 0;               // Bytes read at the moment of latest write operation
    DataTables tables;                    // Info about data tables that should be undiffed
    for (;;) {
        // Check whether next input element is a literal or a match
        if (decoder.is_literal()) {
            // Decode it as a literal
            BYTE c = decoder.getchar();
            print_literal (output-outbuf+offset, c);
            *output++ = c;
            WRITE_DATA_IF (output >= write_end);  // Write next data chunk to outstream if required

        } else {
            // Decode it as a match
            UINT len  = decoder.getlen(minlen);
            UINT dist = decoder.getdist();
            print_match (output-outbuf+offset, len, dist);

            // Check for simple match (i.e. match not requiring any special handling, >99% of matches fall into this category)
            if (output-outbuf>=dist && write_end-output>len) {
                BYTE *p = output-dist;
                do   *output++ = *p++;
                while (--len);

            // Check that it's a proper match
            } else if (len<IMPOSSIBLE_LEN) {
                // Check that compressed data are not broken
                if (dist>bufsize || len>2*_bufsize || (output-outbuf+offset<dist && !offset_overflow))  {errcode=FREEARC_ERRCODE_BAD_COMPRESSED_DATA; goto finished;}
                // Slow match copying route for cases when output-dist points before buffer beginning,
                // or p may wrap at buffer end, or output pointer may run over write point
                BYTE *p  =  output-outbuf>=dist? output-dist : output-dist+bufsize;
                do {
                    *output++ = *p++;
                    if (p==outbuf+bufsize)  p=outbuf;
                    WRITE_DATA_IF (output >= write_end);
                } while (--len);

            // Check for special len/dist code used to encode EOF
            } else if (len==IMPOSSIBLE_LEN && dist==IMPOSSIBLE_DIST) {
                WRITE_DATA_IF (TRUE);  // Flush outbuf
                goto finished;

            // Otherwise it's a special code used to represent info about diffed data tables
            } else {
                len -= IMPOSSIBLE_LEN;
                if (len==0 || dist*len > 2*_bufsize)  {errcode=FREEARC_ERRCODE_BAD_COMPRESSED_DATA; goto finished;}
                stat_only (printf ("\n%d: Start %x, end %x, length %d      ", len, int(output-outbuf+offset), int(output-outbuf+offset+len*dist), len*dist));
                // Add new table to list: len is row length of table and dist is number of rows
                tables.add (len, output, dist);
                // If list of data tables is full then flush it by preprocessing
                // and writing to outstream already filled part of outbuf
                WRITE_DATA_IF (tables.filled() && !compress_all_at_once);
            }
        }
    }
finished:
    PROGRESS(decoder.insize()-prev_insize, output-write_start);
    BigFree(outbuf-PAD_FOR_TABLES);
    // Return decoder error code, errcode or FREEARC_OK
    return decoder.error() < 0 ?  decoder.error() :
           errcode         < 0 ?  errcode
                               :  FREEARC_OK;
}


int tor_decompress (CALLBACK_FUNC *callback, void *auxdata, void *buf, int bytes_to_compress)
{
    int errcode;
    // First 6 bytes of compressed data are encoding method, minimum match length and buffer size
    BYTE header[2];       READ (header, 2);
   {uint encoding_method = header[0];
    uint minlen          = header[1];
    uint bufsize;         READ4 (bufsize);

    switch (encoding_method) {
    case BYTECODER:
            return tor_decompress0 <LZ77_ByteDecoder> (callback, auxdata, bufsize, minlen);

    case BITCODER:
            return tor_decompress0 <LZ77_BitDecoder>  (callback, auxdata, bufsize, minlen);

    case HUFCODER:
            return tor_decompress0 <LZ77_Decoder <HuffmanDecoder<EOB_CODE> > > (callback, auxdata, bufsize, minlen);

    case ARICODER:
            return tor_decompress0 <LZ77_Decoder <ArithDecoder<EOB_CODE> >   > (callback, auxdata, bufsize, minlen);

    default:
            errcode = FREEARC_ERRCODE_BAD_COMPRESSED_DATA;
    }}
finished: return errcode;
}

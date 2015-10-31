// LZ optimal parser
//
// (c) Bulat Ziganshin
// (c) Joachim Henke
// (c) Igor Pavlov
// This code is provided on the GPL license.
// If you need a commercial license to use the code, please write to Bulat.Ziganshin@gmail.com

#define OPTIMAL_WINDOW               (32*kb)  /* used in #if */
#define OPTIMAL_PARSER_MIN_MATCH_LEN  2      /* used in #if */
const UINT MAX_REPDIST = REPDIST_CODES;

#if OPTIMAL_PARSER_MIN_MATCH_LEN > 2
#error Incompatible with the OptimalParser class since evaluate_literal() is not called while p<silence so that positions [p+2,p+OPTIMAL_PARSER_MIN_MATCH_LEN) may remain unpriced
#endif

#if OPTIMAL_WINDOW*PRICE_BITS(63) >= MAX_PRICE  // we presuppose that single literal/match can be encoded in 63 bits or less
#error The real position price in the optimal parser may become higher than MAX_PRICE, in particular with arithmetic coder
#endif

#ifdef STAT  // Optimal parsing statistics
    size_t cnt_eval_lit=0, cnt_eval_repdist=0, cnt_check_match_len=0, cnt_eval_len_in_repdist=0, cnt_eval_match=0, cnt_eval_len_in_match=0;
#endif


template <class Coder>
struct OptimalParser : Coder
{
    static const bool support_tables = FALSE;   // Optimal parsing is incompatible with the table diffing

    struct Info            // Information about one position of the block being processed
    {
        int      len;      // length and distance of best match so far, *ending* in this position (len=1,dist=0 for literal)
        DISTANCE dist;     //            ^^^^^^^^
        DISTANCE prevdist[REPDIST_CODES];  // distances of previous matches (filled after the optimum path to the position was found)
    };
    Info     *x;           // Information for the every position of the block being processed
    PRICE     prices[OPTIMAL_WINDOW+1];    // Best price so far the every position or the MAX_PRICE (separated from the Info just for a small speed optimization)
    DISTANCE *stack_buf;   // Stack used as temporary storage of optimal path from end of block to the beginning
    int       fast_bytes;

    BYTE     *buf, *endbuf;  // Buffer boundaries
    BYTE     *basep;
    BYTE     *lastp;
    CALLBACK_FUNC *callback;
    void     *auxdata;
    uint64    prev_outsize;
    int       errcode;
    // Returns error code if there were any problems in memory allocation or Coder
    int error()               {return (errcode != FREEARC_OK?  errcode  :  Coder::error());}

    OptimalParser (BYTE *_buf, BYTE *_endbuf, int _coder, CALLBACK_FUNC *_callback, void *_auxdata, UINT chunk, UINT pad, int _fast_bytes)
          : Coder (_coder, _callback, _auxdata, chunk, pad),  buf(_buf),  endbuf(_endbuf),  callback(_callback),  auxdata(_auxdata),  fast_bytes(_fast_bytes)
    {
        x         = (Info*)     MidAlloc (sizeof(Info)     * (OPTIMAL_WINDOW+1));
        stack_buf = (DISTANCE*) MidAlloc (sizeof(DISTANCE) * (OPTIMAL_WINDOW*2));
        errcode   = (x==NULL || stack_buf==NULL)?  FREEARC_ERRCODE_NOT_ENOUGH_MEMORY : FREEARC_OK;
        prev_outsize = 0;
    }
    ~OptimalParser()
    {
        stat_only (printf("lit=%.3lf, repdist=%.3lf, check_match_len=%.3lf, len_in_repdist=%.3lf, match=%.3lf, len_in_match=%.3lf\n",
                          double(cnt_eval_lit)/1e6, double(cnt_eval_repdist)/1e6, double(cnt_check_match_len)/1e6, double(cnt_eval_len_in_repdist)/1e6, double(cnt_eval_match)/1e6, double(cnt_eval_len_in_match)/1e6));
        MidFree (stack_buf);
        MidFree (x);
    }

    void start_block  (BYTE *_basep);  // Prepare to collect statistics, required to optimally encode the block starting at address _basep
    void encode_block (BYTE *endp);    // Optimally encode the already evaluated [basep,endp) block

    // Called prior to / after shifting buffer contents toward the beginning
    void before_shift (BYTE *endp)     {encode_block(endp);  Coder::before_shift(endp);}
    void after_shift  (BYTE *_basep)   {Coder::after_shift(_basep);  start_block(_basep);}

    // Replace current encoding starting at the position `p` if the new_price outbids the old one
    void evaluate (BYTE *p, int len, DISTANCE dist, PRICE new_price)
    {
        int src = p-basep,  dst = src+len;
        if (prices[dst]  > new_price) {
            prices[dst]  = new_price;
            x[dst].len   = len;
            x[dst].dist  = dist;
        }
    }

    // Evaluate all the encoding possibilities starting at the position `p`, including literal, matches at repeated distances and all found matches.
    // The [matches,matches_end) buffer should contain length/distance for every match found, starting at the position `p`, in the order of strictly increasing `length`.
    // Returns the maximum match length checked + 1 (or OPTIMAL_PARSER_MIN_MATCH_LEN)
    UINT evaluate_literal_and_repdist (BYTE *p)
    {
        Coder::set_context(p[-1]);     // May be required for calculation of match prices
        int   len   = OPTIMAL_PARSER_MIN_MATCH_LEN;
        int   src   = p - basep;
        PRICE price = prices[src];     // Price of the current position

        // Evaluate literal: price[p+1] = price[p] + the *p literal price
        evaluate (p, 1, 0, price + Coder::lit_price(*p));
        stat_only (cnt_eval_lit++);
        if (!Coder::support_repdist || src==0)  return len;

        // Get the last match distance and pointer to the previous distances
        DISTANCE dist = x[src].dist,  *prevdist_ptr;
        if (dist == 0) {
            prevdist_ptr = x[src-1].prevdist;    // last item was a literal -> load all distances from the previous position
            dist = *prevdist_ptr++;
        } else {
            int dst = src - x[src].len;          // last item was a match - load all but last distances from its destination position
            prevdist_ptr = x[dst].prevdist;
        }

        // Evaluate REPCHAR match - unfortunately, it decreases the compression ratio. May be, it's just a bug
        //if (dist < p-buf  &&  *p == *(p-dist))
            //evaluate (p, 1, 0, price + Coder::lit_price(REPCHAR));

        // Evaluate REPDIST-based matches
        for (int i=0; i<MAX_REPDIST; i++)
        {
            x[src].prevdist[i] = dist;
            if (dist > p-buf)  break;                      // match start was already shifted out of buffer (to do: cyclicBufferSize check)
            stat_only (cnt_eval_repdist++);

            if (p[len-1] == (p-dist)[len-1]) {
                // Evaluate all matches in the [prev_len+1,cur_len] length range
                for (UINT match_len = check_match_len(p, p-dist, lastp-p);  len<=match_len;  len++) {
                    // Evaluate price[p+len] = price[p] + repdist_price(len,i)
                    evaluate (p, len, dist, price + Coder::repdist_price(len-OPTIMAL_PARSER_MIN_MATCH_LEN, i));
                    stat_only (cnt_eval_len_in_repdist++);
                }
                stat_only (cnt_check_match_len++);

                if (len > fast_bytes)  // skip additional time-consuming checks if large match was found
                {
                    for(int j=i+1; j<MAX_REPDIST; j++)             // but first fill up the remaining prevdist[] entries
                        x[src].prevdist[j] = *prevdist_ptr++;
                    break;
                }
            }
            dist = *prevdist_ptr++;
        }
        return len;
    }

    UINT evaluate_matches (BYTE *p, int len, DISTANCE *matches, DISTANCE *matches_end)
    {
        if (matches < matches_end)
        {
            PRICE price = prices[p-basep];     // Price of the current position

            // Evaluate all the provided matches, checking intermediate sizes too
            do {
                int match_len = *matches++;
                DISTANCE dist = *matches++;
                // Evaluate all matches in the [prev_len+1,cur_len] length range
                for ( ; len<=mymin(match_len,lastp-p);  len++) {
                    // Evaluate price[p+len] = price[p] + match_price(len,dist)
                    evaluate (p, len, dist, price + Coder::match_price(len-OPTIMAL_PARSER_MIN_MATCH_LEN, dist));
                    stat_only (cnt_eval_len_in_match++);
                }
                stat_only (cnt_eval_match++);
            } while (matches < matches_end);
        }
        return len;
    }
};

// Prepare to collect statistics, required to optimally encode the block starting at address _basep, with the size up to OPTIMAL_WINDOW bytes
template <class Coder>
void OptimalParser<Coder> :: start_block (BYTE *_basep)
{
    // Pointers to the boundaries of the block we are going to optimally encode
    basep = _basep;
    lastp = basep + mymin(endbuf-basep, OPTIMAL_WINDOW);
    // First block position has the zero price, since we are already here
    prices[0] = 0;
    // We overprice the remaining positions to ensure that these prices will be overbid by the real encoding opportunities
    for (int i=1; i<OPTIMAL_WINDOW+1; i++)
        prices[i] = MAX_PRICE;
    // These distances are used by the REPDIST evaluation
    iterate_var(i,REPDIST_CODES)  x[0].prevdist[i] = UINT_MAX;
}

// Optimally encode the already evaluated [basep,endp) block
template <class Coder>
void OptimalParser<Coder> :: encode_block (BYTE *endp)
{
    if (endp==basep)  return;  // there is nothing to encode, since the block was just started
    DISTANCE *stack = stack_buf;
    // Construct the optimal encoding path from the end of buffer toward the beginning
    for (int i=endp-basep; i>0; ) {
        *stack++ = x[i].dist;  // push to the stack our best deal for the i'th position
        *stack++ = x[i].len;
        i -= x[i].len;         // ... and go back by the length of this literal/match
    }
    // And, finally, encode the optimal path from the start of buffer till the end
    for (BYTE *p = basep;  stack > stack_buf;  ) {
        int      len  = *--stack;
        DISTANCE dist = *--stack;
        // Encode either match or literal
        Coder::set_context(p[-1]);
        Coder::encode (len, p, p-dist, OPTIMAL_PARSER_MIN_MATCH_LEN);
        p += len;
    }

    PROGRESS(endp-basep, Coder::outsize()-prev_outsize);
    prev_outsize = Coder::outsize();
}


// Optimally compress the single chunk of data
template <class MatchFinder, class Coder>
int tor_compress_chunk_optimal (PackMethod m, CALLBACK_FUNC *callback, void *auxdata, byte *buf, int bytes_to_compress)
{
    // Read data in these chunks
    int chunk = compress_all_at_once? m.buffer : mymin (m.shift>0? m.shift:m.buffer, LARGE_BUFFER_SIZE);
    uint64 offset = 0;                        // Current offset of buf[] contents relative to file (increased with every shift() operation)
    int bytes = bytes_to_compress!=-1? bytes_to_compress : callback ("read", buf, chunk, auxdata);   // Number of bytes read by last "read" call
    if (bytes<0)  return bytes;               // Return errcode on error
    BYTE *bufend = buf + bytes;               // Current end of real data in buf[]
    BYTE *matchend = bufend - mymin (MAX_HASHED_BYTES, bytes);   // Maximum pos where match may finish (less than bufend in order to simplify hash updating)
    BYTE *read_point = compress_all_at_once || bytes_to_compress!=-1? bufend : bufend-mymin(LOOKAHEAD,bytes); // Next point where next chunk of data should be read to buf
    // Match finder will search strings similar to current one in previous data
    MatchFinder mf (buf, m.buffer, m.hashsize, m.hash_row_width, m.auxhash_size, m.auxhash_row_width);
    if (mf.error() != FREEARC_OK)  return mf.error();
    // Coder will encode LZ output into bits and put them to outstream
    // Data should be written in HUGE_BUFFER_SIZE chunks (at least) plus chunk*2 bytes should be allocated to ensure that no buffer overflow may occur (because we flush() data only after processing each 'chunk' input bytes)
    OptimalParser<Coder>  coder (buf, buf+m.buffer, m.encoding_method, callback, auxdata, tornado_compressor_outbuf_size (m.buffer, bytes_to_compress), compress_all_at_once? 0:chunk*2, m.fast_bytes);
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
    BYTE *p, *silence = buf;
    for (p=buf; p<buf+4; p++) {
        if (p>=bufend)  goto finished;
        coder.encode (0, p, buf, mf.min_length());
    }
    coder.start_block(p);

    // ========================================================================
    // MAIN CYCLE: FIND AND ENCODE MATCHES UNTIL DATA END
    for (; TRUE; p++) {
        // Read next chunk of data if all data up to read_point was already processed
        if (p >= read_point) {
            if (bytes_to_compress!=-1)  goto finished;  // We shouldn't read/write any data!
            byte *p1=p;  // This trick allows to not take address of p and this buys us a bit better program optimization
            int res = read_next_chunk (m, callback, auxdata, mf, coder, p1, buf, bufend, table_end, last_found, read_point, bytes, chunk, offset, last_checked);
            p=p1, matchend = bufend - mymin (MAX_HASHED_BYTES, bufend-buf);
            if (res==0)  break;            // All input data were successfully compressed
            if (res<0)   return res;       // Error occurred while reading data
            silence = buf;
        }

        // Optimally-encode current block and start the new one if there are enough data processed
        if (p >= coder.lastp) {
            coder.encode_block(p);
            silence = buf;
            coder.start_block(p);
        }

        // Find and evaluate all possible encodings starting at the `p`
        if (p >= silence)
        {
            mf.prefetch_hash(p+1);  // prefetching in CachingMatchFinder improves speed by 5-20%

            UINT len = coder.evaluate_literal_and_repdist (p);
            if (len <= m.fast_bytes)  // skip match search&evaluaion after a long repmatch
            {
                DISTANCE matches[MAX_MATCHES*2];
                DISTANCE *matches_end = mf.find_all_matches (p, matchend, matches);
                len = coder.evaluate_matches (p, len, matches, matches_end);
            }

            // After a long match, skip any search&evaluaion until end of the match, in order to suppress O(n^2) time behavior
            if (len > m.fast_bytes)
                silence = p + len-1;
        }
    }
    // END OF MAIN CYCLE
    // ========================================================================
    coder.encode_block(p);  // flush matches from the last optimal-encoding block

finished:
    stat_only (printf("\nTables %d * %d = %d bytes\n", int(table_count), int(table_sumlen/mymax(table_count,1)), int(table_sumlen)));
    // Return mf/coder error code or mark data end and flush coder
    if (mf.error()    != FREEARC_OK)   return mf.error();
    if (coder.error() != FREEARC_OK)   return coder.error();
    coder.encode (IMPOSSIBLE_LEN, buf, buf-IMPOSSIBLE_DIST, mf.min_length());
    coder.finish();
    return coder.error();
}

// tor_compress template parameterized by MatchFinder and Coder
template <class MatchFinder, class Coder>
int tor_compress0_optimal (PackMethod m, CALLBACK_FUNC *callback, void *auxdata, void *buf0, int bytes_to_compress)
{
    //SET_JMP_POINT( FREEARC_ERRCODE_GENERAL);
    CHECK (FREEARC_ERRCODE_INTERNAL,  MatchFinder::min_length() == OPTIMAL_PARSER_MIN_MATCH_LEN,  (s,"Fatal error: Unsupported MatchFinder type: MatchFinder::min_length()==%d but only value %d is supported by the OptimalParser", MatchFinder::min_length(), OPTIMAL_PARSER_MIN_MATCH_LEN));
    // Make buffer at least 32kb long and round its size up to 4kb chunk
    m.buffer = bytes_to_compress==-1?  (mymax(m.buffer, 32*kb) + 4095) & ~4095  :  bytes_to_compress;
    // If hash is too large - make it smaller
    if (m.hashsize/8     > m.buffer)  m.hashsize     = 1<<lb(m.buffer*8);
    if (m.auxhash_size/8 > m.buffer)  m.auxhash_size = 1<<lb(m.buffer*8);
    // >0: shift data in these chunks, <0: how many old bytes should be kept when buf shifts,
    // -1: don't slide buffer, fill it with new data instead
    m.shift = m.shift?  m.shift  :  m.buffer/16;
    // Allocate buffer for input data
    void *buf = buf0? buf0 : BigAlloc (m.buffer+LOOKAHEAD);       // use calloc() to make Valgrind happy :)  or can we just clear a few bytes after fread?
    if (!buf)  return FREEARC_ERRCODE_NOT_ENOUGH_MEMORY;

    // MAIN COMPRESSION FUNCTION
    int result = tor_compress_chunk_optimal<MatchFinder,Coder> (m, callback, auxdata, (byte*) buf, bytes_to_compress);

    if (!buf0)  BigFree(buf);
    return result;
}

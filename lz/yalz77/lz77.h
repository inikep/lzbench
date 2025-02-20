#ifndef __X_LZ77_H
#define __X_LZ77_H

/*
 * This code is in the public domain, see: http://unlicense.org/
 *
 * Feel free to steal it.
 */

/*
 * This is a variation on the LZ77 compression algorithm.
 * It is designed for code simplicity and clarity.
 * 
 * Highlights:
 *
 *   - Portable, self-contained, tiny implementation in readable C++. 
 *     (Header-only, no ifdefs or CPU dependencies or other stupid tricks.)
 *   - Fast decompression.
 *   - Pretty good compression quality.
 *   - Simple 'one-button' API for realistic use-cases.
 *   - No penalty (only 2 bytes overhead) even when compressing very short strings.
 *   - Fully streamable decompressor: feed it chunks of arbitrarity-sized data, and the
 *     original uncompressed buffers will be reconstructed automatically.
 *
 * Compression performance and quality should be _roughly_ on par with other
 * compression algorithms.
 *
 * (Compression ratio is comparable to other LZ algorithms when at high quality 
 * settings, compression speed is comparable to gzip. Decompression speed is on par
 * with other fast LZ algorithms.)
 *
 */

/* 
 * Usage:

  #include "lz77.h"

  std::string input = ...;
  lz77::compress_t compress;
  std::string compressed = compress.feed(input);

  lz77::decompress_t decompress;

  std::string temp;
  decompress.feed(compressed, temp);

  const std::string& uncompressed = decompress.result();

  NOTE: if you're compressing short strings (on the order of a few kilobytes)
  then instantiating lz77::compress_t with the arguments (12, 4096) will
  give better results.

  --------

  Use decompress.feed(...) for feeding input data step-by-step in chunks.
  For example, if you're trying to decompress a network byte stream:

  lz77::decompress_t decompress;
  std::string extra;

  bool done = decompress.feed(buffer, extra);

  while (!done) {
    done = decompress.feed(buffer, extra);
  }

  std::string result = decompress.result();

  'feed' will start (or continue) decoding a packet of compressed data.
  If it returns true, then all of the message was decompressed.
  If it returns false, then 'feed' needs to be called with more
  compressed data until it returns 'true'.

  'extra' will hold any data that was tacked onto the buffer but wasn't
  part of this packet of compressed data. (Useful when decoding a message
  stream that doesn't have clearly delineated message boundaries; the 
  decompressor will detect message boundaries properly for you.)

  'extra' will be assigned to only when 'feed' returns true.

  'result' is the decompressed message.

  NOTE: calling feed() and result() out of order is undefined
  behaviour and might result in crashes.

*/

 

#include <stdexcept>
#include <string>
#include <vector>

#include <string.h>
#include <stdint.h>


namespace lz77 {

// Constants.
// They were chosen by a series of unscientific empirical tests.
enum {
    DEFAULT_SEARCHLEN = 12,
    DEFAULT_BLOCKSIZE = 64*1024,
    SHORTRUN_BITS = 3,
    SHORTRUN_MAX = (1 << SHORTRUN_BITS),
    MIN_RUN = 5
};


// Utility function: encode a size_t as a variable-sized stream of octets with 7 bits of useful data. 
// (One bit is used to signal an end of stream.)

inline void push_vlq_uint(size_t n, std::string& out) {

    while (1) {
        unsigned char c = n & 0x7F;
        size_t q = n >> 7;

        if (q == 0) {
            out += c;
            break;
        }

        out += (c | 0x80);
        n = q;
    }
}

// Utility function: return common prefix length of two strings.

inline size_t substr_run(const unsigned char* ai, const unsigned char* ae,
                         const unsigned char* bi, const unsigned char* be) {

    size_t n = 0;

    while (1) {

        if (*ai != *bi)
            break;

        ++n;
        ++ai;
        ++bi;

        if (ai == ae || bi == be)
            break;
    }

    return n;
}

// Utility function: Hash the first MIN_RUN bytes of a string into 16-bit ints.
// (MIN_RUN is a magic constant.)
// The hash function itself is important for compression quality.
// This is the FNV hash, a very very simple and quite good hash algorithm.

inline uint32_t fnv32a(const unsigned char* i, size_t len, uint32_t hash = 0x811c9dc5) {

    while (len > 0) {
        hash ^= (uint32_t)(*i);
        hash *= (uint32_t)0x01000193;
        ++i;
        --len;
    }

    return hash;
}

inline void pack_bytes(const unsigned char* i, uint16_t& packed, size_t blocksize) {

    uint32_t a = fnv32a(i, MIN_RUN);

    packed = a % blocksize;
}


// Compute the profit from compression; 'run' is the length of a string at position 'offset'.
// 'run' and 'offset' are numbers encoded as variable-length bitstreams; the sum length of 
// encoded 'run' and 'offset' must be less than 'run'.

inline size_t vlq_length(size_t x) {
    size_t ret = 1;

    if (x > 0x7F)
        ret++;

    if (x > 0x3fff)
        ret++;

    if (x > 0x1fffff)
        ret++;

    return ret;
}

inline size_t gains(size_t run, size_t offset) {
    // Note: this function uses knowledge about the layout of bits in the compressed data format.
    
    size_t gain = run;
    
    offset = offset << (SHORTRUN_BITS + 1);

    size_t loss = vlq_length(offset);
    
    if (run >= SHORTRUN_MAX) {
        loss += vlq_length(run - MIN_RUN + 1);
    }

    if (loss > gain)
        return 0;

    return gain - loss;
}

// Hash table already seen strings; it maps from a hash of a string prefix to
// a list of offsets. (At each offset there is a string with a prefix that hashes
// to the key.)

struct offsets_dict_t {

    typedef std::vector<size_t> offsets_t;
    offsets_t offsets;

    size_t searchlen;
    size_t blocksize;

    offsets_dict_t(size_t sl, size_t bs) : searchlen(sl), blocksize(bs) {

        offsets.resize((searchlen + 1) * blocksize);
    }

    void clear() {
        offsets.assign((searchlen + 1) * blocksize, 0);
    }
        
    // Functions for a simple circular buffer data structure.

    static size_t* prev(size_t* b, size_t* e, size_t* i) {

        if (i == b)
            i = e;

        --i;
        return i;
    }

    static size_t push_back(size_t* b, size_t* e, size_t* head, size_t val) {

        *head = val;
        ++head;

        if (head == e)
            head = b;

        return head - b;
    }
    
    void operator()(uint16_t packed, const unsigned char* i0, const unsigned char* i, const unsigned char* e,
                    size_t& maxrun, size_t& maxoffset, size_t& maxgain) {

        // Select a range of values representing a circular buffer.
        // The first value is the index of the buffer head, the rest are
        // the values of the buffer itself. 

        size_t* cb_start = &offsets[packed * (searchlen + 1)];

        size_t* cb_beg = (cb_start + 1);
        size_t* cb_end = (cb_start + 1 + searchlen);
        size_t* cb_head = cb_beg + *cb_start;

        size_t* cb_i = cb_head;

        while (1) {

            cb_i = prev(cb_beg, cb_end, cb_i);

            if (*cb_i == 0)
                break;

            // The stored value is position + 1 to allow 0 to mean 'uninitialized offset'.
            size_t pos = *cb_i - 1;

            size_t offset = i - i0 - pos;
            size_t run = substr_run(i, e, i0 + pos, e);
            size_t gain = gains(run, offset);

            if (gain > maxgain) {
                maxrun = run;
                maxoffset = offset;
                maxgain = gain;
            }

            if (cb_i == cb_head)
                break;
        }

        *cb_start = push_back(cb_beg, cb_end, cb_head, i - i0 + 1);
    }
};

/*
 * 
 * Entry point for compression.
 * 
 * Inputs: std::string of data to be compressed.
 *
 * Also optionally parameters for tuning speed and quality.
 *
 * There are two parameters: 'searchlen' and 'blocksize'.
 *
 * 'blocksize' is the upper bound for hash table sizes.
 * 'searchlen' is the upper bound for lists of offsets at each hash value.
 *
 * A larger 'searchlen' increases compression quality, running time and memory consumption. 
 * A larger 'blocksize' increases memory consumption and compression quality. 
 *
 * If you want faster compression at the expense of quality, try lowering searchlen.
 *
 * If you only ever compress short strings, try lowering blocksize to save memory.
 *
 * Output: the compressed data as a string.
 */

struct compress_t {

    offsets_dict_t offsets;

    compress_t(size_t searchlen = DEFAULT_SEARCHLEN, size_t blocksize = DEFAULT_BLOCKSIZE) :
        offsets(searchlen, blocksize) {}
    
    std::string feed(const unsigned char* i, const unsigned char* e) {

        const unsigned char* i0 = i;

        std::string ret;
        
        std::string unc;

        push_vlq_uint(e - i, ret);

        offsets.clear();

        size_t blocksize = offsets.blocksize;

        while (i != e) {

            unsigned char c = *i;

            // The last MIN_RUN-1 bytes are uncompressable. (At least MIN_RUN bytes
            // are needed to calculate a prefix hash.)

            if (i > e - MIN_RUN) {

                unc +=c;
                ++i;
                continue;
            }

            size_t maxrun = 0;
            size_t maxoffset = 0;
            size_t maxgain = 0;

            uint16_t packed;

            // The MIN_RUN prefix length was chosen empirically, based on a series
            // of unscientific tests.

            pack_bytes(i, packed, blocksize);

            offsets(packed, i0, i, e, maxrun, maxoffset, maxgain);

            if (maxrun < MIN_RUN) {
                unc += c;
                ++i;
                continue;
            }

            if (unc.size() > 0) {
                // Write a packet of uncompressed data.

                size_t msg = (unc.size() << 1) | 1;
                push_vlq_uint(msg, ret);
                ret += unc;
                unc.clear();
            }

            // A compressed string is a length and an offset.
            // First subtract the minimum length (smaller lengths don't exist).
            // Then check if the length fits in SHORTRUN_BITS bits; if it does, then
            // tack it on to the offset. Otherwise write length and offset separately.
            // The rightmost bit is a zero to differentiate from packets of
            // uncompressed data.

            i += maxrun;
            maxrun = maxrun - MIN_RUN + 1;

            if (maxrun < SHORTRUN_MAX) {

                size_t msg = ((maxoffset << SHORTRUN_BITS) | maxrun) << 1;
                push_vlq_uint(msg, ret);

            } else {

                size_t msg = (maxoffset << (SHORTRUN_BITS + 1));
                push_vlq_uint(msg, ret);
                push_vlq_uint(maxrun, ret);
            }
        }

        if (unc.size() > 0) {

            size_t msg = (unc.size() << 1) | 1;
            push_vlq_uint(msg, ret);
            ret += unc;
            unc.clear();
        }

        return ret;
    }

    std::string feed(const std::string& s) {

        const unsigned char* i = (const unsigned char*)s.data();
        const unsigned char* e = i + s.size();
        return feed(i, e);
    }
};

/*
 * Entry point for decompression.
 * Calling 'feed' and 'result' out of order is undefined behaviour 
 * and will crash your program.
 */

struct decompress_t {

    size_t max_size;
    std::string ret;
    unsigned char* out;
    unsigned char* outb;
    unsigned char* oute;

    struct state_t {
        size_t msg;
        size_t run;
        size_t vlq_num;
        size_t vlq_off;
        enum {
            INIT,
            START,
            READ_DATA,
            READ_RUN
        } state;

        state_t() : msg(0), run(0), vlq_num(0), vlq_off(0), state(INIT) {}
    };

    state_t state;

    // Utility function: decode variable-length-coded unsigned integers.

    bool pop_vlq_uint(const unsigned char*& i, const unsigned char* e, size_t& res) {

        while (1) {

            if (i == e)
                return false;

            size_t c = *i;

            if ((c & 0x80) == 0) {
                state.vlq_num |= (c << state.vlq_off);
                break;
            }

            state.vlq_num |= ((c & 0x7F) << state.vlq_off);
            state.vlq_off += 7;
            ++i;
        }

        res = state.vlq_num;
        state.vlq_num = 0;
        state.vlq_off = 0;

        return true;
    }


    /*
     * max_size is the maximum size of decompressed data you're willing to accept.
     * This is strictly optional and needed for safety reasons, when you're
     * paranoid about accepting data from unknown sources.
     *
     * The default of 0 means no sanity checking is done.
     */

    decompress_t(size_t _max_size = 0) : max_size(_max_size), out(NULL), outb(NULL), oute(NULL) {}

    /*
     * Inputs: the compressed string, as output from 'compress()'.
     * Outputs: 
     *    true if all of the data was decompressed.
     *    false if more input data needs to be fed via 'feed()'.
     *    'remaining' will hold input data that wasn't part of
     *    the compressed message. (Only assigned to when all of
     *    the data was decompressed.)
     */

    bool feed(const std::string& s, std::string& remaining) {

        const unsigned char* i = (const unsigned char*)s.data();
        const unsigned char* e = i + s.size();

        return feed(i, e, remaining);
    }

    bool feed(const unsigned char* i, const unsigned char* e, std::string& remaining) {

        // This function is complex because it is streamable and robust.
        // The routine checks if the input isn't complete and will properly
        // pick up from where we left off when the rest of the input arrives.
        
        if (state.state == state_t::INIT) {

            ret.clear();

            size_t size;
            if (!pop_vlq_uint(i, e, size))
                return true;

            ++i;

            state = state_t();

            if (max_size && size > max_size)
                throw std::length_error("Uncompressed data in message deemed too large");

            ret.resize(size);

            outb = (unsigned char*)ret.data();
            oute = outb + size;
            out = outb;

            state.state = state_t::START;
        }

        while (i != e) {

            if (out == oute) {
                remaining.assign(i, e);
                state.state = state_t::INIT;
                return true;
            }

            if (state.state == state_t::START) {

                if (!pop_vlq_uint(i, e, state.msg))
                    return false;

                ++i;

                state.state = ((state.msg & 1) ? state_t::READ_DATA : state_t::READ_RUN);

                state.msg = state.msg >> 1;
            }

            if (state.state == state_t::READ_DATA) {

                size_t len = state.msg;
                
                if (out + len > oute)
                    throw std::runtime_error("Malformed data while uncompressing");

                if (i == e)
                    return false;

                if (i + len > e) {

                    size_t l = e - i;
                    ::memcpy(out, &(*i), l);
                    out += l;
                    state.msg -= l;

                    return false;
                }

                ::memcpy(out, &(*i), len);
                out += len;
                i += len;

                state.state = state_t::START;

            } else if (state.state == state_t::READ_RUN) {

                size_t shortrun = state.msg & (SHORTRUN_MAX - 1);

                if (shortrun) {

                    state.run = shortrun;

                } else {

                    if (!pop_vlq_uint(i, e, state.run))
                        return false;

                    ++i;
                }

                size_t off = (state.msg >> SHORTRUN_BITS);
                size_t run = state.run + MIN_RUN - 1;

                unsigned char* outi = out - off;

                if (outi >= oute || outi < outb || out + run > oute || out + run < out)
                    throw std::runtime_error("Malformed data while uncompressing");

                if (outi + run < out) {
                    ::memcpy(out, outi, run);
                    out += run;

                } else {

                    while (run > 0) {
                        *out = *outi;
                        ++out;
                        ++outi;
                        --run;
                    }
                }

                state.state = state_t::START;
            }
        }

        if (out == oute) {
            remaining.assign(i, e);
            state.state = state_t::INIT;
            return true;
        }

        return false;
    }

    /*
     * Returns the uncompressed result.
     */

    std::string& result() {
        return ret;
    }

};

}

#endif


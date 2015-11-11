yalz77
======

Yet another LZ77 implementation.

This code is in the public domain, see: http://unlicense.org/

Feel free to steal it.

----

This is a variation on the LZ77 compression algorithm.
It is designed for code simplicity and clarity.

### Highlights: ###

- Portable, self-contained, tiny implementation in readable C++. 
  (Header-only, no ifdefs or CPU dependencies or other stupid tricks.)
- Fast decompression.
- Pretty good compression quality.
- Simple 'one-button' API for realistic use-cases.
- No penalty (only 2 bytes overhead) even when compressing very short strings.
- Fully streamable decompressor: feed it chunks of arbitrarity-sized data, and the
  original uncompressed buffers will be reconstructed automatically.

Compression performance and quality should be _roughly_ on par with other compression algorithms.

(Compression ratio is comparable to other LZ algorithms when at high quality 
settings, compression speed is comparable to gzip. Decompression speed is on par
with other fast LZ algorithms.)

### Usage: ###

    #include "lz77.h"
    
    std::string input = ...;
    lz77::compress_t compress;
    std::string compressed = compress.feed(input);
    
    lz77::decompress_t decompress;
    
    std::string temp;
    decompress.feed(compressed, temp);
    
    const std::string& uncompressed = decompress.result();

_Note_: if you're compressing short strings (on the order of a few kilobytes)
then instantiating `lz77::compress_t compress(8, 4096)` will give better results.

Instantiate `lz77::compress_t compress(1)` if you want compression speed at
the expense of quality.


Use `decompress.feed(...)` for feeding input data step-by-step in chunks.
For example, if you're trying to decompress a network byte stream:

    lz77::decompress_t decompress;
    std::string extra;
    
    bool done = decompress.feed(buffer, extra);
    
    while (!done) {
      buffer = ...
      done = decompress.feed(buffer, extra);
    }
    
    std::string result = decompress.result();

`feed()` will start (or continue) decoding a packet of compressed data.
If it returns true, then all of the message was decompressed.
If it returns false, then `feed()` needs to be called with more
compressed data until it returns `true`.

`extra` will hold any data that was tacked onto the buffer but wasn't
part of this packet of compressed data. (Useful when decoding a message
stream that doesn't have clearly delineated message boundaries; the 
decompressor will detect message boundaries properly for you.)

`extra` will be assigned to only when `feed()` returns true.

`result` is the decompressed message.

_Note_: calling `feed()` and `result()` out of order is undefined
behaviour and might result in crashes.



# Choosing a quality

| Quality | What it does | Typical use |
| --- | --- | --- |
| 0 | One pass, static entropy codes | Fastest, largest output |
| 1 | Two passes, per-block entropy codes | Fast |
| 2 | Greedy matching with the format's fixed codes | Fast |
| 3 | Greedy matching, one prefix code per stream | Balanced |
| 4 | Adds block splitting and histogram optimisation | Balanced, denser |
| 5 | Adds an extensive search and literal context modelling | Densest of these |
| 6 to 9 | Wider match search, more cached distances, richer context models | Denser, slower |
| 10, 11 | Binary-tree matching and a Zopfli dynamic program | Densest, slowest |

[`EncoderConfig::default`] is quality 11, which mirrors the reference
encoder's default and is far slower than most callers want. For online
compression, say so:

```
use mbrotli::{Compressor, EncoderConfig, Quality};

let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q5))?;
let payload = "the quick brown fox ".repeat(500);

let compressed = encoder.compress(payload.as_bytes())?;

assert!(compressed.len() < payload.len() / 100);
# Ok::<(), Box<dyn std::error::Error>>(())
```

# Large Window Brotli

[RFC 9841] widens the sliding window past what RFC 7932 can express. Which
header a stream carries is part of the window itself: build one with
[`Window::standard`] or [`Window::large`], never by widening a number.

```
use mbrotli::{Compressor, EncoderConfig, Quality, Window};

let config = EncoderConfig::default()
    .with_quality(Quality::Q5)
    .with_window(Window::large(30)?);
let mut encoder = Compressor::new(config)?;

let compressed = encoder.compress("large window ".repeat(1000).as_bytes())?;

// The stream carries the RFC 9841 header, so it needs a decoder expecting one.
assert_eq!(compressed[0], 0b0001_0001);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Qualities 0, 1 and 2 write distances through a model built for the RFC 7932
alphabet, so [`Compressor::new`] refuses a Large Window there rather than
quietly dropping the request.

# Shared dictionaries

RFC 9841 also lets a caller attach up to fifteen LZ77 prefix dictionaries in
front of a stream. A [`PreparedDictionary`](dictionary::PreparedDictionary)
is immutable and holds no per-stream state, so any number of compressors may
borrow one at once without a lock.

```
use mbrotli::dictionary::DictionaryBuilder;
use mbrotli::{Compressor, EncoderConfig, Quality};

let dictionary = DictionaryBuilder::new()
    .add_prefix(&b"HTTP/1.1 200 OK\r\nContent-Type: "[..])
    .build()?;
let mut encoder = Compressor::new(EncoderConfig::default().with_quality(Quality::Q5))?;

let payload = b"Content-Type: text/html; charset=utf-8";
assert!(
    encoder.compress_with_dictionary(&dictionary, payload)?.len()
        < encoder.compress(payload)?.len()
);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Below quality five no match finder can consult a dictionary, and one handed
to such a compressor is refused with
[`EncodeError::DictionaryUnsupportedForQuality`] rather than ignored: a
stream compressed without the dictionary it was given decodes perfectly
well, which is what would make the mistake invisible.

The `experimental` feature adds serialized shared dictionaries, custom word
and transform indexes, headerless stream continuations, and the separate
Shared Brotli framing writer. Equivalent-C-streaming byte comparisons do not
cover every extension. Rust API/backend identity and decoder compatibility
remain required for equivalent stream settings.

[RFC 9841]: https://www.rfc-editor.org/rfc/rfc9841.html


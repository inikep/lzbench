## Bitpack Decoder Specification
### Inputs
The decoder for the 'bitpack' transform takes a single serial stream as input.

### Codec Header
The 'bitpack' header must be 1-2 bytes in size.

The first byte stores the number of bits per bitpacked element (`nbBits`) and the width in bytes of elements in the decoded stream (`eltWidth`). The second byte is optional, and if present stores the number of tail elements that should not be decoded.

#### Decoding
#### Byte 1
The high 2 bits store the log2 of the decoded element width (i.e. 3 for `eltWidth = 8`). The lower 6 bits store one less than the number of bits per bitpacked element (i.e. 5 for `nbBits = 6`).

Consider the following encoded input stream |11111010|11011000|11011100| with the first byte of the transform header equal to |01000010|.

The lowest 6 bits of the header encode the `nbBits`. In our example this is 000010, or 2, which means `nbBits = 3`. We can thus split the encoded stream into distinct elements of 3 bits each. The bits should be read in little-endian byte order. Thus, decoding the example stream results in {010, 111, 011, 100, 101, 001, 111, 110} or {2, 7, 3, 4, 5, 1, 7, 6}.

The highest 2 bits of the header encode the width of the original numeric stream. In our example this is 01, or 1, which means that the decoded element width must be 2 bytes.

#### Byte 2
The second byte of the transform header contains the number of extra elements in the input stream that should not be decoded. The number of elements to ignore must not be greater than the total number of elements in the encoded stream.

Consider the example from before with a second header byte present and equal to |00000010| (or 2). This means that we should not decode the last 2 elements in the encoded stream. The encoded stream can thus be thought of as |11111010|11011000|******00| where the last 2 * 3-bit elements (6 bits) are unused, but must be present so that the stream is byte-aligned.

The decoded stream will be {2, 7, 3, 4, 5, 1}.

### Outputs
Let `N` be the number of bits per encoded element, `W` be the decoded element width, `L` be the length in bytes of the encoded input stream, and `E` be the number of elements to ignore.

The output of the decoder is a single numeric stream with elements of width W and number of elements equal to $\lfloor{(8 * L)/N)}\rfloor - E$.

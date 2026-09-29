## Constant Decoder Specification
### Inputs
The decoder for the 'constant' codec takes a single stream as input. The encoded stream must be a serial or struct stream and must contain exactly one element of width >= 1.

### Codec Header
The 'constant' codec header contains the number of elements (`nbElts`) in the decoded stream. This `nbElts` is encoded as a variable-width unsigned integer in little-endian byte order. The header will contain 1-10 bytes depending on the value of `nbElts`.

The varint format is described in detail here: https://protobuf.dev/programming-guides/encoding/#varints.


### Decoding
The decoded stream is `nbElts` copies of the element in the encoded stream.

Consider the struct input stream "ab" with element width equal to 2 and header equal to |11001000|0000001|. Decoding the varint will give us the integer value 200, which means that the output stream should have 200 elements.

This means that the decoded stream will be the struct stream "abababab..." with element width of 2 and 200 elements.

### Outputs
The output of the decoder is a single stream with the same type and element width as the encoded input stream. The output stream should have `nbElts` elements.

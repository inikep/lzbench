### Divide_By Decoder Specification

The decompressor for the 'divide_by' transform takes a numeric stream as an
input, and the divisor used to divide these values on the compressor side is
passed in via the header bytes.

The divisor will be provided in the VARINT format. In particular, the divisor
must be a 64-bit unsigned integer value that is non-zero and otherwise is marked
as corrupted. The VARINT format is described in
https://developers.google.com/protocol-buffers/docs/encoding#varints with each
byte representing 7 bits with the first bit set to 1 if the next byte follows or
0 if othewise. A maximum of 10 bytes is used in the divisor, and any value
exceeding the maximum for unsigned values for the input integer width is
illegal.

The input numeric stream has a defined width of 8-bits, 16-bits, 32-bits or
64-bits, and is a stream of numeric values. The output numeric stream will
preserve the input width. The elements in the numeric stream input must not
overflow when multiplied by the divisor. For example, this means for a given
element with width of 8-bits, its product with the divisor should not exceed the
maximum value for a 8-bit unsigned integer 255. This is because only an invalid
encoding can produce such a result, since we require the input on the encoder
side to be divisible by the divisor.

The output of the decompressor is a numeric stream where each element is the
result of the product of the corresponding element of the input stream and the
divisor.

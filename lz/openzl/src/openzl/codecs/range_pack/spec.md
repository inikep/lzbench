## Range-Pack Decoder Specification
### Inputs
The decoder for the 'range_pack' transform takes a single numeric stream as input. The encoded stream must be a stream of 8, 16, 32, or 64-bit unsigned integers.

### Codec Header
The 'range_pack' codec header contains information on the element width and minimum value of the decoded output stream. The header must be of size 1, 2, 3, 5, or 9 bytes.

#### Byte 1
The first byte of the codec header is always present and encodes the element width of the decoded stream. This value must be equal to a valid integer width of 1, 2, 4, or 8.

Consider the encoded stream {1, 5, 4, 0} with element width 1 and first byte of the codec header |00000100|. In this case, the decoder must produce an output stream with elements of width 4 (i.e. 32-bit integers).

#### Bytes 2-9
The final 0-8 bytes of the codec header encode the minimum unsigned value of the decoded data in little-endian byte order. This `minVal` should be added to each element in the encoded stream in order to recover the original data.

If the total header length is 1 then the `minVal` is 0. Otherwise `minVal` is some non-zero value and the size of the header must be equal to 1 + the width of the decoded data (i.e. the first byte of the codec header).

Consider the encoded stream {1, 5, 4, 0} from before. We already determined that the decoded element width is 4. This means that the codec header must contain either 1 or 5 bytes total.

Let's say the header is 1 byte total. This means that `minVal` is 0 and that decoding would result in the stream {1, 5, 4, 0}, where all values are 32-bit integers.

Now let's say that the header is 5 bytes total, with the last 4 bytes being equal to |00000100|00000001|00000000|00000000|. This means that `minVal` is 260 and thus the decoded stream is {261, 265, 264, 260}, where all values are 32-bit integers.

### Outputs
The output of the decoder is a single numeric stream with the same number of elements as the encoded input stream. The width of elements in the output stream should be equal to the value encoded in the first byte of the stream header and should be a valid integer width of 1, 2, 4, or 8.

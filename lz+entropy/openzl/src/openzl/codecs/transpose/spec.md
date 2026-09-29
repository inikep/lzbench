## Transpose Decoder Specification
### Inputs
The decoder for the 'transpose' codec takes a single struct stream as input. We can describe the dimensions of a struct stream as N x W, where N is the number of elements in the stream and W is the width of each element in bytes.

### Decoding
#### Non-Empty Input
Consider the encoded input stream {"1234", "abcd"} where the number of elements is 2 and the element width is 4 (dimensions 2 x 4).

The decoder can produce the nth element of the decoded output by concatenating the nth byte of each element in the encoded input. For example, to produce the second element of the decoded stream we should concatenate the second byte of "1234", '2', with the second byte of "abcd", 'b', to produce "2b".

This means that the decoded stream from our example would be the struct stream {"1a", "2b", "3c", "4d"}  with dimensions 4 x 2.

#### Empty Input
Consider the encoded input stream {} where `nbElts` is 0 and `eltWidth` is 5. The decoded stream should produce an empty output stream with the same element width.

This means that the decoded stream would be the struct stream {}, with 0 elements and an `eltWidth` of 5.

### Outputs
The output of the decoder is a single struct stream with the same number of bytes as the encoded stream. Let the dimensions of the encoded stream be N x W. The decoded output will have dimensions W x N if N is non-zero and 0 x W otherwise.

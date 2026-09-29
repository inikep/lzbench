## Zig-Zag Decoder Specification
### Inputs
The decoder for the 'zigzag' codec takes a single numeric stream of zig-zag encoded integers as input. The encoded stream must be a stream of 8, 16, 32, or 64-bit unsigned integers.

Zig-Zag encoding is an alternative method for encoding negative numbers. It maps all positive integers p to 2 * p. All negative integers n are mapped to  2 * |n| - 1. This mechanism groups signed integer values of similar magnitudes together as unsigned integers.

The zig-zag format is described in detail here: https://protobuf.dev/programming-guides/encoding/#signed-ints.

### Decoding
Consider the stream of zig-zag encoded integers {1, 3, 4}. The original (signed) integer stream can be recovered with e / 2 for all even integers and (o + 1) / -2 for all odd integers in the encoded stream. This means that decoding the stream {1, 3, 4} would result in {-1, -2, 2}.

### Outputs
The output of the decoder is a single numeric stream where the element width and number of elements are the same as in the input stream.

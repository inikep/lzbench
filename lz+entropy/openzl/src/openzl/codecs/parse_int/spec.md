## Parse-Int Decoder Specification
The decoder for the 'parse_int' transform takes a single numeric stream as input. The numeric stream must be a width-8 64-bit signed integer stream.

The output of the transform is a string stream. Each string in the regenerated string stream corresponds to the decimal representation of each element of the encoded input stream. The string representation of each element has no leading zeros, no double '-' signs, or '+' signs and has a single '-' sign when negative.

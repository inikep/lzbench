## Prefix Decoder Specification

### Inputs
The decoder for the 'prefix' codec takes two streams of the same number of elements as input.
- a string stream containing the suffixes from the original input.
- a numeric stream containing the prefix match sizes.

*Note: The first prefix match size is always 0 and the first element of the suffixes stream is always equal to the first element of the decoded stream.*

### Decoding
Consider the encoded input where the suffix stream is {"apple", "ly", "t", "ly} and the match lengths are {0, 3, 2, 3}. The decoder can produce each element of the decoded stream by prepending `matchLength` bytes from the previously decoded element to the current element of the suffix stream.

```
i | matchsize[i] | suffixes[i] | decoded[i]
------------------------------------------
0 | 0 -> ""      | "apple"     | "apple"
1 | 3 -> "app"   | "ly"        | "apply"
2 | 2 -> "ap"    | "t"         | "apt"
3 | 3 -> "apt"   | "ly"        | "aptly"
```

This means that the decoded stream from our example would be the string stream {"apple", "apply", "apt", "aptly"}.

### Outputs
The output of the decoder is a single string stream. The number of elements in the output stream should be the same as the two input streams.

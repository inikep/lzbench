## Parse Int Manual
### Overview
The parse int codec takes a single string input which it expects contain integer values in string format. These integers can be negative, but must have no-leading zeroes aside from the value "0" and cannot have double negative signs. The value also cannot overflow an 64-bit integer. Specifically, the string cannot have a numeric value more than "9223372036854775807" or if negative, cannot be less than "-9223372036854775808". The integer values are converted into their 2-complement numeric representation in the output. If the string is in an unexpected format as specified above, the codec throws an error.
### Inputs
A single string typed input.

### Outputs
A single numeric typed output of width 8.

### Use Cases
When compressing numeric data received in string format (such as compressing human readable file formats like json or csv), applying the parse int codec on the numeric segments of the string can be an effective step to building a good compressor. Numeric data typically has more structure and can be better compressed than string data.

Often used on string data where dispatchString is called, and integer segments are separated from a main string then sent to parse int to transform the strings to numeric data, followed by sending the numeric data to an lz compressor such as zstd to exploit redundancy characteristic.

## Entropy Manual
### Overview
There are two types of entropy codecs - Finite State Entropy (FSE) and Huffman. Both codecs use redundancy to compress data, building probability tables for symbols as an initial step.

Huffman is theoretically the most efficient binary code for encoding symbols separately, however this encoding model has an assumption that all symbols must be represented in an integer number of bits. In a scenario where one character occuring 90% of the time, Huffman encoding must assign 1 bit to this symbol, where using 0.15 bits is theortically ideal, reducing its compression efficiency. FSE breaks this "1 bit per symbol" limit, to offer improved compression ratio for cases where integer bit lengths per symbol are inefficient, at the cost of slightly worse compression speed.
### Inputs
A single serial input

### Outputs
A single serial output

### Use Cases
In practice, entropy codecs are expected to be used at the final stage of compression after appropriate transformation has been done to capture the known structure of the data. These codecs are a core ingredient of compression, eliminating redundancy in the data. Typically, these codecs will not be used directly unless building a new LZ compressor, however they are used in Field LZ and Zstd compression. When used as a entropy coder for the LZ compressor, the associated graphs for the codecs can be useful as a final stage after match finding.

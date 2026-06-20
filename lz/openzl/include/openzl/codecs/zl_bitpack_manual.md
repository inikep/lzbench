## Bitpack Manual
### Overview
The bitpack codec takes a single numeric input. The numeric input is interpreted as unsigned integers of the given element width (either 1, 2, 4, or 8). The input is transformed so that each element is encoded using `nbBits`, which is the number of bits needed to encode the largest element in the input.

### Inputs
A single numeric input.

### Outputs
A single serial output.

### Use Cases
The bitpack codec is especially useful when handling numeric data which has small values that are represented with a larger numeric width than needed. Bitpack may be useful in lieu of `fieldLZ` or `huffman` as the final codec in a pipeline if the input is non-repetitive or uniformly distributed.

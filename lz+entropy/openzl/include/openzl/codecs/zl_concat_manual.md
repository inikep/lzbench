## Concat Manual
### Overview
There is one codec for each type - serial, numeric, struct or string in the family of 'concat' codecs. All inputs must be of the same type and element width for these 'concat' codecs.

The 'concat' codec takes multiple inputs and concatenates them in the order given, returning the result as a single output.

### Inputs
The 'concat' codec takes $N$ inputs, all of the same type and element width. On format version 17 of OpenZL, the maximum number of inputs is 2048, due to engine limitations.

### Outputs
There are two outputs produced:
- a numeric output, which contains the size of each segment in the input
- an input of type $T$ (either serial, numeric, or struct) and element width $n$, which contains all segments concatenated in order

### Use Cases
Used for combining data with similar formats so they can be compressed in a unified manner. Ideally, entropy statistics of the individual inputs should be similar for compression to be effective.

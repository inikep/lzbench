## Interleave Manual

### Overview
The interleave codec is a data transformation that takes multiple inputs of the same type (and the same length) and interleaves them round-robin-style into a single output.
!!! Note
    Only string input is supported for now.

### Inputs
`interleave` takes a variable number of inputs. As of format version 20, the maximum allowed number of such inputs is 2048. All inputs must be of the same type and have the same length. For numeric/struct inputs, this means having the same number of elements. For string inputs, this means having the same number of strings.

### Output
One output is produced, which is the round-robin interleaving of the inputs. This output has the same type as the inputs and length equal to the sum of the lengths of the inputs.

Example:
Suppose we wish to interleave the following three inputs:
```
A: [1, 2, 3, 4, 5]
B: [6, 7, 8, 9, 10]
C: [11, 12, 13, 14, 15]
```
The output of the interleave codec would be:
```
[1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15]
```

### Parameters
None.

### Use Cases
This codec is useful for scenarios where multiple related data inputs need to be processed together while maintaining their individual structure. In particular, it is useful for zipping multiple correlated columns of a columnar dataset. It is expected that if there is strong correlation between columns (e.g. a "city" column and a "country" column), then successor graphs will be able to take advantage of this correlation to achieve better compression.

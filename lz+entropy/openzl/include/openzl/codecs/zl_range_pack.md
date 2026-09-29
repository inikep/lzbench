## Range-Pack Manual
### Overview
The range-pack codec takes a single stream of integers and subtracts the minimum value from each element. The resulting values are then packed into a numeric stream with the minimum width in bytes required to represent the new range of values.

### Inputs
A single numeric input.

### Outputs
A single numeric output.

### Use Cases
This codec is useful for compressing numeric data that has a range of values that can be represented with a smaller number of bytes.

## Delta Manual
### Overview
The delta codec takes a single numeric input. The numeric input is interpreted as unsigned integers of the given element width (either 1, 2, 4, or 8).

### Inputs
A single numeric input.

### Outputs
A single numeric output.

### Use Cases
The delta codec is useful when handling numeric data which has some consistent relationship between its consecutive elements. For example, the delta codec might be useful for the `timestamp` column of a time series which has monotonically increasing values.

There are many reasonable successors to the delta codec, depending on the relationship between the input elements. `constant` is an ideal successor if the input is an arithmetic series, i.e. has a constant delta. `fieldlz` might be a good successor if there is a lot of repetition of deltas or if there is a consistent pattern. `entropy` could also be a good next state if the deltas are small, but otherwise do not have a lot of repetition. Similarly `transpose + entropy` could be useful for well-clustered larger numbers, since it separates highly predictable high bytes from less predictable ones.

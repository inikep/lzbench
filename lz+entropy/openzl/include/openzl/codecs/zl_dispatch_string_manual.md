## DispatchString Manual
### Overview
DispatchString takes a single string input and a single numeric input. It dispatches the string input to its outputs according to the numeric input, where the $k_{th}$ element of the string input is sent to the output index specified by the $k_{th}$ value in the numeric input. The numeric input is passed to the numeric output.
### Inputs
- a single string input
- a single numeric input
### Outputs
- a single numeric output
- $n$ string outputs
### Use Cases
Suppose there is data given in a structured string format such as csv. If each row represents a sample, grouping the columns with similar entropy statistics into the same input, then passing it to entropy codecs can be beneficial for compression ratio. Using DispatchString, it is possible to output data in this structure.

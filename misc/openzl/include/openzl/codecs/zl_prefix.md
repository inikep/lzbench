## Prefix Manual
### Overview
The prefix codec takes a single input of type string and finds the longest common prefix between each consecutive element.

### Inputs
A single input stream of type string.

### Outputs
The prefix codec outputs two streams:
- A string stream with the unmatched portion of the input between consecutive string elements.
- A numeric stream with the size of the shared prefix between consecutive string elements.

### Use Cases
The prefix codec is useful to compress a sorted list of strings.

## Constant Manual
### Overview
The constant codec encodes a single numeric input with constant elements. It succeeds only if all elements in the input have the same value, compressing it into a single element and a varint representing the number of repetitions written to a header.
### Inputs
A single numeric typed input.

### Outputs
A single numeric typed output with a single element.

### Use Cases
Used for deduplicating constant valued inputs and compress such inputs effectively.

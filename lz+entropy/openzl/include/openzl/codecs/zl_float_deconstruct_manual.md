## Float-Deconstruct Manual
### Overview
The float-deconstruct codec takes a single numeric input. The numeric input is interpreted as floating point numbers of a given width, depending on the codec variant (float16, bfloat16, float32).


### Inputs
A single numeric input of width 2 or 4, depending on the chosen codec variant.

### Outputs
A serial stream containing the exponents and a struct stream containing the sign bits and fractions.

### Use Cases
The float-deconstruct codec is useful for handling floating point data. This is especially relevant where the exponent is expected to have some known distribution while the mantissa is expected to be more random.

There are many reasonable successors to this codec. If the exponents have a lot of repeated values or some consistent pattern, then `fieldLZ` might be a good choice. In most cases, we don't expect the sign bits and fractions to have much structure so `entropy` or `store` might be a good choice.

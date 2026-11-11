## Float-Deconstruct Decoder Specification

### Inputs
The decoder for the 'float_deconstruct' codec takes two streams as input. The streams must have same number of elements, `nbElts`.
- a serial stream containing the exponents
- a struct stream containing the sign bits and fractions  ('signfrac')

The element width (`eltWidth`) of the signfrac stream depends on the element type. The supported element types are float32, bfloat16, and float16. The element type must be encoded in the codec header for frame versions >= 5 and otherwise must be assumed.

### Codec Header
#### Frame Version < 5
There is no codec header. The element type must be float32.

#### Frame Version >=5
The frame header must be present and must be 1 byte in size. This byte contains the encoded element type and must be interpreted as follows.
- 0 -> float32
- 1 -> bfloat16
- 2 -> float16

### Decoding
#### BFloat16
The encoded signfrac stream for bfloat16 will have `eltWidth = 1`.

The bfloat16 format consists of a sign bit, an 8 bit exponent, and a 7 bit fraction.
```
|sign|-------- exponent (8) --------|------- fraction (7) -------|
```
We can construct the `nth` element of the decoded output from the `nth` elements of the encoded signfrac and exponent streams. The decoder should interpret the encoded elements as described below.

- **signfrac[n]**
    - Bit 0 contains the sign bit. (`|*******s|`)
    - Bits 1-7 contain the fraction. (`|fffffff*|`)
- **exponent[n]**
    - Bits 0-7 bits contain the exponent. (`|eeeeeeee|`)


#### Float16
The encoded signfrac stream for float16 will have `eltWidth = 2`. Each element should be interpreted in little-endian byte order.

The float16 format consists of sign bit, a 5 bit exponent, and a 10 bit fraction.
```
|sign|----- exponent (5) -----|---------- fraction (10) ----------|
```

We can construct the `nth` element of the decoded output by reconstructing the float16 format from the `nth` elements of the encoded signfrac and exponent streams.

- **signfrac[n]**
    - Bit 0 contains the sign bit. (`|*******s|********|`)
    - Bits 1-10 contain the fraction. (`|fffffff*|*****fff|`)
    - Bits 11-15 are unused and must be 0. (`|********|00000***|`)
- **exponent[n]**
    - Bits 0-4 contain the exponent. (`|***eeeee|`)
    - Bits 5-7 are unused and must be 0. (`|000*****|`)

#### Float32
The encoded signfrac stream for float32 will have `eltWidth = 3`. Each element should be interpreted in little-endian byte order.

The float32 format consists of sign bit, an 8 bit exponent, and a 23 bit fraction.
```
|sign|--- exponent (8) ---|--------------- fraction (23) ---------------|
```

We can construct the `nth` element of the decoded output by reconstructing the float32 format from the `nth` elements of the encoded signfrac and exponent streams.

- **signfrac[n]**
    - Bit 0 contains the sign bit. (`|*******s|********|********|`)
    - Bits 1-23 contain the fraction. (`|fffffff*|ffffffff|ffffffff|`)
- **exponent[n]**
    - Bits 0-7 bits contain the exponent. (`|eeeeeeee|`)

### Output
The output of the decoder is a single numeric stream. The stream must have the same number of elements as the encoded input (`nbElts`).

The width of elements in the decoded output stream will depend on the encoded float type.
- bfloat16 -> `eltWidth = 2`
- float16 -> `eltWidth = 2`
- float32 -> `eltWidth = 4`

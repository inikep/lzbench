### Dedup_Num Decoder Specification

The decompressor for the 'dedup_num' transform takes 1 numeric stream as input.
There is no restriction of width nor size for this input numeric stream.
This numeric stream will be duplicated N times as outputs of the decoder.
The number N is determined by asking the Graph Engine how many streams must be regenerated, since it's an information that the Graph Engine must know in order to control the reconstruction.

Implementation notes:

- The regenerated streams are effectively references into the original stream,
  so there is no allocation nor `memcpy()` workload.

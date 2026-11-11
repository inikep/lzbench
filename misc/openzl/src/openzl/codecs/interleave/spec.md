# Interleave Decoder Specification

### Inputs
The decoder expects one typed input containing the interleaved data. Numeric, struct, and string data shall be supported. Serial data shall not be supported.

### Codec Header
A 4-byte little-endian unsigned integer containing the number of outputs $n$

### Decoding
The input data are conceptually a combination of $n$ `input`s, each of which contains the same number of $\textit{elements}$. For numeric and struct data, an $\textit{element}$ refers simply to one number or one struct. For string data, an $\textit{element}$ refers to one individual string in the input.

If the input contains $N$ elements, then $n$ must divide $N$. It is an error for $N$ not to be divisible by $n$.

The decoder shall produce $n$ outputs. Each output must contain the same number of elements as every other. The outputs shall be produced by repeatedly dispatching elements in a round-robin fashion from the input. Concretely, using zero-indexed counting, the $i$ -th output will contain the $nk + i$ -th elements from the input, for $k < \frac{N}{n}$. Further, the $j$ -th string in the $i$ -th output will be the $jn + i$ -th string in the input.

### Outputs
$n$ typed outputs, each of the same type as the input. All outputs shall have the same number of elements.

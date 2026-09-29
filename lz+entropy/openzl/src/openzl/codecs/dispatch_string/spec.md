## Decoder Specification for DispatchString
### Inputs
The decompressor shall take as input one numerical input $q$ and a variable number $n$ of inputs $A_i$, $0 \leq i < n$.

### Decoding
#### Frame Version < 21
$q$ is an unsigned **8-bit** numerical input with length equal to the sum of the cardinalities of $a_i$ and represents the ordering in which strings from the inputs are interleaved to create the output. It is an error for any $q_j \in q$ to violate $0 \leq q_j < n$. It is an error for $q_i$ to have width **greater than 8 bits**. It is the responsibility of the decoder to reject malformed input.
#### Frame Version >= 21
$q$ is an unsigned **16-bit** little-endian input with length equal to the sum of the cardinalities of $a_i$ and represents the ordering in which strings from the inputs are interleaved to create the output. It is an error for any $q_j \in q$ to violate $0 \leq q_j < n$. It is an error for $q_i$ to have width **not equal to 16 bits**. It is the responsibility of the decoder to reject malformed input.

In any frame version, semantically, $X$ is constructed by successively taking strings from the inputs (without repeats) according to $q$. Formally, label the $n$ inputs $A_0, A_1, \dots A_{n-1}$. Then the string $X[j]$ is drawn from input $A_{q_j}$. The cardinality of $X$, $|X|$ is equal to the cardinality $|q|$ of $q$. Similarly, $|X| = \sum |A_i|$. Further, $X$ must "use" all the values of $A_i$. That is,
$$\sum_q \mathbb{1}_{\{q_j = i\}} = |A_i|, \; \forall i < n$$
It is the responsibility of the decoder to check these bounds and reject malformed input.

### Outputs
For $n$ of inputs $A_i$, $0 \leq i < n$ the output is the combination of these $A_i$ as one output string output $X$.

### Example
Suppose $n = 4$, and the string inputs are
$$A_0 = \{a, b, c\}\\ A_1 = \{A, B, C, D, E\}\\ A_2 = \{1, 2, 3, 4, 5, 6, 7\}\\ A_3 = \{\alpha, \beta, \gamma, \delta\}$$
If we take
$$q = \{0, 1, 2, 3, 0, 1, 1, 2, 2, 2, 3, 0, 1, 2, 3, 3, 1, 2, 2 \}$$
Then
$$X = \{a, A, 1, \alpha, b, B, C, 2, 3, 4, \beta, c, D, 5, \gamma, \delta, E, 6, 7\}$$

If we take
$$q = \{3, 3, 3, 3, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 1, 2, 1, 2, 2 \}$$
Then
$$X = \{\alpha, \beta, \gamma, \delta, 1, a, A, 2, b, B, 3, c, C, 4, D, 5, E, 6, 7\}$$

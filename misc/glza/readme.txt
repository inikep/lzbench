/***********************************************************************

Copyright 2014-2025 Kennon Conrad

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

***********************************************************************/

GLZA is an experimental grammar compression toolset.  Compression is accomplished by calling GLZAformat, GLZAcompress
  and GLZAencode, in that order.  Decompression is accomplished by calling GLZAdecode.  In general for text-like files
  GLZA achieves Pareto Frontier decompression speeds vs. compression ratio with low decompression RAM use but is slow to
  compress compared to Lempel-Ziv algorithms.
GLZAformat is a preprocessor that detects the data type, performs capital encoding on textual files and
  performs 1, 2 or 4 byte delta filtering on certain strided files.
GLZAcompress performs a grammar transformation by recursively finding sets of production rules until no new
  profitable production rules can be found.  This step may be skipped, in which case GLZA is effectively an
  order-1 compressor.
GLZAencode encodes the grammar using a variation of the Grammatical Ziv-Lempel format.
GLZAdecode decodes the data.

Usage:
   GLZA c|d [-c#] [-d0] [-l0] [-m#] [-p#] [-r#] [-t#] [-v#] [-w0] [-x] [-C#] [-D#] <infilename> <outfilename>
     where
       -c# sets the production cost (default is to estimate it)
       -d0 disables delta transformation
       -l0 disables capital letter lock transformation
       -m0 disables "selective" MTF dictionary coding
       -m1 forces "selective" MTF dictionary encoding
       -o# sets the dedupication candidate score order model.  0.0 is order 0 based, 1.0 is order 1 trailing
           character/leading character based.  Intermediate values are a blend.
       -p# sets the score profit ratio weighting power, where the profit ratio is the profit per substitution divided
           by the order 0 entropy of the string.  The score (estimated profit from creating the rule) is multiplied
           by the profit ratio to the (p) power.  Default is 2.0 for capital encoded or UTF8 compliant files, 0.0 for
           delta encoded files, and 1.0 otherwise.  0.0 is approximately "most compressive bitwise", ie. maximizing
           the order 0 profit of new productions, but somewhat larger values are generally more effective for text
           files.
       -r# sets the approximate RAM usage of GLZAcompress in MB.  Default is 40 MB + 100 x the
           preprocessed file size.
       -t# sets the number of threads for decoding (1 or 2)
       -v1 prints the dictionary to stdout, most frequent first
       -v2 prints the dicitonary to stdout, simple symbols followed by complex symbols in the order they were created
       -w0 disables the first cycle word only search
       -x  enables extreme compression mode
       -C0 disables capital letter transformation
       -C1 forces text processing and capital encoding
       -D# sets an upper limit for the number of grammar rules created

For more details on GLZA and the GLZ format, see http://encode.ru/threads/2427-GLZA.

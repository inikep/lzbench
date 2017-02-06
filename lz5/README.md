LZ5 - Library Files
================================

The __lib__ directory contains several directories.
Depending on target use case, it's enough to include only files from relevant directories.


#### API

LZ5 stable API is exposed within [lz5_compress.h](lz5_compress.h) and [lz5_decompress.h](lz5_decompress.h),
at the root of `lib` directory.


#### Compatibility issues

The raw LZ5 block compression format is detailed within [lz5_Block_format].
To compress an arbitrarily long file or data stream, multiple blocks are required.
Organizing these blocks and providing a common header format to handle their content
is the purpose of the Frame format, defined in [lz5_Frame_format].
`lz5` command line utility produces files or streams compatible with the Frame format.
(_Advanced stuff_ : It's possible to hide xxhash symbols into a local namespace.
This is what `liblz5` does, to avoid symbol duplication
in case a user program would link to several libraries containing xxhash symbols.)

[lz5_Block_format]: ../doc/lz5_Block_format.md
[lz5_Frame_format]: ../doc/lz5_Frame_format.md


#### Various LZ5 builds

Files `lz5_common.h`, `lz5_compress*`, `lz5_parser_*.h`, `lz5_decompress*`, and `entropy\mem.h` are required in all circumstances.

To compile:
- LZ5_raw only with levels 10...29 : use the `-DLZ5_NO_HUFFMAN` compiler flag
- LZ5_raw with levels 10...49 : include also all files from `entropy` directory
- LZ5_frame with levels 10...49 : `lz5frame*` and all files from `entropy` and `xxhash` directories


#### Advanced API 

A more complex `lz5frame_static.h` is also provided.
It contains definitions which are not guaranteed to remain stable within future versions.
It must be used with static linking ***only***.


#### Using MinGW+MSYS to create DLL

DLL can be created using MinGW+MSYS with the `make liblz5` command.
This command creates `dll\liblz5.dll` and the import library `dll\liblz5.lib`.
The import library is only required with Visual C++.
The header files `lz5.h`, `lz5hc.h`, `lz5frame.h` and the dynamic library
`dll\liblz5.dll` are required to compile a project using gcc/MinGW.
The dynamic library has to be added to linking options.
It means that if a project that uses LZ5 consists of a single `test-dll.c`
file it should be compiled with `liblz5.lib`. For example:
```
    gcc $(CFLAGS) -Iinclude/ test-dll.c -o test-dll dll\liblz5.dll
```
The compiled executable will require LZ5 DLL which is available at `dll\liblz5.dll`. 


#### Miscellaneous 

Other files present in the directory are not source code. There are :

 - LICENSE : contains the BSD license text
 - Makefile : script to compile or install lz5 library (static or dynamic)
 - liblz5.pc.in : for pkg-config (make install)
 - README.md : this file


#### License 

All source material within __lib__ directory are BSD 2-Clause licensed.
See [LICENSE](LICENSE) for details.
The license is also repeated at the top of each source file.

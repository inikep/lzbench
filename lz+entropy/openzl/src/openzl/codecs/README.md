The directory `src/openzl/codecs/` hosts OpenZL's Standard Codecs.
Standard Codec are bound by a set of rules that set them apart from Custom ones.
These rules are described in this document

## Directory organization

Each codec is hosted in its own subdirectory, which carries the name of the codec.
Each codec has 2 sides : an encoder, and a decoder.
Additionally, it may feature some shared definitions, such as the Graph Descriptor.

A Standard Codec may optionally depend on resources outside its own directory,
such as `src/openzl/codecs/common/`, which typically hosts definitions for some common graphs.
They may also optionally use private methods from `src/openzl/compress/` or `src/openzl/decompress`.
Finally, they do, of course, use public methods,
typically from `include/openzl/zl_ctransform.h` and `include/openzl/zl_dtransform.h`.

## Coding conventions

`src/openzl/codecs/` is part of the `openzl` projects, and therefore follows all project conventions.
All codes shipped in `src/openzl/codecs/` must be clean `C11`, following best practices,
compiling without a single warning using the extended list of warning flags of the `openzl` project.

## Roles of the Kernel and Binding

Most Standard Codecs are divided into 2 parts :

### The Kernel

This is a minimalist implementation of the codec, sometimes specialized for a specific target.

  + The Kernel concentrates on the processing stage, trying to create the simplest and purest processing loop.
    This is where the bulk of the cpu processing time should be spent.
    Anything outside of this responsability shall be delegated to the Binding.
  + The Kernel publishes a list of requirements that it needs to focus on its one job.
    This can be size requirements, additional buffers, or even limitations on input data (as long as they can be checked by the Binding).
    In best case scenarios, if all conditions are respected, the kernel's processing stage may only be successful.
    If a failure is possible, the Kernel has to provide a clear success/error signal.
  + The Kernel publishes its own interface, optimized for its inner working.
    There is no specific rule for the Kernel interface, outside of being lean and straighforward.
    It's up to the Binding to adapt to it.
  + The Kernel is _not allowed_ to allocate any buffer.
    It can only use provided ones, hence it publishes such requirement as part of its API contract.
    Allocation is a Zstrong-engine level operation.
    It must be delegated to the Binding.
  + The Kernel features a minimalistic implementation *and* dependency graph.
    It should only depend on `<assert.h>`, `<limits.h>`, `<stddef.h>` and `<stdint.h>`.
    `memcpy()` can also be necessary, in which case dependency on `<string.h>` is allowed
    (this last point may change in the future).
    Note the intentional absence of other dependencies, such as `<stdlib.h>`, `<stdio.h>` or any direct `openzl/` dependency.
  + The Kernel should be "transportable" into another context with minimal modification required (and preferably none).

### The Binding

This is the "glue" between the kernel and the OpenZL engine.

  + The Binding's interface is defined by the OpenZL public transform API.
  + It is fully integrated into the `openzl/` project, depends on its resources, and is only expected to compile as part of it.
  + It has to ensure that all conditions required by the Kernel are met.
    The Binding is where pre-condition checks are made, and errors are generated.
  + The Binding is in charge of all allocations required by the kernel.
    In particular, it provides both temporary workspaces, and input and output buffers.
  + When a kernel may fail, the Binding controls operation success,
    and outputs a corresponding return status using its `ZL_Report` return structure.

### Special cases

In some cases, there might not be enough work to justify the existence of a separate `_kernel` file.
In which case, the entire implementation may be hosted in the `_binding` file.
This is not supposed to be a common situation though.
Avoid putting complex processing stages into the `_binding` file, it's not its role.

In other cases, it's possible to define multiple `_kernel` files for a single codec operation.
They may be, for example, different strategies, depending on specific conditions,
such as input size, element size, or even target hardware.
It's then up to the Binding to select the appropriate one.

Finally, a Kernel is not restricted to providing a single function call.
It may provide multiple entry points, for example one for checking, one for sizing, and one for the transformation proper.
The Binding is in charge of orchestrating these operations in the correct order.

## Naming convention

### Directory:

  The directory typically carries the name of the codec

  In some rare cases, an entire category might be hosted within the directory.
  For example, all conversion codecs are hosted in the same directory.
  In which case, the directory carries the name of the category.

### Files:

Each codec has 2 sides: an encoder, and a decoder.
Generally, both have a binding side and a kernel side.
This determines the prefix of the file names:
`encode_xxxxx_binding.h` and `encode_xxxxx_kernel.h`.
Same thing for the decoding side.
Duplicate for the `*.c` implementation.

This means that many codecs feature at least 8 files.

As mentioned, the kernel side is a bit more variable,
as it may be absent, or they may be multiple sub-variants of the kernel.
Adjust the naming accordingly, typically by embedding a variant qualifier in the codec part of the name.
Example : `encode_tokenize2to1_kernel.c`

Some files may fall outside of this categorization,
for example the shared codec's Graph.
Typical name for such file would be `graph_xxxxx.h`.

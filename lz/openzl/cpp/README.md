# C++ Bindings for OpenZL

## Key Ideas

* You must always be able to drop down to C
* You must be able to progressively migrate from C to C++
* It must depend only on the standard library
* It must be C++11
  * the `poly/` subdirectory contains polyfills for useful C++ standard containers.
* It may only depend on public OpenZL headers

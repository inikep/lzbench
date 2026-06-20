# ZStrong tests

## GTest style conventions
**Use the GTest convention for naming unit and integration tests**

Details can be found [here](todo). In particular, the contraindication on underscores in suite and test names shall be strictly followed.

**Use the `zstrong::tests` namespace for unit and integration tests**

All `TEST`, `TEST_F`, and similar classes should be declared within the `zstrong::tests` namespace or, rarely, a sub-namespace if appropriate. Helper functions and classes should be put into an inner anonymous namespace to avoid name collisions. The exception to this rule is utility objects shared between files, which should also be prefixed with the `zstrong::tests` namespace.

**Directory organization (WIP)**

Unit tests shall be placed in the `tests/unittest` directory.

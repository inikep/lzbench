This page documents advanced build options. Most typical applications need not worry about these options.

# Introspection
OpenZL allows the option of attaching code hooks at specified places in the engine execution flow. This is termed "introspection" and is a powerful feature that enables other important tools. However, you may decide it undesirable to allow arbitrary code execution in, for example, a production environment. Hence, it is a togglable feature.
!!! note
    Compression traces and training both rely on introspection. Disabling introspection will cause both of these features to break.

Introspection is conditionally compiled based on the `OPENZL_ALLOW_INTROSPECTION` CMake option. By default, this option is `ON`, but you may disable it with
```
cmake -DOPENZL_ALLOW_INTROSPECTION=OFF ..
```

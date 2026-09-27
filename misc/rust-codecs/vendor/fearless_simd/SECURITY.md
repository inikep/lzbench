# Security Policy

## Supported Versions

Starting with v1.0, we will provide security backports for the latest version of Fearless SIMD for each MSRV (minimum supported Rust version) for *at least* 3 years since its MSRV was released.

This ensures that security updates are just a `cargo update` away, no matter what Rust compiler version you use (as long as it's not older than 3 years).

As an example: if (hypothetically) both 1.0.x and 1.1.x have an MSRV of 1.89, while 1.2.x has MSRV of 1.100, both 1.1.x and 1.2.x will receive security backports, while 1.0.x will not because a fixed version with the same MSRV is already available and will be fetched by `cargo update`. If v2.0.0 is released, versions in v1.x.x series will continue receiving security backports as per the above policy.

Earlier versions may or may not receive security backports at the discretion of Fearless SIMD maintainers.

You can look up when each Rust version was released in the [Rust Release Notes](https://doc.rust-lang.org/stable/releases.html).

## Reporting a Vulnerability

This repository follows a full disclosure policy. Please report security issues publicly on the Github issue tracker.

Entirely AI-generated reports are not permitted. While the use of LLMs for auditing code is welcome,
the issue must be reproduced and understood by a human before reporting.

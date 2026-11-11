Bzip2
=====

This is Bzip2/libbz2; a program and library for lossless, block-sorting data
compression.

This document pertains to the Bzip2 feature development effort hosted on
[GitLab.com](https://gitlab.com/bzip2/bzip2).

The documentation here may differ from that on the Bzip2 1.0.x project page
maintained by Mark Wielaard on[sourceware.org](https://sourceware.org/bzip2/).

Copyright (C) 1996-2010 Julian Seward <jseward@acm.org>

Copyright (C) 2019-2020 Federico Mena Quintero <federico@gnome.org>

Copyright (C) 2021 [Micah Snyder](https://gitlab.com/micahsnyder).

Please read the [WARNING](#warning), [DISCLAIMER](#disclaimer) and
[PATENTS](#patents) sections in this file for important information.

This program is released under the terms of the license contained in the
[COPYING](COPYING) file.

------------------------------------------------------------------

This version is fully compatible with the previous public releases.

Complete documentation is available in Postscript form (manual.ps),
PDF (manual.pdf) or HTML (manual.html).  A plain-text version of the
manual page is available as bzip2.txt.

## Community Code of Conduct

There is a code of conduct for contributors to Bzip2/libbz2.
Please see the [`code-of-conduct.md`](code-of-conduct.md) file.

## Contributing to Bzip2's development

The Bzip2 project is hosted on GitLab for feature development work.
It can be found at https://gitlab.com/bzip2/bzip2

Changes to be included in the next feature version are committed to the
`master` branch.

Feature releases are maintained in `release/*` branches.

Long-term feature and experimental development will occur in feature branches.
*Feature branches are unstable.* Feature branches may be rebased and force-
pushed on occasion to keep them up-to-date and to resolve merge conflicts.

The `rustify` branch is a feature branch that represents an effort to
gradually port Bzip2 to [Rust](https://www.rust-lang.org).

## Report a Bug

Please report bugs via [GitLab Issues](https://gitlab.com/bzip2/bzip2/issues).

Before you create a new issue, please verify that no one else has already
reported the same issue.

## Compiling Bzip2 and libbz2

Please see the [`COMPILING.md`](COMPILING.md) file for details.
This includes instructions for building using Meson or CMake.

## WARNING

This program and library (attempts to) compress data by performing several
non-trivial transformations on it. Unless you are 100% familiar with *all* the
algorithms contained herein, and with the consequences of modifying them, you
should NOT meddle with the compression or decompression machinery.
Incorrect changes can and very likely *will* lead to disastrous loss of data.

**Please contact the maintainers if you want to modify the algorithms.**

## DISCLAIMER

**I TAKE NO RESPONSIBILITY FOR ANY LOSS OF DATA ARISING FROM THE USE OF THIS
PROGRAM/LIBRARY, HOWSOEVER CAUSED.**

Every compression of a file implies an assumption that the compressed file can
be decompressed to reproduce the original. Great efforts in design, coding and
testing have been made to ensure that this program works correctly.

However, the complexity of the algorithms, and, in particular, the presence of
various special cases in the code which occur with very low but non-zero
probability make it impossible to rule out the possibility of bugs remaining in
the program.

DO NOT COMPRESS ANY DATA WITH THIS PROGRAM UNLESS YOU ARE PREPARED TO ACCEPT
THE POSSIBILITY, HOWEVER SMALL, THAT THE DATA WILL NOT BE RECOVERABLE.

That is not to say this program is inherently unreliable.
Indeed, I very much hope the opposite is true.
Bzip2/libbz2 has been carefully constructed and extensively tested.

## PATENTS

To the best of my knowledge, Bzip2/libbz2 does not use any patented algorithms.
However, I do not have the resources to carry out a patent search.
Therefore I cannot give any guarantee of the above statement.

## Maintainers

As of June 2021, [Micah Snyder](https://gitlab.com/micahsnyder) is the
maintainer of Bzip2/libbz2 for feature development work (I.e. versions 1.1+).

The Bzip2 feature development project is hosted on GitLab and can be found at
https://gitlab.com/bzip2/bzip2

Bzip2 version 1.0 is maintained by [Mark Wielaard](https://www.klomp.org/mark/)
at Sourceware and can be found at https://sourceware.org/git/?p=bzip2.git

### Special thanks

Thanks to Julian Seward, the original author of Bzip2/libbz2, for creating the
program and making it a very compelling alternative to previous compression
programs back in the early 2000's. Thanks to Julian also for letting Federico,
Mark, and Micah carry on with the maintainership of the program.

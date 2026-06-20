# unitBench

This program can benchmark _scenarios_,
that are described in `benchList.h`.
It returns a result in unit of time (typically `ms`),
and can extrapolate a speed in MB/s when it makes sense, depending on the scenario.

Each scenario has a name, that can be invoked on the command line,
and is generally followed by a file name,
which contains a dataset on which the scenario will be played.

```
./unitBench scenarioName fileName
```

#### For a list of available commands:
`./unitBench -h`

#### To get the list of already defined scenarios:
`./unitBench --list`

Each scenario can be uniquely defined by its implementer.
Some may only work on one specific file.
For example, the `sao` scenario is only meant to be run on the `sao` file of the Silesia compression corpus.

Others are more generic.
For example, `zs2_decompress` can decompress any data encoded using the Zstrong format.

Documentation on each scenario is terse, and is typically embedded with the scenario's code,
typically referenced from, if not implemented in, `benchList.h`.

## How to add your own scenario

The logic is encapsulated within `benchList.h`.

Look for an array, near the end of the file.
It contains the definitions for all scenarios.
Each scenario is defined as a structure of fields, many of which are optional.

Look at the few paragraphs in front of the array.
They explain the logic to add a new scenario.

You don't have to create your scenario within `benchList.h`.
It's actually recommended that you implement your scenario in your own `*.h` header file, for clarity,
and then just reference it from within `benchList.h`.
You may also prefer to do the implementation into a separate `*.c` unit,
but note that, in this case, the `unitBench` build script will have to be updated.

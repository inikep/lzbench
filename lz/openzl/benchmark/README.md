# ZStrong Benchmarks

## Objectives
Benchmarking in Zstrong is a major part of the regression testing strategy.
They should allow us to:

1. Measure impact of code changes during development and before merge.
The important point here is that these are measurements that should help us make decisions and take actions. That means that they need to be concise and digestible.

2. Record library performance over time to make sure trends are on course.
For example, at some points we might allow minor regressions for functionality or code improvements. But we want to be able to track changes over time to make sure those small regressions don’t pile up together into a big regression.

3. Provide a safeguard against drastic regressions (disallow merge if regression of certain magnitude is introduced).


## Testcases
We generally consider two types of test cases:
1. E2E runs - we want to be able to test Zstrong as a compression engine. That means that we want to be able to collect measurements of E2E runs for complete runs on relevant data in different settings (for example, a list of compression and decompression levels).
There are two types of E2E runs:

    * Real graphs - we want to detect regressions in our known use cases.
    For example, we might want to have a graph similar to the one used for AI checkpointing, with a corpus similar to the AI checkpointing corpus to make sure we maintain performance for this graph.
    Note that we are not going to base this on the actual graphs and actual compressors, as we want to be able to test regressions in release while those graphs compile against release.

    * Small graphs - we want to detect regressions in specific flows and transforms,  these might be harder to pinpoint when they are a part of a larger graph.
    For example, we may want to measure FieldLZ by itself without introducing other dependencies.

2. Microbenchmarking - we need to be able to measure performance sensitive code and to iterate on it quickly. E2E runs won’t be good enough for these cases as they are more noisy. Instead, we need to be able to support running a specific function on pre prepared inputs. This is very similar to benchlist and should replace it.

*Only small E2E graphs are implemented at the moent*

## Code structure
We base our benchmarking suite on Google Benchmark.

Each testcase should implement the `BenchmarkTestcase` base interface and allow itself to register to the Google Benchmark framework. Pay attention to objects lifetimes when creating new types of testcases.

We support common types of generated data using classes defined in `benchmark_data.h`. Any generic type of data should be added there.

E2E benchmarks should be implemented in the `e2e` directory.

When microbenchmarks are implemented, they should be added to a new directory.

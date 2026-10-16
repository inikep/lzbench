// criterion needs a modern toolchain anyway; the 1.51 MSRV does not apply.
#![allow(clippy::incompatible_msrv)]
use branches::{likely, unlikely};
use core::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use std::time::Duration;

fn count_zeros_likely(arr: &[usize]) -> isize {
    let mut count = 0;
    arr.iter().for_each(|&num| {
        if likely(num == 0) {
            count += 1;
        } else {
            count -= 1;
        }
    });
    count
}

fn count_zeros_unlikely(arr: &[usize]) -> isize {
    let mut count = 0;
    arr.iter().for_each(|&num| {
        if unlikely(num == 0) {
            count += 1;
        } else {
            count -= 1;
        }
    });
    count
}

fn count_zeros(arr: &[usize]) -> isize {
    let mut count = 0;
    arr.iter().for_each(|&num| {
        if num == 0 {
            count += 1;
        } else {
            count -= 1;
        }
    });
    count
}

fn bench_zeroes(c: &mut Criterion) {
    // ----- setup -----
    let all_zeros: Vec<usize> = vec![0; 100_000_000];
    let all_ones: Vec<usize> = vec![1; 100_000_000];
    let alternating: Vec<usize> = (0..100_000_000).map(|i| i % 2).collect();

    c.bench_function("count_zeros_likely_all", |b| {
        b.iter(|| {
            let result = count_zeros_likely(black_box(&all_zeros));
            black_box(result);
        })
    });

    c.bench_function("count_zeros_unlikely_all", |b| {
        b.iter(|| {
            let result = count_zeros_unlikely(black_box(&all_zeros));
            black_box(result);
        })
    });

    c.bench_function("count_zeros_default_all", |b| {
        b.iter(|| {
            let result = count_zeros(black_box(&all_zeros));
            black_box(result);
        })
    });

    c.bench_function("count_zeros_likely_none", |b| {
        b.iter(|| {
            let result = count_zeros_likely(black_box(&all_ones));
            black_box(result);
        })
    });

    c.bench_function("count_zeros_unlikely_none", |b| {
        b.iter(|| {
            let result = count_zeros_unlikely(black_box(&all_ones));
            black_box(result);
        })
    });

    c.bench_function("count_zeros_default_none", |b| {
        b.iter(|| {
            let result = count_zeros(black_box(&all_ones));
            black_box(result);
        })
    });

    c.bench_function("count_zeros_likely_alternating", |b| {
        b.iter(|| {
            let result = count_zeros_likely(black_box(&alternating));
            black_box(result);
        })
    });

    c.bench_function("count_zeros_unlikely_alternating", |b| {
        b.iter(|| {
            let result = count_zeros_unlikely(black_box(&alternating));
            black_box(result);
        })
    });

    c.bench_function("count_zeros_default_alternating", |b| {
        b.iter(|| {
            let result = count_zeros(black_box(&alternating));
            black_box(result);
        })
    });
}

fn criterion() -> Criterion {
    Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_millis(250))
}

criterion_group! {
    name = benches;
    config = criterion();
    targets = bench_zeroes
}
criterion_main!(benches);

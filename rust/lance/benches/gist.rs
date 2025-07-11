// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use lance_index::scalar::gist::{GiSTLookup, GiSTOperations, GiSTPredicate, GiSTQuery};
use lance_index::scalar::spatial::{BoundingBox, SpatialGiSTOps, SpatialQuery};
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

fn bench_gist(c: &mut Criterion) {
    let ops = SpatialGiSTOps;
    let num_entries = 10000;
    let predicates: Vec<(u32, Box<dyn GiSTPredicate>)> = (0..num_entries)
        .map(|i| {
            let x = (i % 100) as f64 * 10.0;
            let y = (i / 100) as f64 * 10.0;
            (i, Box::new(BoundingBox::new(x, y, x + 5.0, y + 5.0)) as Box<dyn GiSTPredicate>)
        })
        .collect();

    c.bench_function("gist_build_tree", |b| {
        b.iter(|| GiSTLookup::build_tree(predicates.clone(), &ops).unwrap());
    });

    let lookup = GiSTLookup::build_tree(predicates, &ops).unwrap();
    let query = Box::new(SpatialQuery::Intersects(BoundingBox::new(250.0, 250.0, 350.0, 350.0))) as Box<dyn GiSTQuery>;

    c.bench_function("gist_search_pages", |b| {
        b.iter(|| lookup.search_pages(&*query, &ops));
    });
}

#[cfg(target_os = "linux")]
criterion_group!(
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(10)).sample_size(10).with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_gist
);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(10)).sample_size(10);
    targets = bench_gist
);

criterion_main!(benches);
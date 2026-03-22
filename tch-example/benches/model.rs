use common::BenchmarkConfig;
use common::{bench_predict_many, bench_predict_single, bench_train_batch};
use criterion::{Criterion, criterion_group, criterion_main};
use tch_example::TchRunner;

fn run_benchmarks(c: &mut Criterion) {
    let runner = TchRunner::new(BenchmarkConfig::from_benchmark_args());

    bench_predict_single(c, &runner, "tch");
    bench_predict_many(c, &runner, "tch");
    bench_train_batch(c, &runner, "tch");
}

criterion_group!(benches, run_benchmarks);
criterion_main!(benches);

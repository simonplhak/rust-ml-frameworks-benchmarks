use candle_example::CandleRunner;
use common::{bench_predict_many, bench_predict_single, bench_train_batch, BenchmarkConfig};
use criterion::{criterion_group, criterion_main, Criterion};

fn run_benchmarks(c: &mut Criterion) {
    let runner = CandleRunner::new(BenchmarkConfig::from_benchmark_args());
    bench_predict_single(c, &runner, "candle");
    bench_predict_many(c, &runner, "candle");
    bench_train_batch(c, &runner, "candle");
}

criterion_group!(benches, run_benchmarks);
criterion_main!(benches);

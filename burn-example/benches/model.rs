use burn_example::model_runnner;
use common::{BenchmarkConfig, bench_predict_many, bench_predict_single, bench_train_batch};
use criterion::{Criterion, criterion_group, criterion_main};

fn run_benchmarks(c: &mut Criterion) {
    let config = BenchmarkConfig::from_benchmark_args();
    let runner = model_runnner(config);
    bench_predict_single(c, &runner, "burn");
    bench_predict_many(c, &runner, "burn");
    bench_train_batch(c, &runner, "burn");
}

criterion_group!(benches, run_benchmarks);
criterion_main!(benches);

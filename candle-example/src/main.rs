use candle_example::CandleRunner;
use common::{benchmark_train, BenchmarkConfig};

fn main() {
    let config = BenchmarkConfig::from_args();
    let model_runner = CandleRunner::new(config.clone());
    benchmark_train(model_runner, &config);
}

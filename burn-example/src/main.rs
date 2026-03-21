use burn_example::model_runnner;
use common::{BenchmarkConfig, benchmark_train};

fn main() {
    let config = BenchmarkConfig::from_args();
    let model_runner = model_runnner(config.clone());
    benchmark_train(model_runner, &config);
}

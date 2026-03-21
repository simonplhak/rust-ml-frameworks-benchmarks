use common::{BenchmarkConfig, benchmark_train};
use tch_example::TchRunner;

fn main() {
    let config = BenchmarkConfig::from_args();
    let model_runner = TchRunner::new(config.clone());
    benchmark_train(model_runner, &config);
}

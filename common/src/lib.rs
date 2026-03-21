use burn::data::dataset::{Dataset as _, vision::MnistDataset};
use clap::Parser;
use criterion::{BenchmarkId, Criterion, Throughput};
use serde::Serialize;
use std::{
    hint::black_box,
    path::PathBuf,
    time::{Duration, Instant},
};

pub const INPUT_DIM: usize = 784;
pub const HIDDEN_SIZE_1: usize = 256;
pub const OUTPUT_SIZE: usize = 10;
pub const SEED: u64 = 42;

#[derive(Debug, Clone, Parser)]
#[command(name = "Benchmark CLI")]
#[command(about = "ML Framework Benchmarking Tool")]
pub struct BenchmarkConfig {
    #[arg(long, default_value_t = 256)]
    pub batch_size: usize,

    #[arg(long, default_value_t = 0.01)]
    pub learning_rate: f64,

    #[arg(long, default_value_t = 8)]
    pub num_workers: usize,

    #[arg(long, default_value_t = 30)]
    pub recall_k: usize,

    #[arg(long, default_value_t = 5)]
    pub max_epochs: usize,

    #[arg(long)]
    pub output_csv: Option<PathBuf>,
}

impl BenchmarkConfig {
    pub fn from_args() -> Self {
        Self::parse()
    }

    pub fn from_benchmark_args() -> Self {
        let args = std::env::args()
            .filter(|arg| arg != "--bench" && arg != "bench")
            .collect::<Vec<_>>();
        Self::parse_from(args)
    }
}

pub struct Dataset {
    pub train_xs: Vec<f32>,
    pub train_ys: Vec<usize>,
    pub test_xs: Vec<f32>,
    pub test_ys: Vec<usize>,
}

impl Dataset {
    pub fn load() -> Self {
        println!("  Loading training split...");
        let train_data = MnistDataset::train();
        let test_data = MnistDataset::test();

        let (train_xs, train_ys) = Self::transform(train_data);
        println!(
            "  [✓] Training split loaded: {} samples",
            train_xs.len() / INPUT_DIM
        );

        println!("  Loading test split...");
        let (test_xs, test_ys) = Self::transform(test_data);
        println!(
            "  [✓] Test split loaded: {} samples",
            test_xs.len() / INPUT_DIM
        );

        Self {
            train_xs,
            train_ys,
            test_xs,
            test_ys,
        }
    }

    fn transform(dataset: MnistDataset) -> (Vec<f32>, Vec<usize>) {
        let len = dataset.len();
        // Pre-allocate to avoid re-allocations during benchmark setup
        let mut xs = Vec::with_capacity(len * 784);
        let mut ys = Vec::with_capacity(len);

        for i in 0..len {
            if i % 10000 == 0 && i > 0 {
                println!("    Processing item {}/{}", i, len);
            }
            if let Some(item) = dataset.get(i) {
                // Flatten 2D -> 1D and Normalize 0-255 -> 0.0-1.0
                let flattened_normalized = item
                    .image
                    .iter()
                    .flat_map(|row| row.iter())
                    .map(|&pixel| pixel / 255.0); // Normalization step

                xs.extend(flattened_normalized);
                ys.push(item.label as usize);
            }
        }
        (xs, ys)
    }
}

pub trait RunableModel {
    type TrainModel: Clone;
    type Model;
    type Dataset;
    type Batch: Clone;
    type Optimizer;
    fn model(&self) -> Self::Model;
    fn train_model(&self) -> Self::TrainModel;
    fn optimizer(&self) -> Self::Optimizer;
    fn dataset(&self, xs: &[f32], ys: &[usize]) -> Self::Dataset;
    fn batch(&self, xs: &[f32], ys: &[usize]) -> Self::Batch;
    fn train_batch(
        &self,
        train_model: Self::TrainModel,
        optimizer: &mut Self::Optimizer,
        batch: Self::Batch,
    ) -> Self::TrainModel;
    fn train(&self, dataset: &Self::Dataset, epochs: usize) -> Self::Model;
    fn predict_single(&self, model: &Self::Model, x: &[f32]) -> usize;
    fn predict_many(&self, model: &Self::Model, x: &[f32]) -> Vec<usize>;
}

#[derive(Debug, Serialize)]
pub struct ConvergenceResult {
    pub epochs: usize,
    pub accuracy: f64,
    pub time_training_ms: f64,
}

pub fn accuracy(predictions: &[usize], labels: &[usize]) -> f32 {
    let correct = predictions
        .iter()
        .zip(labels.iter())
        .filter(|&(p, l)| p == l)
        .count();

    (correct as f32) / (labels.len() as f32)
}

pub fn benchmark_train(model_runner: impl RunableModel, config: &BenchmarkConfig) {
    println!("=== Performance Test Configuration ===");
    println!("Input dimension: {}", INPUT_DIM);
    println!("Output dimension: {}", OUTPUT_SIZE);
    println!("Batch size: {}", config.batch_size);
    println!("Learning rate: {}", config.learning_rate);
    println!("=====================================");

    println!("\n[1/5] Loading dataset...");
    let mnist_dataset = Dataset::load();
    let num_samples = mnist_dataset.train_xs.len() / INPUT_DIM;
    let num_queries = mnist_dataset.test_xs.len() / INPUT_DIM;
    let dataset_size_mb = (mnist_dataset.train_xs.len() * 4) as f64 / (1024.0 * 1024.0);
    let queries_size_mb = (mnist_dataset.test_xs.len() * 4) as f64 / (1024.0 * 1024.0);
    println!("[\u{2713}] Dataset loaded:");
    println!("    - Training samples: {}", num_samples);
    println!("    - Training size: {:.2} MB", dataset_size_mb);
    println!("    - Query samples: {}", num_queries);
    println!("    - Query size: {:.2} MB", queries_size_mb);

    let mut results = vec![];

    println!("\n[2/5] Preparing model dataset...");
    // For incremental training, we need to track a model and retrain it
    // Since the trait doesn't support incremental updates, we'll retrain from scratch
    // each time but with the model_runner keeping internal state (framework-specific)
    let train_dataset = model_runner.dataset(&mnist_dataset.train_xs, &mnist_dataset.train_ys);
    println!("[✓] Model dataset prepared");

    println!("\n[3/5] Starting training loop...");
    for epochs in 1..=config.max_epochs {
        println!("Training with {} epoch(s)...", epochs);

        // Train model from scratch for 'epoch' epochs
        let epoch_start = Instant::now();
        let model = model_runner.train(&train_dataset, epochs);
        let epoch_time_ms = epoch_start.elapsed().as_secs_f64() * 1000.0;

        // Perform inference and compute accuracy
        let predictions = model_runner.predict_many(&model, &mnist_dataset.test_xs);

        let accuracy_score = accuracy(&predictions, &mnist_dataset.test_ys) as f64;
        let correct_count = predictions
            .iter()
            .zip(mnist_dataset.test_ys.iter())
            .filter(|&(p, l)| p == l)
            .count();

        println!(
            "  Epoch {}: Accuracy = {:.2}% ({}/{} correct), Time = {:.2} ms",
            epochs,
            accuracy_score * 100.0,
            correct_count,
            num_queries,
            epoch_time_ms
        );
        results.push(ConvergenceResult {
            epochs,
            accuracy: accuracy_score,
            time_training_ms: epoch_time_ms,
        });
    }

    println!("==================================");

    println!("\n[4/5] Saving results...");
    // Save results to CSV if output path is provided
    if let Some(output_path) = &config.output_csv {
        match save_convergence_results(&results, output_path) {
            Ok(_) => {
                println!("[✓] Results saved to: {}", output_path.display());
                println!("    Wrote {} convergence records", results.len());
            }
            Err(e) => eprintln!("Failed to save results: {}", e),
        }
    }

    // Print summary
    println!("\n[5/5] Generating summary...");
    let total_training_time: f64 = results.iter().map(|r| r.time_training_ms).sum();
    let final_accuracy = results.last().map(|r| r.accuracy).unwrap_or(0.0);
    let best_result = results.iter().max_by(|a, b| {
        a.accuracy
            .partial_cmp(&b.accuracy)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!("\n===== INFERENCE SUMMARY =====");
    println!("Total training time:         {:.2} ms", total_training_time);
    println!(
        "Final Accuracy:              {:.2}%",
        final_accuracy * 100.0
    );
    if let Some(best) = best_result {
        println!(
            "Best Epoch:                  {} (Accuracy: {:.2}%)",
            best.epochs,
            best.accuracy * 100.0
        );
    }
    println!("=============================");
}

pub fn save_convergence_results(
    results: &[ConvergenceResult],
    path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut wtr = csv::Writer::from_path(path)?;
    for result in results {
        wtr.serialize(result)?;
    }
    wtr.flush()?;
    Ok(())
}

pub fn bench_predict_single<T: RunableModel>(c: &mut Criterion, runner: &T, name: &str) {
    let dataset = Dataset::load();
    let query = dataset.test_xs[0..INPUT_DIM].to_vec();
    let model = runner.model();
    let mut group = c.benchmark_group("Predict_Single");
    group.sample_size(100);

    group.bench_function(BenchmarkId::new(name, "latency"), |b| {
        b.iter(|| {
            let result = runner.predict_single(&model, black_box(&query));
            black_box(result);
        })
    });

    group.finish();
}

pub fn bench_predict_many<T: RunableModel>(c: &mut Criterion, runner: &T, name: &str) {
    let dataset = Dataset::load();
    let model = runner.model();
    let mut group = c.benchmark_group("Predict_Many");
    group.sample_size(100);

    group.bench_function(BenchmarkId::new(name, "latency"), |b| {
        b.iter(|| {
            let result = runner.predict_many(&model, black_box(&dataset.test_xs));
            black_box(result);
        })
    });

    group.finish();
}

pub fn bench_train_batch<T: RunableModel>(c: &mut Criterion, runner: &T, name: &str) {
    let mut group = c.benchmark_group("Train_Batch_Step");

    // For training, we use fewer samples because each step is heavy
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(3));

    // Experimenting with different batch sizes is a core part of your thesis
    let batch_sizes = [32, 64, 128];

    for size in batch_sizes {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new(name, size), &size, |b, &s| {
            // SETUP: Create model, optimizer, and a dummy batch for this size
            // (Using dummy data here to isolate framework compute speed from disk I/O)
            let dummy_xs = vec![0.5f32; s * 784];
            let dummy_ys = vec![1usize; s];

            let mut train_model = runner.train_model();
            let mut optimizer = runner.optimizer(); // You may need to add this to your trait
            let batch = runner.batch(&dummy_xs, &dummy_ys);

            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    // The actual training step: Forward -> Backward -> Update
                    train_model = runner.train_batch(
                        black_box(train_model.clone()),
                        &mut optimizer,
                        black_box(batch.clone()),
                    );
                }
                start.elapsed()
            });
        });
    }
    group.finish();
}

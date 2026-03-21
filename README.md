# Rust ML Frameworks Benchmark Suite

A comprehensive Python-based benchmarking and visualization suite for comparing Rust ML frameworks.

## Overview

This suite runs benchmarks for three Rust machine learning frameworks:

- **Burn** - Pure Rust ML framework with multiple backend support
- **TCH-rs** - Rust bindings for PyTorch
- **Candle** - Hugging Face's Rust ML framework

All three frameworks are benchmarked on the same task: **MNIST digit classification** using a 2-layer neural network (784→256→10).

### Benchmarking Focus: CPU Performance

⚠️ **Important:** This benchmark suite focuses exclusively on **CPU performance**. All three frameworks are configured to run on CPU:

- **Burn** - Uses ndarray backend (pure Rust, CPU only)
- **TCH-rs** - Configured for CPU inference/training
- **Candle** - Runs on CPU (GPU support available but not configured)

GPU acceleration is **not enabled** for this benchmark. Results measure CPU-bound compute performance and are not indicative of GPU performance. If you need GPU benchmarks, framework-specific GPU backends can be configured separately.

## Experiments & Benchmarks

This suite runs **4 distinct experiments** to comprehensively evaluate framework performance:

### Experiment 1: Training Convergence (Binary Execution)

**What it measures:** How quickly each framework trains the MNIST classifier over 5 epochs.

**Configuration:**

- Framework: Each framework's `main.rs` compiled with `cargo run --release`
- Task: Train a 2-layer neural network (784→256→10) on MNIST
- Epochs: 1 to 5 (incremental - each run trains from scratch for N epochs)
- Batch size: 256
- Learning rate: 0.01
- Optimizer: Adam (framework-dependent implementation)
- Num workers: 8

**Metrics collected (per epoch):**

- **Epoch:** Iteration number (1-20)
- **Training Time (ms):** Cumulative milliseconds to complete N epochs
- **Accuracy:** Classification accuracy on 10,000 test samples

**Output:** CSV file with convergence data for all 5 epochs

**Why this matters:** Shows which framework trains most efficiently and achieves best accuracy fastest.

#### Convergence Results Visualization

```
[PLACEHOLDER: Convergence curves graph]
Type: Line plot with markers
X-axis: Epoch (1-20), Y-axis: Accuracy (0.0-1.0)
Series: Three lines (Burn, TCH, Candle)
Interpretation: Steepness = faster convergence
```

---

### Experiment 2: Single-Sample Inference Latency

**What it measures:** How long it takes to classify a single MNIST image (cold start).

**Criterion Configuration:**
- Benchmark name: `predict_single`
- Sample size: 100 measurements
- Warm-up: 3 seconds, Measurement time: 5 seconds
- Input: One flattened 784-dimensional MNIST image

**Metrics collected:**
- **Mean latency (ms):** Average inference time with 95% confidence interval
- **Median latency (ms):** 50th percentile with 95% confidence interval
- **Std Dev (ms):** Standard deviation in inference times
- **Median Absolute Deviation (MAD, ms):** Robust spread measure
- **Slope (ms):** Linear regression coefficient

**Why this matters:** Measures framework overhead and startup cost for small inputs. Critical for real-time inference applications.

#### Single-Sample Inference Latency Visualization

```
[PLACEHOLDER: Single-sample inference latency]
Type: Grouped bar chart with error bars
X-axis: Framework (Burn, TCH, Candle)
Y-axis: Latency (ms)
Interpretation: Lower bar = faster inference
```

---

### Experiment 3: Batch Inference Throughput

**What it measures:** How long it takes to classify 10,000 MNIST images in a batch.

**Criterion Configuration:**
- Benchmark name: `predict_many`
- Sample size: 100 measurements
- Warm-up: 3 seconds, Measurement time: 5 seconds
- Input: All 10,000 test MNIST images (~31.4 MB)

**Metrics collected:**
- **Mean latency (ms):** Average time with 95% confidence interval
- **Median latency (ms):** 50th percentile with 95% confidence interval
- **Std Dev (ms):** Standard deviation
- **Median Absolute Deviation (MAD, ms):** Robust spread measure
- **Throughput:** Samples per second

**Why this matters:** Shows framework efficiency with large batches. Important for inference pipelines. Typically faster per-sample than single inference due to vectorization.

#### Batch Inference Throughput Visualization

```
[PLACEHOLDER: Batch inference throughput]
Type: Grouped bar chart
X-axis: Experiment (Single, Batch)
Y-axis: Latency (ms)
Interpretation: Batch should have lower per-sample latency
```

---

### Experiment 4: Training Step Performance (Variable Batch Sizes)

**What it measures:** Forward pass + backward pass + optimizer step takes at different batch sizes.

**Criterion Configuration:**
- Benchmark name: `train_batch`
- Sample size: 20 measurements (training is expensive)
- Warm-up: 3 seconds, Measurement time: 5 seconds
- Batch sizes tested: [32, 64, 128]
- Input: Random dummy data (isolates compute speed from I/O)

**Metrics collected (per batch size):**
- **Mean latency (ms):** Average training step time with 95% CI
- **Median latency (ms):** 50th percentile with 95% CI
- **Std Dev (ms):** Standard deviation
- **Median Absolute Deviation (MAD, ms):** Robust spread measure
- **Throughput (samples/sec):** Derived from inverse of latency

**Batch sizes tested:**
- **32:** Small batch (memory efficient)
- **64:** Medium batch (balanced)
- **128:** Large batch (maximizes parallelism)

**Why this matters:** Different applications use different batch sizes. Shows which framework scales best with batch size. Important for production deployment decisions.

#### Training Throughput vs. Batch Size Visualization

```
[PLACEHOLDER: Training throughput across batch sizes]
Type: Grouped bar chart
X-axis: Batch Size (32, 64, 128)
Y-axis: Throughput (samples/sec)
Interpretation: Higher bars = faster, slope = scaling efficiency
```

---

## Metrics Summary

**All Criterion metrics extracted per benchmark:**

- **Mean** - Average value with 95% confidence interval (± ~2-5%)
- **Median** - 50th percentile with 95% confidence interval (more robust than mean)
- **Standard Deviation** - Measure of variance in measurements (higher = less stable)
- **Median Absolute Deviation (MAD)** - Robust alternative to std dev (less sensitive to outliers)
- **Slope** - Linear regression coefficient (if applicable)
- **Unit conversion:** Criterion outputs nanoseconds (ns), all converted to **milliseconds (ms)** for consistency

## Setup

### Prerequisites

- Python 3.12+
- Rust toolchain
- UV package manager

### Installation

1. **Install dependencies:**

```bash
# UV will install all Python dependencies from pyproject.toml
uv sync
```

## Usage

### Running Benchmarks

```bash
# Run benchmarks for all three frameworks
# Results stored in results/{YYYY-MM-DD_HH-MM-SS}/
uv run python run_benchmarks.py
```

The script will:

1. Run `cargo run --release` for each framework → generates convergence CSV
2. Run `cargo bench` for each framework → generates Criterion JSON results
3. Parse all outputs and standardize units (ns → ms)
4. Save timestamped results in `results/` directory
5. **Abort immediately if ANY framework fails** (no partial results, no retries)

**Output Structure:**

```
results/
└── 2026-03-21_14-30-45/
    ├── metadata.json          # Timestamp, git commit, Rust version
    ├── burn_results.json      # Convergence + Criterion metrics
    ├── tch_results.json
    └── candle_results.json
```

### Generating Visualizations

```bash
# Generates plots from the latest benchmark run
uv run python visualize_results.py
```

**Output Visualizations** (saved to `visualizations/{run_timestamp}/`):

1. **01_convergence_curves.{html,png}** - Training accuracy over epochs
2. **02_inference_latency.{html,png}** - Predict single/batch latency comparison
3. **03_training_throughput.{html,png}** - Training throughput comparison
4. **04_latency_distribution.{html,png}** - Latency MAD comparison
5. **05_confidence_intervals.{html,png}** - Mean latency with CI bounds

**Data Exports** (CSV format):

- `convergence_data.csv` - Epoch-by-epoch training data
- `benchmark_metrics.csv` - All Criterion metrics with confidence intervals
- `summary.csv` - Overall statistics per framework
- `speedup_ratios.csv` - Relative performance vs. baseline (Burn)

All plots are:

- **Interactive HTML** - Zoom, pan, hover for exact values
- **High-resolution PNG** - 300+ DPI suitable for thesis inclusion
- **Publication-quality** - Plotly graph_objects with full formatting control

## Results Storage

### Metadata (`metadata.json`)

```json
{
  "benchmark_suite_run": "2026-03-21_14-30-45",
  "system_info": {
    "timestamp": "2026-03-21T14:30:45.123456",
    "git_commit": "abc123def456...",
    "rust_version": "rustc 1.XX.X"
  },
  "frameworks": ["burn-example", "tch-example", "candle-example"]
}
```

### Framework Results (`{framework}_results.json`)

```json
{
  "framework": "burn-example",
  "convergence": {
    "convergence": [
      {"epoch": 1, "training_time_ms": 234.5, "accuracy": 0.7234},
      ...
    ],
    "total_epochs": 20,
    "final_accuracy": 0.9834,
    "total_training_time_ms": 4567.8
  },
  "benchmarks": {
    "predict_single": {
      "original_unit": "ns",
      "metrics": {
        "mean": {
          "estimate_ms": 0.345,
          "ci_lower_ms": 0.340,
          "ci_upper_ms": 0.350,
          "standard_error_ms": 0.002
        },
        ...
      }
    },
    ...
  }
}
```

## Troubleshooting

### Dependencies Not Installed

```bash
uv sync
```

### Criterion Benchmark Timeout

If benchmarks timeout (>10 min), increase timeout in `run_benchmarks.py`:

```python
timeout=1200,  # Change from 600 to 1200 seconds
```

### PNG Export Failing

Requires `kaleido` system dependency:

```bash
# macOS
brew install kaleido

# Linux
apt-get install kaleido

# Or via uv
uv add kaleido
```

## Performance Notes

- **Total runtime:** ~10-30 minutes depending on machine and configuration
  - Cargo build/check: ~2-5 min per framework
  - Binary execution: ~3-8 min per framework
  - Criterion benchmarks: ~2-5 min per framework
  
- **Criterion defaults:**
  - 100 samples per benchmark
  - 3 second warm-up
  - 5 second measurement time
  - 95% confidence level

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for the full text.

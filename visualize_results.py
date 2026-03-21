#!/usr/bin/env python3
"""
Visualization script for ML framework benchmark results.
Generates publication-quality plots using Plotly's graph_objects.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


RESULT_DIR = Path("results")
OUTPUT_DIR = Path("visualizations")
FRAMEWORKS = ["burn", "tch", "candle"]
FRAMEWORK_COLORS = {
    "burn": "#E91E63",
    "tch": "#2196F3",
    "candle": "#FFC107",
}


def load_results(result_dir: Path) -> Dict[str, Any]:
    """Load all framework results from a benchmark run."""
    results = {}

    for framework in FRAMEWORKS:
        result_file = result_dir / f"{framework}_results.json"
        if not result_file.exists():
            raise FileNotFoundError(f"Missing results file: {result_file}")

        try:
            with open(result_file, "r") as f:
                results[framework] = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load {framework} results: {e}") from e

    return results


def load_metadata(result_dir: Path) -> Dict[str, Any]:
    """Load metadata from a benchmark run."""
    metadata_file = result_dir / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_file}")

    try:
        with open(metadata_file, "r") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load metadata: {e}") from e


def get_latest_result_dir() -> Path:
    """Get the latest benchmark result directory."""
    if not RESULT_DIR.exists():
        raise FileNotFoundError(f"Results directory not found: {RESULT_DIR}")

    result_dirs = sorted(RESULT_DIR.glob("*"))
    if not result_dirs:
        raise FileNotFoundError(f"No benchmark results found in {RESULT_DIR}")

    return result_dirs[-1]


def plot_convergence_curves(results: Dict[str, Any]) -> go.Figure:
    """Create convergence curves (epoch vs accuracy) for all frameworks."""
    fig = go.Figure()

    for framework, data in results.items():
        convergence = data["convergence"]["convergence"]
        epochs = [d["epoch"] for d in convergence]
        accuracies = [d["accuracy"] for d in convergence]

        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=accuracies,
                mode="lines+markers",
                name=framework.upper(),
                line=dict(color=FRAMEWORK_COLORS[framework], width=2.5),
                marker=dict(size=6),
            )
        )

    fig.update_layout(
        title="Training Convergence: Accuracy vs. Epoch",
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        hovermode="x unified",
        template="plotly_white",
        font=dict(size=12),
        height=600,
        width=900,
    )

    return fig


def plot_inference_latency(results: Dict[str, Any]) -> go.Figure:
    """Create inference latency comparison (predict_single and predict_many)."""
    benchmarks = ["predict_single", "predict_many"]
    frameworks_list = list(results.keys())

    # Extract latency data
    data_to_plot = {bench: {} for bench in benchmarks}

    for framework, data in results.items():
        for bench in benchmarks:
            if bench in data["benchmarks"]:
                metrics = data["benchmarks"][bench]["metrics"]
                if "mean" in metrics:
                    data_to_plot[bench][framework] = metrics["mean"]["estimate_ms"]

    # Create grouped bar chart
    fig = go.Figure()

    for i, benchmark in enumerate(benchmarks):
        x_pos = np.arange(len(frameworks_list)) + (i * 0.35)
        latencies = [data_to_plot[benchmark].get(f, 0) for f in frameworks_list]

        fig.add_trace(
            go.Bar(
                x=frameworks_list,
                y=latencies,
                name=benchmark.replace("_", " ").title(),
                marker=dict(
                    color=(
                        FRAMEWORK_COLORS[frameworks_list[0]]
                        if i == 0
                        else FRAMEWORK_COLORS[frameworks_list[1]]
                    )
                ),
                opacity=0.7 + (i * 0.15),
            )
        )

    fig.update_layout(
        title="Inference Latency Comparison",
        xaxis_title="Framework",
        yaxis_title="Latency (ms)",
        barmode="group",
        hovermode="x unified",
        template="plotly_white",
        font=dict(size=12),
        height=600,
        width=900,
    )

    return fig


def plot_training_throughput(results: Dict[str, Any]) -> go.Figure:
    """Create training throughput comparison across batch sizes."""
    fig = go.Figure()

    # Note: train_batch benchmark data contains metrics per batch size
    # This is a placeholder - actual implementation depends on how
    # train_batch data is structured in Criterion output

    batch_sizes = [32, 64, 128]
    frameworks_list = list(results.keys())

    for framework, data in results.items():
        if "train_batch" in data["benchmarks"]:
            metrics = data["benchmarks"]["train_batch"]["metrics"]
            if "mean" in metrics:
                latency_ms = metrics["mean"]["estimate_ms"]
                # Calculate throughput as samples/second
                # Assuming each batch is one training step
                throughput = 1000.0 / latency_ms if latency_ms > 0 else 0

                fig.add_trace(
                    go.Bar(
                        x=["Train Batch"],
                        y=[throughput],
                        name=framework.upper(),
                        marker=dict(color=FRAMEWORK_COLORS[framework]),
                    )
                )

    fig.update_layout(
        title="Training Throughput (Samples/Second)",
        yaxis_title="Throughput (samples/sec)",
        hovermode="x unified",
        template="plotly_white",
        font=dict(size=12),
        height=600,
        width=900,
    )

    return fig


def plot_latency_distribution(results: Dict[str, Any]) -> go.Figure:
    """Create latency distribution box plot using median_abs_dev."""
    fig = go.Figure()

    benchmarks = ["predict_single", "predict_many", "train_batch"]

    for framework, data in results.items():
        mad_values = []
        benchmark_names = []

        for bench in benchmarks:
            if bench in data["benchmarks"]:
                metrics = data["benchmarks"][bench]["metrics"]
                if "median_abs_dev" in metrics:
                    mad = metrics["median_abs_dev"]["estimate_ms"]
                    if mad is not None:
                        mad_values.append(mad)
                        benchmark_names.append(bench.replace("_", "\n"))

        fig.add_trace(
            go.Bar(
                x=benchmark_names,
                y=mad_values,
                name=framework.upper(),
                marker=dict(color=FRAMEWORK_COLORS[framework]),
            )
        )

    fig.update_layout(
        title="Latency Distribution: Median Absolute Deviation",
        xaxis_title="Benchmark",
        yaxis_title="MAD (ms)",
        barmode="group",
        hovermode="x unified",
        template="plotly_white",
        font=dict(size=12),
        height=600,
        width=900,
    )

    return fig


def plot_confidence_intervals(results: Dict[str, Any]) -> go.Figure:
    """Create confidence interval visualization for mean latencies."""
    fig = go.Figure()

    benchmarks = ["predict_single", "predict_many", "train_batch"]
    frameworks_list = list(results.keys())

    for i, benchmark in enumerate(benchmarks):
        for j, framework in enumerate(frameworks_list):
            if benchmark in results[framework]["benchmarks"]:
                metrics = results[framework]["benchmarks"][benchmark]["metrics"]
                if "mean" in metrics:
                    mean = metrics["mean"]
                    estimate = mean["estimate_ms"]
                    lower = mean["ci_lower_ms"]
                    upper = mean["ci_upper_ms"]

                    # Add point
                    fig.add_trace(
                        go.Scatter(
                            x=[benchmark],
                            y=[estimate],
                            mode="markers",
                            marker=dict(size=10, color=FRAMEWORK_COLORS[framework]),
                            name=framework.upper(),
                            showlegend=(i == 0),
                        )
                    )

                    # Add error bars
                    if lower is not None and upper is not None:
                        fig.add_trace(
                            go.Scatter(
                                x=[benchmark, benchmark],
                                y=[lower, upper],
                                mode="lines",
                                line=dict(color=FRAMEWORK_COLORS[framework], width=2),
                                showlegend=False,
                                hoverinfo="skip",
                            )
                        )

    fig.update_layout(
        title="Mean Latency with 95% Confidence Intervals",
        xaxis_title="Benchmark",
        yaxis_title="Latency (ms)",
        hovermode="x unified",
        template="plotly_white",
        font=dict(size=12),
        height=600,
        width=900,
    )

    return fig


def compute_speedup_ratios(results: Dict[str, Any]) -> pd.DataFrame:
    """Compute speedup ratios (relative to baseline framework)."""
    baseline_framework = "burn"  # Use burn as baseline

    speedup_data = []

    benchmarks = ["predict_single", "predict_many", "train_batch"]

    for bench in benchmarks:
        if bench not in results[baseline_framework]["benchmarks"]:
            continue

        baseline_metrics = results[baseline_framework]["benchmarks"][bench]["metrics"]
        if "mean" not in baseline_metrics:
            continue

        baseline_latency = baseline_metrics["mean"]["estimate_ms"]

        for framework in results.keys():
            if framework not in results:
                continue

            if bench not in results[framework]["benchmarks"]:
                continue

            metrics = results[framework]["benchmarks"][bench]["metrics"]
            if "mean" not in metrics:
                continue

            latency = metrics["mean"]["estimate_ms"]
            speedup = baseline_latency / latency if latency > 0 else 1.0

            speedup_data.append(
                {
                    "Benchmark": bench,
                    "Framework": framework.upper(),
                    "Speedup": f"{speedup:.2f}x",
                    "Latency (ms)": f"{latency:.2f}",
                }
            )

    return pd.DataFrame(speedup_data)


def create_summary_table(
    results: Dict[str, Any], metadata: Dict[str, Any]
) -> pd.DataFrame:
    """Create summary table of all metrics."""
    summary_data = []

    for framework, data in results.items():
        convergence = data["convergence"]

        row = {
            "Framework": framework.upper(),
            "Final Accuracy": f"{convergence['final_accuracy']:.4f}",
            "Total Training Time (ms)": f"{convergence['total_training_time_ms']:.2f}",
            "Total Epochs": convergence["total_epochs"],
        }

        # Add benchmark metrics
        for bench in ["predict_single", "predict_many", "train_batch"]:
            if bench in data["benchmarks"]:
                metrics = data["benchmarks"][bench]["metrics"]
                if "mean" in metrics:
                    latency = metrics["mean"]["estimate_ms"]
                    row[f"{bench} (ms)"] = f"{latency:.2f}"

        summary_data.append(row)

    return pd.DataFrame(summary_data)


def save_results_csv(results: Dict[str, Any], output_dir: Path):
    """Save all results to CSV files."""
    # Convergence data
    convergence_data = []
    for framework, data in results.items():
        for row in data["convergence"]["convergence"]:
            convergence_data.append(
                {
                    "Framework": framework.upper(),
                    "Epoch": row["epoch"],
                    "Training Time (ms)": row["training_time_ms"],
                    "Accuracy": row["accuracy"],
                }
            )

    df_convergence = pd.DataFrame(convergence_data)
    df_convergence.to_csv(output_dir / "convergence_data.csv", index=False)
    print(f"✓ Saved convergence data to convergence_data.csv")

    # Benchmark metrics
    benchmark_data = []
    for framework, data in results.items():
        for bench_name, bench_data in data["benchmarks"].items():
            metrics = bench_data["metrics"]
            if "mean" in metrics:
                mean = metrics["mean"]
                benchmark_data.append(
                    {
                        "Framework": framework.upper(),
                        "Benchmark": bench_name,
                        "Mean (ms)": mean["estimate_ms"],
                        "CI Lower (ms)": mean["ci_lower_ms"],
                        "CI Upper (ms)": mean["ci_upper_ms"],
                    }
                )

    df_benchmarks = pd.DataFrame(benchmark_data)
    df_benchmarks.to_csv(output_dir / "benchmark_metrics.csv", index=False)
    print(f"✓ Saved benchmark metrics to benchmark_metrics.csv")

    # Summary table
    summary_df = create_summary_table(results, {})
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    print(f"✓ Saved summary to summary.csv")

    # Speedup ratios
    speedup_df = compute_speedup_ratios(results)
    speedup_df.to_csv(output_dir / "speedup_ratios.csv", index=False)
    print(f"✓ Saved speedup ratios to speedup_ratios.csv")


def save_figure(
    fig: go.Figure, output_dir: Path, name: str, format_types: List[str] = None
):
    """Save figure in multiple formats."""
    if format_types is None:
        format_types = ["html", "png"]

    for fmt in format_types:
        try:
            if fmt == "html":
                filepath = output_dir / f"{name}.html"
                fig.write_html(str(filepath))
                print(f"✓ Saved {name}.html")
            elif fmt == "png":
                filepath = output_dir / f"{name}.png"
                fig.write_image(str(filepath), width=1200, height=750, scale=2)
                print(f"✓ Saved {name}.png (300+ DPI)")
            elif fmt == "pdf":
                filepath = output_dir / f"{name}.pdf"
                fig.write_image(str(filepath), width=1200, height=750)
                print(f"✓ Saved {name}.pdf")
        except Exception as e:
            print(f"⚠ Warning: Failed to save {name}.{fmt}: {e}")


def main():
    """Main visualization logic."""
    print("ML Frameworks Benchmark Visualization")
    print("=" * 60)

    # Get latest result directory
    try:
        result_dir = get_latest_result_dir()
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

    print(f"Loading results from: {result_dir}")

    # Load results and metadata
    try:
        results = load_results(result_dir)
        metadata = load_metadata(result_dir)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

    # Create output directory
    output_dir = OUTPUT_DIR / result_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Visualizations will be saved to: {output_dir}")

    print("\nGenerating plots...")

    # Generate plots
    try:
        # Convergence curves
        fig_convergence = plot_convergence_curves(results)
        save_figure(fig_convergence, output_dir, "01_convergence_curves")

        # Inference latency
        fig_latency = plot_inference_latency(results)
        save_figure(fig_latency, output_dir, "02_inference_latency")

        # Training throughput
        fig_throughput = plot_training_throughput(results)
        save_figure(fig_throughput, output_dir, "03_training_throughput")

        # Latency distribution
        fig_distribution = plot_latency_distribution(results)
        save_figure(fig_distribution, output_dir, "04_latency_distribution")

        # Confidence intervals
        fig_ci = plot_confidence_intervals(results)
        save_figure(fig_ci, output_dir, "05_confidence_intervals")

    except Exception as e:
        print(f"✗ Error generating plots: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\nGenerating data exports...")

    # Save CSV results
    try:
        save_results_csv(results, output_dir)
    except Exception as e:
        print(f"⚠ Warning: Failed to save CSV results: {e}")

    print(f"\n{'='*60}")
    print(f"✓ Visualization complete!")
    print(f"All plots and data saved to: {output_dir.absolute()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

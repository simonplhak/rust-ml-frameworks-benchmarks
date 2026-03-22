#!/usr/bin/env python3
"""
Benchmark orchestrator for Rust ML frameworks.
Runs binaries and benchmarks for burn-example, tch-example, and candle-example.
Parses results from CSV (main.rs output) and Criterion JSON.
Stores results in timestamped JSON structure.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import traceback


# Framework configurations
# todo: use all benchmarks
# FRAMEWORKS = ["burn-example", "tch-example", "candle-example"]
FRAMEWORKS = ["tch-example"]
RESULT_DIR = Path("results")
CRITERION_BENCHMARKS = ["Predict_Single", "Predict_Many", "Train_Batch_Step"]


def run_command(cmd: list, cwd: Optional[Path] = None) -> str:
    """Execute shell command and return output. Print output to console. Abort on failure."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=False,
            text=True,
            timeout=1200,
        )

        if result.returncode != 0:
            error_msg = f"Command failed: {' '.join(cmd)}\n"
            error_msg += f"Return code: {result.returncode}"
            raise RuntimeError(error_msg)
        return result.stdout
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Command timed out: {' '.join(cmd)}") from e
    except Exception as e:
        raise RuntimeError(
            f"Failed to execute command: {' '.join(cmd)}: {str(e)}"
        ) from e


def find_csv_file(framework_dir: Path) -> Optional[Path]:
    """Find convergence CSV file in framework directory."""
    csv_file = framework_dir / "convergence_results.csv"
    if csv_file.exists():
        return csv_file
    raise Exception(f"CSV file not found in {framework_dir}")


def parse_convergence_csv(csv_path: Path) -> Dict[str, Any]:
    """Parse convergence CSV file from main.rs output."""
    convergence_data = []
    try:
        with open(csv_path, "r") as f:
            lines = f.readlines()
            if len(lines) < 2:
                raise ValueError("CSV has no data rows")

            # Skip header
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) >= 3:
                    convergence_data.append(
                        {
                            "epoch": int(parts[0]),
                            "accuracy": float(parts[1]),
                            "training_time_ms": float(parts[2]),
                        }
                    )

        if not convergence_data:
            raise ValueError("No valid convergence data found in CSV")

        return {
            "convergence": convergence_data,
            "total_epochs": len(convergence_data),
            "final_accuracy": convergence_data[-1]["accuracy"],
            "total_training_time_ms": convergence_data[-1]["training_time_ms"],
        }
    except Exception as e:
        raise RuntimeError(
            f"Failed to parse convergence CSV {csv_path}: {str(e)}"
        ) from e


def get_criterion_unit(estimates: Dict[str, Any]) -> str:
    """Extract unit from Criterion estimates.json. Returns 'ns', 'us', 'ms', or 'ms' (default)."""
    # Try to find unit in any metric's confidence_interval
    for metric in ["mean", "median", "std_dev", "slope"]:
        if metric in estimates and isinstance(estimates[metric], dict):
            ci = estimates[metric].get("confidence_interval", {})
            if "unit" in ci:
                return ci["unit"]
    return "ns"  # Default to nanoseconds


def convert_to_ms(value: float, from_unit: str) -> float:
    """Convert time value to milliseconds."""
    conversions = {
        "ns": value / 1_000_000,
        "us": value / 1_000,
        "ms": value,
        "s": value * 1_000,
    }
    return conversions.get(from_unit, value)


def parse_criterion_estimates(
    estimates_json: Dict[str, Any], unit: str
) -> Dict[str, Any]:
    """Parse Criterion estimates.json file. All times converted to ms."""
    metrics = {}

    for metric_name in ["mean", "median", "std_dev", "median_abs_dev", "slope"]:
        if metric_name not in estimates_json or estimates_json[metric_name] is None:
            continue

        metric_data = estimates_json[metric_name]

        # Extract point estimate
        point_estimate = metric_data.get("point_estimate")
        if point_estimate is None:
            continue

        # Convert to ms
        point_estimate_ms = convert_to_ms(point_estimate, unit)

        # Extract confidence interval
        ci = metric_data.get("confidence_interval", {})
        lower_bound = ci.get("lower_bound")
        upper_bound = ci.get("upper_bound")

        if lower_bound is not None:
            lower_bound = convert_to_ms(lower_bound, unit)
        if upper_bound is not None:
            upper_bound = convert_to_ms(upper_bound, unit)

        metrics[metric_name] = {
            "estimate_ms": point_estimate_ms,
            "ci_lower_ms": lower_bound,
            "ci_upper_ms": upper_bound,
            "standard_error_ms": (
                convert_to_ms(metric_data.get("standard_error", 0), unit)
                if metric_data.get("standard_error")
                else None
            ),
        }

    return metrics


def parse_criterion_results(framework_dir: Path, framework_name: str) -> Dict[str, Any]:
    """Parse all Criterion benchmark results for a framework."""
    criterion_dir = Path("target") / "criterion"

    if not criterion_dir.exists():
        raise RuntimeError(f"Criterion directory not found: {criterion_dir}")

    benchmark_results = {}
    framework_short = framework_name.split("-")[0]

    for bench_name in CRITERION_BENCHMARKS:
        bench_dir = criterion_dir / bench_name / framework_short

        if not bench_dir.exists():
            raise RuntimeError(f"Benchmark directory not found: {bench_dir}")

        if bench_name == "Train_Batch_Step":
            # Parametrized benchmark with batch sizes
            batch_sizes = [32, 64, 128]
            benchmark_results[bench_name] = {}

            for batch_size in batch_sizes:
                param_dir = bench_dir / str(batch_size)
                estimates_file = param_dir / "new" / "estimates.json"

                if not estimates_file.exists():
                    raise RuntimeError(
                        f"Criterion estimates.json not found for {bench_name} (batch_size={batch_size}): {estimates_file}"
                    )

                try:
                    with open(estimates_file, "r") as f:
                        estimates = json.load(f)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to parse Criterion estimates for {bench_name} (batch_size={batch_size}): {str(e)}"
                    ) from e

                unit = get_criterion_unit(estimates)
                metrics = parse_criterion_estimates(estimates, unit)

                benchmark_results[bench_name][str(batch_size)] = {
                    "original_unit": unit,
                    "metrics": metrics,
                }
        else:
            # Non-parametrized benchmarks
            estimates_file = bench_dir / "latency" / "new" / "estimates.json"

            if not estimates_file.exists():
                raise RuntimeError(
                    f"Criterion estimates.json not found for {bench_name}: {estimates_file}"
                )

            try:
                with open(estimates_file, "r") as f:
                    estimates = json.load(f)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to parse Criterion estimates for {bench_name}: {str(e)}"
                ) from e

            unit = get_criterion_unit(estimates)
            metrics = parse_criterion_estimates(estimates, unit)

            benchmark_results[bench_name] = {
                "original_unit": unit,
                "metrics": metrics,
            }

    return benchmark_results


def run_framework_benchmarks(framework_name: str) -> Dict[str, Any]:
    """Run binary and benchmarks for a single framework. Abort on any failure."""
    framework_dir = Path(framework_name)

    if not framework_dir.exists():
        raise RuntimeError(f"Framework directory not found: {framework_dir}")

    print(f"\n{'='*60}")
    print(f"Processing {framework_name}")
    print(f"{'='*60}")

    # Step 1: Run binary (cargo run --release) to generate convergence CSV
    print(f"[1/2] Running {framework_name} binary...")
    try:
        csv_output_path = framework_dir / "convergence_results.csv"
        cmd = ["cargo", "run", "--release", "-p", framework_name]
        if framework_name == "burn-example":
            cmd.extend(["--features", "ndarray"])
        cmd.extend(["--", "--output-csv", str(csv_output_path)])
        run_command(cmd)
        print(f"✓ {framework_name} binary completed")
    except RuntimeError as e:
        print(f"✗ Failed to run {framework_name} binary")
        raise

    # Step 2: Parse convergence CSV
    print(f"[2/3] Parsing convergence data...")
    csv_file = find_csv_file(framework_dir)

    try:
        convergence_data = parse_convergence_csv(csv_file)
        print(
            f"✓ Parsed convergence data: {convergence_data['total_epochs']} epochs, "
            f"final accuracy: {convergence_data['final_accuracy']:.4f}"
        )
    except RuntimeError as e:
        print(f"✗ Failed to parse convergence CSV")
        raise

    # Step 3: Run benchmarks (cargo bench)
    print(f"[3/3] Running Criterion benchmarks...")
    try:
        cmd = ["cargo", "bench", "-p", framework_name]
        if framework_name == "burn-example":
            cmd.extend(["--features", "ndarray"])
        run_command(cmd)
        print(f"✓ Criterion benchmarks completed")
    except RuntimeError as e:
        print(f"✗ Failed to run Criterion benchmarks")
        raise

    # Step 4: Parse Criterion results
    print(f"[4/3] Parsing Criterion results...")
    try:
        criterion_results = parse_criterion_results(framework_dir, framework_name)
        print(f"✓ Parsed Criterion results for {len(criterion_results)} benchmarks")
    except RuntimeError as e:
        print(f"✗ Failed to parse Criterion results")
        raise

    return {
        "framework": framework_name,
        "convergence": convergence_data,
        "benchmarks": criterion_results,
    }


def get_system_info() -> Dict[str, str]:
    """Collect system information."""
    system_info = {
        "timestamp": datetime.now().isoformat(),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H-%M-%S"),
    }

    # Try to get git commit hash
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        system_info["git_commit"] = git_commit
    except Exception:
        system_info["git_commit"] = "unknown"

    # Try to get Rust version
    try:
        rust_version = subprocess.run(
            ["rustc", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        system_info["rust_version"] = rust_version
    except Exception:
        system_info["rust_version"] = "unknown"

    return system_info


def main():
    """Main orchestration logic."""
    print("Rust ML Frameworks Benchmark Suite")
    print("=" * 60)

    # Create results directory with timestamp
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = RESULT_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results will be saved to: {run_dir}")

    # Collect system info
    system_info = get_system_info()

    all_results = {}

    # Run benchmarks for each framework
    for framework in FRAMEWORKS:
        try:
            results = run_framework_benchmarks(framework)
            all_results[framework] = results
        except RuntimeError as e:
            print(f"\n✗ FATAL ERROR in {framework}:")
            print(f"  {str(e)}")
            print("\n✗ Full traceback:")
            traceback.print_exc()
            print("\n✗ Aborting entire benchmark run due to framework failure.")
            sys.exit(1)

    # Save metadata
    metadata = {
        "benchmark_suite_run": timestamp,
        "system_info": system_info,
        "frameworks": list(all_results.keys()),
    }

    metadata_file = run_dir / "metadata.json"
    try:
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"\n✓ Saved metadata to {metadata_file}")
    except Exception as e:
        print(f"✗ Failed to save metadata: {e}")
        sys.exit(1)

    # Save framework results
    for framework, results in all_results.items():
        result_file = run_dir / f"{framework.replace('-example', '')}_results.json"
        try:
            with open(result_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"✓ Saved {framework} results to {result_file}")
        except Exception as e:
            print(f"✗ Failed to save {framework} results: {e}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"✓ All benchmarks completed successfully!")
    print(f"Results saved to: {run_dir.absolute()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

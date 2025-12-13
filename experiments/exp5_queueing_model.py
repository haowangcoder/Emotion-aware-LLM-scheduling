#!/usr/bin/env python3
"""
Exp-5: Queueing/Regression Modeling

Purpose:
- Build a simple predictive model for waiting time
- Validate M/G/1 approximation or regression model
- Explain why queue_length/ρ works as online control signal

Two approaches:
A. M/G/1 Queueing Approximation:
   E[Wq] = (λ * E[S^2]) / (2 * (1 - ρ))
   where ρ = λ * E[S]

B. Regression Model:
   predicted_wait = f(queue_length, predicted_S, k)

Usage:
    python experiments/exp5_queueing_model.py \
        --input_dir results/experiments/exp2_load_sweep \
        --output_dir results/experiments/exp5_queueing
"""

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import matplotlib.pyplot as plt  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    plt = None

# Optional sklearn for Random Forest
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error as sklearn_mae
    from sklearn.metrics import r2_score as sklearn_r2
    HAS_SKLEARN = True
except ModuleNotFoundError:  # pragma: no cover
    HAS_SKLEARN = False


def mg1_expected_wait(arrival_rate: float, mean_service: float, var_service: float) -> float:
    """
    Compute expected waiting time using M/G/1 formula.

    Args:
        arrival_rate: λ (arrivals per second)
        mean_service: E[S] (mean service time)
        var_service: Var[S] (variance of service time)

    Returns:
        Expected waiting time E[Wq]
    """
    rho = arrival_rate * mean_service
    if rho >= 1.0:
        raise ValueError("M/G/1 steady-state requires rho < 1.0")

    # Pollaczek-Khinchin formula
    second_moment = var_service + mean_service ** 2
    expected_wait = (arrival_rate * second_moment) / (2 * (1 - rho))
    return expected_wait


@dataclass(frozen=True)
class JobRecord:
    job_id: int
    arrival_time: float
    start_time: Optional[float]
    finish_time: Optional[float]
    waiting_time: Optional[float]
    predicted_serving_time: Optional[float]
    actual_serving_time: Optional[float]
    affect_weight: Optional[float]
    urgency: Optional[float]
    arousal: Optional[float]
    valence: Optional[float]


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return default


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_result_files(result_dir: Path, pattern: str) -> List[Path]:
    return sorted(result_dir.glob(pattern))


def _pick_file_by_scheduler(files: Sequence[Path], scheduler: Optional[str]) -> Optional[Path]:
    if not files:
        return None
    if scheduler:
        # Prefer filename prefix match (fast path).
        prefix = f"{scheduler}_"
        for path in files:
            if path.name.startswith(prefix):
                return path
        # Fallback: read metadata.scheduler.
        for path in files:
            try:
                summary = _read_json(path)
            except Exception:
                continue
            sched = summary.get("metadata", {}).get("scheduler")
            if sched == scheduler:
                return path
    return files[0]


def _read_jobs_csv(path: Path) -> List[JobRecord]:
    jobs: List[JobRecord] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            jobs.append(
                JobRecord(
                    job_id=_safe_int(row.get("job_id")),
                    arrival_time=_safe_float(row.get("arrival_time")) or 0.0,
                    start_time=_safe_float(row.get("start_time")),
                    finish_time=_safe_float(row.get("finish_time")),
                    waiting_time=_safe_float(row.get("waiting_time")),
                    predicted_serving_time=_safe_float(
                        row.get("predicted_serving_time") or row.get("predicted_service_time")
                    ),
                    actual_serving_time=_safe_float(
                        row.get("actual_serving_time") or row.get("actual_service_time")
                    ),
                    affect_weight=_safe_float(row.get("affect_weight")),
                    urgency=_safe_float(row.get("urgency")),
                    arousal=_safe_float(row.get("arousal")),
                    valence=_safe_float(row.get("valence")),
                )
            )
    return jobs


def _mean_and_var(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        raise ValueError("expected non-empty values")
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, 0.0
    return mean, statistics.variance(values)


def _fcfs_avg_wait_from_samples(arrival_times: Sequence[float], service_times: Sequence[float]) -> float:
    if len(arrival_times) != len(service_times):
        raise ValueError("arrival_times and service_times must have same length")
    pairs = sorted(zip(arrival_times, service_times), key=lambda p: p[0])
    server_free = 0.0
    total_wait = 0.0
    for arrival_time, service_time in pairs:
        if server_free < arrival_time:
            server_free = arrival_time
        total_wait += server_free - arrival_time
        server_free += service_time
    return total_wait / len(pairs) if pairs else 0.0


def _simulate_fcfs_avg_wait(
    arrival_rate: float,
    service_samples: Sequence[float],
    num_jobs: int,
    rng: random.Random,
) -> float:
    if arrival_rate <= 0:
        return 0.0
    if not service_samples:
        return 0.0
    t = 0.0
    server_free = 0.0
    total_wait = 0.0

    for _ in range(num_jobs):
        u = rng.random()
        inter_arrival = -math.log(1.0 - u) / arrival_rate
        t += inter_arrival

        service_time = rng.choice(service_samples)
        if server_free < t:
            server_free = t
        total_wait += server_free - t
        server_free += service_time

    return total_wait / num_jobs if num_jobs else 0.0


def _solve_linear_system(a: List[List[float]], b: List[float]) -> List[float]:
    """
    Solve A x = b via Gaussian elimination with partial pivoting.
    Assumes A is square and non-singular.
    """
    n = len(a)
    if n == 0:
        return []
    # Make copies (we'll mutate)
    a = [row[:] for row in a]
    b = b[:]

    for col in range(n):
        # Pivot
        pivot_row = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot_row][col]) < 1e-12:
            raise ValueError("singular matrix")
        if pivot_row != col:
            a[col], a[pivot_row] = a[pivot_row], a[col]
            b[col], b[pivot_row] = b[pivot_row], b[col]

        # Eliminate
        pivot = a[col][col]
        for r in range(col + 1, n):
            factor = a[r][col] / pivot
            if factor == 0:
                continue
            for c in range(col, n):
                a[r][c] -= factor * a[col][c]
            b[r] -= factor * b[col]

    # Back-substitute
    x = [0.0] * n
    for r in reversed(range(n)):
        s = b[r] - sum(a[r][c] * x[c] for c in range(r + 1, n))
        x[r] = s / a[r][r]
    return x


def _ridge_fit(
    x: Sequence[Sequence[float]],
    y: Sequence[float],
    alpha: float = 1.0,
) -> Tuple[List[float], float]:
    """
    Fit ridge regression with intercept (intercept not regularized).

    Returns: (coefficients, intercept)
    """
    if not x or not y:
        return [], 0.0
    if len(x) != len(y):
        raise ValueError("X and y must have same length")

    n = len(x)
    p = len(x[0])

    # Augment design matrix with intercept term.
    z = [[1.0] + list(row) for row in x]
    dim = p + 1

    # Compute A = Z^T Z + alpha * R, b = Z^T y
    a = [[0.0 for _ in range(dim)] for __ in range(dim)]
    b_vec = [0.0 for _ in range(dim)]

    for i in range(n):
        zi = z[i]
        yi = y[i]
        for r in range(dim):
            b_vec[r] += zi[r] * yi
            for c in range(dim):
                a[r][c] += zi[r] * zi[c]

    # Regularize coefficients only (not intercept)
    for i in range(1, dim):
        a[i][i] += alpha

    beta = _solve_linear_system(a, b_vec)
    intercept = beta[0]
    coef = beta[1:]
    return coef, intercept


def _predict_linear(x: Sequence[Sequence[float]], coef: Sequence[float], intercept: float) -> List[float]:
    preds: List[float] = []
    for row in x:
        preds.append(intercept + sum(ci * xi for ci, xi in zip(coef, row)))
    return preds


def _mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if not y_true:
        return 0.0
    return statistics.fmean(abs(a - b) for a, b in zip(y_true, y_pred))


def _r2(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if not y_true:
        return 0.0
    mean_y = statistics.fmean(y_true)
    ss_tot = sum((v - mean_y) ** 2 for v in y_true)
    if ss_tot == 0:
        return 0.0
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    return 1.0 - (ss_res / ss_tot)


def plot_prediction_scatter(
    actual: Sequence[float],
    predicted: Sequence[float],
    output_path: Path,
    title: str = "Model Prediction vs Actual"
) -> None:
    """
    Plot predicted vs actual scatter with y=x reference line.
    """
    if plt is None:  # pragma: no cover
        print("matplotlib not installed; skipping plot:", output_path)
        return
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(actual, predicted, alpha=0.5, s=30)

    # y=x reference line
    lims = [min(min(actual), min(predicted)), max(max(actual), max(predicted))]
    ax.plot(lims, lims, 'r--', linewidth=2, label='y=x (perfect prediction)')

    ax.set_xlabel('Actual Waiting Time (seconds)', fontsize=12)
    ax.set_ylabel('Predicted Waiting Time (seconds)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_mg1_validation(
    load_values: List[float],
    actual_waits: List[float],
    predicted_waits: List[Optional[float]],
    output_path: Path
) -> None:
    """
    Plot M/G/1 model validation: predicted vs actual across loads.
    """
    if plt is None:  # pragma: no cover
        print("matplotlib not installed; skipping plot:", output_path)
        return
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(load_values, actual_waits, 'o-', label='Actual (simulation)', linewidth=2, markersize=8)
    masked_pred = [p if p is not None else float("nan") for p in predicted_waits]
    ax.plot(load_values, masked_pred, 's--', label='M/G/1 prediction (steady-state)', linewidth=2, markersize=8)

    ax.axvline(x=1.0, color='red', linestyle=':', alpha=0.7, label='Saturation (ρ=1)')

    ax.set_xlabel('System Load (ρ)', fontsize=12)
    ax.set_ylabel('Average Waiting Time (seconds)', fontsize=12)
    ax.set_title('M/G/1 Model Validation', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_residual_vs_load(
    load_values: List[float],
    residuals: List[Optional[float]],
    output_path: Path
) -> None:
    """
    Plot residuals vs load to identify model limitations.
    """
    if plt is None:  # pragma: no cover
        print("matplotlib not installed; skipping plot:", output_path)
        return
    fig, ax = plt.subplots(figsize=(10, 5))

    masked_residuals = [r if r is not None else 0.0 for r in residuals]
    ax.bar(load_values, masked_residuals, width=0.08, color='#4a90d9', edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    ax.set_xlabel('System Load (ρ)', fontsize=12)
    ax.set_ylabel('Residual (Actual - Predicted)', fontsize=12)
    ax.set_title('Model Residuals by Load\n(Shows where model over/under-predicts)', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_model_report(
    mg1_results: Dict,
    regression_metrics: Dict,
    rf_metrics: Dict,
    output_path: Path
) -> Dict:
    """Generate modeling report."""
    report = {
        'experiment': 'exp5_queueing_model',
        'purpose': 'Validate predictive models for waiting time',
        'mg1_model': mg1_results,
        'regression_model': regression_metrics,
        'random_forest_model': rf_metrics,
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved: {output_path}")
    return report


def load_exp2_results(
    input_dir: Path,
    scheduler: Optional[str],
) -> Tuple[List[float], List[float], Dict[str, Any]]:
    """
    Load results from exp2_load_sweep directories.

    Args:
        input_dir: Path to exp2_load_sweep results

    Returns:
        Tuple of (loads, actual_waits, combined_data)
    """
    loads = []
    actual_waits = []
    run_records: List[Dict[str, Any]] = []
    all_service_times: List[float] = []
    all_jobs: List[JobRecord] = []

    # Find all load* directories (handles both load0.5 and load_0.5 patterns)
    load_dirs = sorted(list(input_dir.glob("load_*")) + list(input_dir.glob("load[0-9]*")))
    # Remove duplicates
    load_dirs = sorted(set(load_dirs), key=lambda p: p.name)

    for load_dir in load_dirs:
        # Extract load value from directory name (e.g., load0.7 or load_0.7 -> 0.7)
        try:
            load_str = load_dir.name.replace("load_", "").replace("load", "")
            load_value = float(load_str)
        except ValueError:
            continue

        # Load JSON summary
        summary_candidates = _find_result_files(load_dir, "*_summary.json")
        summary_path = _pick_file_by_scheduler(summary_candidates, scheduler)
        if summary_path is None:
            continue

        summary = _read_json(summary_path)
        meta = summary.get("metadata", {})
        run_metrics = meta.get("run_metrics") or summary.get("run_metrics") or {}
        avg_wait = _safe_float(run_metrics.get("avg_waiting_time")) or 0.0
        num_jobs = _safe_int(run_metrics.get("total_jobs")) or 0
        arrival_rate = _safe_float(meta.get("arrival_rate"))
        nominal_load = _safe_float(meta.get("system_load")) or load_value
        expected_service_time = None
        if arrival_rate and arrival_rate > 0:
            expected_service_time = nominal_load / arrival_rate

        loads.append(load_value)
        actual_waits.append(avg_wait)

        # Load CSV for detailed data
        csv_candidates = sorted(load_dir.glob("*_jobs.csv")) + sorted(load_dir.glob("*_fixed_jobs.csv"))
        jobs_path = _pick_file_by_scheduler(csv_candidates, scheduler)

        jobs: List[JobRecord] = []
        if jobs_path is not None and jobs_path.exists():
            jobs = _read_jobs_csv(jobs_path)
            for j in jobs:
                if j.actual_serving_time is not None:
                    all_service_times.append(j.actual_serving_time)
            all_jobs.extend(jobs)

        run_records.append(
            {
                "load": load_value,
                "nominal_load": nominal_load,
                "arrival_rate": arrival_rate,
                "expected_service_time": expected_service_time,
                "avg_wait": avg_wait,
                "num_jobs": num_jobs,
                "summary_path": str(summary_path),
                "jobs_path": str(jobs_path) if jobs_path else None,
                "jobs": jobs,
            }
        )

    combined = {
        "runs": run_records,
        "all_service_times": all_service_times,
        "all_jobs": all_jobs,
    }

    return loads, actual_waits, combined


def main():
    parser = argparse.ArgumentParser(description="Exp-5: Queueing Model Analysis")
    parser.add_argument("--input_dir", type=str, default="results/experiments/exp2_load_sweep")
    parser.add_argument("--output_dir", type=str, default="results/experiments/exp5_queueing")
    parser.add_argument("--scheduler", type=str, default="AW-SSJF",
                        help="Which scheduler's exp2 outputs to analyze (e.g., AW-SSJF or SJF).")
    parser.add_argument("--mc_reps", type=int, default=2000,
                        help="Monte Carlo reps for finite-horizon FCFS model.")
    parser.add_argument("--reg_target", choices=["raw", "log1p"], default="log1p",
                        help="Regression target transform (log1p is usually better for heavy-tail waits).")
    parser.add_argument("--reg_poly", action=argparse.BooleanOptionalAction, default=True,
                        help="Add simple polynomial/interaction terms for regression.")
    parser.add_argument("--reg_alpha", type=float, default=1.0,
                        help="Ridge regularization strength for regression.")
    parser.add_argument("--rf_estimators", type=int, default=200,
                        help="Number of trees for Random Forest.")
    parser.add_argument("--rf_max_depth", type=int, default=15,
                        help="Max depth for Random Forest trees.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print("Exp-5: Queueing Model Analysis")
    print("========================================")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print("========================================")

    # Load real data from exp2_load_sweep
    loads, actual_waits, data = load_exp2_results(input_dir, scheduler=args.scheduler)

    if not loads:
        print("\nWARNING: No exp2 results found. Using mock data for demonstration.")
        loads = [0.5, 0.7, 0.9, 0.95]
        mean_S = 2.0
        var_S = 0.5
        actual_waits = []
        rng = random.Random(42)
        for rho in loads:
            arrival_rate = rho / mean_S
            try:
                pred = mg1_expected_wait(arrival_rate, mean_S, var_S)
            except ValueError:
                pred = 50.0
            actual = pred * (1 + 0.1 * rng.uniform(-1, 1)) + 2 * rho
            actual_waits.append(actual)
        data = {"runs": [], "all_service_times": [], "all_jobs": []}
    else:
        print(f"\nLoaded {len(loads)} load configurations from exp2")

    # Estimate global service time distribution from job logs (more stable than per-load).
    service_times = data.get("all_service_times", [])
    if service_times:
        mean_S, var_S = _mean_and_var(service_times)
        print(f"  Service time (empirical): mean={mean_S:.3f}s, var={var_S:.3f}")
    else:
        mean_S, var_S = 2.0, 0.5
        print(f"  Using default service time: mean={mean_S:.3f}s, var={var_S:.3f}")

    # Collect per-load parameters from summaries.
    runs: List[Dict[str, Any]] = data.get("runs", [])
    runs_by_load = {float(r.get("load")): r for r in runs if r.get("load") is not None}

    arrival_rates: List[Optional[float]] = []
    rho_effective: List[Optional[float]] = []
    expected_service_times: List[Optional[float]] = []
    mg1_predicted: List[Optional[float]] = []
    fcfs_sample_predicted: List[Optional[float]] = []
    fcfs_mc_predicted: List[Optional[float]] = []

    # Precompute bootstrap service samples for Monte Carlo model.
    service_samples = service_times[:] if service_times else [mean_S]

    for load in loads:
        rec = runs_by_load.get(load, {})
        lam = rec.get("arrival_rate")
        if lam is not None:
            lam = float(lam)
        arrival_rates.append(lam)
        expected_service_times.append(rec.get("expected_service_time"))

        if lam is None:
            rho_effective.append(None)
            mg1_predicted.append(None)
            fcfs_sample_predicted.append(None)
            fcfs_mc_predicted.append(None)
            continue

        rho_eff = lam * mean_S
        rho_effective.append(rho_eff)

        # M/G/1 steady-state expectation (valid only if rho < 1).
        try:
            mg1_pred = mg1_expected_wait(lam, mean_S, var_S)
        except ValueError:
            mg1_pred = None
        mg1_predicted.append(mg1_pred)

        # FCFS sample-path avg wait using realized arrivals/services (finite horizon; always finite).
        jobs: List[JobRecord] = rec.get("jobs") or []
        arrivals = [j.arrival_time for j in jobs]
        services = [j.actual_serving_time for j in jobs if j.actual_serving_time is not None]
        if jobs and len(services) == len(jobs):
            fcfs_sample = _fcfs_avg_wait_from_samples(arrivals, services)  # type: ignore[arg-type]
        else:
            fcfs_sample = None
        fcfs_sample_predicted.append(fcfs_sample)

        # Finite-horizon FCFS Monte Carlo expectation.
        n_jobs = int(rec.get("num_jobs") or (len(jobs) if jobs else 80))
        rng = random.Random(42)
        reps = max(1, int(args.mc_reps))
        mc_avg = statistics.fmean(
            _simulate_fcfs_avg_wait(lam, service_samples, n_jobs, rng) for _ in range(reps)
        )
        fcfs_mc_predicted.append(mc_avg)

    print(f"\n=== M/G/1 Validation ===")
    for i, load in enumerate(loads):
        lam = arrival_rates[i]
        rho_eff = rho_effective[i]
        pred = mg1_predicted[i]
        if pred is None:
            print(
                f"  load={load:.2f}: λ={lam:.3f} ρ_eff={rho_eff:.3f} "
                f"Actual={actual_waits[i]:.2f}s, M/G/1=undefined (ρ>=1)"
            )
        else:
            residual = actual_waits[i] - pred
            print(
                f"  load={load:.2f}: λ={lam:.3f} ρ_eff={rho_eff:.3f} "
                f"Actual={actual_waits[i]:.2f}s, M/G/1={pred:.2f}s, Residual={residual:+.2f}s"
            )

    # Generate plots
    print("\n=== Generating Plots ===")
    plot_mg1_validation(loads, actual_waits, mg1_predicted, output_dir / "mg1_validation.png")

    residuals: List[Optional[float]] = []
    for a, p in zip(actual_waits, mg1_predicted):
        residuals.append(None if p is None else a - p)
    plot_residual_vs_load(loads, residuals, output_dir / "residual_vs_load.png")

    # Regression model
    print("\n=== Regression Model ===")
    feature_names = [
        "system_load",
        "arrival_rate",
        "predicted_serving_time",
        "affect_weight",
        "queue_len_at_arrival",
        "num_in_system_at_arrival",
        "pred_workload_at_arrival",
    ]
    if args.reg_poly:
        feature_names += [
            "predicted_serving_time*queue_len_at_arrival",
            "predicted_serving_time*num_in_system_at_arrival",
            "pred_workload_at_arrival*predicted_serving_time",
            "pred_workload_at_arrival*num_in_system_at_arrival",
            "pred_workload_at_arrival*queue_len_at_arrival",
            "pred_workload_at_arrival^2",
            "num_in_system_at_arrival^2",
            "queue_len_at_arrival^2",
        ]

    x_rows: List[List[float]] = []
    y_rows: List[float] = []

    def predicted_remaining(job: JobRecord, t: float) -> float:
        pred = job.predicted_serving_time
        if pred is None:
            pred = job.actual_serving_time
        if pred is None:
            return 0.0
        if job.start_time is not None and job.finish_time is not None and job.start_time <= t < job.finish_time:
            served = max(0.0, t - job.start_time)
            return max(0.0, pred - served)
        return pred

    for run in runs:
        load = float(run.get("load") or 0.0)
        lam = run.get("arrival_rate")
        lam = float(lam) if lam is not None else 0.0
        jobs: List[JobRecord] = run.get("jobs") or []

        # O(N^2) is fine here: N=80 per run.
        for job in jobs:
            if job.waiting_time is None:
                continue
            t = job.arrival_time
            prev_active = [
                j
                for j in jobs
                if j.job_id != job.job_id
                and j.arrival_time < t
                and j.finish_time is not None
                and j.finish_time > t
            ]
            queue_len = sum(1 for j in prev_active if j.start_time is None or j.start_time > t)
            num_in_system = len(prev_active)
            workload = sum(predicted_remaining(j, t) for j in prev_active)

            pred_s = job.predicted_serving_time or (job.actual_serving_time or 0.0)
            affect_w = job.affect_weight or 1.0
            q_len = float(queue_len)
            n_sys = float(num_in_system)
            w_pred = float(workload)

            feats = [
                load,
                lam,
                pred_s,
                affect_w,
                q_len,
                n_sys,
                w_pred,
            ]
            if args.reg_poly:
                feats += [
                    pred_s * q_len,
                    pred_s * n_sys,
                    w_pred * pred_s,
                    w_pred * n_sys,
                    w_pred * q_len,
                    w_pred ** 2,
                    n_sys ** 2,
                    q_len ** 2,
                ]

            x_rows.append(feats)
            y_rows.append(float(job.waiting_time))

    rf_metrics: Dict[str, Any] = {}
    if len(y_rows) < 20:
        print("  Not enough job-level rows for regression; skipping.")
        reg_metrics: Dict[str, Any] = {}
    else:
        # Train/test split
        idx = list(range(len(y_rows)))
        rnd = random.Random(42)
        rnd.shuffle(idx)
        test_size = max(1, int(0.2 * len(idx)))
        test_idx = set(idx[:test_size])

        x_train = [x_rows[i] for i in range(len(x_rows)) if i not in test_idx]
        y_train_raw = [y_rows[i] for i in range(len(y_rows)) if i not in test_idx]
        x_test = [x_rows[i] for i in range(len(x_rows)) if i in test_idx]
        y_test_raw = [y_rows[i] for i in range(len(y_rows)) if i in test_idx]

        if args.reg_target == "log1p":
            y_train = [math.log1p(v) for v in y_train_raw]
        else:
            y_train = y_train_raw

        # Standardize features based on train set.
        cols = list(zip(*x_train))
        means = [statistics.fmean(col) for col in cols]
        stds = [statistics.pstdev(col) or 1.0 for col in cols]

        def zscore_rows(rows: Sequence[Sequence[float]]) -> List[List[float]]:
            out: List[List[float]] = []
            for row in rows:
                out.append([(v - m) / s for v, m, s in zip(row, means, stds)])
            return out

        x_train_z = zscore_rows(x_train)

        coef_z, intercept_z = _ridge_fit(x_train_z, y_train, alpha=float(args.reg_alpha))

        # Convert coefficients back to original feature scale.
        coef = [cz / s for cz, s in zip(coef_z, stds)]
        intercept = intercept_z - sum((cz * m) / s for cz, m, s in zip(coef_z, means, stds))

        y_pred_test = _predict_linear(x_test, coef, intercept)
        if args.reg_target == "log1p":
            y_pred_test = [max(0.0, math.expm1(v)) for v in y_pred_test]

        r2 = _r2(y_test_raw, y_pred_test)
        mae = _mae(y_test_raw, y_pred_test)

        reg_metrics = {
            "mae": mae,
            "r2": r2,
            "target_transform": args.reg_target,
            "poly_features": bool(args.reg_poly),
            "alpha": float(args.reg_alpha),
            "feature_names": feature_names,
            "coefficients": dict(zip(feature_names, coef)),
            "intercept": intercept,
            "n_samples": len(y_rows),
            "n_train": len(y_train_raw),
            "n_test": len(y_test_raw),
        }

        print(f"  [Ridge] R2 Score: {r2:.3f}")
        print(f"  [Ridge] MAE: {mae:.3f}s")

        y_pred_all = _predict_linear(x_rows, coef, intercept)
        if args.reg_target == "log1p":
            y_pred_all = [max(0.0, math.expm1(v)) for v in y_pred_all]
        plot_prediction_scatter(y_rows, y_pred_all, output_dir / "regression_scatter.png",
                                title=f"Ridge Regression (R²={r2:.3f})")

        # Random Forest Model
        if HAS_SKLEARN:
            print("\n=== Random Forest Model ===")
            # Use base features only (no polynomial terms needed for RF)
            base_feature_names = [
                "system_load",
                "arrival_rate",
                "predicted_serving_time",
                "affect_weight",
                "queue_len_at_arrival",
                "num_in_system_at_arrival",
                "pred_workload_at_arrival",
            ]
            x_base = [row[:7] for row in x_rows]  # First 7 features
            x_train_rf = [x_base[i] for i in range(len(x_base)) if i not in test_idx]
            x_test_rf = [x_base[i] for i in range(len(x_base)) if i in test_idx]

            rf = RandomForestRegressor(
                n_estimators=args.rf_estimators,
                max_depth=args.rf_max_depth,
                min_samples_split=3,
                random_state=42,
                n_jobs=-1,
            )
            rf.fit(x_train_rf, y_train_raw)
            y_pred_rf = rf.predict(x_test_rf)

            rf_mae = sklearn_mae(y_test_raw, y_pred_rf)
            rf_r2 = sklearn_r2(y_test_raw, y_pred_rf)

            print(f"  [RF] R2 Score: {rf_r2:.3f}")
            print(f"  [RF] MAE: {rf_mae:.3f}s")

            # Feature importances
            importances = dict(zip(base_feature_names, rf.feature_importances_))
            print(f"  [RF] Feature Importances:")
            for name, imp in sorted(importances.items(), key=lambda x: -x[1]):
                print(f"       {name}: {imp:.4f}")

            rf_metrics = {
                "mae": rf_mae,
                "r2": rf_r2,
                "n_estimators": args.rf_estimators,
                "max_depth": args.rf_max_depth,
                "feature_names": base_feature_names,
                "feature_importances": importances,
                "n_samples": len(y_rows),
                "n_train": len(y_train_raw),
                "n_test": len(y_test_raw),
            }

            # Plot RF predictions
            y_pred_rf_all = rf.predict(x_base)
            plot_prediction_scatter(
                y_rows, list(y_pred_rf_all),
                output_dir / "rf_scatter.png",
                title=f"Random Forest (R²={rf_r2:.3f})"
            )

            # Model comparison summary
            print("\n=== Model Comparison ===")
            print(f"  Ridge Regression:  MAE={mae:.3f}s, R²={r2:.3f}")
            print(f"  Random Forest:     MAE={rf_mae:.3f}s, R²={rf_r2:.3f}")
            improvement = ((mae - rf_mae) / mae) * 100 if mae > 0 else 0
            print(f"  RF MAE improvement: {improvement:.1f}%")
        else:
            print("\n  [Note] sklearn not installed; skipping Random Forest.")
            rf_metrics = {"note": "sklearn not installed"}

    # Generate report
    print("\n=== Generating Report ===")
    # mg1 MAE: compute only where M/G/1 is defined (rho<1).
    mae_values = [
        abs(a - p) for a, p in zip(actual_waits, mg1_predicted) if p is not None
    ]
    mg1_mae = statistics.fmean(mae_values) if mae_values else None

    # FCFS baselines
    fcfs_sample_mae = None
    sample_pairs = [(a, p) for a, p in zip(actual_waits, fcfs_sample_predicted) if p is not None]
    if sample_pairs:
        fcfs_sample_mae = statistics.fmean(abs(a - p) for a, p in sample_pairs)

    fcfs_mc_mae = statistics.fmean(abs(a - p) for a, p in zip(actual_waits, fcfs_mc_predicted) if p is not None)

    mg1_results = {
        'loads': loads,
        'actual_waits': actual_waits,
        'predicted_waits': mg1_predicted,
        'arrival_rates': arrival_rates,
        'rho_effective': rho_effective,
        'expected_service_time': expected_service_times,
        'mean_service_time': mean_S,
        'var_service_time': var_S,
        'residuals': residuals,
        'mean_absolute_error': mg1_mae,
        'fcfs_sample_path_predicted_waits': fcfs_sample_predicted,
        'fcfs_sample_path_mae': fcfs_sample_mae,
        'fcfs_monte_carlo_predicted_waits': fcfs_mc_predicted,
        'fcfs_monte_carlo_mae': fcfs_mc_mae,
        'notes': [
            "M/G/1 steady-state expectation is only defined for rho<1 (unstable points are null).",
            "If exp2 uses few jobs (e.g., 80) and a single seed, sample variance can make any expectation look 'off'.",
            "FCFS baselines are included to show finite-horizon behavior and reduce dependence on arbitrary caps.",
        ],
    }
    generate_model_report(mg1_results, reg_metrics, rf_metrics, output_dir / "exp5_report.json")

    print(f"\n=== Summary ===")
    if mg1_mae is None:
        print("M/G/1 MAE: N/A (no stable rho<1 points)")
    else:
        print(f"M/G/1 MAE (stable only): {mg1_mae:.3f}s")
    if fcfs_sample_mae is not None:
        print(f"FCFS sample-path MAE: {fcfs_sample_mae:.3f}s")
    print(f"FCFS Monte Carlo MAE: {fcfs_mc_mae:.3f}s")
    if reg_metrics:
        print(f"Ridge Regression: MAE={reg_metrics.get('mae', 'N/A'):.3f}s, R²={reg_metrics.get('r2', 'N/A'):.3f}")
    if rf_metrics and rf_metrics.get('r2') is not None:
        print(f"Random Forest:    MAE={rf_metrics.get('mae', 'N/A'):.3f}s, R²={rf_metrics.get('r2', 'N/A'):.3f}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()

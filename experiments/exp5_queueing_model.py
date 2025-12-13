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
        --output_dir results/experiments/exp5_model
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils.result_aggregator import (
    load_json_summary,
    load_csv_results,
    find_result_files,
)


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
        return float('inf')  # System unstable

    # Pollaczek-Khinchin formula
    second_moment = var_service + mean_service ** 2
    expected_wait = (arrival_rate * second_moment) / (2 * (1 - rho))
    return expected_wait


def extract_queueing_params(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract queueing parameters from job-level data.

    Args:
        df: DataFrame with job-level data

    Returns:
        Dict with arrival_rate, mean_service, var_service
    """
    # Estimate arrival rate from inter-arrival times
    if 'arrival_time' in df.columns:
        arrival_times = df['arrival_time'].sort_values()
        inter_arrivals = arrival_times.diff().dropna()
        arrival_rate = 1.0 / inter_arrivals.mean() if len(inter_arrivals) > 0 else 0.5
    else:
        arrival_rate = 0.5  # Default

    # Service time statistics
    if 'actual_serving_time' in df.columns:
        mean_service = df['actual_serving_time'].mean()
        var_service = df['actual_serving_time'].var()
    else:
        mean_service = 2.0
        var_service = 1.0

    return {
        'arrival_rate': arrival_rate,
        'mean_service': mean_service,
        'var_service': var_service,
        'rho': arrival_rate * mean_service,
    }


def fit_regression_model(
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: List[str] = None
) -> Tuple[Ridge, Dict]:
    """
    Fit a regression model to predict waiting time.

    Args:
        features: Feature matrix (queue_length, predicted_S, k, etc.)
        targets: Target waiting times
        feature_names: Names of features for reporting

    Returns:
        Tuple of (fitted model, metrics dict)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'coefficients': dict(zip(feature_names or [], model.coef_)) if feature_names else list(model.coef_),
        'intercept': model.intercept_,
    }

    return model, metrics


def plot_prediction_scatter(
    actual: np.ndarray,
    predicted: np.ndarray,
    output_path: Path,
    title: str = "Model Prediction vs Actual"
) -> None:
    """
    Plot predicted vs actual scatter with y=x reference line.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(actual, predicted, alpha=0.5, s=30)

    # y=x reference line
    lims = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
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
    predicted_waits: List[float],
    output_path: Path
) -> None:
    """
    Plot M/G/1 model validation: predicted vs actual across loads.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(load_values, actual_waits, 'o-', label='Actual (simulation)', linewidth=2, markersize=8)
    ax.plot(load_values, predicted_waits, 's--', label='M/G/1 prediction', linewidth=2, markersize=8)

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
    residuals: List[float],
    output_path: Path
) -> None:
    """
    Plot residuals vs load to identify model limitations.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(load_values, residuals, width=0.08, color='#4a90d9', edgecolor='black')
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
    output_path: Path
) -> Dict:
    """Generate modeling report."""
    report = {
        'experiment': 'exp5_queueing_model',
        'purpose': 'Validate predictive models for waiting time',
        'mg1_model': mg1_results,
        'regression_model': regression_metrics,
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=float)
    print(f"Saved: {output_path}")
    return report


def load_exp2_results(input_dir: Path) -> Tuple[List[float], List[float], Dict]:
    """
    Load results from exp2_load_sweep directories.

    Args:
        input_dir: Path to exp2_load_sweep results

    Returns:
        Tuple of (loads, actual_waits, combined_data)
    """
    loads = []
    actual_waits = []
    all_jobs_data = []

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
        json_files = find_result_files(load_dir, "*_summary.json")
        if not json_files:
            continue

        summary = load_json_summary(json_files[0])
        run_metrics = summary.get('run_metrics', summary.get('overall_metrics', {}))
        avg_wait = run_metrics.get('avg_waiting_time', 0)

        loads.append(load_value)
        actual_waits.append(avg_wait)

        # Load CSV for detailed data
        csv_files = list(load_dir.glob("*_jobs.csv")) + list(load_dir.glob("*_fixed_jobs.csv"))
        if csv_files:
            df = pd.read_csv(csv_files[0])
            df['system_load'] = load_value
            all_jobs_data.append(df)

    combined_df = pd.concat(all_jobs_data, ignore_index=True) if all_jobs_data else pd.DataFrame()

    return loads, actual_waits, {'combined_df': combined_df}


def main():
    parser = argparse.ArgumentParser(description="Exp-5: Queueing Model Analysis")
    parser.add_argument("--input_dir", type=str, default="results/experiments/exp2_load_sweep")
    parser.add_argument("--output_dir", type=str, default="results/experiments/exp5_queueing")
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
    loads, actual_waits, data = load_exp2_results(input_dir)

    if not loads:
        print("\nWARNING: No exp2 results found. Using mock data for demonstration.")
        np.random.seed(42)
        loads = [0.5, 0.7, 0.9, 0.95]
        mean_S = 2.0
        var_S = 0.5
        actual_waits = []
        for rho in loads:
            if rho < 1.0:
                arrival_rate = rho / mean_S
                pred = mg1_expected_wait(arrival_rate, mean_S, var_S)
            else:
                pred = 50.0
            actual = pred * (1 + 0.1 * np.random.randn()) + 2 * rho
            actual_waits.append(actual)
        data = {'combined_df': pd.DataFrame()}
    else:
        print(f"\nLoaded {len(loads)} load configurations from exp2")

    # Estimate service time parameters from data
    combined_df = data.get('combined_df', pd.DataFrame())
    if not combined_df.empty and 'actual_serving_time' in combined_df.columns:
        mean_S = combined_df['actual_serving_time'].mean()
        var_S = combined_df['actual_serving_time'].var()
        print(f"  Service time: mean={mean_S:.3f}s, var={var_S:.3f}")
    elif not combined_df.empty and 'execution_duration' in combined_df.columns:
        mean_S = combined_df['execution_duration'].mean()
        var_S = combined_df['execution_duration'].var()
        print(f"  Service time (from execution_duration): mean={mean_S:.3f}s, var={var_S:.3f}")
    else:
        mean_S = 2.0
        var_S = 0.5
        print(f"  Using default service time: mean={mean_S:.3f}s, var={var_S:.3f}")

    # Compute M/G/1 predictions
    mg1_predicted = []
    for rho in loads:
        if rho < 1.0:
            arrival_rate = rho / mean_S
            pred = mg1_expected_wait(arrival_rate, mean_S, var_S)
        else:
            pred = 100.0  # Cap for stability
        mg1_predicted.append(pred)

    print(f"\n=== M/G/1 Validation ===")
    for i, rho in enumerate(loads):
        residual = actual_waits[i] - mg1_predicted[i]
        print(f"  ρ={rho:.2f}: Actual={actual_waits[i]:.2f}s, M/G/1={mg1_predicted[i]:.2f}s, Residual={residual:+.2f}s")

    # Generate plots
    print("\n=== Generating Plots ===")
    plot_mg1_validation(loads, actual_waits, mg1_predicted, output_dir / "mg1_validation.png")

    residuals = [a - p for a, p in zip(actual_waits, mg1_predicted)]
    plot_residual_vs_load(loads, residuals, output_dir / "residual_vs_load.png")

    # Regression model
    print("\n=== Regression Model ===")
    if not combined_df.empty and 'waiting_time' in combined_df.columns:
        # Build features from real data
        feature_cols = []
        if 'system_load' in combined_df.columns:
            feature_cols.append('system_load')
        if 'execution_duration' in combined_df.columns:
            feature_cols.append('execution_duration')
        elif 'predicted_service_time' in combined_df.columns:
            feature_cols.append('predicted_service_time')

        if feature_cols:
            features = combined_df[feature_cols].values
            targets = combined_df['waiting_time'].values

            # Remove NaN
            mask = ~np.isnan(features).any(axis=1) & ~np.isnan(targets)
            features = features[mask]
            targets = targets[mask]

            if len(targets) > 50:
                model, reg_metrics = fit_regression_model(features, targets, feature_cols)
                print(f"  R2 Score: {reg_metrics['r2']:.3f}")
                print(f"  MAE: {reg_metrics['mae']:.3f}s")
                print(f"  Coefficients: {reg_metrics['coefficients']}")

                y_pred_all = model.predict(features)
                plot_prediction_scatter(targets, y_pred_all, output_dir / "regression_scatter.png",
                                        title=f"Regression Model (R²={reg_metrics['r2']:.3f})")
            else:
                print("  Not enough data for regression model")
                reg_metrics = {}
        else:
            print("  No suitable features found for regression")
            reg_metrics = {}
    else:
        print("  No job-level data available for regression model")
        # Use synthetic demonstration
        np.random.seed(42)
        n_samples = 200
        features = np.column_stack([
            np.random.uniform(0.5, 0.95, n_samples),  # system_load
            np.random.uniform(1, 4, n_samples),   # predicted_S
        ])
        targets = 5 + 20 * features[:, 0] + 1.5 * features[:, 1] + np.random.randn(n_samples) * 2

        model, reg_metrics = fit_regression_model(features, targets, ['system_load', 'predicted_S'])
        print(f"  (Synthetic demo) R2 Score: {reg_metrics['r2']:.3f}")
        y_pred_all = model.predict(features)
        plot_prediction_scatter(targets, y_pred_all, output_dir / "regression_scatter.png",
                                title="Regression Model (Synthetic Demo)")

    # Generate report
    print("\n=== Generating Report ===")
    mg1_results = {
        'loads': loads,
        'actual_waits': actual_waits,
        'predicted_waits': mg1_predicted,
        'mean_service_time': mean_S,
        'var_service_time': var_S,
        'residuals': residuals,
        'mean_absolute_error': np.mean(np.abs(residuals)),
    }
    generate_model_report(mg1_results, reg_metrics, output_dir / "exp5_report.json")

    print(f"\n=== Summary ===")
    print(f"M/G/1 MAE: {np.mean(np.abs(residuals)):.3f}s")
    if reg_metrics:
        print(f"Regression R²: {reg_metrics.get('r2', 'N/A')}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()

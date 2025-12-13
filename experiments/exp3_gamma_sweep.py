#!/usr/bin/env python3
"""
Exp-3: gamma_panic (DUAL_CHANNEL) Sweep

Purpose:
- Demonstrate DUAL_CHANNEL value: gamma_panic controls Panic channel weight
- Show how gamma_panic affects per-quadrant waiting times
- Verify Depression isn't significantly harmed when increasing Panic priority

Parameters:
- gamma_panic ∈ {0.0, 0.15, 0.3, 0.5}
- Fixed: k=4.0, w_max=2.0, gamma_dep=1.0, system_load=0.9

Usage:
    bash scripts/run_exp3_gamma_sweep.sh
    python experiments/exp3_gamma_sweep.py --input_dir results/experiments/exp3_gamma_sweep
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils.result_aggregator import (
    load_json_summary,
    find_result_files,
    extract_quadrant_metrics,
)


GAMMA_VALUES = [0.0, 0.15, 0.3, 0.5]
QUADRANTS = ['depression', 'panic', 'excited', 'calm']
QUADRANT_COLORS = {
    'depression': '#ff6b6b',
    'panic': '#ff9f43',
    'excited': '#4a90d9',
    'calm': '#6bcf7f',
}


def load_gamma_sweep_results(input_dir: Path) -> Dict[float, Dict]:
    """Load results for each gamma_panic value."""
    results = {}
    for gamma in GAMMA_VALUES:
        patterns = [f"gamma{gamma}", f"gamma_{gamma}", f"gamma={gamma}"]
        for pattern in patterns:
            subdir = input_dir / pattern
            if subdir.exists():
                json_files = find_result_files(subdir, "*_summary.json")
                if json_files:
                    results[gamma] = load_json_summary(json_files[0])
                    break
    return results


def extract_quadrant_waits(results: Dict[float, Dict]) -> pd.DataFrame:
    """Extract per-quadrant waiting times for each gamma value."""
    data = {}
    for gamma, summary in sorted(results.items()):
        data[gamma] = {}
        for quadrant in QUADRANTS:
            data[gamma][quadrant] = extract_quadrant_metrics(
                summary, quadrant, 'avg_waiting_time'
            )
    return pd.DataFrame(data).T


def plot_quadrant_bars(df: pd.DataFrame, output_path: Path) -> None:
    """Plot grouped bar chart: gamma_panic vs quadrant wait times."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(df.index))
    width = 0.2
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for i, quadrant in enumerate(QUADRANTS):
        ax.bar(x + offsets[i] * width, df[quadrant], width,
               label=quadrant.capitalize(), color=QUADRANT_COLORS[quadrant])

    ax.set_xlabel('gamma_panic', fontsize=12)
    ax.set_ylabel('Average Waiting Time (seconds)', fontsize=12)
    ax.set_title('Per-Quadrant Waiting Time vs gamma_panic\n(DUAL_CHANNEL Mode)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{g:.2f}' for g in df.index])
    ax.legend(title='Quadrant')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_panic_depression_tradeoff(df: pd.DataFrame, output_path: Path) -> None:
    """Plot Panic vs Depression waiting time trade-off."""
    fig, ax = plt.subplots(figsize=(10, 6))

    gammas = df.index.tolist()
    ax.plot(gammas, df['depression'], 'o-', label='Depression', color='#ff6b6b', linewidth=2, markersize=8)
    ax.plot(gammas, df['panic'], 's--', label='Panic', color='#ff9f43', linewidth=2, markersize=8)

    ax.set_xlabel('gamma_panic', fontsize=12)
    ax.set_ylabel('Average Waiting Time (seconds)', fontsize=12)
    ax.set_title('Panic vs Depression Waiting Time Trade-off', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.annotate('gamma_panic↑ → Panic wait↓\nbut Depression should not rise much',
                xy=(0.3, df.loc[0.3, 'panic']), xytext=(0.35, df.loc[0.3, 'panic'] * 1.3),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_gamma_report(results: Dict[float, Dict], output_path: Path) -> Dict:
    """Generate JSON report."""
    df = extract_quadrant_waits(results)

    report = {
        'experiment': 'exp3_gamma_sweep',
        'purpose': 'Demonstrate DUAL_CHANNEL gamma_panic effect',
        'gamma_values': list(results.keys()),
        'quadrant_waits': df.to_dict(orient='index'),
        'findings': {
            'depression_stable': df['depression'].std() < 2.0,  # Check stability
            'panic_improvement': df.loc[0.0, 'panic'] - df.loc[0.5, 'panic'] if 0.0 in df.index and 0.5 in df.index else None,
        }
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=float)
    print(f"Saved: {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Exp-3: gamma_panic Sweep Analysis")
    parser.add_argument("--input_dir", type=str, default="results/experiments/exp3_gamma_sweep")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading gamma sweep results from: {input_dir}")
    results = load_gamma_sweep_results(input_dir)

    if not results:
        print(f"Error: No results found. Expected: gamma0.0/, gamma0.15/, gamma0.3/, gamma0.5/")
        sys.exit(1)

    print(f"Found gamma values: {sorted(results.keys())}")

    df = extract_quadrant_waits(results)
    print("\n=== Quadrant Waiting Times ===")
    print(df.to_string())

    print("\n=== Generating Plots ===")
    plot_quadrant_bars(df, output_dir / "quadrant_wait_vs_gamma.png")
    plot_panic_depression_tradeoff(df, output_dir / "panic_depression_tradeoff.png")

    print("\n=== Generating Report ===")
    generate_gamma_report(results, output_dir / "exp3_report.json")


if __name__ == "__main__":
    main()

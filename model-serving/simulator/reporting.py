import os
from typing import List

import numpy as np

from analysis.fairness_metrics import analyze_fairness_comprehensive
from analysis.logger import EmotionAwareLogger
from core.job import Job


def print_summary_metrics(completed_jobs: List[Job], run_metrics: dict) -> None:
    """
    Print overall performance and latency metrics.
    """
    waiting_times = [
        j.waiting_duration
        for j in completed_jobs
        if j.waiting_duration is not None
    ]
    turnaround_times = [
        (j.completion_time - j.arrival_time)
        for j in completed_jobs
        if j.completion_time is not None
    ]

    print(f"\n" + "=" * 80)
    print("Results (Fixed-Rate Arrival)")
    print("=" * 80)

    print(f"\nOverall Performance Metrics:")
    print(f"  Total completed jobs: {run_metrics['total_jobs']}")
    print(
        f"  Jobs by deadline ({run_metrics['simulation_time']:.2f}s): "
        f"{run_metrics['jobs_by_deadline']}"
    )
    print(f"  Jobs after deadline: {run_metrics['jobs_after_deadline']}")
    print(f"  Total run time: {run_metrics['total_time']:.2f}s")
    print(
        f"  Effective throughput (by deadline): "
        f"{run_metrics['effective_throughput']:.3f} jobs/sec"
    )
    print(
        f"  Total throughput: {run_metrics['total_throughput']:.3f} jobs/sec"
    )

    print(f"\nLatency Metrics:")
    if waiting_times:
        print(f"  Avg waiting time: {np.mean(waiting_times):.3f}")
        print(f"  Std waiting time: {np.std(waiting_times):.3f}")
        print(f"  P50 waiting time: {np.percentile(waiting_times, 50):.3f}")
        print(f"  P95 waiting time: {np.percentile(waiting_times, 95):.3f}")
        print(f"  P99 waiting time: {np.percentile(waiting_times, 99):.3f}")
    else:
        print("  No waiting time data available")

    print(f"\nTurnaround Time:")
    if turnaround_times:
        print(f"  Avg: {np.mean(turnaround_times):.3f}")
        print(f"  P99: {np.percentile(turnaround_times, 99):.3f}")
    else:
        print("  No turnaround time data available")


def print_fairness_analysis(completed_jobs: List[Job]) -> None:
    """
    Print fairness analysis based on completed jobs.
    """
    print(f"\nFairness Analysis:")
    fairness_analysis = analyze_fairness_comprehensive(completed_jobs)

    waiting_fairness = fairness_analysis["waiting_time_fairness"]
    print(f"  Waiting Time Fairness:")
    print(f"    Jain Index: {waiting_fairness['jain_index']:.4f}")
    print(f"    CV: {waiting_fairness['coefficient_of_variation']:.4f}")
    print(f"    Max/Min Ratio: {waiting_fairness['max_min_ratio']:.4f}")

    print(f"\n  Per-Emotion-Class Waiting Time:")
    for emotion_class, avg_wait in waiting_fairness["per_class_values"].items():
        print(f"    {emotion_class}: {avg_wait:.3f}")

    per_class = fairness_analysis["per_class_metrics"]
    print(f"\nPer-Emotion-Class Detailed Metrics:")
    print(
        f"  {'Class':<10} {'Count':<8} "
        f"{'Avg Wait':<12} {'P99 Wait':<12} {'Avg Service':<12}"
    )
    print(f"  {'-'*60}")
    for emotion_class, metrics in per_class.items():
        if emotion_class != "overall":
            print(
                f"  {emotion_class:<10} {metrics['count']:<8} "
                f"{metrics['avg_waiting_time']:<12.3f} "
                f"{metrics['p99_waiting_time']:<12.3f} "
                f"{metrics['avg_execution_time']:<12.3f}"
            )


def save_results(
    args,
    config,
    completed_jobs: List[Job],
    run_metrics: dict,
    arrival_rate: float,
) -> None:
    """
    Save experiment results to disk using EmotionAwareLogger.
    """
    output_dir = args.output_dir if args.output_dir is not None else config.output.results_dir
    if not output_dir:
        return

    print(f"\nSaving results to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    logger = EmotionAwareLogger(
        output_dir=output_dir,
        experiment_name=(
            f"{config.scheduler.algorithm}_"
            f"{run_metrics['total_jobs']}jobs_"
            f"load{config.scheduler.system_load:.2f}_"
            f"time{config.experiment.simulation_time:.0f}s"
        ),
    )

    metadata = vars(args).copy()
    metadata["arrival_rate"] = arrival_rate
    metadata["run_metrics"] = run_metrics
    logger.set_metadata(metadata)

    logger.log_jobs_batch(completed_jobs)
    logger.save_job_logs()
    logger.save_summary_statistics(completed_jobs)

    print("Results saved successfully!")


def report_and_save_results(
    args,
    config,
    completed_jobs: List[Job],
    run_metrics: dict,
    arrival_rate: float,
) -> None:
    """
    High-level helper to print and persist experiment results.
    """
    print_summary_metrics(completed_jobs, run_metrics)
    print_fairness_analysis(completed_jobs)
    save_results(args, config, completed_jobs, run_metrics, arrival_rate)


__all__ = [
    "print_summary_metrics",
    "print_fairness_analysis",
    "save_results",
    "report_and_save_results",
]

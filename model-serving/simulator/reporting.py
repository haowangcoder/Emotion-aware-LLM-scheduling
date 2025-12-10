import os
from typing import List

import numpy as np

from analysis.fairness_metrics import analyze_fairness_comprehensive
from analysis.logger import EmotionAwareLogger, percentile_throughput
from core.job import Job


def print_summary_metrics(completed_jobs: List[Job], run_metrics: dict) -> None:
    """
    Print overall performance and latency metrics for both fixed-jobs and time-window experiments.

    This function is defensive: it uses dict.get with defaults so that missing
    keys never cause a crash (e.g., when no jobs are completed).
    """
    print(f"\n" + "=" * 80)
    mode = "Fixed-Jobs" if "num_jobs" in run_metrics else "Time-Window"
    mode_info = (
        f"{run_metrics.get('total_jobs', len(completed_jobs))} jobs"
        if mode == "Fixed-Jobs"
        else f"{run_metrics.get('total_time', 0.0):.0f}s window"
    )
    print(f"Results ({mode} Mode: {mode_info})")
    print("=" * 80)

    # ---- Overall performance ----
    print(f"\nOverall Performance Metrics:")
    total_jobs = run_metrics.get("total_jobs", len(completed_jobs))
    total_time = run_metrics.get("total_time", 0.0)
    throughput = run_metrics.get("throughput", 0.0)

    print(f"  Total completed jobs: {total_jobs}")
    print(f"  Total run time: {total_time:.2f}s")
    print(f"  Throughput: {throughput:.3f} jobs/sec")

    # Percentile throughput based on completed_jobs
    thr_p25 = percentile_throughput(completed_jobs, 25)
    thr_p50 = percentile_throughput(completed_jobs, 50)
    thr_p75 = percentile_throughput(completed_jobs, 75)
    print(f"  Throughput P25: {thr_p25:.3f} jobs/sec")
    print(f"  Throughput P50: {thr_p50:.3f} jobs/sec")
    print(f"  Throughput P75: {thr_p75:.3f} jobs/sec")

    # ---- Latency metrics ----
    avg_wait = run_metrics.get("avg_waiting_time", 0.0)
    p50_wait = run_metrics.get("p50_waiting_time", 0.0)
    p95_wait = run_metrics.get("p95_waiting_time", 0.0)
    p99_wait = run_metrics.get("p99_waiting_time", 0.0)

    print(f"\nLatency Metrics:")
    print(f"  Avg waiting time: {avg_wait:.3f}s")
    print(f"  P50 waiting time: {p50_wait:.3f}s")
    print(f"  P95 waiting time: {p95_wait:.3f}s")
    print(f"  P99 waiting time: {p99_wait:.3f}s")

    # ---- JCT metrics ----
    avg_jct = run_metrics.get("avg_jct", 0.0)
    p50_jct = run_metrics.get("p50_jct", 0.0)
    p95_jct = run_metrics.get("p95_jct", 0.0)
    p99_jct = run_metrics.get("p99_jct", 0.0)

    print(f"\nJob Completion Time (JCT):")
    print(f"  Avg JCT: {avg_jct:.3f}s")
    print(f"  P50 JCT: {p50_jct:.3f}s")
    print(f"  P95 JCT: {p95_jct:.3f}s")
    print(f"  P99 JCT: {p99_jct:.3f}s")

    # ---- Affect-weighted JCT (AW-JCT) ----
    print_affect_weighted_jct(completed_jobs)


def print_affect_weighted_jct(completed_jobs: List[Job]) -> None:
    """
    Print Affect-Weighted JCT (AW-JCT) metric.

    AW-JCT = sum(w_i * JCT_i) / sum(w_i)

    This weights job completion times by their affect weights,
    prioritizing depression-quadrant users in the aggregate metric.
    """
    if not completed_jobs:
        return

    weighted_jcts = []
    weights = []

    for job in completed_jobs:
        if job.completion_time is not None and job.arrival_time is not None:
            jct = job.completion_time - job.arrival_time
            weight = getattr(job, 'affect_weight', 1.0)
            weighted_jcts.append(weight * jct)
            weights.append(weight)

    if weights:
        aw_jct = sum(weighted_jcts) / sum(weights)
        print(f"\nAffect-Weighted JCT (AW-JCT):")
        print(f"  AW-JCT: {aw_jct:.3f}s")
        print(f"  Total weight: {sum(weights):.2f}")
        print(f"  Avg weight: {np.mean(weights):.3f}")


def print_fairness_analysis(completed_jobs: List[Job]) -> None:
    """
    Print fairness analysis based on completed jobs with Russell quadrant grouping.
    """
    print(f"\nFairness Analysis:")
    fairness_analysis = analyze_fairness_comprehensive(completed_jobs)

    waiting_fairness = fairness_analysis["waiting_time_fairness"]
    print(f"  Waiting Time Fairness:")
    print(f"    Jain Index: {waiting_fairness['jain_index']:.4f}")
    print(f"    CV: {waiting_fairness['coefficient_of_variation']:.4f}")
    print(f"    Max/Min Ratio: {waiting_fairness['max_min_ratio']:.4f}")

    # Per-quadrant waiting time analysis
    print_per_quadrant_analysis(completed_jobs)

    # Per-class detailed metrics (legacy compatibility)
    per_class = fairness_analysis.get("per_class_metrics", {})
    if per_class:
        print(f"\nPer-Emotion-Class Detailed Metrics:")
        print(
            f"  {'Class':<12} {'Count':<8} "
            f"{'Avg Wait':<12} {'P99 Wait':<12} {'Avg Service':<12}"
        )
        print(f"  {'-'*60}")
        for emotion_class, metrics in per_class.items():
            if emotion_class != "overall":
                avg_service = metrics.get('avg_predicted_service_time', metrics.get('avg_execution_time', 0))
                print(
                    f"  {emotion_class:<12} {metrics['count']:<8} "
                    f"{metrics['avg_waiting_time']:<12.3f} "
                    f"{metrics['p99_waiting_time']:<12.3f} "
                    f"{avg_service:<12.3f}"
                )


def print_per_quadrant_analysis(completed_jobs: List[Job]) -> None:
    """
    Print per-Russell-quadrant analysis of waiting times and JCT.
    """
    if not completed_jobs:
        return

    # Group jobs by Russell quadrant
    quadrant_jobs = {
        'excited': [],
        'calm': [],
        'panic': [],
        'depression': [],
    }

    for job in completed_jobs:
        quadrant = getattr(job, 'russell_quadrant', None)
        if quadrant in quadrant_jobs:
            quadrant_jobs[quadrant].append(job)

    print(f"\nPer-Russell-Quadrant Analysis:")
    print(f"  {'Quadrant':<12} {'Count':<8} {'Avg Wait':<12} {'Avg JCT':<12} {'Avg Weight':<12}")
    print(f"  {'-'*60}")

    for quadrant in ['excited', 'calm', 'panic', 'depression']:
        jobs = quadrant_jobs[quadrant]
        if jobs:
            wait_times = [j.waiting_duration for j in jobs if j.waiting_duration is not None]
            jcts = [j.completion_time - j.arrival_time for j in jobs
                    if j.completion_time is not None and j.arrival_time is not None]
            weights = [getattr(j, 'affect_weight', 1.0) for j in jobs]

            avg_wait = np.mean(wait_times) if wait_times else 0
            avg_jct = np.mean(jcts) if jcts else 0
            avg_weight = np.mean(weights) if weights else 1.0

            print(f"  {quadrant:<12} {len(jobs):<8} {avg_wait:<12.3f} {avg_jct:<12.3f} {avg_weight:<12.3f}")
        else:
            print(f"  {quadrant:<12} {0:<8} {'N/A':<12} {'N/A':<12} {'N/A':<12}")

    # Compute inter-quadrant fairness
    quadrant_avg_waits = []
    for quadrant, jobs in quadrant_jobs.items():
        if jobs:
            wait_times = [j.waiting_duration for j in jobs if j.waiting_duration is not None]
            if wait_times:
                quadrant_avg_waits.append(np.mean(wait_times))

    if len(quadrant_avg_waits) >= 2:
        jain_index = compute_jain_index(quadrant_avg_waits)
        cv = np.std(quadrant_avg_waits) / np.mean(quadrant_avg_waits) if np.mean(quadrant_avg_waits) > 0 else 0
        print(f"\n  Inter-Quadrant Fairness:")
        print(f"    Jain Index: {jain_index:.4f}")
        print(f"    CV: {cv:.4f}")


def compute_jain_index(values: List[float]) -> float:
    """Compute Jain's fairness index for a list of values."""
    if not values or len(values) == 0:
        return 1.0
    n = len(values)
    sum_values = sum(values)
    sum_squares = sum(v ** 2 for v in values)
    if sum_squares == 0:
        return 1.0
    return (sum_values ** 2) / (n * sum_squares)


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

    mode = config.experiment.mode
    if mode == "fixed_jobs":
        exp_suffix = f"{run_metrics['total_jobs']}jobs_load{config.scheduler.system_load:.2f}_fixed"
    else:
        exp_suffix = f"{int(config.experiment.simulation_duration)}s_load{config.scheduler.system_load:.2f}_window"

    logger = EmotionAwareLogger(
        output_dir=output_dir,
        experiment_name=f"{config.scheduler.algorithm}_{exp_suffix}",
    )

    # Set metadata with experiment mode
    metadata = vars(args).copy()
    metadata["experiment_mode"] = mode
    if mode == "time_window":
        metadata["simulation_duration"] = config.experiment.simulation_duration
    metadata["num_jobs"] = run_metrics["total_jobs"]
    metadata["arrival_rate"] = arrival_rate
    metadata["run_metrics"] = run_metrics

    # Add affect weight parameters for AW-SSJF and Weight-Only
    if config.scheduler.algorithm in ("AW-SSJF", "Weight-Only"):
        affect_cfg = config.scheduler.affect_weight
        metadata["w_max"] = affect_cfg.w_max
        metadata["p"] = affect_cfg.p
        metadata["q"] = affect_cfg.q
        metadata["use_confidence"] = affect_cfg.use_confidence

    # Add percentile throughput to metrics
    thr_p25 = percentile_throughput(completed_jobs, 25)
    thr_p50 = percentile_throughput(completed_jobs, 50)
    thr_p75 = percentile_throughput(completed_jobs, 75)
    run_metrics["throughput_p25"] = thr_p25
    run_metrics["throughput_p50"] = thr_p50
    run_metrics["throughput_p75"] = thr_p75

    # Add AW-JCT metric
    weighted_jcts = []
    weights = []
    for job in completed_jobs:
        if job.completion_time is not None and job.arrival_time is not None:
            jct = job.completion_time - job.arrival_time
            weight = getattr(job, 'affect_weight', 1.0)
            weighted_jcts.append(weight * jct)
            weights.append(weight)
    if weights:
        run_metrics["aw_jct"] = sum(weighted_jcts) / sum(weights)

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
    "print_per_quadrant_analysis",
    "print_affect_weighted_jct",
    "save_results",
    "report_and_save_results",
]

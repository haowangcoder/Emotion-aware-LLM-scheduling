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



def print_fairness_analysis(completed_jobs: List[Job], valence_beta: float = None) -> None:
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
            # Use new key name (avg_predicted_service_time), fallback to old key for backwards compatibility
            avg_service = metrics.get('avg_predicted_service_time', metrics.get('avg_execution_time', 0))
            print(
                f"  {emotion_class:<10} {metrics['count']:<8} "
                f"{metrics['avg_waiting_time']:<12.3f} "
                f"{metrics['p99_waiting_time']:<12.3f} "
                f"{avg_service:<12.3f}"
            )

    # Optional Phase II: valence-weighted fairness
    if valence_beta is not None:
        from analysis.fairness_metrics import calculate_valence_fairness

        valence_fairness = calculate_valence_fairness(
            completed_jobs, beta=valence_beta, metric="waiting_time"
        )
        if valence_fairness.get("per_valence_values"):
            print(f"\nValence-weighted Fairness (β={valence_beta}):")
            print(f"  Weighted Jain Index: {valence_fairness['weighted_jain_index']:.4f}")
            print(f"  Weights: {valence_fairness['weights']}")
            for vcls, val in valence_fairness["per_valence_values"].items():
                print(f"    {vcls}: avg_wait={val:.3f}")


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

    # logger = EmotionAwareLogger(
    #     output_dir=output_dir,
    #     experiment_name=(
    #         f"{config.scheduler.algorithm}_"
    #         f"{run_metrics['total_jobs']}jobs_"
    #         f"load{config.scheduler.system_load:.2f}_"
    #         f"fixedjobs"
    #     ),
    # )

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
    # metadata["experiment_mode"] = "fixed_jobs"
    metadata["experiment_mode"] = mode
    if mode == "time_window":
        metadata["simulation_duration"] = config.experiment.simulation_duration
    metadata["num_jobs"] = run_metrics["total_jobs"]
    metadata["arrival_rate"] = arrival_rate
    metadata["run_metrics"] = run_metrics
    if config.scheduler.algorithm in ("SSJF-Valence", "SSJF-Combined"):
        metadata["valence_beta"] = config.scheduler.valence_priority.beta

    # Add percentile throughput to metrics
    thr_p25 = percentile_throughput(completed_jobs, 25)
    thr_p50 = percentile_throughput(completed_jobs, 50)
    thr_p75 = percentile_throughput(completed_jobs, 75)
    run_metrics["throughput_p25"] = thr_p25
    run_metrics["throughput_p50"] = thr_p50
    run_metrics["throughput_p75"] = thr_p75

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
    valence_beta = None
    try:
        if getattr(config.scheduler, "algorithm", "") == "SSJF-Valence":
            valence_beta = getattr(config.scheduler.valence_priority, "beta", None)
    except Exception:
        valence_beta = None
    print_fairness_analysis(completed_jobs, valence_beta=valence_beta)
    save_results(args, config, completed_jobs, run_metrics, arrival_rate)


__all__ = [
    "print_summary_metrics",
    "print_fairness_analysis",
    "save_results",
    "report_and_save_results",
]

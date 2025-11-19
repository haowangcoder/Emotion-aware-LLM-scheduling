"""
Emotion-aware Logging Module for LLM Scheduling

This module provides logging functionality for emotion-aware job scheduling experiments.
It records detailed information about each job including emotion attributes, timing data,
and performance metrics to CSV files for later analysis and visualization.
"""

import csv
import os
import json
import math
import copy
from typing import List, Dict, Optional
from datetime import datetime

import numpy as np

from core.job import Job
from analysis.fairness_metrics import (
    calculate_per_class_metrics,
    calculate_fairness_across_emotions,
    analyze_fairness_comprehensive
)


def percentile_throughput(completed_jobs: List[Job], percentage: float = 25.0) -> float:
    """
    Calculate throughput at a given percentile of job completions.

    This metric measures the throughput when a certain percentage of jobs have completed,
    useful for comparing how quickly different schedulers deliver early results.

    Args:
        completed_jobs: List of completed jobs with completion_time set
        percentage: The percentile to calculate (e.g., 25 for first quartile)

    Returns:
        Throughput (jobs/sec) at the given percentile
    """
    finish_times = [j.completion_time for j in completed_jobs if j.completion_time is not None]
    if not finish_times:
        return 0.0

    finish_times.sort()
    n = len(finish_times)

    # Calculate the index for the given percentile
    idx_list = list(range(n))
    pos = int(np.percentile(idx_list, percentage))

    if pos <= 0 or finish_times[pos] <= 0:
        return 0.0

    # Throughput = number of jobs completed / time to complete them
    return (pos + 1) / finish_times[pos]


class EmotionAwareLogger:
    """
    Logger for emotion-aware scheduling experiments

    Logs job-level data and experiment-level statistics to CSV and JSON files
    """

    def __init__(self, output_dir: str = 'results/emotion_aware/',
                 experiment_name: str = None):
        """
        Initialize logger

        Args:
            output_dir: Directory to save log files
            experiment_name: Name of the experiment (used in filenames)
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Data storage
        self.job_logs = []
        self.experiment_metadata = {}

    def log_job(self, job: Job):
        """
        Log information about a single job

        Args:
            job: Completed Job object
        """
        # Calculate actual service time used for completion
        actual_service = job.actual_execution_duration if job.actual_execution_duration else job.execution_duration

        log_entry = {
            'job_id': job.job_id,
            'emotion_label': job.emotion_label,
            'arousal': job.arousal,
            'emotion_class': job.emotion_class,
            'arrival_time': job.arrival_time,
            'predicted_serving_time': job.execution_duration,
            'actual_serving_time': job.actual_execution_duration,
            'start_time': (job.completion_time - actual_service) if job.completion_time else None,
            'finish_time': job.completion_time,
            'waiting_time': job.waiting_duration,
            'turnaround_time': (job.completion_time - job.arrival_time) if job.completion_time else None,
        }

        # Add LLM inference fields if available
        if hasattr(job, 'response_text'):
            log_entry['response_text'] = job.response_text
            log_entry['output_token_length'] = job.output_token_length
            log_entry['cached'] = job.cached
            log_entry['error_msg'] = job.error_msg
            log_entry['fallback_used'] = job.fallback_used
            log_entry['model_name'] = job.model_name

            # Add conversation context (truncated for readability)
            if job.conversation_context:
                # Store first 200 chars of context as preview
                log_entry['conversation_context_preview'] = job.conversation_context[:200]

        self.job_logs.append(log_entry)

    def log_jobs_batch(self, job_list: List[Job]):
        """
        Log information about multiple jobs

        Args:
            job_list: List of Job objects
        """
        for job in job_list:
            self.log_job(job)

    def set_metadata(self, metadata: Dict):
        """
        Set experiment metadata

        Args:
            metadata: Dictionary with experiment configuration and parameters
        """
        self.experiment_metadata.update(metadata)

    def save_job_logs(self, filename: str = None) -> str:
        """
        Save job logs to CSV file

        Args:
            filename: Custom filename (optional)

        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"{self.experiment_name}_jobs.csv"

        filepath = os.path.join(self.output_dir, filename)

        if not self.job_logs:
            print(f"Warning: No job logs to save")
            return filepath

        # Write to CSV
        fieldnames = list(self.job_logs[0].keys())

        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.job_logs)

        print(f"Job logs saved to: {filepath}")
        return filepath

    def save_summary_statistics(self, job_list: List[Job], filename: str = None) -> str:
        """
        Calculate and save summary statistics

        Args:
            job_list: List of completed Job objects
            filename: Custom filename (optional)

        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"{self.experiment_name}_summary.json"

        filepath = os.path.join(self.output_dir, filename)

        # Calculate various statistics
        completed_jobs = [j for j in job_list if j.completion_time is not None]

        if not completed_jobs:
            print(f"Warning: No completed jobs to summarize")
            return filepath

        # Basic statistics
        waiting_times = [j.waiting_duration for j in completed_jobs if j.waiting_duration is not None]
        turnaround_times = [(j.completion_time - j.arrival_time) for j in completed_jobs]
        service_times = [j.execution_duration for j in completed_jobs]

        import numpy as np

        # Prepare metadata with strict-JSON friendly values
        # Do not mutate original metadata in-place
        sanitized_metadata = copy.deepcopy(self.experiment_metadata)
        try:
            st = sanitized_metadata.get('starvation_threshold', None)
            if isinstance(st, (int, float)) and not math.isfinite(float(st)):
                # Represent Infinity as string "inf" for clarity
                if math.isinf(st):
                    sanitized_metadata['starvation_threshold'] = "inf" if st > 0 else "-inf"
                elif math.isnan(st):
                    sanitized_metadata['starvation_threshold'] = "nan"
        except Exception:
            # If any issue occurs, fall back without special handling
            pass

        summary = {
            'experiment_name': self.experiment_name,
            'metadata': sanitized_metadata,
            'num_jobs': len(completed_jobs),
            'total_run_time': max([j.completion_time for j in completed_jobs]) if completed_jobs else 0,
            'overall_metrics': {
                'avg_waiting_time': float(np.mean(waiting_times)) if waiting_times else 0,
                'std_waiting_time': float(np.std(waiting_times)) if waiting_times else 0,
                'p50_waiting_time': float(np.percentile(waiting_times, 50)) if waiting_times else 0,
                'p95_waiting_time': float(np.percentile(waiting_times, 95)) if waiting_times else 0,
                'p99_waiting_time': float(np.percentile(waiting_times, 99)) if waiting_times else 0,
                'avg_turnaround_time': float(np.mean(turnaround_times)),
                'std_turnaround_time': float(np.std(turnaround_times)),
                'p99_turnaround_time': float(np.percentile(turnaround_times, 99)),
                'avg_service_time': float(np.mean(service_times)),
                'throughput': len(completed_jobs) / max([j.completion_time for j in completed_jobs]) if completed_jobs else 0,
            }
        }

        # Add run_metrics if available (Fixed-rate arrival metrics)
        if 'run_metrics' in sanitized_metadata:
            summary['run_metrics'] = sanitized_metadata['run_metrics']

        # Add LLM-specific metrics if using real model inference
        if completed_jobs and hasattr(completed_jobs[0], 'response_text'):
            # Collect LLM-related data
            jobs_with_llm = [j for j in completed_jobs if j.response_text is not None]

            if jobs_with_llm:
                actual_times = [j.actual_execution_duration for j in jobs_with_llm if j.actual_execution_duration is not None]
                output_lengths = [j.output_token_length for j in jobs_with_llm if j.output_token_length is not None]
                cached_count = sum(1 for j in jobs_with_llm if j.cached)
                fallback_count = sum(1 for j in jobs_with_llm if j.fallback_used)
                error_count = sum(1 for j in jobs_with_llm if j.error_msg is not None)

                llm_metrics = {
                    'num_llm_jobs': len(jobs_with_llm),
                    'avg_actual_execution_time': float(np.mean(actual_times)) if actual_times else 0,
                    'std_actual_execution_time': float(np.std(actual_times)) if actual_times else 0,
                    'avg_output_token_length': float(np.mean(output_lengths)) if output_lengths else 0,
                    'std_output_token_length': float(np.std(output_lengths)) if output_lengths else 0,
                    'cache_hit_count': cached_count,
                    'cache_hit_rate': cached_count / len(jobs_with_llm) if jobs_with_llm else 0,
                    'fallback_used_count': fallback_count,
                    'error_count': error_count,
                }

                # Prediction accuracy metrics using execution_duration as predicted value
                predicted_from_execution = [j.execution_duration for j in jobs_with_llm if j.actual_execution_duration is not None]
                actual_for_comparison = [j.actual_execution_duration for j in jobs_with_llm if j.actual_execution_duration is not None]

                if actual_for_comparison and predicted_from_execution and len(actual_for_comparison) == len(predicted_from_execution):
                    prediction_errors = [abs(a - p) for a, p in zip(actual_for_comparison, predicted_from_execution)]
                    relative_errors = [abs(a - p) / a if a > 0 else 0 for a, p in zip(actual_for_comparison, predicted_from_execution)]

                    llm_metrics['prediction_accuracy'] = {
                        'avg_absolute_error': float(np.mean(prediction_errors)),
                        'std_absolute_error': float(np.std(prediction_errors)),
                        'avg_relative_error': float(np.mean(relative_errors)),
                        'median_relative_error': float(np.median(relative_errors)),
                        'max_absolute_error': float(np.max(prediction_errors)),
                    }

                summary['llm_metrics'] = llm_metrics

        # Per-emotion-class metrics
        per_class = calculate_per_class_metrics(completed_jobs, completed_only=True)
        # Convert numpy types to Python types for JSON serialization
        per_class_serializable = {}
        for emotion_class, metrics in per_class.items():
            per_class_serializable[emotion_class] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in metrics.items()
            }
        summary['per_emotion_class_metrics'] = per_class_serializable

        # Fairness analysis
        fairness_analysis = analyze_fairness_comprehensive(completed_jobs)
        # Convert to serializable format
        fairness_serializable = {}
        for key, value in fairness_analysis.items():
            if isinstance(value, dict):
                fairness_serializable[key] = {
                    k: float(v) if isinstance(v, (np.floating, np.integer))
                    else (dict(v) if isinstance(v, dict) else v)
                    for k, v in value.items()
                }
            else:
                fairness_serializable[key] = value
        summary['fairness_analysis'] = fairness_serializable

        # Generic sanitizer to ensure strict JSON (convert NaN/Infinity to strings)
        def _sanitize(obj):
            # Numbers
            if isinstance(obj, float):
                if math.isfinite(obj):
                    return obj
                elif math.isinf(obj):
                    return "inf" if obj > 0 else "-inf"
                else:  # NaN
                    return "nan"
            # Ints are always finite
            if isinstance(obj, int):
                return obj
            # Containers
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple, set)):
                return [_sanitize(v) for v in list(obj)]
            # Numpy types and arrays
            try:
                import numpy as _np  # local import to avoid top-level dependency
                if isinstance(obj, (_np.floating, _np.integer)):
                    val = float(obj)
                    if math.isfinite(val):
                        return val
                    elif math.isinf(val):
                        return "inf" if val > 0 else "-inf"
                    else:  # NaN
                        return "nan"
                if isinstance(obj, _np.ndarray):
                    return [_sanitize(v) for v in obj.tolist()]
            except Exception:
                pass
            # Everything else unchanged
            return obj

        summary_strict = _sanitize(summary)

        # Save to JSON (disallow NaN/Infinity explicitly)
        with open(filepath, 'w') as jsonfile:
            json.dump(summary_strict, jsonfile, indent=2, allow_nan=False)

        print(f"Summary statistics saved to: {filepath}")
        return filepath

    def print_summary(self, job_list: List[Job]):
        """
        Print summary statistics to console

        Args:
            job_list: List of completed Job objects
        """
        completed_jobs = [j for j in job_list if j.completion_time is not None]

        if not completed_jobs:
            print("No completed jobs to summarize")
            return

        print("\n" + "=" * 70)
        print(f"Experiment Summary: {self.experiment_name}")
        print("=" * 70)

        # Overall metrics
        waiting_times = [j.waiting_duration for j in completed_jobs if j.waiting_duration is not None]
        turnaround_times = [(j.completion_time - j.arrival_time) for j in completed_jobs]

        import numpy as np

        print(f"\nOverall Metrics:")
        print(f"  Total jobs: {len(completed_jobs)}")
        print(f"  Avg waiting time: {np.mean(waiting_times):.3f}" if waiting_times else "  N/A")
        print(f"  P99 waiting time: {np.percentile(waiting_times, 99):.3f}" if waiting_times else "  N/A")
        print(f"  Avg turnaround time: {np.mean(turnaround_times):.3f}")
        print(f"  P99 turnaround time: {np.percentile(turnaround_times, 99):.3f}")
        print(f"  Throughput: {len(completed_jobs) / max([j.completion_time for j in completed_jobs]):.3f} jobs/sec"
              if completed_jobs else "  N/A")

        # Per-emotion-class metrics
        per_class = calculate_per_class_metrics(completed_jobs)

        print(f"\nPer-Emotion-Class Metrics:")
        print(f"  {'Class':<10} {'Count':<8} {'Avg Wait':<12} {'Avg Turnaround':<15}")
        print(f"  {'-'*50}")
        for emotion_class, metrics in per_class.items():
            if emotion_class != 'overall':
                print(f"  {emotion_class:<10} {metrics['count']:<8} "
                      f"{metrics['avg_waiting_time']:<12.3f} {metrics['avg_turnaround_time']:<15.3f}")

        # Fairness metrics
        fairness = calculate_fairness_across_emotions(completed_jobs, metric='waiting_time')
        print(f"\nFairness Metrics (Waiting Time):")
        print(f"  Jain Fairness Index: {fairness['jain_index']:.4f}")
        print(f"  Coefficient of Variation: {fairness['coefficient_of_variation']:.4f}")
        print(f"  Max/Min Ratio: {fairness['max_min_ratio']:.4f}")

        print("=" * 70 + "\n")

    def reset(self):
        """Clear all logged data"""
        self.job_logs = []
        self.experiment_metadata = {}


# Example usage and testing
if __name__ == '__main__':
    from job import Job
    import random

    print("=" * 70)
    print("Emotion-aware Logger Test")
    print("=" * 70)

    # Create test jobs with emotions
    random.seed(42)
    emotions = ['excited', 'sad', 'angry', 'calm', 'neutral'] * 4
    arousals = [0.9, -0.6, 0.8, -0.3, 0.0] * 4
    emotion_classes = ['high', 'low', 'high', 'low', 'medium'] * 4

    jobs = []
    current_time = 0

    for i in range(20):
        service_time = 2.0 * (1 + 0.5 * arousals[i])
        job = Job(
            job_id=i,
            execution_duration=service_time,
            arrival_time=i * 0.5,
            emotion_label=emotions[i],
            arousal=arousals[i],
            emotion_class=emotion_classes[i]
        )

        # Simulate execution
        waiting = max(0, current_time - job.arrival_time)
        job.waiting_duration = waiting
        job.completion_time = current_time + service_time
        current_time = job.completion_time

        jobs.append(job)

    # Test logger
    print("\n1. Creating logger and logging jobs")
    logger = EmotionAwareLogger(
        output_dir='results/test/',
        experiment_name='test_emotion_logging'
    )

    # Set metadata
    logger.set_metadata({
        'scheduler': 'SSJF-Emotion',
        'alpha': 0.5,
        'gamma': 0.3,
        'num_jobs': len(jobs)
    })

    # Log jobs
    logger.log_jobs_batch(jobs)
    print(f"   Logged {len(jobs)} jobs")

    # Save logs
    print("\n2. Saving logs to files")
    job_log_path = logger.save_job_logs()
    summary_path = logger.save_summary_statistics(jobs)

    # Print summary
    print("\n3. Printing summary to console")
    logger.print_summary(jobs)

    print("=" * 70)

from typing import List, Tuple

import numpy as np

from core.emotion import EmotionConfig
from core.job import Job
from workload.service_time_mapper import ServiceTimeConfig


def run_scheduling_loop(
    scheduler,
    arrival_rate: float,
    simulation_time: float,
    emotion_config: EmotionConfig,
    service_time_config: ServiceTimeConfig,
    enable_emotion: bool = True,
    verbose: bool = False,
    llm_handler=None,
    llm_skip_on_error: bool = True,
    pre_generated_jobs: List[Job] | None = None,
) -> Tuple[List[Job], dict]:
    """
    Run emotion-aware job scheduling loop with fixed-rate arrivals.

    This function implements the core simulation mechanics and is intentionally
    kept independent from configuration loading, logging, and LLM setup.
    """
    from workload.task_generator import generate_job_on_demand

    if verbose:
        print(f"\nStarting scheduling run with {scheduler.name} scheduler")
        print(f"  Arrival rate: {arrival_rate:.3f} req/sec")
        print(f"  Simulation time: {simulation_time:.2f}s")
        if pre_generated_jobs:
            print(f"  Using pre-generated jobs: {len(pre_generated_jobs)}")

    # Scheduling state
    current_time = 0.0
    waiting_queue: List[Job] = []
    completed_jobs: List[Job] = []
    next_job_id = 0
    next_arrival_time = 0.0

    # Track whether we've entered drain phase
    drain_phase_started = False

    # Pre-generated jobs setup
    if pre_generated_jobs:
        # Sort by arrival time to ensure correct order
        pre_generated_jobs = sorted(pre_generated_jobs, key=lambda j: j.arrival_time)
        job_index = 0
        total_jobs = len(pre_generated_jobs)
        if total_jobs > 0:
            next_arrival_time = pre_generated_jobs[0].arrival_time
    else:
        job_index = None
        total_jobs = None

    # Phase 1: Generate arrivals until simulation_time
    if verbose:
        print(f"\n=== Phase 1: Arrival Phase (0 - {simulation_time:.2f}s) ===")

    while current_time < simulation_time or waiting_queue:
        # Generate new arrivals if we're still in the arrival phase
        while current_time >= next_arrival_time and next_arrival_time < simulation_time:
            if pre_generated_jobs and job_index is not None and job_index < total_jobs:  # type: ignore[operator]
                # Use pre-generated job
                new_job = pre_generated_jobs[job_index]
                waiting_queue.append(new_job)

                if verbose and job_index % 50 == 0:
                    print(
                        f"  Time {new_job.arrival_time:.2f}: Job {new_job.job_id} arrived "
                        f"(emotion: {new_job.emotion_label}, queue: {len(waiting_queue)})"
                    )

                job_index += 1
                if job_index < total_jobs:  # type: ignore[operator]
                    next_arrival_time = pre_generated_jobs[job_index].arrival_time
                else:
                    next_arrival_time = simulation_time  # No more jobs
            else:
                # Generate job on-demand
                new_job = generate_job_on_demand(
                    job_id=next_job_id,
                    arrival_time=next_arrival_time,
                    emotion_config=emotion_config,
                    service_time_config=service_time_config,
                    enable_emotion=enable_emotion,
                )
                waiting_queue.append(new_job)

                if verbose and next_job_id % 50 == 0:
                    print(
                        f"  Time {next_arrival_time:.2f}: Job {next_job_id} arrived "
                        f"(emotion: {new_job.emotion_label}, queue: {len(waiting_queue)})"
                    )

                # Schedule next arrival using exponential distribution
                inter_arrival_time = np.random.exponential(1.0 / arrival_rate)
                next_arrival_time += inter_arrival_time
                next_job_id += 1

        # Check if we've transitioned to drain phase
        if current_time >= simulation_time and not drain_phase_started:
            # Mark the transition to drain phase
            drain_phase_started = True
            if verbose:
                print(f"\n=== Phase 2: Drain Phase (starting at {current_time:.2f}s) ===")
                print(f"  Jobs still waiting: {len(waiting_queue)}")

        # If queue is empty, advance time
        if not waiting_queue:
            if next_arrival_time < simulation_time:
                # Jump to next arrival
                current_time = next_arrival_time
            else:
                # No more arrivals and queue empty - done
                break
            continue

        # Schedule next job
        selected_job = scheduler.schedule(waiting_queue, current_time=current_time)

        if selected_job is None:
            print(
                f"Warning: Scheduler returned None with non-empty queue "
                f"at time {current_time}"
            )
            break

        # Remove from queue
        waiting_queue.remove(selected_job)

        # Execute job
        selected_job.status = "RUNNING"
        selected_job.waiting_duration = current_time - selected_job.arrival_time
        scheduler.on_job_scheduled(selected_job, current_time)

        # Use real LLM inference if handler provided
        if llm_handler is not None:
            if verbose and len(completed_jobs) % 10 == 0:
                print(
                    f"  Time {current_time:.2f}: Executing job {selected_job.job_id} "
                    f"with LLM (emotion: {selected_job.emotion_label}, "
                    f"predicted: {selected_job.execution_duration:.2f}s)"
                )

            # Execute with real LLM model
            success = llm_handler.execute_job(selected_job)

            if not success and llm_skip_on_error:
                # Skip this job if it failed
                if verbose:
                    print(
                        f"  WARNING: Job {selected_job.job_id} failed, skipping: "
                        f"{selected_job.error_msg}"
                    )
                continue
            elif not success:
                # Fail entire run
                print(f"ERROR: Job {selected_job.job_id} failed: {selected_job.error_msg}")
                print("Set LLM_SKIP_ON_ERROR=True to skip failed jobs instead")
                break

        # Advance time (uses actual LLM time if llm_handler was used)
        current_time += selected_job.execution_duration
        selected_job.completion_time = current_time
        selected_job.status = "COMPLETED"

        scheduler.on_job_completed(selected_job, current_time)
        completed_jobs.append(selected_job)

    # Calculate metrics for both phases
    # IMPORTANT: Only count jobs that completed within the time window
    # This correctly reflects the "effective throughput" in a fixed time window
    jobs_within_window = [
        j for j in completed_jobs if j.completion_time is not None and j.completion_time <= simulation_time
    ]
    jobs_after_window = [
        j for j in completed_jobs if j.completion_time is not None and j.completion_time > simulation_time
    ]

    total_jobs = len(completed_jobs)
    jobs_by_deadline_count = len(jobs_within_window)
    jobs_after_deadline = len(jobs_after_window)

    metrics = {
        "total_jobs": total_jobs,
        "jobs_by_deadline": jobs_by_deadline_count,
        "jobs_after_deadline": jobs_after_deadline,
        "simulation_time": simulation_time,
        "total_time": current_time,
        "effective_throughput": jobs_by_deadline_count / simulation_time
        if simulation_time > 0
        else 0,
        "total_throughput": total_jobs / current_time if current_time > 0 else 0,
        "arrival_rate": arrival_rate,
    }

    if verbose:
        print(f"\n=== Run Completed ===")
        print(f"  Total simulation time: {current_time:.2f}s")
        print(f"  Jobs completed by deadline: {jobs_by_deadline_count}")
        print(f"  Jobs completed after deadline: {jobs_after_deadline}")
        print(f"  Total completed jobs: {total_jobs}")
        print(f"  Effective throughput: {metrics['effective_throughput']:.3f} jobs/sec")

    return completed_jobs, metrics


__all__ = ["run_scheduling_loop"]


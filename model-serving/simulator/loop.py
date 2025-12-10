from typing import List, Tuple, Dict

import numpy as np

from core.job import Job


def run_scheduling_loop(
    scheduler,
    jobs: List[Job],
    verbose: bool = False,
    llm_handler=None,
    llm_skip_on_error: bool = True,
    early_prompt_generator=None,
) -> Tuple[List[Job], dict]:
    """
    Run fixed-num_jobs scheduling loop.

    This function implements the core simulation mechanics for fixed job count experiments.
    The loop continues until all jobs are completed, with no time window cutoff.

    Args:
        scheduler: Scheduler instance to use for job selection
        jobs: List of pre-generated jobs to schedule
        verbose: Whether to print progress information
        llm_handler: Optional LLM handler for real inference
        llm_skip_on_error: Whether to skip failed jobs or abort
        early_prompt_generator: Optional EarlyPromptGenerator for BERT prediction

    Returns:
        Tuple of (completed_jobs, metrics)
    """
    if not jobs:
        return [], {"total_jobs": 0, "total_time": 0.0, "throughput": 0.0}

    # Sort jobs by arrival time
    jobs = sorted(jobs, key=lambda j: j.arrival_time)
    total_jobs = len(jobs)

    if verbose:
        print(f"\nStarting scheduling run with {scheduler.name} scheduler")
        print(f"  Total jobs: {total_jobs}")

    # Scheduling state
    current_time = 0.0
    waiting_queue: List[Job] = []
    completed_jobs: List[Job] = []
    next_index = 0

    # Main scheduling loop - continue until all jobs completed
    while len(completed_jobs) < total_jobs:
        # Add arrived jobs to waiting queue
        while next_index < total_jobs and jobs[next_index].arrival_time <= current_time:
            new_job = jobs[next_index]

            # === Early prompt generation and BERT prediction ===
            # Check if job already has predicted_service_time (from trace)
            if new_job.predicted_service_time is None and early_prompt_generator is not None:
                prompt, predicted_time, conv_idx = early_prompt_generator.generate_prompt_and_predict(new_job)
                new_job.set_prompt(prompt)
                new_job.set_conversation_context(prompt)
                new_job.predicted_service_time = predicted_time
                # Use BERT-predicted service time as simulated execution duration
                new_job.execution_duration = predicted_time
                if new_job.conversation_index is None:
                    new_job.conversation_index = conv_idx

                if verbose:
                    # Distinguish true BERT predictions from default fallback
                    is_true_prediction = (
                        hasattr(early_prompt_generator, "is_prediction_available")
                        and early_prompt_generator.is_prediction_available()
                    )
                    if is_true_prediction:
                        print(
                            f"    BERT predicted: {predicted_time:.3f}s for Job {new_job.job_id}"
                        )
                    else:
                        print(
                            f"    Using default service time: {predicted_time:.3f}s "
                            f"(BERT not available) for Job {new_job.job_id}"
                        )

            waiting_queue.append(new_job)

            if verbose and next_index % 50 == 0:
                print(
                    f"  Time {new_job.arrival_time:.2f}: Job {new_job.job_id} arrived "
                    f"(emotion: {new_job.emotion_label}, queue: {len(waiting_queue)})"
                )

            next_index += 1

        # If queue is empty and there are more jobs, fast-forward time
        if not waiting_queue and next_index < total_jobs:
            current_time = jobs[next_index].arrival_time
            continue

        # If queue is empty and no more jobs, we're done
        if not waiting_queue:
            break

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

        # Print job info in verbose mode
        if verbose:
            quadrant = getattr(selected_job, 'russell_quadrant', 'N/A')
            weight = getattr(selected_job, 'affect_weight', 1.0)
            print(
                f"  [{len(completed_jobs)+1}/{total_jobs}] Time {current_time:.2f}s: "
                f"Job {selected_job.job_id} | "
                f"emotion={selected_job.emotion_label} ({quadrant}) | "
                f"weight={weight:.2f} | "
                f"arrival={selected_job.arrival_time:.2f}s | "
                f"wait={selected_job.waiting_duration:.2f}s | "
                f"pred_exec={selected_job.execution_duration:.2f}s | "
                f"queue={len(waiting_queue)}"
            )

        # Use real LLM inference if handler provided
        if llm_handler is not None:

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

        # Advance time (uses actual LLM time if available, otherwise predicted)
        actual_time = selected_job.actual_execution_duration or selected_job.execution_duration
        current_time += actual_time
        selected_job.completion_time = current_time
        selected_job.status = "COMPLETED"

        scheduler.on_job_completed(selected_job, current_time)
        completed_jobs.append(selected_job)

    # Calculate metrics
    metrics = compute_fixed_jobs_metrics(completed_jobs)

    if verbose:
        print(f"\n=== Run Completed ===")
        print(f"  Total time: {metrics['total_time']:.2f}s")
        print(f"  Total completed jobs: {metrics['total_jobs']}")
        print(f"  Throughput: {metrics['throughput']:.3f} jobs/sec")
        print(f"  Avg waiting time: {metrics['avg_waiting_time']:.3f}s")
        print(f"  Avg JCT: {metrics['avg_jct']:.3f}s")

    return completed_jobs, metrics


def compute_fixed_jobs_metrics(completed_jobs: List[Job]) -> dict:
    """
    Compute metrics for fixed-num_jobs experiment.

    Args:
        completed_jobs: List of completed jobs with timing information

    Returns:
        Dictionary of metrics
    """
    if not completed_jobs:
        return {
            "total_jobs": 0,
            "total_time": 0.0,
            "throughput": 0.0,
            "avg_waiting_time": 0.0,
            "avg_jct": 0.0,
            "p50_waiting_time": 0.0,
            "p95_waiting_time": 0.0,
            "p99_waiting_time": 0.0,
            "p50_jct": 0.0,
            "p95_jct": 0.0,
            "p99_jct": 0.0,
            "estimated_arrival_rate": 0.0,
            "avg_actual_execution_time": None,
            "effective_load": None,
        }

    total_jobs = len(completed_jobs)

    # Get completion times and calculate total time
    completion_times = [j.completion_time for j in completed_jobs if j.completion_time is not None]
    total_time = max(completion_times) if completion_times else 0.0

    # Calculate waiting times
    waiting_times = [j.waiting_duration for j in completed_jobs if j.waiting_duration is not None]

    # Calculate JCT (Job Completion Time = waiting + execution)
    jcts = []
    for j in completed_jobs:
        if j.completion_time is not None and j.arrival_time is not None:
            jct = j.completion_time - j.arrival_time
            jcts.append(jct)

    # Compute statistics
    avg_waiting_time = np.mean(waiting_times) if waiting_times else 0.0
    avg_jct = np.mean(jcts) if jcts else 0.0

    # Percentiles for waiting time
    p50_waiting = np.percentile(waiting_times, 50) if waiting_times else 0.0
    p95_waiting = np.percentile(waiting_times, 95) if waiting_times else 0.0
    p99_waiting = np.percentile(waiting_times, 99) if waiting_times else 0.0

    # Percentiles for JCT
    p50_jct = np.percentile(jcts, 50) if jcts else 0.0
    p95_jct = np.percentile(jcts, 95) if jcts else 0.0
    p99_jct = np.percentile(jcts, 99) if jcts else 0.0

    # Throughput
    throughput = total_jobs / total_time if total_time > 0 else 0.0

    # Calculate effective load (based on actual measured execution times)
    # effective_load = arrival_rate * E[S_actual]
    # Estimate arrival rate from arrival time span
    arrival_times = [j.arrival_time for j in completed_jobs if j.arrival_time is not None]
    if len(arrival_times) >= 2:
        arrival_span = max(arrival_times) - min(arrival_times)
        estimated_arrival_rate = (len(arrival_times) - 1) / arrival_span if arrival_span > 0 else 0
    else:
        estimated_arrival_rate = 0

    # Get average actual execution time (measured from LLM inference)
    actual_exec_times = [
        j.actual_execution_duration
        for j in completed_jobs
        if hasattr(j, 'actual_execution_duration') and j.actual_execution_duration is not None
    ]
    avg_actual_exec = np.mean(actual_exec_times) if actual_exec_times else None
    effective_load = estimated_arrival_rate * avg_actual_exec if avg_actual_exec else None

    metrics = {
        "total_jobs": total_jobs,
        "total_time": total_time,
        "throughput": throughput,
        "avg_waiting_time": float(avg_waiting_time),
        "avg_jct": float(avg_jct),
        "p50_waiting_time": float(p50_waiting),
        "p95_waiting_time": float(p95_waiting),
        "p99_waiting_time": float(p99_waiting),
        "p50_jct": float(p50_jct),
        "p95_jct": float(p95_jct),
        "p99_jct": float(p99_jct),
        # Load metrics for clarification
        "estimated_arrival_rate": float(estimated_arrival_rate),
        "avg_actual_execution_time": float(avg_actual_exec) if avg_actual_exec else None,
        "effective_load": float(effective_load) if effective_load else None,
    }

    return metrics


def run_scheduling_loop_time_window(
    scheduler,
    job_trace: List[Dict],
    simulation_duration: float,
    emotion_config,
    verbose: bool = False,
    llm_handler=None,
    llm_skip_on_error: bool = True,
    early_prompt_generator=None,
) -> Tuple[List[Job], dict]:
    """
    Run time-window scheduling loop with pre-generated trace.

    Jobs arrive according to the pre-generated trace, but only jobs completed
    within the simulation_duration are counted.

    Args:
        scheduler: Scheduler instance
        job_trace: Pre-generated job trace (list of dicts)
        simulation_duration: Time window duration (seconds)
        emotion_config: EmotionConfig for arousal classification
        verbose: Print progress
        llm_handler: Optional LLM handler
        llm_skip_on_error: Skip failed jobs
        early_prompt_generator: Optional EarlyPromptGenerator for BERT prediction

    Returns:
        Tuple of (completed_jobs, metrics)
    """
    from workload.task_generator import create_jobs_from_trace
    
    if not job_trace:
        # No jobs in the trace: return an empty list and a full metrics dict
        # with all keys present (so reporting never crashes).
        metrics = compute_time_window_metrics([], simulation_duration)
        return [], metrics
    
    if verbose:
        print(f"\nStarting time-window scheduling with {scheduler.name}")
        print(f"  Simulation duration: {simulation_duration}s")
        print(f"  Total jobs in trace: {len(job_trace)}")
    
    # State
    current_time = 0.0
    waiting_queue: List[Job] = []
    completed_jobs: List[Job] = []
    next_trace_index = 0
    
    # Main loop
    while current_time < simulation_duration:
        # Add all jobs that have arrived up to current_time
        while (next_trace_index < len(job_trace) and
               job_trace[next_trace_index]['arrival_time'] <= current_time):

            job_config = job_trace[next_trace_index]
            arousal = job_config.get('arousal', 0.0)
            valence = job_config.get('valence', 0.0)
            russell_quadrant = job_config.get('russell_quadrant')
            affect_weight = job_config.get('affect_weight', 1.0)
            urgency = job_config.get('urgency', 0.0)

            # Classify Russell quadrant if not in trace
            if russell_quadrant is None:
                russell_quadrant = emotion_config.classify_russell_quadrant(arousal, valence)

            # Check if trace already has predicted_service_time
            predicted_service_time = job_config.get('predicted_service_time')
            prompt_from_trace = job_config.get('prompt')
            conversation_index = job_config.get('conversation_index')

            new_job = Job(
                job_id=job_config['job_id'],
                # Prefer BERT-predicted service time if available
                execution_duration=predicted_service_time
                if predicted_service_time is not None
                else job_config['service_time'],
                arrival_time=job_config['arrival_time'],
                emotion_label=job_config['emotion'],
                arousal=arousal,
                valence=valence,
                russell_quadrant=russell_quadrant,
                affect_weight=affect_weight,
                urgency=urgency,
                predicted_service_time=predicted_service_time,
            )

            # Set conversation_index and prompt if available from trace
            if conversation_index is not None:
                new_job.conversation_index = conversation_index
            if prompt_from_trace is not None:
                new_job.set_prompt(prompt_from_trace)
                new_job.set_conversation_context(prompt_from_trace)

            # === Early prompt generation and BERT prediction ===
            # Only if not already in trace and generator available
            if new_job.predicted_service_time is None and early_prompt_generator is not None:
                prompt, predicted_time, conv_idx = early_prompt_generator.generate_prompt_and_predict(new_job)
                new_job.set_prompt(prompt)
                new_job.set_conversation_context(prompt)
                new_job.predicted_service_time = predicted_time
                # Use BERT-predicted service time as simulated execution duration
                new_job.execution_duration = predicted_time
                if new_job.conversation_index is None:
                    new_job.conversation_index = conv_idx

                if verbose:
                    # Distinguish true BERT predictions from default fallback
                    is_true_prediction = (
                        hasattr(early_prompt_generator, "is_prediction_available")
                        and early_prompt_generator.is_prediction_available()
                    )
                    if is_true_prediction:
                        print(f"    BERT predicted: {predicted_time:.3f}s for Job {new_job.job_id}")
                    else:
                        print(
                            f"    Using default service time: {predicted_time:.3f}s "
                            f"(BERT not available) for Job {new_job.job_id}"
                        )

            waiting_queue.append(new_job)

            if verbose and next_trace_index % 50 == 0:
                print(f"  Time {new_job.arrival_time:.2f}: Job {new_job.job_id} arrived")

            next_trace_index += 1
        
        # If queue empty, jump to next arrival
        if not waiting_queue:
            if next_trace_index < len(job_trace):
                next_arrival = job_trace[next_trace_index]['arrival_time']
                if next_arrival < simulation_duration:
                    current_time = next_arrival
                    continue
                else:
                    # No more jobs within window
                    break
            else:
                # No more jobs
                break
        
        # Schedule next job
        selected_job = scheduler.schedule(waiting_queue, current_time=current_time)
        
        if selected_job is None:
            print(f"Warning: Scheduler returned None at time {current_time}")
            break
        
        waiting_queue.remove(selected_job)
        
        # Execute job
        selected_job.status = "RUNNING"
        selected_job.waiting_duration = current_time - selected_job.arrival_time
        scheduler.on_job_scheduled(selected_job, current_time)
        
        if verbose:
            quadrant = getattr(selected_job, 'russell_quadrant', 'N/A')
            weight = getattr(selected_job, 'affect_weight', 1.0)
            print(
                f"  Time {current_time:.2f}s: Job {selected_job.job_id} | "
                f"emotion={selected_job.emotion_label} ({quadrant}) | "
                f"weight={weight:.2f} | "
                f"wait={selected_job.waiting_duration:.2f}s | "
                f"service={selected_job.execution_duration:.2f}s | "
                f"queue={len(waiting_queue)}"
            )
        
        # LLM inference if available
        if llm_handler is not None:
            success = llm_handler.execute_job(selected_job)
            
            if not success and llm_skip_on_error:
                if verbose:
                    print(f"  WARNING: Job {selected_job.job_id} failed, skipping")
                continue
            elif not success:
                print(f"ERROR: Job {selected_job.job_id} failed")
                break
        
        # Advance time
        actual_time = selected_job.actual_execution_duration or selected_job.execution_duration
        current_time += actual_time
        selected_job.completion_time = current_time
        selected_job.status = "COMPLETED"
        
        scheduler.on_job_completed(selected_job, current_time)
        completed_jobs.append(selected_job)
        
        # Stop if exceeded time window
        if current_time >= simulation_duration:
            if verbose:
                print(f"  Reached time window limit: {simulation_duration}s")
            break
    
    # Compute metrics
    metrics = compute_time_window_metrics(completed_jobs, simulation_duration)
    
    if verbose:
        print(f"\n=== Time Window Completed ===")
        print(f"  Simulation duration: {simulation_duration}s")
        print(f"  Completed jobs: {metrics['total_jobs']}")
        print(f"  Throughput: {metrics['throughput']:.3f} jobs/sec")
        print(f"  Avg waiting time: {metrics['avg_waiting_time']:.3f}s")
    
    return completed_jobs, metrics


def compute_time_window_metrics(completed_jobs: List[Job], simulation_duration: float) -> dict:
    """
    Compute metrics for time-window experiment.
    
    Args:
        completed_jobs: List of completed jobs
        simulation_duration: Total simulation time
    
    Returns:
        Dictionary of metrics
    """
    if not completed_jobs:
        return {
            "total_jobs": 0,
            "total_time": simulation_duration,
            "throughput": 0.0,
            "avg_waiting_time": 0.0,
            "avg_jct": 0.0,
            "p50_waiting_time": 0.0,
            "p95_waiting_time": 0.0,
            "p99_waiting_time": 0.0,
            "p50_jct": 0.0,
            "p95_jct": 0.0,
            "p99_jct": 0.0,
            "estimated_arrival_rate": 0.0,
            "avg_actual_execution_time": None,
            "effective_load": None,
        }

    total_jobs = len(completed_jobs)

    # Waiting times
    waiting_times = [j.waiting_duration for j in completed_jobs if j.waiting_duration is not None]

    # JCT
    jcts = []
    for j in completed_jobs:
        if j.completion_time is not None and j.arrival_time is not None:
            jct = j.completion_time - j.arrival_time
            jcts.append(jct)

    # Statistics
    avg_waiting_time = np.mean(waiting_times) if waiting_times else 0.0
    avg_jct = np.mean(jcts) if jcts else 0.0

    p50_waiting = np.percentile(waiting_times, 50) if waiting_times else 0.0
    p95_waiting = np.percentile(waiting_times, 95) if waiting_times else 0.0
    p99_waiting = np.percentile(waiting_times, 99) if waiting_times else 0.0

    p50_jct = np.percentile(jcts, 50) if jcts else 0.0
    p95_jct = np.percentile(jcts, 95) if jcts else 0.0
    p99_jct = np.percentile(jcts, 99) if jcts else 0.0

    # Throughput (jobs per second)
    throughput = total_jobs / simulation_duration if simulation_duration > 0 else 0.0

    # Calculate effective load (based on actual measured execution times)
    arrival_times = [j.arrival_time for j in completed_jobs if j.arrival_time is not None]
    if len(arrival_times) >= 2:
        arrival_span = max(arrival_times) - min(arrival_times)
        estimated_arrival_rate = (len(arrival_times) - 1) / arrival_span if arrival_span > 0 else 0
    else:
        estimated_arrival_rate = 0

    actual_exec_times = [
        j.actual_execution_duration
        for j in completed_jobs
        if hasattr(j, 'actual_execution_duration') and j.actual_execution_duration is not None
    ]
    avg_actual_exec = np.mean(actual_exec_times) if actual_exec_times else None
    effective_load = estimated_arrival_rate * avg_actual_exec if avg_actual_exec else None

    return {
        "total_jobs": total_jobs,
        "total_time": simulation_duration,
        "throughput": throughput,
        "avg_waiting_time": float(avg_waiting_time),
        "avg_jct": float(avg_jct),
        "p50_waiting_time": float(p50_waiting),
        "p95_waiting_time": float(p95_waiting),
        "p99_waiting_time": float(p99_waiting),
        "p50_jct": float(p50_jct),
        "p95_jct": float(p95_jct),
        "p99_jct": float(p99_jct),
        "estimated_arrival_rate": float(estimated_arrival_rate),
        "avg_actual_execution_time": float(avg_actual_exec) if avg_actual_exec else None,
        "effective_load": float(effective_load) if effective_load else None,
    }


__all__ = [
    "run_scheduling_loop",
    "compute_fixed_jobs_metrics",
    "run_scheduling_loop_time_window",
    "compute_time_window_metrics",
]

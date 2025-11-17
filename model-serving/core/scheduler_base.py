"""
Abstract Scheduler Base Class for LLM Task Scheduling

This module defines the base interface for all scheduling algorithms in the
emotion-aware LLM scheduling system. It provides a unified API that allows
different scheduling strategies to be easily plugged in and compared.

Scheduling algorithms implement the core `schedule()` method to determine
which job(s) should be executed next from the waiting queue.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from core.job import Job


class SchedulerBase(ABC):
    """
    Abstract base class for all scheduling strategies

    All schedulers must implement the schedule() method which selects
    the next job(s) to execute from the waiting queue.
    """

    def __init__(self, name: str = "Base Scheduler"):
        """
        Initialize scheduler

        Args:
            name: Name of the scheduling algorithm
        """
        self.name = name
        self.scheduled_count = 0  # Track number of scheduling decisions
        self.total_waiting_time = 0  # Track cumulative waiting time

    @abstractmethod
    def schedule(self, waiting_queue: List[Job], current_time: float = 0) -> Optional[Job]:
        """
        Select the next job to execute from the waiting queue

        Args:
            waiting_queue: List of jobs waiting to be executed
            current_time: Current time (used for starvation prevention)

        Returns:
            The selected Job object, or None if queue is empty
        """
        pass

    def schedule_batch(self, waiting_queue: List[Job], batch_size: int,
                       current_time: float = 0) -> List[Job]:
        """
        Select a batch of jobs to execute together

        Default implementation: repeatedly call schedule() to fill batch.
        Subclasses can override for more sophisticated batch selection.

        Args:
            waiting_queue: List of jobs waiting to be executed
            batch_size: Maximum number of jobs in the batch
            current_time: Current time

        Returns:
            List of selected Job objects (up to batch_size)
        """
        selected_jobs = []
        remaining_queue = waiting_queue.copy()

        for _ in range(min(batch_size, len(remaining_queue))):
            job = self.schedule(remaining_queue, current_time)
            if job is None:
                break
            selected_jobs.append(job)
            remaining_queue.remove(job)

        return selected_jobs

    def on_job_scheduled(self, job: Job, current_time: float):
        """
        Callback when a job is scheduled for execution

        Can be overridden by subclasses to track statistics or update state

        Args:
            job: The scheduled job
            current_time: Current time
        """
        self.scheduled_count += 1
        waiting_time = current_time - job.arrival_time
        self.total_waiting_time += waiting_time
        job.curr_waiting_time = waiting_time

    def on_job_completed(self, job: Job, current_time: float):
        """
        Callback when a job completes execution

        Can be overridden by subclasses to update internal state

        Args:
            job: The completed job
            current_time: Current time
        """
        pass

    def get_statistics(self) -> dict:
        """
        Get statistics about scheduler performance

        Returns:
            Dictionary with scheduler statistics
        """
        avg_waiting_time = (self.total_waiting_time / self.scheduled_count
                            if self.scheduled_count > 0 else 0)

        return {
            'scheduler_name': self.name,
            'scheduled_count': self.scheduled_count,
            'total_waiting_time': self.total_waiting_time,
            'avg_waiting_time': avg_waiting_time,
        }

    def reset(self):
        """Reset scheduler statistics"""
        self.scheduled_count = 0
        self.total_waiting_time = 0

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class FCFSScheduler(SchedulerBase):
    """
    First-Come-First-Served (FCFS) Scheduler

    Jobs are scheduled in the order they arrive (by arrival_time).
    This serves as the baseline scheduling strategy.
    """

    def __init__(self):
        super().__init__(name="FCFS")

    def schedule(self, waiting_queue: List[Job], current_time: float = 0) -> Optional[Job]:
        """
        Select the job with the earliest arrival time

        Args:
            waiting_queue: List of jobs waiting to be executed
            current_time: Current time (unused in FCFS)

        Returns:
            Job with earliest arrival time, or None if queue is empty
        """
        if not waiting_queue:
            return None

        # Sort by arrival time and return the earliest
        earliest_job = min(waiting_queue, key=lambda j: j.arrival_time)
        return earliest_job


class SJFScheduler(SchedulerBase):
    """
    Shortest-Job-First (SJF) Scheduler

    Jobs are scheduled based on their execution duration (shortest first).
    This minimizes average waiting time but can cause starvation for long jobs.
    """

    def __init__(self, use_prediction: bool = False,
                 starvation_threshold: float = float('inf')):
        """
        Initialize SJF scheduler

        Args:
            use_prediction: Use predicted execution duration instead of actual
            starvation_threshold: Max waiting time before forcing job execution
        """
        super().__init__(name="SJF" + ("P" if use_prediction else ""))
        self.use_prediction = use_prediction
        self.starvation_threshold = starvation_threshold

    def schedule(self, waiting_queue: List[Job], current_time: float = 0) -> Optional[Job]:
        """
        Select the job with the shortest execution duration

        Implements starvation prevention: if any job has waited longer than
        the threshold, schedule it regardless of its duration.

        Args:
            waiting_queue: List of jobs waiting to be executed
            current_time: Current time

        Returns:
            Job with shortest duration (or starving job), or None if queue is empty
        """
        if not waiting_queue:
            return None

        # Check for starving jobs
        for job in waiting_queue:
            waiting_time = current_time - job.arrival_time
            if waiting_time >= self.starvation_threshold:
                return job

        # Select job with shortest execution duration
        if self.use_prediction:
            # Use predicted execution duration
            shortest_job = min(waiting_queue,
                               key=lambda j: j.predicted_execution_duration
                               if j.predicted_execution_duration is not None
                               else j.execution_duration)
        else:
            # Use actual execution duration
            shortest_job = min(waiting_queue, key=lambda j: j.execution_duration)

        return shortest_job


# Alias for backward compatibility and clarity
SSJFScheduler = SJFScheduler  # Speculative SJF uses prediction

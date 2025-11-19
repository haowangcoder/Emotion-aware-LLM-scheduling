"""
Test for scheduler base module.
"""

from core.job import Job
from core.scheduler_base import FCFSScheduler, SJFScheduler


def test_scheduler_base():
    """Test FCFS and SJF schedulers."""
    print("=" * 70)
    print("Scheduler Base Class Test")
    print("=" * 70)

    # Create test jobs
    jobs = [
        Job(job_id=0, execution_duration=5.0, arrival_time=0.0),
        Job(job_id=1, execution_duration=2.0, arrival_time=1.0),
        Job(job_id=2, execution_duration=8.0, arrival_time=2.0),
        Job(job_id=3, execution_duration=3.0, arrival_time=3.0),
        Job(job_id=4, execution_duration=1.0, arrival_time=4.0),
    ]

    # Test FCFS Scheduler
    print("\n1. FCFS Scheduler Test")
    fcfs = FCFSScheduler()
    queue = jobs.copy()
    print(f"   Queue: {[j.job_id for j in queue]}")
    print(f"   Scheduling order: ", end="")
    while queue:
        job = fcfs.schedule(queue, current_time=0)
        print(f"{job.job_id} ", end="")
        queue.remove(job)
    print()

    # Test SJF Scheduler
    print("\n2. SJF Scheduler Test")
    sjf = SJFScheduler()
    queue = jobs.copy()
    print(f"   Queue: {[j.job_id for j in queue]}")
    print(f"   Durations: {[j.execution_duration for j in queue]}")
    print(f"   Scheduling order: ", end="")
    while queue:
        job = sjf.schedule(queue, current_time=0)
        print(f"{job.job_id} ", end="")
        queue.remove(job)
    print()

    # Test starvation prevention
    print("\n3. Starvation Prevention Test")
    sjf_safe = SJFScheduler(starvation_threshold=10.0)
    queue = jobs.copy()
    current_time = 15.0  # Job 0 has been waiting 15 seconds
    print(f"   Current time: {current_time}")
    print(f"   Starvation threshold: {sjf_safe.starvation_threshold}")
    print(f"   Job 0 arrival: {jobs[0].arrival_time}, waiting: {current_time - jobs[0].arrival_time}")
    selected = sjf_safe.schedule(queue, current_time=current_time)
    print(f"   Selected job (should be 0 due to starvation): {selected.job_id}")

    # Test batch scheduling
    print("\n4. Batch Scheduling Test")
    fcfs_batch = FCFSScheduler()
    queue = jobs.copy()
    batch = fcfs_batch.schedule_batch(queue, batch_size=3, current_time=0)
    print(f"   Batch size: {len(batch)}")
    print(f"   Batch jobs: {[j.job_id for j in batch]}")

    # Test statistics
    print("\n5. Scheduler Statistics Test")
    scheduler = FCFSScheduler()
    for i, job in enumerate(jobs):
        scheduler.on_job_scheduled(job, current_time=i*2)
    stats = scheduler.get_statistics()
    print(f"   {stats}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    test_scheduler_base()

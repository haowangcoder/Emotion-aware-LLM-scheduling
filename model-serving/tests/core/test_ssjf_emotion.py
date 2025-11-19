"""
Test for SSJF emotion scheduler module.
"""

from core.job import Job
from core.ssjf_emotion import SSJFEmotionScheduler, SSJFEmotionPriorityScheduler
from core.scheduler_base import FCFSScheduler
import random


def test_ssjf_emotion():
    """Test SSJF emotion-aware schedulers."""
    print("=" * 70)
    print("SSJF Emotion Scheduler Test")
    print("=" * 70)

    # Create emotion-aware test jobs
    random.seed(42)
    emotions = ['excited', 'sad', 'angry', 'calm', 'neutral']
    arousals = [0.9, -0.6, 0.8, -0.3, 0.0]
    emotion_classes = ['high', 'low', 'high', 'low', 'medium']

    # Service times computed from arousal
    # S_i = 2.0 * (1 + 0.5 * a_i)
    service_times = [2.0 * (1 + 0.5 * a) for a in arousals]

    jobs = []
    for i in range(5):
        job = Job(
            job_id=i,
            execution_duration=service_times[i],
            arrival_time=i * 1.0,
            emotion_label=emotions[i],
            arousal=arousals[i],
            emotion_class=emotion_classes[i]
        )
        jobs.append(job)

    # Test 1: Basic SSJF Emotion scheduling
    print("\n1. Basic SSJF Emotion Scheduling")
    scheduler = SSJFEmotionScheduler()
    queue = jobs.copy()

    print(f"\n   {'Job':<5} {'Emotion':<10} {'Arousal':<10} {'Class':<10} {'Service Time':<15}")
    print(f"   {'-'*55}")
    for job in queue:
        print(f"   {job.job_id:<5} {job.emotion_label:<10} {job.arousal:<10.2f} "
              f"{job.emotion_class:<10} {job.execution_duration:<15.3f}")

    print(f"\n   Scheduling order (by service time): ", end="")
    scheduled_order = []
    while queue:
        job = scheduler.schedule(queue, current_time=0)
        print(f"{job.job_id} ", end="")
        scheduled_order.append(job.job_id)
        scheduler.on_job_scheduled(job, current_time=len(scheduled_order))
        queue.remove(job)
    print()

    # Test 2: Starvation prevention
    print("\n2. Starvation Prevention Test")
    scheduler_safe = SSJFEmotionScheduler(
        starvation_threshold=10.0,
        starvation_coefficient=3.0
    )
    queue = jobs.copy()

    # Simulate: jobs arrived at different times, current time is 15
    current_time = 15.0
    for job in queue:
        job.arrival_time = 0  # All arrived at time 0

    print(f"   Current time: {current_time}")
    print(f"   Starvation threshold: {scheduler_safe.starvation_threshold}")
    print(f"   All jobs waiting for: {current_time} seconds")

    selected = scheduler_safe.schedule(queue, current_time=current_time)
    print(f"   Selected job (should trigger starvation): Job {selected.job_id}")
    print(f"   Job {selected.job_id} service time: {selected.execution_duration:.3f}")

    # Test 3: Emotion class statistics
    print("\n3. Emotion Class Statistics")
    scheduler_stats = SSJFEmotionScheduler()
    queue = jobs.copy()

    current_time = 0
    while queue:
        job = scheduler_stats.schedule(queue, current_time=current_time)
        scheduler_stats.on_job_scheduled(job, current_time=current_time)
        current_time += job.execution_duration
        queue.remove(job)

    stats = scheduler_stats.get_statistics()
    print(f"\n   Overall Statistics:")
    print(f"     Scheduled count: {stats['scheduled_count']}")
    print(f"     Average waiting time: {stats['avg_waiting_time']:.3f}")

    print(f"\n   Per-Emotion-Class Statistics:")
    for emotion_class, class_stats in stats['emotion_class_stats'].items():
        print(f"     {emotion_class}:")
        print(f"       Scheduled: {class_stats['scheduled']}")
        print(f"       Avg waiting time: {class_stats['avg_waiting_time']:.3f}")

    # Test 4: Priority-adjusted scheduling
    print("\n4. Priority-Adjusted SSJF Scheduling")
    # Give higher weight (priority) to low arousal emotions
    priority_scheduler = SSJFEmotionPriorityScheduler(
        priority_weights={'high': 0.8, 'medium': 1.0, 'low': 1.5}
    )
    queue = jobs.copy()

    print(f"   Priority weights: {priority_scheduler.priority_weights}")
    print(f"   (Higher weight = higher priority = scheduled sooner)")
    print(f"\n   Scheduling order: ", end="")

    while queue:
        job = priority_scheduler.schedule(queue, current_time=0)
        print(f"{job.job_id}({job.emotion_class}) ", end="")
        queue.remove(job)
    print()

    # Test 5: Compare SSJF vs FCFS order
    print("\n5. Comparison: SSJF-Emotion vs FCFS")

    fcfs = FCFSScheduler()
    ssjf = SSJFEmotionScheduler()

    queue_fcfs = jobs.copy()
    queue_ssjf = jobs.copy()

    print(f"\n   FCFS order: ", end="")
    while queue_fcfs:
        job = fcfs.schedule(queue_fcfs)
        print(f"{job.job_id} ", end="")
        queue_fcfs.remove(job)

    print(f"\n   SSJF order: ", end="")
    while queue_ssjf:
        job = ssjf.schedule(queue_ssjf)
        print(f"{job.job_id} ", end="")
        queue_ssjf.remove(job)
    print()

    print("\n" + "=" * 70)


if __name__ == '__main__':
    test_ssjf_emotion()

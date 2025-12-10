# Affect-Aware LLM Scheduling System (AW-SSJF)

An affect-aware GPU task scheduling system for LLM inference workloads that implements the **Affect-Weighted Shortest-Service-Job-First (AW-SSJF)** algorithm. The system uses BERT-based length prediction for service time estimation and Russell's Circumplex Model for emotion-aware scheduling prioritization.

## Overview

This project implements a novel scheduling framework that:

- **BERT-based Length Prediction**: Uses a BERT proxy model to predict LLM output token length (service time)
- **Depression-First Strategy**: Prioritizes users with negative emotional states (low valence + low arousal)
- **WSPT Scheduling**: Implements Weighted Shortest Processing Time rule: `Score = S / w`
- **Russell Quadrant Analysis**: Evaluates fairness across four emotional quadrants (excited, calm, panic, depression)

### Key Design Principles

| Component | Design |
|-----------|--------|
| **Service Time** | BERT proxy model prediction (content-based) |
| **Emotion Role** | Affects scheduling weight (priority) |
| **Scheduling** | Unified WSPT framework |
| **Fairness Goal** | Depression-First prioritization |

## Quick Start

**Run with FCFS baseline:**
```bash
uv run python run_simulation.py --scheduler FCFS --num_jobs 50 --verbose
```

**Run AW-SSJF scheduler (main algorithm):**
```bash
uv run python run_simulation.py --scheduler AW-SSJF --num_jobs 50 --w_max 2.0
```

**Compare schedulers:**
```bash
# FCFS baseline
uv run python run_simulation.py --scheduler FCFS --num_jobs 100 --random_seed 42 --output_dir results/

# SJF (pure shortest-job-first, equivalent to w_max=1)
uv run python run_simulation.py --scheduler SJF --num_jobs 100 --output_dir results/

# AW-SSJF (affect-weighted)
uv run python run_simulation.py --scheduler AW-SSJF --num_jobs 100 --w_max 2.0 --output_dir results/

# Weight-Only (pure affect priority, ignores service time)
uv run python run_simulation.py --scheduler Weight-Only --num_jobs 100 --w_max 2.0 --output_dir results/
```

## Table of Contents

- [Installation](#installation)
- [System Architecture](#system-architecture)
- [Key Algorithms](#key-algorithms)
- [Configuration](#configuration)
- [Usage](#usage)
- [Scheduling Algorithms](#scheduling-algorithms)
- [Evaluation Metrics](#evaluation-metrics)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast and reliable Python package management.

### 1. Install uv

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Setup Project Environment

```bash
cd Emotion-aware-LLM-scheduling

# Sync dependencies (creates virtual environment automatically)
uv sync
```

### 3. Additional Setup

**HuggingFace Authentication** (required for gated models like LLaMA):
```bash
uv run huggingface-cli login
```

**Download EmpatheticDialogues Dataset:**
```bash
mkdir -p dataset
# Download from Kaggle and extract to dataset/
# Expected files: train.csv, valid.csv, test.csv
```

---

## System Architecture

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Affect-Aware LLM Scheduling System                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Input     │    │  Emotion    │    │   Length    │    │  Scheduler  │  │
│  │  (Prompt)   │───▶│  Analyzer   │───▶│  Predictor  │───▶│   (Core)    │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│        │                  │                  │                  │          │
│        │            ┌─────┴─────┐      ┌─────┴─────┐      ┌─────┴─────┐    │
│        │            │ arousal   │      │ predicted │      │  Score =  │    │
│        │            │ valence   │      │ service   │      │   S / w   │    │
│        │            │ quadrant  │      │ time (Ŝ)  │      │           │    │
│        │            └───────────┘      └───────────┘      └───────────┘    │
│        │                  │                                     │          │
│        │                  ▼                                     │          │
│        │            ┌───────────┐                               │          │
│        │            │  Affect   │                               │          │
│        │            │  Weight   │───────────────────────────────┘          │
│        │            │ Calculator│                                          │
│        │            │  (w ≥ 1)  │                                          │
│        │            └───────────┘                                          │
│        │                                                                   │
│        ▼                                                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                    │
│  │    LLM      │    │   Results   │    │  Fairness   │                    │
│  │  Executor   │───▶│   Logger    │───▶│  Analyzer   │                    │
│  └─────────────┘    └─────────────┘    └─────────────┘                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Project Structure

```
Emotion-aware-LLM-scheduling/
├── run_simulation.py           # Main entry point
├── pyproject.toml              # uv project configuration
├── dataset/                    # EmpatheticDialogues dataset
│
├── model-serving/              # Main package
│   ├── config/                 # Configuration
│   │   ├── default.yaml        # YAML configuration (single source of truth)
│   │   └── config_loader.py    # Configuration loader
│   │
│   ├── core/                   # Core scheduling logic
│   │   ├── job.py              # Job data structure
│   │   ├── emotion.py          # Emotion sampling & Russell quadrants
│   │   ├── affect_weight.py    # Depression-First weight calculation
│   │   ├── scheduler_base.py   # Base scheduler & FCFS/SJF
│   │   ├── aw_ssjf_scheduler.py    # AW-SSJF scheduler (main)
│   │   └── weight_only_scheduler.py # Weight-Only scheduler (ablation)
│   │
│   ├── predictor/              # BERT length prediction
│   │   ├── bert_predictor.py   # BERT model wrapper
│   │   └── length_estimator.py # Unified prediction interface
│   │
│   ├── llm/                    # LLM integration
│   │   ├── engine.py           # Model loading & generation
│   │   ├── dataset_loader.py   # Dataset loading
│   │   ├── prompt_builder.py   # Prompt construction
│   │   └── response_cache.py   # Response caching
│   │
│   ├── workload/               # Workload generation
│   │   └── task_generator.py   # Job generation with quadrant sampling
│   │
│   ├── simulator/              # Simulation framework
│   │   ├── cli.py              # CLI argument parsing
│   │   ├── experiment.py       # Experiment orchestration
│   │   ├── loop.py             # Core scheduling loop
│   │   └── reporting.py        # Results reporting
│   │
│   └── analysis/               # Results analysis
│       ├── fairness_metrics.py # Fairness calculations
│       └── logger.py           # Logging utilities
│
└── results/                    # Experiment outputs
```

---

## Key Algorithms

### 1. BERT Length Prediction

Service time is predicted by a BERT proxy model (content-based, emotion-independent):

```
Service Time = const_latency + predicted_tokens × per_token_latency
```

Where:
- `predicted_tokens`: BERT regression model output (0-512 tokens)
- `per_token_latency`: Time per generated token (default: 0.02s)
- `const_latency`: Fixed overhead (default: 0.1s)

### 2. Depression-First Affect Weight

The affect weight prioritizes users in "depression" state (low valence + low arousal):

**Step 1: Compute Urgency**
```
n = max(0, -valence)    # Unpleasant intensity ∈ [0, 1]
ℓ = max(0, -arousal)    # Low arousal intensity ∈ [0, 1]
u = n^p × ℓ^q           # Depression urgency ∈ [0, 1]
```

**Step 2: Compute Weight**
```
w = 1 + (w_max - 1) × c × u
```

Where:
- `w_max`: Maximum weight (recommended: 1.5-2.5)
- `p, q`: Exponents controlling curve shape (default: 1.0 = linear)
- `c`: Emotion recognition confidence (optional discount)

**Key Property**: Only users with BOTH negative valence AND low arousal get priority boost.

### 3. WSPT Scheduling Rule

Jobs are sorted by WSPT score (lower = higher priority):

```
Score_i = S_i / w_i
```

Where:
- `S_i`: Predicted service time (from BERT)
- `w_i`: Affect weight (from Depression-First formula)

### 4. Russell Quadrant Classification

Emotions are classified into four quadrants based on valence and arousal:

| Quadrant | Valence | Arousal | Example Emotions | Priority Boost |
|----------|---------|---------|------------------|----------------|
| **Excited** | ≥ 0 | ≥ 0 | excited, joyful, surprised | None (w=1) |
| **Calm** | ≥ 0 | < 0 | hopeful, grateful, content | None (w=1) |
| **Panic** | < 0 | ≥ 0 | terrified, anxious, angry | None (w=1) |
| **Depression** | < 0 | < 0 | sad, lonely, disappointed | Yes (w>1) |

---

## Configuration

### Configuration File: `model-serving/config/default.yaml`

```yaml
# Scheduler Configuration
scheduler:
  algorithm: 'FCFS'  # FCFS | SJF | SSJF | AW-SSJF | Weight-Only
  system_load: 0.6

  # Affect Weight Parameters (Depression-First)
  affect_weight:
    w_max: 2.0         # Maximum weight [1.2, 3.0] recommended
    p: 1.0             # Negative valence exponent
    q: 1.0             # Low arousal exponent
    use_confidence: true

  # Starvation Prevention
  starvation_prevention:
    threshold: .inf
    coefficient: 3.0

# BERT Length Predictor
length_predictor:
  enabled: false       # Disabled by default (requires trained model)
  model_path: 'predictor/models/bert_regression.pth'
  model_name: 'bert-base-uncased'
  device: 'cuda'
  per_token_latency: 0.02
  const_latency: 0.1
  default_service_time: 2.0

# Workload Generation
workload:
  service_time:
    base_service_time: 2.0
    min_service_time: 0.1

  emotion:
    enable_emotion_aware: true
    use_stratified_sampling: true
    quadrant_distribution: uniform  # Balanced across 4 quadrants
```

### Command-Line Arguments

```bash
uv run python run_simulation.py \
  --scheduler AW-SSJF \
  --num_jobs 100 \
  --w_max 2.0 \
  --p 1.0 \
  --q 1.0 \
  --system_load 0.6 \
  --output_dir results/
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--scheduler` | Algorithm: FCFS, SJF, SSJF, AW-SSJF, Weight-Only | FCFS |
| `--num_jobs` | Number of jobs to run | 50 |
| `--w_max` | Maximum affect weight | 2.0 |
| `--p` | Negative valence exponent | 1.0 |
| `--q` | Low arousal exponent | 1.0 |
| `--use_confidence` / `--no_confidence` | Enable/disable confidence discount | enabled |
| `--system_load` | Target system utilization (ρ) | 0.6 |
| `--random_seed` | Random seed for reproducibility | None |
| `--output_dir` | Results output directory | results/ |
| `--verbose` | Print detailed progress | False |

---

## Scheduling Algorithms

### Available Schedulers

| Scheduler | Score Formula | Description |
|-----------|---------------|-------------|
| **FCFS** | Arrival time | First-Come-First-Serve baseline |
| **SJF** | `S` | Shortest-Job-First (service time only) |
| **SSJF** | `S` | Speculative SJF (alias for SJF) |
| **AW-SSJF** | `S / w` | **Main algorithm**: balances efficiency and fairness |
| **Weight-Only** | `-w` | Pure affect priority (ignores service time) |

### Experimental Comparisons

| Comparison | Purpose | Expected Result |
|------------|---------|-----------------|
| FCFS vs SJF | Verify SJF efficiency | SJF has lower avg latency |
| SJF vs AW-SSJF | Verify affect weight effect | AW-SSJF fairer to depression users |
| Weight-Only vs AW-SSJF | Verify service time effect | AW-SSJF more efficient overall |
| Weight-Only vs SJF | Fairness vs efficiency tradeoff | Weight-Only fairer but less efficient |

**Expected Results:**
```
Overall Mean JCT:     SJF ≈ AW-SSJF < Weight-Only < FCFS
Depression Group JCT: Weight-Only < AW-SSJF < SJF < FCFS
Fairness (Jain Index): Weight-Only > AW-SSJF > SJF ≈ FCFS
```

---

## Evaluation Metrics

### Performance Metrics

- **Average JCT**: Mean job completion time (arrival to completion)
- **Average Waiting Time**: Mean time jobs wait before execution
- **P50/P95/P99 Latency**: Percentile latencies
- **Throughput**: Jobs completed per unit time

### Affect-Weighted JCT (AW-JCT)

A fairness-aware aggregate metric that weights JCT by affect weight:

```
AW-JCT = Σ(w_i × JCT_i) / Σ(w_i)
```

This metric gives more importance to depression-quadrant users.

### Fairness Metrics

**Jain Fairness Index:**
```
J = (Σx_i)² / (n × Σx_i²)
```
Range: [1/n, 1] where 1 = perfect fairness

**Per-Quadrant Analysis:**
- Average waiting time per Russell quadrant
- Inter-quadrant Jain index
- Coefficient of variation across quadrants

---

## Output Files

Each experiment produces:

```
results/
├── <SCHEDULER>_<N>jobs_load<LOAD>_fixed_jobs.csv   # Per-job logs
├── <SCHEDULER>_<N>jobs_load<LOAD>_fixed_summary.json # Statistics
└── cache/
    ├── responses.json      # Cached LLM responses
    └── job_configs.json    # Saved job configurations
```

### CSV Log Fields

| Field | Description |
|-------|-------------|
| `job_id` | Unique job identifier |
| `emotion_label` | Emotion from EmpatheticDialogues |
| `arousal`, `valence` | NRC-VAD values [-1, 1] |
| `russell_quadrant` | excited/calm/panic/depression |
| `affect_weight` | Computed weight (w ≥ 1) |
| `urgency` | Depression urgency (u ∈ [0, 1]) |
| `predicted_service_time` | BERT prediction (or default) |
| `actual_execution_duration` | Measured execution time |
| `arrival_time`, `completion_time` | Timestamps |
| `waiting_duration` | Time in queue |

### Summary JSON Structure

```json
{
  "experiment_config": {
    "scheduler": "AW-SSJF",
    "w_max": 2.0,
    "p": 1.0,
    "q": 1.0
  },
  "performance_metrics": {
    "avg_jct": 5.32,
    "avg_waiting_time": 2.45,
    "p99_jct": 12.34,
    "throughput": 3.2
  },
  "fairness_metrics": {
    "jain_index": 0.92,
    "aw_jct": 4.87,
    "quadrant_analysis": {
      "depression": {"avg_wait": 1.8, "count": 12},
      "panic": {"avg_wait": 2.5, "count": 15},
      "calm": {"avg_wait": 2.6, "count": 11},
      "excited": {"avg_wait": 2.9, "count": 12}
    }
  }
}
```

---

## Troubleshooting

### Issue 1: BERT Predictor Not Available

**Symptoms:**
```
length_predictor.enabled: false
Using default_service_time: 2.0
```

**Solution:**
The BERT predictor requires a trained model. Either:
1. Train the model using `output-token-len-prediction/` scripts
2. Use the default service time (system still works without BERT)

### Issue 2: CUDA Out of Memory

**Solutions:**
```bash
# Use 8-bit quantization
uv run python run_simulation.py --load_in_8bit

# Use CPU
uv run python run_simulation.py --device_map cpu

# Use smaller model
uv run python run_simulation.py --model_name "mistralai/Mistral-7B-Instruct-v0.2"
```

### Issue 3: Import Errors

**Solution:**
```bash
uv sync
uv run python -c "import torch; print(torch.__version__)"
```

---

## References

### Core Papers

1. **Russell's Circumplex Model**: Russell, J. A. (1980). A circumplex model of affect. Journal of Personality and Social Psychology, 39(6), 1161-1178.

2. **WSPT Rule**: Smith, W. E. (1956). Various optimizers for single-stage production. Naval Research Logistics Quarterly, 3(1-2), 59-66.

3. **NRC-VAD-Lexicon**: Mohammad, S. M. (2018). Obtaining Reliable Human Ratings of Valence, Arousal, and Dominance for 20,000 English Words. ACL 2018.

4. **EmpatheticDialogues**: Rashkin et al. (2019). Towards Empathetic Open-domain Conversation Models. ACL 2019.

5. **Jain Fairness Index**: Jain et al. (1984). A Quantitative Measure of Fairness and Discrimination for Resource Allocation.

### LLM Serving

6. **Proxy Model Prediction**: Qiu et al. "Efficient Interactive LLM Serving with Proxy Model-based Sequence Length Prediction"

---

## License

This project is licensed under the Apache License 2.0.

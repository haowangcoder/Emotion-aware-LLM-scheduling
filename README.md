# Affect-Aware LLM Scheduling System (AW-SSJF)

An affect-aware GPU task scheduling system for LLM inference workloads that implements the **Affect-Weighted Shortest-Service-Job-First (AW-SSJF)** algorithm. The system uses BERT-based length prediction for service time estimation and Russell's Circumplex Model for emotion-aware scheduling prioritization.

## Overview

This project implements a novel scheduling framework that:

- **BERT Bucket Length Prediction**: Uses a BERT bucket classifier + expected value to estimate output length → service time
- **Affect Weighting**: Depression-First urgency (v1 hard gating) plus optional v2 soft/dual-channel variants
- **WSPT-Style Scheduling**: Scores jobs with `Score = S / w^k` (k = `weight_exponent`, configurable)
- **Fairness Analysis**: Reports per-Russell-quadrant latency and Jain fairness indices
- **Experiment Modes**: Fixed job count (`fixed_jobs`) and throughput-in-window (`time_window`), plus optional bursty arrivals (MMPP)

### Key Design Principles

| Component | Design |
|-----------|--------|
| **Service Time** | Prompt-based length→time prediction (content-based) |
| **Emotion Role** | Affects scheduling weight (priority) |
| **Scheduling** | `S / w^k` with optional robustness knobs |
| **Fairness Goal** | Prioritize vulnerable users without collapsing throughput |

## Quick Start

Note: `run_simulation.py` runs real HuggingFace model inference (and may download weights on first run).

**Run with FCFS baseline (fixed-jobs mode):**
```bash
uv run python run_simulation.py --mode fixed_jobs --scheduler FCFS --num_jobs 50 --output_dir results/quickstart --force_new_job_config
```

**Run AW-SSJF scheduler on the same saved workload (apples-to-apples):**
```bash
uv run python run_simulation.py --mode fixed_jobs --scheduler AW-SSJF --num_jobs 50 --w_max 2.0 --weight_exponent 4 --output_dir results/quickstart --use_saved_job_config
```

**Time-window throughput mode:**
```bash
uv run python run_simulation.py --mode time_window --scheduler AW-SSJF --simulation_duration 120 --num_jobs 80 --output_dir results/quickstart_time_window
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

**Python Version**: This repo targets Python **3.12** (see `.python-version`).

**HuggingFace Authentication** (required for gated models like LLaMA):
```bash
uv run huggingface-cli login
```

**Download EmpatheticDialogues Dataset:**

```bash
# Download and extract (expects CSVs at dataset/train.csv, dataset/valid.csv, dataset/test.csv)
mkdir -p dataset
wget https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz
tar -xzf empatheticdialogues.tar.gz
mv empatheticdialogues/*.csv dataset/

# Verify
ls dataset/
# Expected: train.csv, valid.csv, test.csv
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
│        │            │ valence   │      │ service   │      │ S / w^k  │    │
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
│   │   ├── affect_weight.py    # Depression-First weight calculation (v1)
│   │   ├── affect_weight_v2.py # Soft/dual-channel weights + presets (v2)
│   │   ├── scheduler_base.py   # Base scheduler & FCFS/SJF
│   │   ├── aw_ssjf_scheduler.py    # AW-SSJF scheduler (main)
│   │   ├── weight_only_scheduler.py # Weight-Only scheduler (ablation)
│   │   └── adaptive_k_controller.py # Online k control (optional)
│   │
│   ├── predictor/              # BERT length prediction
│   │   ├── bert_predictor.py   # BERT model wrapper
│   │   ├── length_estimator.py # Unified prediction interface
│   │   └── early_prompt_generator.py # Prompt + early prediction
│   │
│   ├── llm/                    # LLM integration
│   │   ├── engine.py           # Model loading & generation
│   │   ├── dataset_loader.py   # Dataset loading
│   │   ├── inference_handler.py # LLM execution + caching
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
├── scripts/                    # Experiment runner scripts (bash)
├── experiments/                # Experiment sweep + aggregation scripts (Python)
├── analysis/                   # Plotting scripts (Python)
├── slurm/                      # Slurm jobs (optional)
├── tools/                      # NRC-VAD extraction utilities
└── results/                    # Experiment outputs
```

---

## Key Algorithms

### 1. BERT Length Prediction

Service time is predicted by a BERT bucket predictor (content-based, emotion-independent):

```
T_mean = Σ(q_i × m_i)                    # expected output token length
Service Time = const_latency + T_mean × per_token_latency
```

Where:
- `q_i`: predicted probability of token-length bin i (bucket classifier)
- `m_i`: midpoint of bin i (from `model-serving/predictor/models/bin_edges.npy`)
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

**v2.0 (optional)**: The scheduler also supports soft gating (`soft`) and dual-channel weighting (`dual`) that can give limited priority to the panic quadrant via `gamma_panic` (see `model-serving/core/affect_weight_v2.py`).

### 3. WSPT Scheduling Rule

Jobs are sorted by WSPT score (lower = higher priority):

```
Score_i = S_i / w_i^k
```

Where:
- `S_i`: Predicted service time (from BERT)
- `w_i`: Affect weight (from Depression-First formula)
- `k`: Weight exponent (`scheduler.weight_exponent`) to tune fairness/throughput trade-off

### 4. Russell Quadrant Classification

Emotions are classified into four quadrants based on valence and arousal:

| Quadrant | Valence | Arousal | Example Emotions | Priority Boost |
|----------|---------|---------|------------------|----------------|
| **Excited** | ≥ 0 | ≥ 0 | excited, joyful, surprised | None (w=1) |
| **Calm** | ≥ 0 | < 0 | hopeful, grateful, content | None (w=1) |
| **Panic** | < 0 | ≥ 0 | terrified, anxious, angry | None in v1 hard gating; optional boost in v2 dual-channel |
| **Depression** | < 0 | < 0 | sad, lonely, disappointed | Yes (w>1) |

---

## Configuration

### Configuration File: `model-serving/config/default.yaml`

```yaml
# Scheduler Configuration
scheduler:
  algorithm: 'FCFS'  # FCFS | SJF | SSJF | AW-SSJF | Weight-Only
  system_load: 0.6

  # AW-SSJF scoring: Score = S / w^k
  weight_exponent: 4.0

  # Optional robustness knobs (AW-SSJF)
  use_robust_scoring: false            # Score = log(S+1) / w^k
  use_conservative_prediction: false   # S := margin * S before scoring
  conservative_margin: 1.3

  # Affect Weight Parameters
  affect_weight:
    # Can be 'hard' | 'soft' | 'dual' or a preset name (see core/affect_weight_v2.py)
    weight_mode: 'dual_channel_balanced'
    w_max: 2.0         # Maximum weight [1.2, 3.0] recommended
    p: 1.0             # Negative valence exponent
    q: 1.0             # Low arousal exponent
    gamma_panic: 0.3   # Dual-channel only
    use_confidence: true

  # Starvation Prevention
  starvation_prevention:
    threshold: .inf
    coefficient: 3.0

# BERT Bucket Length Predictor
length_predictor:
  enabled: true
  model_path: 'predictor/models/bert_bucket'        # HuggingFace model directory
  bin_edges_path: 'predictor/models/bin_edges.npy'  # Bin edges
  model_name: 'distilbert-base-uncased'
  device: 'cuda'
  num_bins: 5
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

dataset:
  emotion_dataset_path: './dataset'

output:
  results_dir: 'results/llm_runs/'
```

### Command-Line Arguments

```bash
uv run python run_simulation.py \
  --scheduler AW-SSJF \
  --mode fixed_jobs \
  --num_jobs 100 \
  --w_max 2.0 \
  --weight_exponent 4 \
  --p 1.0 \
  --q 1.0 \
  --system_load 0.6 \
  --output_dir results/llm_runs/
```

Defaults come from `model-serving/config/default.yaml`; CLI flags override config.

| Argument | Purpose |
|----------|---------|
| `--scheduler` | Scheduler: FCFS, SJF, SSJF, AW-SSJF, Weight-Only |
| `--mode` | `fixed_jobs` (latency) or `time_window` (throughput-in-window) |
| `--num_jobs` | Jobs to generate (fixed_jobs) / trace size seed (time_window) |
| `--simulation_duration` | Window length in seconds (time_window only) |
| `--system_load` | Target system load ρ (sets arrival rate λ = ρ / E[S]) |
| `--w_max`, `--p`, `--q` | Affect weight parameters |
| `--weight_exponent` | k in `S / w^k` (AW-SSJF only) |
| `--weight_mode` | `hard` / `soft` / `dual` (v2 weights) |
| `--mmpp_enabled`, `--mmpp_*` | Enable bursty arrivals (MMPP) |
| `--disable_predictor` | Disable length predictor (constant service time) |
| `--model_name`, `--device_map`, `--dtype`, `--load_in_8bit` | LLM runtime options |
| `--output_dir` | Output directory (writes `cache/` under it) |

Run `uv run python run_simulation.py --help` for the full list.

---

## Usage

### Fixed-Jobs vs Time-Window

- `fixed_jobs`: schedules a fixed number of jobs until all complete; best for JCT/waiting-time comparisons.
- `time_window`: schedules jobs for a fixed wall-clock window and counts completions; best for throughput comparisons. The arrival trace is cached to `<output_dir>/cache/time_window_trace.json`.

### Reproducible Comparisons

For fair comparisons across schedulers in `fixed_jobs` mode, reuse the same job config file by keeping a shared `--output_dir`:

```bash
# Generate and save job config (writes <output_dir>/cache/job_configs.json)
uv run python run_simulation.py --mode fixed_jobs --scheduler FCFS --num_jobs 80 --output_dir results/compare --force_new_job_config

# Reuse the exact same jobs/prompts
uv run python run_simulation.py --mode fixed_jobs --scheduler AW-SSJF --num_jobs 80 --output_dir results/compare --use_saved_job_config
```

### Experiment Scripts

- `bash scripts/run_experiments_fixed.sh` (fixed-jobs comparisons + plots)
- `bash scripts/run_experiments_time-window.sh` (time-window comparisons + plots)

## Scheduling Algorithms

### Available Schedulers

| Scheduler | Score Formula | Description |
|-----------|---------------|-------------|
| **FCFS** | Arrival time | First-Come-First-Serve baseline |
| **SJF** | `S` | Shortest-Job-First (service time only) |
| **SSJF** | `S` | Speculative SJF (alias for SJF) |
| **AW-SSJF** | `S / w^k` | **Main algorithm**: balances efficiency and fairness |
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
<output_dir>/
├── experiment_<timestamp>.log        # Full stdout log
├── <experiment_name>_jobs.csv        # Per-job logs
├── <experiment_name>_summary.json    # Aggregated metrics + fairness analysis
├── adaptive_k_trajectory.json        # Present when adaptive k is enabled
└── cache/
    ├── responses.json                # Cached LLM responses
    ├── job_configs.json              # Saved job configurations (fixed_jobs mode)
    └── time_window_trace.json        # Saved arrival trace (time_window mode)
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
| `predicted_serving_time` | Predicted service time used for scheduling |
| `actual_serving_time` | Measured inference time (cache hit or real generation) |
| `arrival_time`, `start_time`, `finish_time` | Timestamps |
| `waiting_time`, `turnaround_time` | Queueing + completion metrics |
| `output_token_length`, `cached`, `model_name` | LLM-related fields |

### Summary JSON

`<experiment_name>_summary.json` includes run metadata, overall metrics (waiting/JCT/throughput), per-quadrant metrics, and a serialized fairness analysis (Jain index, CV, and related breakdowns).

---

## Troubleshooting

### Issue 1: Length Predictor Not Available / Disabled

**Symptoms:**
```
⚠ Predictor not available, using default service time
```

**Solution:**
- Ensure `model-serving/predictor/models/bert_bucket/` and `model-serving/predictor/models/bin_edges.npy` exist.
- To run without the predictor, pass `--disable_predictor` (uses `default_service_time`).
- To retrain, see `model-serving/predictor/training/` (helper script: `scripts/predictor/train.sh`).

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

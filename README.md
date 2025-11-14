# Emotion-aware LLM Scheduling System

An emotion-aware GPU task scheduling system for LLM inference workloads that integrates real language models with emotion-based scheduling strategies. The system maps user emotions (specifically arousal levels) to predict task characteristics and evaluates different scheduling algorithms for fairness and efficiency.

## Overview

This project explores how emotional context affects LLM inference task scheduling by:

- **Real LLM Integration**: Runs actual HuggingFace models (e.g., Meta-Llama-3-8B-Instruct) generating empathetic responses
- **Emotion-aware Workload**: Uses the EmpatheticDialogues dataset with 32 emotion categories mapped to arousal levels
- **Scheduling Algorithms**: Compares FCFS (baseline) vs SSJF-Emotion (prioritizes predicted shorter tasks)
- **Fairness Analysis**: Evaluates scheduling fairness across emotion categories using comprehensive metrics

## Quick Start

**Run with real LLM (10 tasks):**
```bash
uv run python run_simulation.py --scheduler FCFS --num_jobs 10 --verbose
```

**Run SSJF-Emotion scheduler:**
```bash
uv run python run_simulation.py --scheduler SSJF-Emotion --num_jobs 50
```

**Compare schedulers:**
```bash
# Run FCFS
uv run python run_simulation.py --scheduler FCFS --num_jobs 100 --output_dir results/llm_runs

# Run SSJF-Emotion (uses cached responses for fair comparison)
uv run python run_simulation.py --scheduler SSJF-Emotion --num_jobs 100 --output_dir results/llm_runs
```

## Table of Contents

- [Installation](#installation)
- [System Architecture](#system-architecture)
- [Key Formulas](#key-formulas)
- [Configuration](#configuration)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)
- [Research Directions](#research-directions)
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
# Clone the repository
cd Emotion-aware-LLM-scheduling

# Sync dependencies (creates virtual environment automatically)
uv sync
```

### 3. Additional Setup for LLM Integration

**HuggingFace Authentication** (required for gated models like LLaMA):
```bash
# Install HuggingFace CLI
uv pip install huggingface-hub

# Login with your token
uv run huggingface-cli login
# Get token from: https://huggingface.co/settings/tokens
```

**Accept model license** (for Llama):
1. Go to https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
2. Click "Agree and access repository"
3. Fill out the form

**Download and setup dataset** (EmpatheticDialogues):

1. Download the dataset from Kaggle:
```bash
# Download the dataset using Kaggle API
curl -L -o ~/Downloads/empathetic-dialogues-facebook-ai.zip \
  https://www.kaggle.com/api/v1/datasets/download/atharvjairath/empathetic-dialogues-facebook-ai
```

2. Extract and setup:
```bash
# Unzip the downloaded file
unzip ~/Downloads/empathetic-dialogues-facebook-ai.zip -d ~/Downloads/empathetic-dialogues

# Create dataset directory in project root
mkdir -p dataset

# Copy and rename the files
cp ~/Downloads/empathetic-dialogues/train.csv dataset/train.csv
cp ~/Downloads/empathetic-dialogues/valid.csv dataset/valid.csv
cp ~/Downloads/empathetic-dialogues/test.csv dataset/test.csv
```

3. Verify dataset:
```bash
ls -la dataset/
# Expected: train.csv, valid.csv, test.csv
```

---

## System Architecture

### Project Structure

```
Emotion-aware-LLM-scheduling/
├── run_simulation.py           # Main entry point
├── pyproject.toml              # uv project configuration
├── dataset/                    # EmpatheticDialogues dataset
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
├── model-serving/              # Main package (refactored)
│   ├── simulator.py            # Scheduling simulator
│   ├── config/                 # Configuration modules
│   │   ├── __init__.py        # Unified config exports
│   │   ├── base.py            # General settings
│   │   ├── llm_config.py      # LLM model & generation
│   │   ├── scheduler_config.py # Scheduling parameters
│   │   └── workload_config.py  # Task generation settings
│   ├── core/                   # Core scheduling logic
│   │   ├── emotion.py         # Emotion sampling & mapping
│   │   ├── job.py             # Job data structure
│   │   ├── scheduler_base.py  # Base scheduler & FCFS
│   │   └── ssjf_emotion.py    # SSJF-Emotion scheduler
│   ├── llm/                    # LLM integration
│   │   ├── engine.py          # Model loading & generation
│   │   ├── dataset_loader.py  # Dataset loading
│   │   ├── prompt_builder.py  # Prompt construction
│   │   ├── response_cache.py  # Response caching
│   │   └── inference_handler.py # LLM orchestration
│   ├── workload/               # Workload generation
│   │   ├── service_time_mapper.py # Arousal to time mapping
│   │   └── task_generator.py      # Job generation
│   └── analysis/               # Results analysis
│       ├── logger.py          # Logging utilities
│       └── fairness_metrics.py # Fairness calculations
├── analysis/                   # Visualization tools
│   └── plot_emotion_results.py
└── results/                    # Experiment outputs
    ├── llm_runs/              # Experiment results
    └── cache/                 # Response cache
        └── responses.json
```

### Data Flow

```
1. Configuration Loading (config/)
   ↓
2. Emotion Sampling (core/emotion.py)
   - Sample from 32 emotion categories
   - Map emotion → arousal value [-1, 1]
   ↓
3. Task Generation (workload/task_generator.py)
   - Generate arrival times (emotion-adjusted rates)
   - Map arousal → predicted service time
   ↓
4. LLM Initialization (llm/engine.py, llm/inference_handler.py)
   - Load HuggingFace model
   - Load EmpatheticDialogues dataset
   ↓
5. Scheduling (core/scheduler_base.py, core/ssjf_emotion.py)
   - FCFS: First-come-first-served
   - SSJF-Emotion: Shortest predicted service time first
   ↓
6. Job Execution (simulator.py)
   - Build empathetic prompt (llm/prompt_builder.py)
   - Generate response with real LLM (llm/engine.py)
   - Measure actual execution time
   - Cache responses (llm/response_cache.py)
   ↓
7. Logging & Analysis (analysis/)
   - Calculate performance metrics
   - Compute fairness indices
   - Generate visualizations
```

---

## Key Formulas

### Service Time Mapping
```
S_i = L_0 * (1 + α * a_i)
```
Where:
- `S_i`: Predicted service time for task i
- `L_0`: Base service time (default: 2.0s)
- `α` (alpha): Sensitivity coefficient (default: 0.5, range: 0-1)
- `a_i`: Arousal value for task i (range: [-1, 1])

**Note:** This is the *predicted* service time used for scheduling. Actual service time is measured from real LLM inference.

### Arrival Rate Modification
```
λ(a) = λ_0 * (1 + γ * a)
```
Where:
- `λ(a)`: Arrival rate for arousal level a
- `λ_0`: Base arrival rate
- `γ` (gamma): Sensitivity coefficient (default: 0.3, range: 0-1)

### System Load
```
ρ = (λ * E[S]) / N
```
Where:
- `ρ` (rho): System utilization/load
- `λ`: Task arrival rate
- `E[S]`: Expected service time
- `N`: Number of GPUs (N=1 in current setup)

---

## Configuration

The system uses a modular configuration structure organized in `model-serving/config/`:

### Core Configuration Files

**`config/base.py`**: General experiment settings
- `ENABLE_EMOTION_AWARE`: Enable emotion features (default: True)
- `EMOTION_DATASET_PATH`: Path to EmpatheticDialogues dataset
- `RESULTS_DIR`: Output directory for results
- `CACHE_DIR`: Cache directory for responses

**`config/llm_config.py`**: LLM model and generation settings
- `LLM_MODEL_NAME`: HuggingFace model ID (default: 'meta-llama/Meta-Llama-3-8B-Instruct')
- `LLM_DEVICE_MAP`: Device mapping ('auto', 'cuda', 'cpu')
- `LLM_MAX_NEW_TOKENS`: Max tokens to generate (default: 64)
- `LLM_TEMPERATURE`: Sampling temperature (default: 0.7)
- `USE_RESPONSE_CACHE`: Enable caching (default: True)

**`config/scheduler_config.py`**: Scheduling algorithm settings
- `SCHEDULER_ALGORITHM`: 'FCFS' or 'SSJF-Emotion'
- `SYSTEM_LOAD`: Target system utilization (default: 0.6)
- `NUM_GPUS`: Number of GPUs (default: 1)

**`config/workload_config.py`**: Workload generation parameters
- `BASE_SERVICE_TIME`: L_0 base time (default: 2.0s)
- `ALPHA`: Arousal impact on service time (default: 0.5)
- `GAMMA`: Arousal impact on arrival rate (default: 0.3)
- `RHO`: Emotion-system correlation (default: 1.0)

### Environment Variables

Override any configuration using environment variables:

```bash
# LLM Configuration
export LLM_MODEL_NAME_ENV="mistralai/Mistral-7B-Instruct-v0.2"
export LLM_DEVICE_MAP_ENV="cpu"
export LLM_MAX_NEW_TOKENS_ENV=128

# Scheduling Configuration
export SCHEDULER_ALGORITHM_ENV="SSJF-Emotion"
export SYSTEM_LOAD_ENV=0.8

# Workload Configuration
export ALPHA_ENV=0.7
export GAMMA_ENV=0.5

uv run python run_simulation.py --num_jobs 100
```

---

## Usage

### Basic Usage

**Run FCFS scheduler:**
```bash
uv run python run_simulation.py \
  --scheduler FCFS \
  --num_jobs 50 \
  --verbose
```

**Run SSJF-Emotion scheduler:**
```bash
uv run python run_simulation.py \
  --scheduler SSJF-Emotion \
  --num_jobs 50 \
  --verbose
```

### Advanced Options

**Use CPU (if no GPU available):**
```bash
uv run python run_simulation.py \
  --scheduler FCFS \
  --num_jobs 10 \
  --device_map cpu \
  --verbose
```

**Use 8-bit quantization (save memory):**
```bash
uv run python run_simulation.py \
  --scheduler FCFS \
  --num_jobs 20 \
  --load_in_8bit \
  --verbose
```

**Use different model:**
```bash
uv run python run_simulation.py \
  --scheduler FCFS \
  --num_jobs 10 \
  --model_name "mistralai/Mistral-7B-Instruct-v0.2" \
  --verbose
```

**Adjust system parameters:**
```bash
uv run python run_simulation.py \
  --scheduler SSJF-Emotion \
  --num_jobs 100 \
  --system_load 0.8 \
  --alpha 0.7 \
  --gamma 0.5
```

### Comparing Schedulers

For fair comparison, use response caching:

```bash
# First run: FCFS (generates and caches responses)
uv run python run_simulation.py \
  --scheduler FCFS \
  --num_jobs 100 \
  --output_dir results/llm_runs/

# Second run: SSJF-Emotion (uses cached responses)
uv run python run_simulation.py \
  --scheduler SSJF-Emotion \
  --num_jobs 100 \
  --output_dir results/llm_runs/
```

### Generating Visualizations

```bash
# Generate comparison plots from results
uv run python analysis/plot_emotion_results.py
```

---

## Evaluation Metrics

### Performance Metrics

- **Average Waiting Time (R̄)**: Mean time jobs wait before execution
- **P99 Tail Latency**: 99th percentile of waiting/completion time
- **Throughput**: Jobs completed per unit time
- **Average Turnaround Time**: Mean time from arrival to completion

### Fairness Metrics

**Jain Fairness Index (J)**:
```
J = (Σx_i)² / (n * Σx_i²)
```
Range: [1/n, 1] where 1 = perfect fairness

**Additional Fairness Metrics**:
- Per-emotion-class statistics (high/medium/low arousal groups)
- Coefficient of Variation (CV): Std deviation / mean
- Max/Min Ratio: Ratio of maximum to minimum class performance

### LLM-specific Metrics

- **Prediction Accuracy**: Compare predicted vs actual execution time
- **Cache Hit Rate**: Percentage of responses served from cache
- **Average Output Token Length**: Mean length of generated responses
- **Error Rate**: Percentage of failed generations

---

## Output Files

Each experiment produces:

```
results/llm_runs/
├── <SCHEDULER>_<N>jobs_load<LOAD>_jobs.csv      # Per-job detailed logs
├── <SCHEDULER>_<N>jobs_load<LOAD>_summary.json  # Aggregated statistics
└── cache/
    └── responses.json                            # Cached LLM responses (shared)
```

### CSV Log Format

**Standard fields:**
- `job_id`, `emotion`, `arousal`, `valence`
- `arrival_time`, `predicted_service_time`, `actual_execution_duration`
- `start_time`, `completion_time`, `waiting_time`, `turnaround_time`

**LLM-specific fields:**
- `response_text`: Full generated response from LLM
- `output_token_length`: Number of tokens in response
- `cached`: Whether response was from cache
- `error_msg`: Error message if generation failed
- `model_name`: HuggingFace model identifier
- `conversation_context_preview`: First 200 chars of prompt

### Summary JSON Structure

```json
{
  "experiment_config": {
    "scheduler": "SSJF-Emotion",
    "num_jobs": 100,
    "system_load": 0.6,
    "alpha": 0.5,
    "gamma": 0.3
  },
  "performance_metrics": {
    "avg_waiting_time": 15.32,
    "p99_waiting_time": 45.67,
    "avg_turnaround_time": 18.21,
    "throughput": 5.49
  },
  "fairness_metrics": {
    "jain_fairness_index": 0.87,
    "coefficient_of_variation": 0.42
  },
  "llm_metrics": {
    "avg_actual_execution_time": 2.89,
    "avg_output_token_length": 52.4,
    "cache_hit_rate": 0.45,
    "prediction_accuracy": {
      "avg_relative_error": 0.12,
      "median_relative_error": 0.08
    }
  }
}
```

---

## Troubleshooting

### Issue 1: Import Errors

**Symptoms:**
```
ImportError: No module named 'torch'
ModuleNotFoundError: No module named 'transformers'
```

**Solution:**
```bash
# Re-sync dependencies
uv sync

# Verify installation
uv run python -c "import torch; print(torch.__version__)"
uv run python -c "import transformers; print(transformers.__version__)"
```

### Issue 2: HuggingFace Authentication

**Symptoms:**
```
Repository not found
401 Client Error: Unauthorized
```

**Solution:**
```bash
# Login to HuggingFace
uv pip install huggingface-hub
uv run huggingface-cli login

# Accept model license at:
# https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
```

### Issue 3: CUDA Out of Memory

**Symptoms:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**

**Use 8-bit quantization:**
```bash
uv run python run_simulation.py \
  --scheduler FCFS \
  --num_jobs 10 \
  --load_in_8bit
```

**Use CPU:**
```bash
uv run python run_simulation.py \
  --scheduler FCFS \
  --num_jobs 10 \
  --device_map cpu
```

**Use smaller model:**
```bash
uv run python run_simulation.py \
  --scheduler FCFS \
  --num_jobs 10 \
  --model_name "mistralai/Mistral-7B-Instruct-v0.2"
```

### Issue 4: Dataset Not Found

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'dataset/train.csv'
```

**Solution:**
```bash
# Verify dataset exists
ls -la dataset/
# Should show: train.csv, valid.csv, test.csv

# Ensure running from project root
cd /path/to/Emotion-aware-LLM-scheduling
uv run python run_simulation.py --num_jobs 10
```

### Issue 5: Slow First Run

**Explanation:**
This is normal for first run:
- Model download: 15-30 GB, takes 10-30 minutes (one-time)
- Model loading: 1-2 minutes each run
- First inference: 30-60s (PyTorch compilation, then faster)

**Solution:**
Use caching for subsequent runs:
```bash
# First run: generates and caches
uv run python run_simulation.py --num_jobs 50

# Second run: uses cache (much faster)
uv run python run_simulation.py --num_jobs 50
```

### Diagnostic Commands

**Full system check:**
```bash
echo "=== Python Version ==="
uv run python --version

echo "=== Dependencies ==="
uv run python -c "import torch; print('PyTorch:', torch.__version__)"
uv run python -c "import transformers; print('Transformers:', transformers.__version__)"

echo "=== CUDA Check ==="
uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

echo "=== Dataset Check ==="
ls -lh dataset/*.csv

echo "=== Component Test ==="
uv run python -c "from model-serving.llm.engine import LLMEngine; print('LLM Engine: OK')"
```

---

## Emotion Categories

Based on EmpatheticDialogues dataset (32 emotions mapped to arousal levels):

### High Arousal (0.6 to 1.0)
- **Positive**: excited, joyful, surprised, anticipating, impressed, proud
- **Negative**: terrified, afraid, anxious, angry, furious, annoyed, disgusted

### Medium Arousal (-0.3 to 0.6)
- hopeful, trusting, caring, grateful, confident, jealous, embarrassed, sentimental, nostalgic, content, prepared, apprehensive, guilty, ashamed

### Low Arousal (-1.0 to -0.3)
- sad, lonely, disappointed, devastated, depressed, bored

---

## References

### Core Papers

1. **EmpatheticDialogues Dataset**: Rashkin et al., "Towards Empathetic Open-domain Conversation Models: a New Benchmark and Dataset", ACL 2019
2. **Jain Fairness Index**: Jain et al., "A Quantitative Measure of Fairness and Discrimination for Resource Allocation in Shared Computer Systems", 1984
3. **Russell's Circumplex Model**: For arousal-valence mapping in affective computing
4. **GPU Scheduling**: Qiu et al., "Efficient Interactive LLM Serving with Proxy Model-based Sequence Length Prediction"

### LLM Serving & Inference

5. **HuggingFace Transformers**: https://huggingface.co/docs/transformers
6. **Meta LLaMA**: https://huggingface.co/meta-llama
7. **LLM Inference Optimization**: Pope et al., "Efficiently Scaling Transformer Inference", MLSys 2023
8. **Quantization**: Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale", NeurIPS 2022

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for the full text or visit http://www.apache.org/licenses/LICENSE-2.0.

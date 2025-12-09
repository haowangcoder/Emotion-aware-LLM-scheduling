#!/bin/bash
#SBATCH --job-name=emotion_multi_seed
#SBATCH --partition=gpu-invest
#SBATCH --nodelist=gnode26
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=Emotion-aware-LLM-scheduling/slurm/slurm_logs/%x_%j.out
#SBATCH --error=Emotion-aware-LLM-scheduling/slurm/slurm_logs/%x_%j.err

set -e

# Change to project directory
cd /storage/homefs/hw24w089/Emotion-aware-LLM-scheduling

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

echo "=== GPU Info ==="
nvidia-smi
echo ""

# Run multi-seed robustness testing
./scripts/run_multi_seed.sh

echo ""
echo "=== Job completed at $(date) ==="

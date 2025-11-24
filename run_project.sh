#!/bin/bash
# ==========================================================
# Emotion-aware-LLM-scheduling 自动化运行脚本
# 版本：2025-11
# 功能：
#   1. 自动加载 CUDA 模块（若可用）
#   2. 自动激活虚拟环境
#   3. 自动检测 HuggingFace 登录（如未登录则用 Token）
#   4. 自动运行项目主脚本
#   5. 打印运行信息与结果路径
# ==========================================================

# --- Step 1. 加载 CUDA 环境（如果可用） ---
if module avail CUDA >/dev/null 2>&1; then
    module load CUDA/12.1.1 2>/dev/null && echo "[INFO] CUDA 12.1.1 environment loaded."
else
    echo "[WARN] CUDA module not found — running in CPU mode."
fi

# --- Step 2. 激活虚拟环境 ---
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "env" ]; then
    source env/bin/activate
else
    echo "[ERROR] No virtual environment found. Please run 'uv sync' first."
    exit 1
fi
echo "[INFO] Virtual environment activated."

# --- Step 3. HuggingFace 登录检测 ---
echo "[INFO] Checking HuggingFace authentication..."
HF_DIR="$HOME/.huggingface"
HF_TOKEN_FILE="$HF_DIR/token"

# 若不存在 token 文件，则尝试从环境变量创建
if [ ! -f "$HF_TOKEN_FILE" ]; then
    if [ -z "$HUGGINGFACE_TOKEN" ]; then
        echo "[ERROR] HuggingFace token not found!"
        echo "Please export it first, e.g.:"
        echo "  export HUGGINGFACE_TOKEN='hf_xxxxxxxxxxxxxxxxx'"
        exit 1
    fi
    echo "[INFO] No token file found — creating from environment variable..."
    mkdir -p "$HF_DIR"
    echo "$HUGGINGFACE_TOKEN" > "$HF_TOKEN_FILE"
    chmod 600 "$HF_TOKEN_FILE"
    echo "[INFO] HuggingFace token saved to $HF_TOKEN_FILE"
else
    echo "[INFO] Found existing HuggingFace token file."
fi

# 验证登录状态
python - <<'EOF'
from huggingface_hub import whoami
try:
    info = whoami()
    print(f"[INFO] Logged in to HuggingFace as: {info['name']}")
except Exception as e:
    print(f"[WARN] HuggingFace login check failed: {e}")
EOF

# --- Step 4. 参数设置 ---
MODEL_NAME=${1:-"mistralai/Mistral-7B-Instruct-v0.2"}  # 默认公开模型
SCHEDULER=${2:-"FCFS"}
NUM_JOBS=${3:-10}
DEVICE=${4:-"cpu"}  # 可改为 cuda
LOAD_IN_8BIT=${5:-""}

echo "-----------------------------------------------"
echo "[INFO] Configuration:"
echo "  Model:     $MODEL_NAME"
echo "  Scheduler: $SCHEDULER"
echo "  Jobs:      $NUM_JOBS"
echo "  Device:    $DEVICE"
echo "-----------------------------------------------"

# --- Step 5. 运行主程序 ---
CMD="uv run python run_simulation.py \
  --scheduler $SCHEDULER \
  --num_jobs $NUM_JOBS \
  --model_name \"$MODEL_NAME\" \
  --device_map $DEVICE \
  --verbose"

# 如果要求使用 8-bit 模式
if [ "$LOAD_IN_8BIT" == "8bit" ]; then
  CMD="$CMD --load_in_8bit"
fi

echo "[INFO] Executing: $CMD"
eval $CMD

# --- Step 6. 输出总结 ---
echo "==============================================="
echo "[DONE] Simulation finished."
echo "Results are stored under: results/llm_runs/"
echo "==============================================="


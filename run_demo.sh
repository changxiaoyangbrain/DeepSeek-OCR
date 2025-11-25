#!/usr/bin/env bash
set -euo pipefail

# One-click launcher for DeepSeek-OCR vLLM Gradio demo

# Activate conda env
if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
  source /root/miniconda3/etc/profile.d/conda.sh
fi

conda activate deepseek-ocr || {
  echo "[ERR] conda env 'deepseek-ocr' not found" >&2
  exit 1
}

# Offline + CUDA envs
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export DEMO_PORT=${DEMO_PORT:-7860}

# 启动时预热模型，使其常驻显存，首次识别无需等待加载
export WARMUP_ON_START=${WARMUP_ON_START:-1}

WATCH_MODE=${1:-}

if [[ "$WATCH_MODE" == "--watch" ]]; then
  echo "[INFO] Launching Gradio vLLM Demo with watch mode on port ${DEMO_PORT}"
  echo "[INFO] Auto-restart on file changes under current directory"
  # lightweight watcher using inotifywait if available; otherwise fallback to loop
  if command -v inotifywait >/dev/null 2>&1; then
    # initial start
    python -u gradio_vllm_demo.py &
    PID=$!
    EXCLUDE_REGEX='^./outputs/.*|^./models/.*'
    while true; do
      inotifywait -e modify,create,delete,move -r --exclude "$EXCLUDE_REGEX" . >/dev/null 2>&1 || true
      echo "[INFO] Change detected. Restarting service..."
      kill "$PID" >/dev/null 2>&1 || true
      # graceful wait up to 10s
      for i in {1..20}; do
        kill -0 "$PID" 2>/dev/null || break
        sleep 0.5
      done
      kill -9 "$PID" >/dev/null 2>&1 || true
      sleep 0.5
      python -u gradio_vllm_demo.py &
      PID=$!
      sleep 1
    done
  else
    echo "[WARN] inotifywait not found. Using naive polling every 2s."
    # naive checksum-based polling
    checksum() {
      find . -type f \
        \( -name "*.py" -o -name "*.md" -o -name "*.json" -o -name "*.yaml" -o -name "*.yml" \) \
        -not -path "./outputs/*" -not -path "./models/*" \
        -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}';
    }
    CUR=$(checksum)
    python -u gradio_vllm_demo.py &
    PID=$!
    while true; do
      sleep 2
      NEW=$(checksum)
      if [[ "$NEW" != "$CUR" ]]; then
        echo "[INFO] Change detected. Restarting service..."
        CUR="$NEW"
        kill "$PID" >/dev/null 2>&1 || true
        for i in {1..20}; do
          kill -0 "$PID" 2>/dev/null || break
          sleep 0.5
        done
        kill -9 "$PID" >/dev/null 2>&1 || true
        sleep 0.5
        python -u gradio_vllm_demo.py &
        PID=$!
      fi
    done
  fi
else
  echo "[INFO] Launching Gradio vLLM Demo on port ${DEMO_PORT}"
  python -u gradio_vllm_demo.py
fi
#!/usr/bin/env bash
set -euo pipefail

# One-click launcher for DeepSeek-OCR vLLM Gradio demo

# ============================================
# 颜色定义
# ============================================
# 基础颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'

# 粗体颜色
BOLD_RED='\033[1;31m'
BOLD_GREEN='\033[1;32m'
BOLD_YELLOW='\033[1;33m'
BOLD_BLUE='\033[1;34m'
BOLD_CYAN='\033[1;36m'

# 背景颜色
BG_RED='\033[41m'
BG_GREEN='\033[42m'
BG_YELLOW='\033[43m'

# 重置
NC='\033[0m'

# ============================================
# 日志函数
# ============================================
log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓ OK]${NC} $1"
}

log_warning() {
    echo -e "${BOLD_YELLOW}[⚠ WARN]${NC} ${YELLOW}$1${NC}"
}

log_error() {
    echo -e "${BOLD_RED}[✗ ERROR]${NC} ${RED}$1${NC}"
}

log_fatal() {
    echo -e "${BG_RED}${WHITE}[FATAL]${NC} ${BOLD_RED}$1${NC}"
}

log_step() {
    echo -e "${BLUE}📍${NC} $1"
}

log_detail() {
    echo -e "   ${GREEN}✅${NC} $1"
}

log_detail_warn() {
    echo -e "   ${YELLOW}⚠️${NC}  $1"
}

# 默认端口
DEMO_PORT=${DEMO_PORT:-7860}

# ============================================
# 优雅退出处理函数
# ============================================
cleanup() {
    echo ""
    echo -e "${BOLD_CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD_YELLOW}🛑 正在优雅退出...${NC}"
    echo -e "${BOLD_CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # 1. 杀掉子进程
    if [[ -n "${PID:-}" ]] && kill -0 "$PID" 2>/dev/null; then
        log_step "正在停止 Gradio 服务 (PID: $PID)..."
        kill -TERM "$PID" 2>/dev/null || true
        # 等待进程优雅退出（最多10秒）
        for i in {1..20}; do
            kill -0 "$PID" 2>/dev/null || break
            sleep 0.5
        done
        # 强制杀掉
        kill -9 "$PID" 2>/dev/null || true
        log_detail "Gradio 服务已停止"
    fi
    
    # 2. 释放端口
    log_step "正在释放端口 ${DEMO_PORT}..."
    local port_pids=$(lsof -ti:${DEMO_PORT} 2>/dev/null || true)
    if [[ -n "$port_pids" ]]; then
        echo "$port_pids" | xargs -r kill -9 2>/dev/null || true
        log_detail "端口 ${DEMO_PORT} 已释放"
    else
        log_detail "端口 ${DEMO_PORT} 已空闲"
    fi
    
    # 3. 清理显存
    log_step "正在清理 GPU 显存..."
    python3 -c "
import gc
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if hasattr(torch.cuda, 'ipc_collect'):
            torch.cuda.ipc_collect()
        print('   \033[0;32m✅\033[0m GPU 显存已清理')
    else:
        print('   \033[0;36mℹ️\033[0m  无可用 GPU')
except Exception as e:
    print(f'   \033[0;33m⚠️\033[0m  显存清理时出现警告: {e}')
gc.collect()
" 2>/dev/null || echo -e "   ${YELLOW}⚠️${NC}  显存清理跳过"
    
    # 4. 杀掉所有相关的 Python 进程
    log_step "正在清理残留进程..."
    pkill -f "gradio_vllm_demo.py" 2>/dev/null || true
    log_detail "残留进程已清理"
    
    echo -e "${BOLD_CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD_GREEN}✅ 清理完成，再见！${NC}"
    echo -e "${BOLD_CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    exit 0
}

# 捕获 SIGINT (Ctrl+C) 和 SIGTERM 信号
trap cleanup SIGINT SIGTERM

# ============================================
# 启动前清理（确保端口可用、清理残留进程和显存）
# ============================================
pre_cleanup() {
    log_step "启动前检查..."
    
    # 1. 杀掉残留的 gradio_vllm_demo.py 进程
    local old_pids=$(pgrep -f "gradio_vllm_demo.py" 2>/dev/null || true)
    if [[ -n "$old_pids" ]]; then
        log_detail_warn "发现残留进程，正在清理..."
        pkill -9 -f "gradio_vllm_demo.py" 2>/dev/null || true
        sleep 1
    fi
    
    # 2. 释放端口
    local port_pids=$(lsof -ti:${DEMO_PORT} 2>/dev/null || true)
    if [[ -n "$port_pids" ]]; then
        log_detail_warn "端口 ${DEMO_PORT} 被占用，正在释放..."
        echo "$port_pids" | xargs -r kill -9 2>/dev/null || true
        sleep 1
        log_detail "端口已释放"
    else
        log_detail "端口 ${DEMO_PORT} 可用"
    fi
    
    # 3. 清理可能残留的显存
    python3 -c "
import gc
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'ipc_collect'):
            torch.cuda.ipc_collect()
except:
    pass
gc.collect()
" 2>/dev/null || true
    
    echo ""
}

# ============================================
# 主程序
# ============================================

# Activate conda env
if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
  source /root/miniconda3/etc/profile.d/conda.sh
fi

conda activate deepseek-ocr || {
  log_fatal "conda env 'deepseek-ocr' not found"
  exit 1
}

# Offline + CUDA envs
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export DEMO_PORT

# 修复 WSL 环境下的 hostname 警告
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# 启动时预热模型，使其常驻显存，首次识别无需等待加载
export WARMUP_ON_START=${WARMUP_ON_START:-1}

# 启动前清理端口
pre_cleanup

WATCH_MODE=${1:-}

echo -e "${BOLD_CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD_GREEN}🏥 长小养照护智能资源数字化平台${NC}"
echo -e "${BOLD_CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${WHITE}📍 服务地址:${NC} ${BOLD_CYAN}http://localhost:${DEMO_PORT}${NC}"
echo -e "${WHITE}📍 按${NC} ${BOLD_YELLOW}Ctrl+C${NC} ${WHITE}优雅退出并清理资源${NC}"
echo -e "${BOLD_CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# ============================================
# 日志过滤器 - 添加颜色
# ============================================
colorize_logs() {
    while IFS= read -r line; do
        # 致命错误 - 红色背景高亮
        if [[ "$line" =~ (FATAL|Fatal|fatal|Traceback|OutOfMemoryError|OOM|CUDA.*error) ]]; then
            echo -e "${BG_RED}${WHITE}$line${NC}"
        # 错误信息 - 红色高亮
        elif [[ "$line" =~ (ERROR|Error|error|Exception|exception|FAILED|Failed|failed) ]]; then
            echo -e "${BOLD_RED}$line${NC}"
        # 警告信息 - 黄色
        elif [[ "$line" =~ (WARNING|Warning|warning|WARN|Warn|warn|\[W[0-9]) ]]; then
            echo -e "${YELLOW}$line${NC}"
        # 成功/完成信息 - 绿色
        elif [[ "$line" =~ (SUCCESS|Success|success|DONE|Done|done|Completed|completed|finished|Finished|100%|Running\ on) ]]; then
            echo -e "${BOLD_GREEN}$line${NC}"
        # INFO 信息 - 青色
        elif [[ "$line" =~ ^INFO|^\[INFO\] ]]; then
            echo -e "${CYAN}$line${NC}"
        # 加载进度 - 蓝色
        elif [[ "$line" =~ (Loading|loading|Initializing|initializing|Starting|starting|Capturing) ]]; then
            echo -e "${BLUE}$line${NC}"
        # Gradio 服务启动信息 - 紫色
        elif [[ "$line" =~ ^\*|localhost|public\ link ]]; then
            echo -e "${BOLD_GREEN}$line${NC}"
        # 普通输出
        else
            echo "$line"
        fi
    done
}

if [[ "$WATCH_MODE" == "--watch" ]]; then
  log_info "启动 Watch 模式 - 文件变更时自动重启"
  # lightweight watcher using inotifywait if available; otherwise fallback to loop
  if command -v inotifywait >/dev/null 2>&1; then
    # initial start
    python -u gradio_vllm_demo.py 2>&1 | colorize_logs &
    PID=$!
    EXCLUDE_REGEX='^./outputs/.*|^./models/.*'
    while true; do
      inotifywait -e modify,create,delete,move -r --exclude "$EXCLUDE_REGEX" . >/dev/null 2>&1 || true
      log_info "检测到文件变更，正在重启服务..."
      kill "$PID" >/dev/null 2>&1 || true
      # graceful wait up to 10s
      for i in {1..20}; do
        kill -0 "$PID" 2>/dev/null || break
        sleep 0.5
      done
      kill -9 "$PID" >/dev/null 2>&1 || true
      sleep 0.5
      python -u gradio_vllm_demo.py 2>&1 | colorize_logs &
      PID=$!
      sleep 1
    done
  else
    log_warning "inotifywait 未安装，使用轮询模式（每2秒检查一次）"
    # naive checksum-based polling
    checksum() {
      find . -type f \
        \( -name "*.py" -o -name "*.md" -o -name "*.json" -o -name "*.yaml" -o -name "*.yml" \) \
        -not -path "./outputs/*" -not -path "./models/*" \
        -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}';
    }
    CUR=$(checksum)
    python -u gradio_vllm_demo.py 2>&1 | colorize_logs &
    PID=$!
    while true; do
      sleep 2
      NEW=$(checksum)
      if [[ "$NEW" != "$CUR" ]]; then
        log_info "检测到文件变更，正在重启服务..."
        CUR="$NEW"
        kill "$PID" >/dev/null 2>&1 || true
        for i in {1..20}; do
          kill -0 "$PID" 2>/dev/null || break
          sleep 0.5
        done
        kill -9 "$PID" >/dev/null 2>&1 || true
        sleep 0.5
        python -u gradio_vllm_demo.py 2>&1 | colorize_logs &
        PID=$!
      fi
    done
  fi
else
  log_info "启动 Gradio vLLM ..."
  echo ""
  python -u gradio_vllm_demo.py 2>&1 | colorize_logs &
  PID=$!
  # 等待子进程结束
  wait $PID
fi

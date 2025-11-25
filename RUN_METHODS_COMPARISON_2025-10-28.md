# DeepSeek-OCR 运行方法对比（HF vs vLLM）

日期：2025-10-28

## 概览

- 路径：
  - HF（Transformers）：`/root/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-hf/`
  - vLLM：`/root/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/`
- 核心差异：
  - 框架：HF 直接使用 `transformers.AutoModel/AutoTokenizer`；vLLM 使用 `LLM/AsyncLLMEngine` 与 KV Cache、高并发能力。
  - 模型代码来源：HF 默认 `trust_remote_code=True` 并从 HuggingFace 加载自定义模型代码；vLLM 通过本地 `ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)`，`trust_remote_code=False`。
  - 权重/分词器路径：HF 默认从 `deepseek-ai/DeepSeek-OCR` 远程拉取或本地快照；vLLM 使用本地权重目录 `MODEL_PATH` 与本地快照分词器 `TOKENIZER_PATH`。
  - 输入类型：HF 脚本示例为单张图片；vLLM 支持单图、PDF 转图、多图批量。
  - 离线能力：本仓库已为 vLLM 加入离线优先逻辑；HF 可通过环境变量与本地快照实现离线。

---

## 环境与前提

- 需要 CUDA 可用的 GPU（例如 4090D）。
- 已安装并启用 Conda 环境（示例：`conda activate deepseek-ocr`）。
- 本地权重与快照：
  - 权重目录：`/root/DeepSeek-OCR/models/DeepSeek-OCR`（含 `config.json`、`model-00001-of-000001.safetensors`）。
  - HF 本地快照（示例）：`~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR/snapshots/<latest>`（含 `tokenizer.json`、`modeling_deepseekocr.py` 等）。

---

## HF 运行方法（Transformers）

### 单图脚本（run_dpsk_ocr.py）

1) 编辑脚本参数：

```python
# 文件：DeepSeek-OCR-master/DeepSeek-OCR-hf/run_dpsk_ocr.py
model_name = 'deepseek-ai/DeepSeek-OCR'  # 或者改为本地快照路径
prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = 'your_image.jpg'
output_path = 'your/output/dir'
```

2) 运行：

```bash
conda activate deepseek-ocr
CUDA_VISIBLE_DEVICES=0 python -u DeepSeek-OCR-master/DeepSeek-OCR-hf/run_dpsk_ocr.py
```

### HF 离线运行建议

- 将 `model_name` 指向本地快照路径（例如：`~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR/snapshots/<latest>`）。
- 设置环境变量并仅用本地文件：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_ENABLE_HF_TRANSFER=1
```

示例（在脚本中传参）：

```python
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True, local_files_only=True)
```

### Gradio Demo（已改造为离线优先）

```bash
conda activate deepseek-ocr
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_ENABLE_HF_TRANSFER=1 CUDA_VISIBLE_DEVICES=0
python -u gradio_demo.py
# 访问： http://0.0.0.0:7860
```

---

## vLLM 运行方法

配置文件：`DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py`

- 关键项：
  - `MODEL_PATH = '/root/DeepSeek-OCR/models/DeepSeek-OCR'`（本地权重目录）。
  - `TOKENIZER_PATH` 自动解析为本地 HF 快照路径（找不到则回退 `deepseek-ai/DeepSeek-OCR`）。
  - `PROMPT`、`INPUT_PATH`、`OUTPUT_PATH` 控制输入与输出。
  - 已启用离线环境变量（脚本内统一 `HF_HUB_OFFLINE=1` 等）。

### 单图（run_dpsk_ocr_image.py）

```bash
conda activate deepseek-ocr
export CUDA_VISIBLE_DEVICES=0
python -u DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py
# 读取 INPUT_PATH（默认 /root/DeepSeek-OCR/assets/show1.jpg），输出到 OUTPUT_PATH（默认 /root/DeepSeek-OCR/outputs/vllm_image_test）
```

### PDF（run_dpsk_ocr_pdf.py）

```python
# config.py 中设置：INPUT_PATH = '/path/to/file.pdf'
```

```bash
conda activate deepseek-ocr
export CUDA_VISIBLE_DEVICES=0
python -u DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_pdf.py
# 输出：/root/DeepSeek-OCR/outputs/vllm_pdf_test/ 内的 *_det.mmd、*.mmd、*_layouts.pdf 与 images/
```

### 批量图片（run_dpsk_ocr_eval_batch.py）

```python
# config.py 中设置：INPUT_PATH = '/root/DeepSeek-OCR/assets'  # 目录
# 已过滤非图片文件，仅处理 .jpg/.jpeg
```

```bash
conda activate deepseek-ocr
export CUDA_VISIBLE_DEVICES=0
python -u DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_eval_batch.py
# 输出：/root/DeepSeek-OCR/outputs/vllm_image_test/ 下按文件名生成 *.md 与 *_det.md
```

---

## 4090D 参数与性能建议

- `gpu_memory_utilization=0.9`；若 OOM 降到 `0.85`。
- 并发：`MAX_CONCURRENCY=8~16` 稳妥；vLLM 会根据 `max_model_len/max_tokens` 估算并发上限（常见日志显示 8192 tokens 可达 ~30x）。
- 裁剪：`CROP_MODE=True` 降显存；速度优先可关闭但内存压力上升。
- 长度：`max_model_len=8192` 与 `max_tokens=8192` 在 4090D 通常可用；OOM 时优先降低 `max_tokens`。

---

## 典型差异对比

| 项目 | HF（Transformers） | vLLM |
|---|---|---|
| 框架 | `AutoModel/AutoTokenizer` | `LLM/AsyncLLMEngine` + KV Cache |
| 模型代码 | 远程 `trust_remote_code=True` | 本地注册，`trust_remote_code=False` |
| 权重路径 | 远程或本地快照 | `MODEL_PATH` 本地权重目录 |
| 分词器路径 | 远程或本地快照 | `TOKENIZER_PATH` 本地快照优先 |
| 输入类型 | 单图示例 | 单图、PDF、批量目录 |
| 离线配置 | 需手动设置及使用本地快照 | 已在脚本中统一启用离线变量 |
| 并发能力 | 单请求为主 | 高并发、KV Cache、高吞吐 |
| 输出位置 | 用户自设 `output_path` | `OUTPUT_PATH`（按脚本/场景） |

---

## 常见问题与提示

- HF 离线报错：若找不到本地快照，会因 `local_files_only=True` 报错；请确认快照目录存在并包含 `tokenizer.json`、`modeling_deepseekocr.py` 等。
- vLLM `fused_moe` 提示：日志中关于 MoE config 的提示为性能相关信息，不影响运行。
- 批量脚本非图片：已过滤 `.jpg/.jpeg`，避免 `PIL.UnidentifiedImageError`。
- `destroy_process_group` 警告：退出时的 NCCL 警告可忽略；如需消除可在退出处显式调用 `destroy_process_group()`。

---

## 路径速查

- 本地权重：`/root/DeepSeek-OCR/models/DeepSeek-OCR`
- 本地快照示例：`~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR/snapshots/<latest>`
- 默认单图输出：`/root/DeepSeek-OCR/outputs/vllm_image_test`
- PDF 输出：`/root/DeepSeek-OCR/outputs/vllm_pdf_test`

---

## 参考运行命令（统一离线环境）

```bash
conda activate deepseek-ocr
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_ENABLE_HF_TRANSFER=1 CUDA_VISIBLE_DEVICES=0

# HF 单图（编辑脚本内 image_file/output_path）
python -u DeepSeek-OCR-master/DeepSeek-OCR-hf/run_dpsk_ocr.py

# vLLM 单图
python -u DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py

# vLLM PDF（先在 config.py 设置 INPUT_PATH=.pdf）
python -u DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_pdf.py

# vLLM 批量（先在 config.py 设置 INPUT_PATH=目录）
python -u DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_eval_batch.py
```
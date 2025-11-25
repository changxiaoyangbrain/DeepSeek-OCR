# CLAUDE.md

此文件为 Claude Code (claude.ai/code) 提供在本仓库中工作的指导。

## 概述

DeepSeek-OCR 是一个用于光学字符识别（OCR）的视觉-语言模型，具有上下文压缩能力。支持 **vLLM**（高性能批量推理）和 **Transformers**（HuggingFace）两种后端。

**关键特性：**
- 多分辨率模式：Tiny (512×512)、Small (640×640)、Base (1024×1024)、Large (1280×1280)、Gundam（动态裁剪，推荐）
- 离线优先：优先使用本地模型缓存 `~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR/snapshots/`
- 技术栈：CUDA 11.8 + Torch 2.6.0 + vLLM 0.8.5
- 输出格式：Markdown + 布局定位标签 `<|ref|>...<|/ref|><|det|>...<|/det|>`

## 环境配置

**创建环境：**
```bash
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr
```

**安装依赖：**
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl  # 从 vLLM releases 下载
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation
```

**离线环境变量：**
```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export CUDA_VISIBLE_DEVICES=0
```

## 运行推理

### 配置文件

**核心配置：** [DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py](DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py)

运行任何 vLLM 脚本前需更新：
- `MODEL_PATH`：本地模型权重路径（默认：`/root/DeepSeek-OCR/models/DeepSeek-OCR`）
- `INPUT_PATH`：输入文件路径（图片或 PDF）
- `OUTPUT_PATH`：输出目录
- `BASE_SIZE`, `IMAGE_SIZE`, `CROP_MODE`：模型分辨率模式
- `MAX_CONCURRENCY`：vLLM 批处理大小（默认：100）
- `PROMPT`：任务相关的提示词模板

### vLLM 推理（生产环境推荐）

**单图处理（流式输出）：**
```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python run_dpsk_ocr_image.py
```

**PDF 批量处理（A100-40G 上约 2500 tokens/s）：**
```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python run_dpsk_ocr_pdf.py
```
- 输出文件：`<文件名>_det.mmd`（原始）、`<文件名>.mmd`（清理后）、`<文件名>_layouts.pdf`（带标注）

**批量评估：**
```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python run_dpsk_ocr_eval_batch.py
```

### Gradio Demo（vLLM）

**启动：**
```bash
./run_demo.sh
```
访问地址：`http://0.0.0.0:7860/`

**启动并启用自动重启（监听模式）：**
```bash
./run_demo.sh --watch
```

**功能：**
- 单图上传，支持提示词选择（Free OCR / Markdown / 自定义）
- 批量目录处理（`.jpg/.jpeg` 文件）
- PDF 上传与布局提取
- 高级参数：并发数（`max_num_seqs`）、GPU 显存利用率、`max_tokens`

**推荐设置（4090D）：**
- `gpu_memory_utilization=0.88~0.92`
- `max_num_seqs=12~24`
- `CROP_MODE=True`

### Transformers 推理（HuggingFace）

**直接使用 Python：**
```python
from transformers import AutoModel, AutoTokenizer
import torch

model_name = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2',
                                   trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)

prompt = "<image>\n<|grounding|>Convert the document to markdown."
res = model.infer(tokenizer, prompt=prompt, image_file='path/to/image.jpg',
                  output_path='output/dir', base_size=1024, image_size=640,
                  crop_mode=True, save_results=True, test_compress=True)
```

**使用脚本：**
```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-hf
python run_dpsk_ocr.py
```

## 架构设计

### 目录结构

```
DeepSeek-OCR-master/
├── DeepSeek-OCR-vllm/          # vLLM 后端（生产环境）
│   ├── config.py               # 核心配置文件
│   ├── deepseek_ocr.py         # vLLM 模型实现（通过 ModelRegistry 注册）
│   ├── run_dpsk_ocr_image.py   # 单图推理（异步流式）
│   ├── run_dpsk_ocr_pdf.py     # PDF 批量处理
│   ├── run_dpsk_ocr_eval_batch.py  # 基准评估
│   ├── deepencoder/            # 视觉编码器
│   │   ├── sam_vary_sdpa.py    # 基于 SAM 的编码器
│   │   ├── clip_sdpa.py        # CLIP 编码器
│   │   └── build_linear.py     # MLP 投影器
│   └── process/
│       ├── image_process.py    # DeepseekOCRProcessor（动态裁剪、宽高比处理）
│       └── ngram_norepeat.py   # NoRepeatNGramLogitsProcessor（防重复）
└── DeepSeek-OCR-hf/            # Transformers 后端
    └── run_dpsk_ocr.py         # HuggingFace 推理脚本
```

### 核心组件

**1. 模型注册（vLLM）**
- [deepseek_ocr.py:38](DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepseek_ocr.py#L38)：`ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)`
- 允许 vLLM 使用本地模型类而非 HuggingFace 的（设置 `trust_remote_code=False`）

**2. 图像预处理**
- [DeepseekOCRProcessor](DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/image_process.py)：处理动态宽高比裁剪
- `dynamic_preprocess()`：将高分辨率图片切分为 `image_size×image_size` 的瓦片（2-6 个瓦片）
- `count_tiles()`：基于宽高比计算最优瓦片网格
- [config.py:1-6](DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py#L1-L6)：分辨率模式配置（Tiny/Small/Base/Large/Gundam）

**3. Logits 处理**
- [NoRepeatNGramLogitsProcessor](DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/ngram_norepeat.py)：防止表格 token 重复
- `ngram_size=30, window_size=90, whitelist_token_ids={128821, 128822}`（针对 `<td>`、`</td>`）

**4. 输出后处理**
- 正则模式：`<|ref|>(.*?)<|/ref|><|det|>(.*?)<|/det|>` 提取布局边界框
- [run_dpsk_ocr_image.py:57-69](DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py#L57-L69)：`re_match()` 分离图片引用和其他布局元素
- `draw_bounding_boxes()`：在图片上渲染布局标注
- 特殊处理 `<|ref|>image<|/ref|>` 标签 → 提取并保存到 `outputs/images/`

**5. Tokenizer 回退机制**
- [config.py:46-79](DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py#L46-L79)：自动检测本地 HuggingFace 快照，设置离线模式
- 优先使用 `~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR/snapshots/<hash>/`，避免远程下载

### 双推理路径

| 后端 | 适用场景 | 特性 |
|------|---------|------|
| **vLLM** (`DeepSeek-OCR-vllm/`) | 生产环境、批量处理、高吞吐 | AsyncLLMEngine、批处理、KV 缓存、流式输出 |
| **Transformers** (`DeepSeek-OCR-hf/`) | 研究、交互式、单文件推理 | 直接使用 model.infer() API，配置简单 |

## 提示词工程

**常用提示词** ([config.py:31-38](DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py#L31-L38))：
```python
# 文档 OCR（带布局）：
"<image>\n<|grounding|>Convert the document to markdown."

# 通用 OCR（无布局）：
"<image>\nFree OCR."

# 其他图片类型（照片、图表）：
"<image>\n<|grounding|>OCR this image."

# 从文档中提取图形：
"<image>\nParse the figure."

# 图像描述：
"<image>\nDescribe this image in detail."

# 目标定位：
"<image>\nLocate <|ref|>xxxx<|/ref|> in the image."
```

## 分辨率模式

通过 [config.py](DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py#L1-L10) 中的 `BASE_SIZE`、`IMAGE_SIZE`、`CROP_MODE` 控制：

| 模式 | base_size | image_size | crop_mode | 视觉 Tokens | 使用场景 |
|------|-----------|------------|-----------|-------------|---------|
| Tiny | 512 | 512 | False | 64 | 快速推理、低显存 |
| Small | 640 | 640 | False | 100 | 平衡性能 |
| Base | 1024 | 1024 | False | 256 | 高质量 |
| Large | 1280 | 1280 | False | 400 | 最高质量 |
| **Gundam**（默认） | 1024 | 640 | **True** | 256 + n×100 (n=2-6 瓦片) | 动态分辨率，推荐使用 |

## 开发注意事项

**CUDA 11.8 兼容性：**
- [run_dpsk_ocr_image.py:6-19](DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py#L6-L19)：自动设置 `TRITON_PTXAS_PATH` 用于 Triton 内核
- [run_dpsk_ocr_pdf.py:11-12](DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_pdf.py#L11-L12)：处理混合 CUDA 安装的回退逻辑

**vLLM v1 模式：**
- 强制使用旧版模式：`os.environ['VLLM_USE_V1'] = '0'`（vLLM 0.8.5 需要）

**显存优化：**
- 有限显存时降低 `MAX_CONCURRENCY`
- 降低 `gpu_memory_utilization`（推荐 0.75-0.9）
- 限制 `MAX_CROPS`（默认：6，最大：9）

**离线 Tokenizer：**
- [config.py:62-79](DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py#L62-L79)：自动解析本地快照
- 设置 `HF_HUB_OFFLINE=1` 防止远程调用

**输出格式：**
- `result_ori.mmd`：带定位标签的原始模型输出
- `result.mmd`：清理后的 Markdown（带图片链接）
- `result_with_boxes.jpg`：带布局框的标注图片

## 测试

**快速测试（单图）：**
1. 更新 [config.py](DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py#L26-L27)：
   ```python
   INPUT_PATH = '/path/to/test/image.jpg'
   OUTPUT_PATH = '/path/to/output/dir'
   ```
2. 运行：
   ```bash
   cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
   python run_dpsk_ocr_image.py
   ```

**Gradio Demo 测试：**
```bash
./run_demo.sh
# 在 http://0.0.0.0:7860/ 上传图片
```

## 已知问题

**vLLM 版本冲突：**
- README 说明："如果希望 vLLM 和 transformers 代码在同一环境中运行，不需要担心安装错误，如：vllm 0.8.5+cu118 requires transformers>=4.51.1"
- 根本原因：vLLM 使用本地模型类（`trust_remote_code=False`）

**PDF 模式跳页：**
- [run_dpsk_ocr_pdf.py:296-299](DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_pdf.py#L296-L299)：如果 `SKIP_REPEAT=True`，没有 EOS token 的页面会被跳过
- 通过在 config 中设置 `SKIP_REPEAT=False` 禁用

**几何图形绘制处理：**
- [run_dpsk_ocr_image.py:270-320](DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py#L270-L320)：几何图形的特殊渲染（圆、线端点）
- 仅在输出包含 `"line_type"` 键时激活

## 4090D 参数建议

基于 [RUN_METHODS_COMPARISON_2025-10-28.md](RUN_METHODS_COMPARISON_2025-10-28.md#L125-L130)：
- `gpu_memory_utilization=0.9`；若 OOM 降至 `0.85`
- 并发：`MAX_CONCURRENCY=8~16` 较稳妥；vLLM 会根据 `max_model_len/max_tokens` 估算并发上限（8192 tokens 可达约 30x）
- 裁剪：`CROP_MODE=True` 降低显存占用；优先速度可关闭但显存压力上升
- 长度：`max_model_len=8192` 与 `max_tokens=8192` 在 4090D 通常可用；OOM 时优先降低 `max_tokens`

## 常用命令参考

**统一离线环境运行：**
```bash
conda activate deepseek-ocr
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_ENABLE_HF_TRANSFER=1 CUDA_VISIBLE_DEVICES=0

# HF 单图（需编辑脚本内的 image_file/output_path）
python -u DeepSeek-OCR-master/DeepSeek-OCR-hf/run_dpsk_ocr.py

# vLLM 单图
python -u DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py

# vLLM PDF（先在 config.py 设置 INPUT_PATH=.pdf）
python -u DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_pdf.py

# vLLM 批量（先在 config.py 设置 INPUT_PATH=目录）
python -u DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_eval_batch.py

# Gradio vLLM Demo（推荐）
./run_demo.sh
```

## 参考资料

- 论文：[DeepSeek_OCR_paper.pdf](DeepSeek_OCR_paper.pdf) | [arXiv:2510.18234](https://arxiv.org/abs/2510.18234)
- 模型：[deepseek-ai/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- 上游 vLLM 支持：[vLLM recipes](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html)
- 本仓库运行方法对比：[RUN_METHODS_COMPARISON_2025-10-28.md](RUN_METHODS_COMPARISON_2025-10-28.md)

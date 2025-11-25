# DeepSeek-OCR Agents

快速给多代理/自动化系统准备的角色说明与启动指令。所有命令默认在仓库根目录执行，已假设你使用 `conda activate deepseek-ocr` 并拥有本地权重与缓存（`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 等）。

## 通用前置
- GPU：已配置 `CUDA_VISIBLE_DEVICES`（示例 `0`）。
- 权重：`models/DeepSeek-OCR/` 内含 `config.json` 与 safetensors；本地分词器快照已存在（否则回退到远程）。
- 输出：脚本默认写入 `outputs/`，请确保目录可写。

## Agent: HF 单图推理
- 目标：单张图片转 Markdown，依赖 `transformers`。
- 入口：`DeepSeek-OCR-master/DeepSeek-OCR-hf/run_dpsk_ocr.py`（修改 `model_name/prompt/image_file/output_path`）。
- 命令：
```bash
CUDA_VISIBLE_DEVICES=0 python -u DeepSeek-OCR-master/DeepSeek-OCR-hf/run_dpsk_ocr.py
```

## Agent: vLLM 通用推理
- 目标：高吞吐推理（单图/PDF/批量），依赖本地 `vllm`。
- 配置：`DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py` 中调整 `MODEL_PATH`、`INPUT_PATH`、`OUTPUT_PATH`、`PROMPT` 等。
- 常用入口：
  - 单图：`python -u DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py`
  - PDF：`python -u DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_pdf.py`
  - 批量图片：`python -u DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_eval_batch.py`

## Agent: Gradio vLLM Demo
- 目标：交互式 Web Demo，默认端口 `7860`，离线优先。
- 快捷脚本：`./run_demo.sh`（支持 `--watch` 自动重启）。
- 直接运行：
```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_ENABLE_HF_TRANSFER=1
export CUDA_VISIBLE_DEVICES=0 DEMO_PORT=7860
python -u gradio_vllm_demo.py
```
- 功能：单图/批量/PDF 处理，输出至 `outputs/vllm_gradio_*/*`。

## Agent: 诊断与维护
- 目标：检查环境与本地资源，辅助其他代理。
- 参考命令：
```bash
nvidia-smi
conda info --envs | grep deepseek-ocr
ls -la models/DeepSeek-OCR
```
- 常见问题：权重路径错误、未设置离线变量、显存不足（可在 `config.py` 调低 `gpu_memory_utilization` 或 `max_num_seqs`）。

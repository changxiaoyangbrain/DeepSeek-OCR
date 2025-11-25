import os
import sys
import time
import json
import traceback
from datetime import datetime
import gradio as gr
from typing import Optional, List, Tuple

import torch
import gc
from PIL import Image, ImageDraw, ImageFont
import io
import re
import glob
import numpy as np
import zipfile
import shutil


# ============================================
# 日志辅助函数
# ============================================
def log_info(msg: str):
    """输出带时间戳的 INFO 日志"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [INFO] {msg}")

def log_success(msg: str):
    """输出带时间戳的成功日志"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [✓ OK] {msg}")

def log_warning(msg: str):
    """输出带时间戳的警告日志"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [⚠ WARN] {msg}")

def log_error(msg: str):
    """输出带时间戳的错误日志"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [✗ ERROR] {msg}")

def log_progress(current: int, total: int, task: str, extra: str = ""):
    """输出进度日志"""
    ts = datetime.now().strftime("%H:%M:%S")
    pct = (current / total * 100) if total > 0 else 0
    bar_len = 20
    filled = int(bar_len * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_len - filled)
    extra_str = f" | {extra}" if extra else ""
    print(f"[{ts}] [{bar}] {current}/{total} ({pct:.1f}%) {task}{extra_str}")


# Add vLLM module directory to import path
ROOT_DIR = os.path.dirname(__file__)
VLLM_DIR = os.path.join(ROOT_DIR, "DeepSeek-OCR-master", "DeepSeek-OCR-vllm")
if VLLM_DIR not in sys.path:
    sys.path.append(VLLM_DIR)

from config import (
    MODEL_PATH,
    TOKENIZER_PATH,
    PROMPT,
    CROP_MODE,
    MAX_CONCURRENCY,
)

from deepseek_ocr import DeepseekOCRForCausalLM
from vllm.model_executor.models.registry import ModelRegistry
from vllm import LLM, SamplingParams
from process.image_process import DeepseekOCRProcessor
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor


llm: Optional[LLM] = None
current_engine_cfg = {
    "max_concurrency": None,
    "gpu_memory_utilization": None,
    "max_model_len": None,
}

# Model size presets (accuracy/speed tradeoff)
size_configs = {
    "极速（Tiny）": {"base_size": 512, "image_size": 512, "crop_mode": False},
    "快速（Small）": {"base_size": 640, "image_size": 640, "crop_mode": False},
    "标准（Base）": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
    "精细（Large）": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
    "高达模式（推荐）": {"base_size": 1024, "image_size": 640, "crop_mode": True},
}


def _ensure_offline_env():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def _setup_cuda_env():
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    try:
        if getattr(torch.version, "cuda", None) == "11.8":
            for p in [
                "/usr/local/cuda-11.8/bin/ptxas",
                "/usr/local/cuda/bin/ptxas",
            ]:
                if os.path.exists(p):
                    os.environ["TRITON_PTXAS_PATH"] = p
                    break
    except Exception:
        pass


def init_llm(
    max_concurrency: int,
    gpu_memory_utilization: float,
    max_model_len: int = 8192,
    force_reinit: bool = False,
):
    global llm, current_engine_cfg

    if (
        llm is not None
        and not force_reinit
        and current_engine_cfg.get("max_concurrency") == max_concurrency
        and current_engine_cfg.get("gpu_memory_utilization") == gpu_memory_utilization
        and current_engine_cfg.get("max_model_len") == max_model_len
    ):
        return llm

    # Cleanup previous engine if reconfiguring
    if llm is not None:
        try:
            llm.sleep()
        except Exception:
            pass
        try:
            del llm
        except Exception:
            pass
        llm = None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        except Exception:
            pass
        try:
            gc.collect()
        except Exception:
            pass
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        time.sleep(0.5)

    _ensure_offline_env()
    _setup_cuda_env()

    ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

    try:
        llm = LLM(
            model=MODEL_PATH,
            tokenizer=TOKENIZER_PATH,
            hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
            block_size=256,
            enforce_eager=False,
            trust_remote_code=False,
            max_model_len=max_model_len,
            swap_space=0,
            max_num_seqs=max_concurrency,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            disable_mm_preprocessor_cache=True,
        )
    except AssertionError as ae:
        # Retry once after aggressive cleanup to handle vLLM memory profiling assertion
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        except Exception:
            pass
        try:
            gc.collect()
        except Exception:
            pass
        time.sleep(0.5)
        llm = LLM(
            model=MODEL_PATH,
            tokenizer=TOKENIZER_PATH,
            hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
            block_size=256,
            enforce_eager=False,
            trust_remote_code=False,
            max_model_len=max_model_len,
            swap_space=0,
            max_num_seqs=max_concurrency,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            disable_mm_preprocessor_cache=True,
        )

    current_engine_cfg = {
        "max_concurrency": max_concurrency,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
    }
    return llm


def process_image(
    image: Image.Image,
    prompt_type: str,
    custom_prompt: str,
    model_size: str,
    crop_mode: bool,
    max_concurrency: int,
    gpu_memory_utilization: float,
    max_tokens: int,
):
    try:
        # Guard empty input
        if image is None:
            return "未检测到图片，请先上传图片后再点击处理。"
        
        single_start_time = time.time()
        log_info("=" * 50)
        log_info(f"📷 开始单图识别")
        log_info("=" * 50)
        log_info(f"   识别模式: {prompt_type}")
        log_info(f"   模型档位: {model_size}")
        log_info(f"   裁剪模式: {'开启' if crop_mode else '关闭'}")
        log_info(f"   图片尺寸: {image.size}")
        
        llm_local = init_llm(
            max_concurrency=max_concurrency,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=8192,
        )

        # 根据官方文档设置 prompt
        # 文档: <image>\n<|grounding|>Convert the document to markdown.
        # 纯文字: <image>\nFree OCR.
        # 其他图片: <image>\n<|grounding|>OCR this image.
        # 图表: <image>\nParse the figure.
        # 通用描述: <image>\nDescribe this image in detail.
        if prompt_type == "自由识别":
            prompt = "<image>\nFree OCR. "
        elif prompt_type == "Markdown转换":
            prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        elif prompt_type == "图片OCR":
            prompt = "<image>\n<|grounding|>OCR this image. "
        elif prompt_type == "图表解析":
            prompt = "<image>\nParse the figure. "
        elif prompt_type == "图像描述":
            prompt = "<image>\nDescribe this image in detail. "
        elif prompt_type == "自定义":
            prompt = f"<image>\n{custom_prompt}"
        else:
            prompt = "<image>\nFree OCR. "
        
        log_info(f"   Prompt: {prompt[:50]}...")

        # Apply size preset
        preset = size_configs.get(model_size, size_configs["高达模式（推荐）"])
        base_size = preset["base_size"]
        image_size = preset["image_size"]
        # Use current checkbox for cropping (updated by preset change)
        image = image.convert("RGB")
        
        log_info(f"🔧 正在预处理图片...")
        preprocess_start = time.time()
        proc = DeepseekOCRProcessor(image_size=image_size, base_size=base_size)
        image_features = proc.tokenize_with_images(
            images=[image], bos=True, eos=True, cropping=crop_mode
        )
        preprocess_time = time.time() - preprocess_start
        log_success(f"   预处理完成, 耗时 {preprocess_time:.2f} 秒")

        logits_processors = [
            NoRepeatNGramLogitsProcessor(
                ngram_size=40, window_size=90, whitelist_token_ids={128821, 128822}
            )
        ]

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logits_processors=logits_processors,
            skip_special_tokens=False,
        )

        cache_item = {
            "prompt": prompt,
            "multi_modal_data": {"image": image_features},
        }

        log_info(f"🚀 开始OCR推理...")
        inference_start = time.time()
        outputs = llm_local.generate(
            [cache_item], sampling_params=sampling_params
        )
        inference_time = time.time() - inference_start
        log_success(f"   推理完成, 耗时 {inference_time:.2f} 秒")

        content = outputs[0].outputs[0].text
        
        # 清理结果：移除结束标记
        if "<｜end▁of▁sentence｜>" in content:
            content = content.replace("<｜end▁of▁sentence｜>", "")
        
        total_time = time.time() - single_start_time
        log_info("=" * 50)
        log_success(f"📷 单图识别完成！")
        log_info(f"   总耗时: {total_time:.2f} 秒")
        log_info(f"   输出长度: {len(content)} 字符")
        log_info("=" * 50)
        
        return content

    except Exception as e:
        log_error(f"单图识别失败: {str(e)}")
        return f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"

def clean_formula(text: str) -> str:
    formula_pattern = r"\\\[(.*?)\\\]"

    def process_formula(match):
        formula = match.group(1)
        formula = re.sub(r"\\quad\s*\([^)]*\)", "", formula)
        formula = formula.strip()
        return r"\[" + formula + r"\]"

    cleaned_text = re.sub(formula_pattern, process_formula, text)
    return cleaned_text

def re_match(text: str) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
    matches = re.findall(pattern, text, re.DOTALL)
    mathes_other = []
    for a_match in matches:
        mathes_other.append(a_match[0])
    return matches, mathes_other

def re_match_pdf(text: str) -> Tuple[List[Tuple[str, str, str]], List[str], List[str]]:
    pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
    matches = re.findall(pattern, text, re.DOTALL)
    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def _is_image(path: str) -> bool:
    """检查是否为支持的图片格式"""
    return os.path.splitext(path)[1].lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def _list_images_in_dir(dir_path: str) -> list:
    """
    列出目录中所有图片文件，兼容中文路径。
    使用 os.listdir 代替 glob.glob 以避免中文路径问题。
    """
    if not os.path.isdir(dir_path):
        return []
    try:
        files = os.listdir(dir_path)
        images = []
        for f in files:
            full_path = os.path.join(dir_path, f)
            if os.path.isfile(full_path) and _is_image(full_path):
                images.append(full_path)
        return sorted(images)
    except Exception:
        return []


def process_batch_upload(
    uploaded_files: List[str],
    prompt_type: str,
    custom_prompt: str,
    model_size: str,
    crop_mode: bool,
    max_concurrency: int,
    gpu_memory_utilization: float,
    max_tokens: int,
):
    """处理上传的多张图片（支持远程客户端）"""
    try:
        if not uploaded_files:
            return "请先上传图片文件", "", None
        
        batch_start_time = time.time()
        log_info("=" * 50)
        log_info(f"📚 开始批量处理任务")
        log_info(f"   文件数量: {len(uploaded_files)}")
        log_info(f"   识别模式: {prompt_type}")
        log_info(f"   模型精度: {model_size}")
        log_info(f"   智能裁剪: {'是' if crop_mode else '否'}")
        log_info("=" * 50)
        
        log_info("正在初始化推理引擎...")
        llm_local = init_llm(
            max_concurrency=max_concurrency,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=8192,
        )
        log_success("推理引擎就绪")

        log_info("正在加载图片...")
        images = []
        valid_paths = []
        for idx, file_path in enumerate(uploaded_files):
            try:
                image = Image.open(file_path).convert("RGB")
                images.append(image)
                valid_paths.append(file_path)
                log_progress(idx + 1, len(uploaded_files), "加载图片", os.path.basename(file_path))
            except Exception as e:
                log_warning(f"跳过文件: {os.path.basename(file_path)} - {e}")

        if not images:
            return "没有可处理的有效图片文件", "", None
        
        log_success(f"成功加载 {len(images)} 张图片")

        # 根据官方文档设置 prompt
        if prompt_type == "自由识别":
            prompt = "<image>\nFree OCR. "
        elif prompt_type == "Markdown转换":
            prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        elif prompt_type == "图片OCR":
            prompt = "<image>\n<|grounding|>OCR this image. "
        elif prompt_type == "图表解析":
            prompt = "<image>\nParse the figure. "
        elif prompt_type == "图像描述":
            prompt = "<image>\nDescribe this image in detail. "
        elif prompt_type == "自定义":
            prompt = f"<image>\n{custom_prompt}"
        else:
            prompt = "<image>\nFree OCR. "

        preset = size_configs.get(model_size, size_configs["高达模式（推荐）"])
        base_size = preset["base_size"]
        image_size = preset["image_size"]
        
        log_info(f"正在预处理图片 (base_size={base_size}, image_size={image_size})...")
        preprocess_start = time.time()
        proc = DeepseekOCRProcessor(image_size=image_size, base_size=base_size)
        batch_inputs = []
        for idx, img in enumerate(images):
            image_features = proc.tokenize_with_images(
                images=[img], bos=True, eos=True, cropping=crop_mode
            )
            cache_item = {
                "prompt": prompt,
                "multi_modal_data": {"image": image_features},
            }
            batch_inputs.append(cache_item)
            if (idx + 1) % 5 == 0 or idx == len(images) - 1:
                log_progress(idx + 1, len(images), "预处理图片")
        preprocess_time = time.time() - preprocess_start
        log_success(f"预处理完成，耗时 {preprocess_time:.2f} 秒")

        logits_processors = [
            NoRepeatNGramLogitsProcessor(
                ngram_size=40, window_size=90, whitelist_token_ids={128821, 128822}
            )
        ]

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logits_processors=logits_processors,
            skip_special_tokens=False,
        )

        log_info(f"🚀 开始批量推理 ({len(batch_inputs)} 张图片)...")
        inference_start = time.time()
        outputs_list = llm_local.generate(batch_inputs, sampling_params=sampling_params)
        inference_time = time.time() - inference_start
        avg_time = inference_time / len(batch_inputs) if batch_inputs else 0
        log_success(f"推理完成，总耗时 {inference_time:.2f} 秒，平均 {avg_time:.2f} 秒/张")

        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("outputs", "vllm_gradio_batch", ts)
        os.makedirs(out_dir, exist_ok=True)

        log_info("正在保存识别结果...")
        preview_texts = []
        for idx, (output, file_path) in enumerate(zip(outputs_list, valid_paths)):
            content = output.outputs[0].text
            base_name = os.path.basename(file_path)
            name_no_ext = os.path.splitext(base_name)[0]
            # 避免文件名冲突，添加序号
            safe_name = f"{idx+1:03d}_{name_no_ext}"

            mmd_det_path = os.path.join(out_dir, f"{safe_name}_det.md")
            with open(mmd_det_path, "w", encoding="utf-8") as afile:
                afile.write(content)

            content_clean = clean_formula(content)
            matches_ref, mathes_other = re_match(content_clean)
            for a_match_other in mathes_other:
                content_clean = (
                    content_clean.replace(a_match_other, "")
                    .replace("\\n\\n\\n\\n", "\\n\\n")
                    .replace("\\n\\n\\n", "\\n\\n")
                    .replace("<center>", "")
                    .replace("</center>", "")
                )

            mmd_path = os.path.join(out_dir, f"{safe_name}.md")
            with open(mmd_path, "w", encoding="utf-8") as afile:
                afile.write(content_clean)

            if len(preview_texts) < 3:
                preview_texts.append(f"## {base_name}\n\n" + content_clean[:2000])

        # 创建 zip 文件供下载
        zip_path = os.path.join("outputs", "vllm_gradio_batch", f"batch_result_{ts}.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(out_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, out_dir)
                    zf.write(file_path, arcname)

        total_time = time.time() - batch_start_time
        log_info("=" * 50)
        log_success(f"📚 批量处理完成！")
        log_info(f"   处理图片: {len(images)} 张")
        log_info(f"   总耗时: {total_time:.2f} 秒")
        log_info(f"   平均速度: {total_time/len(images):.2f} 秒/张")
        log_info(f"   输出目录: {out_dir}")
        log_info("=" * 50)
        
        return f"✅ 已处理 {len(images)} 张图片\n⏱️ 总耗时: {total_time:.1f} 秒\n📁 结果保存到: {out_dir}", "\n\n".join(preview_texts), zip_path

    except Exception as e:
        log_error(f"批量处理失败: {str(e)}")
        return f"处理出错: {str(e)}\n{traceback.format_exc()}", "", None


def process_batch(
    dir_path: str,
    prompt_type: str,
    custom_prompt: str,
    model_size: str,
    crop_mode: bool,
    max_concurrency: int,
    gpu_memory_utilization: float,
    max_tokens: int,
):
    """处理服务器本地目录（保留向后兼容）"""
    try:
        llm_local = init_llm(
            max_concurrency=max_concurrency,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=8192,
        )

        # 使用兼容中文路径的方法列出图片
        images_path = _list_images_in_dir(dir_path)
        if not images_path:
            return f"目录中未找到图片文件（支持 jpg/png/webp/bmp/tiff）：{dir_path}", ""

        images = []
        for image_path in images_path:
            try:
                image = Image.open(image_path).convert("RGB")
                images.append(image)
            except Exception as e:
                print(f"skip file: {image_path} due to error: {e}")

        # 根据官方文档设置 prompt
        if prompt_type == "自由识别":
            prompt = "<image>\nFree OCR. "
        elif prompt_type == "Markdown转换":
            prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        elif prompt_type == "图片OCR":
            prompt = "<image>\n<|grounding|>OCR this image. "
        elif prompt_type == "图表解析":
            prompt = "<image>\nParse the figure. "
        elif prompt_type == "图像描述":
            prompt = "<image>\nDescribe this image in detail. "
        elif prompt_type == "自定义":
            prompt = f"<image>\n{custom_prompt}"
        else:
            prompt = "<image>\nFree OCR. "

        preset = size_configs.get(model_size, size_configs["高达模式（推荐）"])
        base_size = preset["base_size"]
        image_size = preset["image_size"]
        proc = DeepseekOCRProcessor(image_size=image_size, base_size=base_size)
        batch_inputs = []
        for img in images:
            image_features = proc.tokenize_with_images(
                images=[img], bos=True, eos=True, cropping=crop_mode
            )
            cache_item = {
                "prompt": prompt,
                "multi_modal_data": {"image": image_features},
            }
            batch_inputs.append(cache_item)

        logits_processors = [
            NoRepeatNGramLogitsProcessor(
                ngram_size=40, window_size=90, whitelist_token_ids={128821, 128822}
            )
        ]

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logits_processors=logits_processors,
            skip_special_tokens=False,
        )

        outputs_list = llm_local.generate(batch_inputs, sampling_params=sampling_params)

        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("outputs", "vllm_gradio_batch", ts)
        os.makedirs(out_dir, exist_ok=True)

        preview_texts = []
        for output, image_path in zip(outputs_list, images_path):
            content = output.outputs[0].text
            base_name = os.path.basename(image_path)
            name_no_ext = os.path.splitext(base_name)[0]

            mmd_det_path = os.path.join(out_dir, f"{name_no_ext}_det.md")
            with open(mmd_det_path, "w", encoding="utf-8") as afile:
                afile.write(content)

            content_clean = clean_formula(content)
            matches_ref, mathes_other = re_match(content_clean)
            for a_match_other in mathes_other:
                content_clean = (
                    content_clean.replace(a_match_other, "")
                    .replace("\\n\\n\\n\\n", "\\n\\n")
                    .replace("\\n\\n\\n", "\\n\\n")
                    .replace("<center>", "")
                    .replace("</center>", "")
                )

            mmd_path = os.path.join(out_dir, f"{name_no_ext}.md")
            with open(mmd_path, "w", encoding="utf-8") as afile:
                afile.write(content_clean)

            if len(preview_texts) < 3:
                preview_texts.append(f"## {base_name}\n\n" + content_clean[:2000])

        return f"已写入 {len(images_path)} 个结果到: {out_dir}", "\n\n".join(preview_texts)

    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}", ""

def pdf_to_images_high_quality(pdf_path: str, dpi: int = 144, image_format: str = "PNG") -> List[Image.Image]:
    import fitz
    images: List[Image.Image] = []
    pdf_document = fitz.open(pdf_path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        Image.MAX_IMAGE_PIXELS = None
        img_data = pixmap.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        if img.mode in ("RGBA", "LA"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            img = background
        images.append(img)
    pdf_document.close()
    return images

def pil_to_pdf_img2pdf(pil_images: List[Image.Image], output_path: str):
    import img2pdf
    if not pil_images:
        return
    image_bytes_list = []
    for img in pil_images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="JPEG", quality=95)
        img_bytes = img_buffer.getvalue()
        image_bytes_list.append(img_bytes)
    try:
        pdf_bytes = img2pdf.convert(image_bytes_list)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)
    except Exception as e:
        print(f"error: {e}")

def extract_coordinates_and_label(ref_text: Tuple[str, str, str], image_width: int, image_height: int):
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(e)
        return None
    return (label_type, cor_list)

def draw_bounding_boxes(image: Image.Image, refs: List[Tuple[str, str, str]], jdx: int, save_dir: str):
    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    overlay = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    img_idx = 0
    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result
                color = (
                    np.random.randint(0, 200),
                    np.random.randint(0, 200),
                    np.random.randint(0, 255),
                )
                color_a = color + (20,)
                for points in points_list:
                    x1, y1, x2, y2 = points
                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)
                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)
                    if label_type == "image":
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            cropped.save(os.path.join(save_dir, "images", f"{jdx}_{img_idx}.jpg"))
                        except Exception as e:
                            print(e)
                        img_idx += 1
                    try:
                        if label_type == "title":
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle(
                                [x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1
                            )
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle(
                                [x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1
                            )
                        text_x = x1
                        text_y = max(0, y1 - 15)
                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle(
                            [text_x, text_y, text_x + text_width, text_y + text_height],
                            fill=(255, 255, 255, 30),
                        )
                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except Exception:
                        pass
        except Exception:
            continue
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw

def process_pdf(
    pdf_path: str,
    dpi: int,
    prompt_type: str,
    custom_prompt: str,
    model_size: str,
    crop_mode: bool,
    max_concurrency: int,
    gpu_memory_utilization: float,
    max_tokens: int,
    export_layout_pdf: bool,
):
    try:
        # Guard empty input
        if not pdf_path:
            return "未检测到 PDF 文件，请先上传后再点击处理。", "", None
        if isinstance(pdf_path, str) and not os.path.exists(pdf_path):
            return f"文件不存在：{pdf_path}", "", None
        
        pdf_start_time = time.time()
        pdf_name = os.path.basename(pdf_path)
        log_info("=" * 60)
        log_info(f"📄 开始处理 PDF: {pdf_name}")
        log_info("=" * 60)
        log_info(f"   DPI: {dpi}")
        log_info(f"   识别模式: {prompt_type}")
        log_info(f"   模型档位: {model_size}")
        log_info(f"   裁剪模式: {'开启' if crop_mode else '关闭'}")
        log_info(f"   最大Token: {max_tokens}")
        log_info(f"   导出布局PDF: {'是' if export_layout_pdf else '否'}")
        
        llm_local = init_llm(
            max_concurrency=max_concurrency,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=8192,
        )

        # 根据官方文档设置 prompt
        if prompt_type == "自由识别":
            prompt = "<image>\nFree OCR. "
        elif prompt_type == "Markdown转换":
            prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        elif prompt_type == "图片OCR":
            prompt = "<image>\n<|grounding|>OCR this image. "
        elif prompt_type == "图表解析":
            prompt = "<image>\nParse the figure. "
        elif prompt_type == "图像描述":
            prompt = "<image>\nDescribe this image in detail. "
        elif prompt_type == "自定义":
            prompt = f"<image>\n{custom_prompt}"
        else:
            prompt = "<image>\nFree OCR. "

        log_info(f"📖 正在转换 PDF 为图片 (DPI={dpi})...")
        convert_start = time.time()
        images = pdf_to_images_high_quality(pdf_path, dpi=dpi)
        convert_time = time.time() - convert_start
        if not images:
            return "PDF 中无可处理页面", "", None
        log_success(f"   PDF转换完成: {len(images)} 页, 耗时 {convert_time:.2f} 秒")

        preset = size_configs.get(model_size, size_configs["高达模式（推荐）"])
        base_size = preset["base_size"]
        image_size = preset["image_size"]
        proc = DeepseekOCRProcessor(image_size=image_size, base_size=base_size)
        
        log_info(f"🔧 正在预处理 {len(images)} 页...")
        preprocess_start = time.time()
        batch_inputs = []
        for idx, img in enumerate(images):
            image_features = proc.tokenize_with_images(
                images=[img], bos=True, eos=True, cropping=crop_mode
            )
            cache_item = {
                "prompt": prompt,
                "multi_modal_data": {"image": image_features},
            }
            batch_inputs.append(cache_item)
            if (idx + 1) % 5 == 0 or idx == len(images) - 1:
                log_progress(idx + 1, len(images), "预处理")
        preprocess_time = time.time() - preprocess_start
        log_success(f"   预处理完成, 耗时 {preprocess_time:.2f} 秒")

        logits_processors = [
            NoRepeatNGramLogitsProcessor(
                ngram_size=20, window_size=50, whitelist_token_ids={128821, 128822}
            )
        ]
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logits_processors=logits_processors,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
        )

        log_info(f"🚀 开始OCR推理 ({len(images)} 页)...")
        inference_start = time.time()
        outputs_list = llm_local.generate(batch_inputs, sampling_params=sampling_params)
        inference_time = time.time() - inference_start
        avg_time = inference_time / len(images) if images else 0
        log_success(f"   推理完成, 总耗时 {inference_time:.2f} 秒, 平均 {avg_time:.2f} 秒/页")

        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("outputs", "vllm_gradio_pdf", ts)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

        log_info(f"💾 正在保存结果...")
        contents_det = ""
        contents = ""
        draw_images: List[Image.Image] = []

        jdx = 0
        for output, img in zip(outputs_list, images):
            content = output.outputs[0].text
            if "<｜end▁of▁sentence｜>" in content:
                content = content.replace("<｜end▁of▁sentence｜>", "")

            page_num = f"\n<--- Page Split --->"
            contents_det += content + f"\n{page_num}\n"

            image_draw = img.copy()
            matches_ref, matches_images, _ = re_match_pdf(content)
            result_image = draw_bounding_boxes(image_draw, matches_ref, jdx, out_dir)
            draw_images.append(result_image)

            for idx, a_match in enumerate(matches_images):
                content = content.replace(
                    a_match,
                    f"![](images/" + str(jdx) + "_" + str(idx) + ".jpg)\n",
                )
            content = (
                content.replace("\\coloneqq", ":=")
                .replace("\\eqqcolon", "=:")
                .replace("\n\n\n\n", "\n\n")
                .replace("\n\n\n", "\n\n")
            )

            contents += content + f"\n{page_num}\n"
            jdx += 1

        base_name = os.path.basename(pdf_path)
        mmd_det_path = os.path.join(out_dir, base_name.replace(".pdf", "_det.mmd"))
        mmd_path = os.path.join(out_dir, base_name.replace(".pdf", ".mmd"))
        pdf_out_path = os.path.join(out_dir, base_name.replace(".pdf", "_layouts.pdf"))

        with open(mmd_det_path, "w", encoding="utf-8") as afile:
            afile.write(contents_det)
        with open(mmd_path, "w", encoding="utf-8") as afile:
            afile.write(contents)

        # 创建 zip 文件供下载
        zip_path = os.path.join("outputs", "vllm_gradio_pdf", f"pdf_result_{ts}.zip")
        
        if export_layout_pdf:
            log_info(f"📊 正在生成布局PDF...")
            pil_to_pdf_img2pdf(draw_images, pdf_out_path)
        
        # 打包所有结果到 zip
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(out_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, out_dir)
                    zf.write(file_path, arcname)
        
        total_time = time.time() - pdf_start_time
        log_info("=" * 60)
        log_success(f"📄 PDF处理完成！")
        log_info(f"   文件名: {pdf_name}")
        log_info(f"   处理页数: {len(images)} 页")
        log_info(f"   总耗时: {total_time:.2f} 秒")
        log_info(f"   平均速度: {total_time/len(images):.2f} 秒/页")
        log_info(f"   输出目录: {out_dir}")
        log_info("=" * 60)
        
        return contents, contents_det, zip_path

    except Exception as e:
        log_error(f"PDF处理失败: {str(e)}")
        return f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}", "", None



def create_demo():
    # 自定义 CSS 样式 - 长小养照护智能资源数字化平台主题
    custom_css = """
    /* 全局样式 */
    .gradio-container {
        font-family: 'Microsoft YaHei', 'PingFang SC', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        min-height: 100vh;
    }
    
    /* 主容器 */
    .main {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 20px !important;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15) !important;
        margin: 20px !important;
        padding: 30px !important;
    }
    
    /* 页面头部样式 */
    .header-banner {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #1e3c72 100%);
        padding: 40px 30px;
        border-radius: 16px;
        margin-bottom: 25px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(30, 60, 114, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .header-banner::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Ccircle cx='30' cy='30' r='4'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        pointer-events: none;
    }
    
    .header-banner h1 {
        color: #ffffff !important;
        font-size: 2.5em !important;
        font-weight: 700 !important;
        margin: 0 0 15px 0 !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        letter-spacing: 3px;
    }
    
    .header-banner p {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1.1em !important;
        margin: 8px 0 !important;
        line-height: 1.6;
    }
    
    .header-banner .subtitle {
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        padding: 8px 20px;
        border-radius: 25px;
        margin-top: 10px;
        backdrop-filter: blur(10px);
    }
    
    /* 提示框样式 */
    .tips-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 4px solid #4caf50;
        border-radius: 0 12px 12px 0;
        padding: 15px 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.15);
    }
    
    .tips-box .tips-title {
        color: #2e7d32 !important;
        margin: 0 0 10px 0 !important;
        font-size: 1em;
        font-weight: 600;
    }
    
    .tips-box .tips-title strong {
        color: #2e7d32 !important;
    }
    
    .tips-box .tips-content {
        color: #1b5e20 !important;
        margin: 6px 0 !important;
        font-size: 0.9em;
        line-height: 1.5;
    }
    
    /* 选项卡样式 */
    .tabs > .tab-nav {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
        border-radius: 12px !important;
        padding: 5px !important;
        margin-bottom: 20px !important;
    }
    
    .tabs > .tab-nav > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        padding: 12px 24px !important;
        color: #1e3c72 !important;
        background: transparent !important;
    }
    
    .tabs > .tab-nav > button:hover {
        background: rgba(30, 60, 114, 0.1) !important;
        color: #1e3c72 !important;
    }
    
    .tabs > .tab-nav > button.selected {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(30, 60, 114, 0.3) !important;
    }
    
    /* 按钮样式 */
    .primary {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 1.05em !important;
        padding: 12px 28px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(30, 60, 114, 0.3) !important;
    }
    
    .primary:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(30, 60, 114, 0.4) !important;
    }
    
    button.secondary {
        background: linear-gradient(135deg, #6c757d 0%, #495057 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        color: white !important;
        transition: all 0.3s ease !important;
    }
    
    button.secondary:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* 输入框样式 */
    textarea, input[type="text"] {
        border: 2px solid #e9ecef !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
    }
    
    textarea:focus, input[type="text"]:focus {
        border-color: #2a5298 !important;
        box-shadow: 0 0 0 3px rgba(42, 82, 152, 0.15) !important;
    }
    
    /* 滑块样式 */
    input[type="range"] {
        accent-color: #2a5298 !important;
    }
    
    /* 文件上传区域 */
    .file-upload {
        border: 2px dashed #2a5298 !important;
        border-radius: 12px !important;
        background: rgba(42, 82, 152, 0.03) !important;
        transition: all 0.3s ease !important;
    }
    
    .file-upload:hover {
        background: rgba(42, 82, 152, 0.08) !important;
        border-color: #1e3c72 !important;
    }
    
    /* 图片上传区域 */
    .image-upload {
        border-radius: 12px !important;
        overflow: hidden;
    }
    
    /* Accordion 手风琴样式 */
    .accordion {
        border: 1px solid #e9ecef !important;
        border-radius: 12px !important;
        overflow: hidden;
    }
    
    .accordion > .label-wrap {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
        padding: 12px 16px !important;
    }
    
    /* 设置面板 */
    .settings-panel {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid #e9ecef;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    }
    
    /* 隐藏 Gradio 原生 footer */
    footer.svelte-1rjryqp,
    footer.svelte-mpyp5e,
    .gradio-container > footer,
    footer[class*="svelte"],
    .built-with {
        display: none !important;
    }
    
    /* 页脚样式 */
    .footer {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        margin-top: 30px;
        text-align: center;
    }
    
    .footer p {
        margin: 5px 0 !important;
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    .footer .copyright {
        font-size: 0.95em;
        opacity: 0.9;
    }
    
    .footer .tech-info {
        font-size: 0.85em;
        opacity: 0.7;
        margin-top: 10px !important;
    }
    
    /* 单选按钮组 */
    .radio-group {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 10px;
    }
    
    /* 复选框 */
    input[type="checkbox"] {
        accent-color: #2a5298 !important;
    }
    
    /* 状态文本框 */
    .status-box textarea {
        background: #f8f9fa !important;
        font-family: 'Consolas', 'Monaco', monospace !important;
    }
    
    /* 响应式调整 */
    @media (max-width: 768px) {
        .header-banner h1 {
            font-size: 1.8em !important;
        }
        .main {
            margin: 10px !important;
            padding: 15px !important;
        }
    }
    
    /* 动画效果 */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .gradio-container > * {
        animation: fadeIn 0.5s ease-out;
    }
    """
    
    # Gradio 6.0+ 使用 launch(css=...) 而不是 Blocks(css=...)
    with gr.Blocks(title="长小养照护智能资源数字化平台") as demo:
        # 页面头部 Banner
        gr.HTML(
            """
            <div class="header-banner">
                <h1>🏥 长小养照护智能资源数字化平台</h1>
                <p>智能文档识别 · 高效数字化转换 · 专业照护知识管理</p>
                <div class="subtitle">📄 支持图片OCR · 批量处理 · PDF智能解析</div>
            </div>
            """
        )
        
        # 提示信息
        gr.HTML(
            """
            <div class="tips-box">
                <p class="tips-title">💡 <strong>使用提示</strong></p>
                <p class="tips-content">• <b>Markdown转换</b>：文档/论文识别，保留版面结构、表格、公式（推荐）</p>
                <p class="tips-content">• <b>自由识别</b>：纯文字提取，不含布局信息</p>
                <p class="tips-content">• <b>图片OCR</b>：通用图片中的文字识别</p>
                <p class="tips-content">• <b>图表解析</b>：专门解析图表、流程图等</p>
                <p class="tips-content">• <b>图像描述</b>：获取图片的详细描述</p>
            </div>
            """
        )
        
        # 设置区域标题
        gr.Markdown("### ⚙️ 通用设置")

        with gr.Row():
            with gr.Column(scale=1):
                prompt_type = gr.Radio(
                    choices=[
                        "Markdown转换",
                        "自由识别",
                        "图片OCR",
                        "图表解析",
                        "图像描述",
                        "自定义",
                    ],
                    value="Markdown转换",
                    label="📝 识别模式",
                    info="根据内容类型选择：文档用Markdown、纯文字用自由识别、图表用图表解析"
                )
                custom_prompt = gr.Textbox(
                    label="自定义指令（选择「自定义」时生效）",
                    placeholder="例如: Locate <|ref|>关键词<|/ref|> in the image.",
                    lines=2,
                    visible=False,
                )
            with gr.Column(scale=1):
                crop_mode = gr.Checkbox(
                    label="📐 启用智能裁剪",
                    value=bool(CROP_MODE),
                    info="适用于大尺寸文档图片"
                )
                model_size = gr.Radio(
                    choices=[
                        "极速（Tiny）",
                        "快速（Small）",
                        "标准（Base）",
                        "精细（Large）",
                        "高达模式（推荐）",
                    ],
                    value="标准（Base）",
                    label="🎯 模型精度",
                    info="高达模式平衡速度与精度，推荐使用"
                )
            with gr.Column(scale=1):
                with gr.Accordion("⚡ 高级参数", open=False):
                    # 动态解析并发滑条的默认值与上限，避免默认值越界
                    try:
                        _default_concurrency = int(MAX_CONCURRENCY) if MAX_CONCURRENCY else 12
                    except Exception:
                        _default_concurrency = 12
                    _concurrency_max = max(16, _default_concurrency)
                    _concurrency_default = min(_default_concurrency, _concurrency_max)
                    max_concurrency = gr.Slider(
                        minimum=1,
                        maximum=_concurrency_max,
                        step=1,
                        value=_concurrency_default,
                        label="并发数量",
                        info="建议根据显存大小调整"
                    )
                    gpu_memory_utilization = gr.Slider(
                        minimum=0.5,
                        maximum=0.98,
                        step=0.01,
                        value=0.85,
                        label="显存利用率",
                        info="建议保持在0.85左右"
                    )
                    max_tokens = gr.Slider(
                        minimum=256,
                        maximum=16384,
                        step=512,
                        value=16384,
                        label="最大生成长度",
                        info="文档较长时请增大此值"
                    )
                    with gr.Row():
                        restart_btn = gr.Button("♻️ 重启引擎", variant="secondary")
                        estimate_btn = gr.Button("🧮 估算并发", variant="secondary")
                    restart_service_btn = gr.Button("🔄 重启服务", variant="secondary")
                    engine_status = gr.Textbox(
                        label="引擎状态",
                        value="✅ 已就绪",
                        lines=2,
                        interactive=False,
                    )

        with gr.Tabs():
            with gr.Tab("📷 单图识别"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="📤 上传图片（支持拖拽/粘贴）",
                            type="pil",
                            sources=["upload", "clipboard"],
                        )
                        process_btn_single = gr.Button("🚀 开始识别", variant="primary")
                    with gr.Column(scale=1):
                        output_text_single = gr.Textbox(
                            label="📄 识别结果",
                            lines=20,
                            max_lines=30,
                        )

            with gr.Tab("📚 批量处理"):
                gr.HTML(
                    '''
                    <div style="background:linear-gradient(135deg,#e8f4fd,#d4e9f7);padding:15px 20px;border-radius:10px;margin-bottom:15px;border-left:4px solid #1e3c72;">
                        <p style="margin:0;"><span style="color:#1e3c72 !important;font-weight:bold;font-size:1.1em;">📂 批量识别模式</span> <span style="color:#333;">- 支持同时上传多张图片进行批处理</span></p>
                        <p style="margin:5px 0 0 0;color:#555;font-size:0.9em;">支持格式: JPG, PNG, WebP, BMP, TIFF</p>
                    </div>
                    '''
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        batch_files = gr.File(
                            label="📤 上传图片（可多选）",
                            file_count="multiple",
                            file_types=["image"],
                            type="filepath",
                        )
                        process_btn_batch = gr.Button("🚀 开始批量识别", variant="primary")
                    with gr.Column(scale=1):
                        batch_outdir_text = gr.Textbox(
                            label="📊 处理状态",
                            lines=2,
                            interactive=False,
                        )
                        batch_download = gr.File(
                            label="📥 下载识别结果（ZIP压缩包）",
                            interactive=False,
                        )
                        batch_preview_text = gr.Textbox(
                            label="👀 结果预览（前3项）",
                            lines=15,
                            max_lines=25,
                        )

            with gr.Tab("📑 PDF解析"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML(
                            '''
                            <div style="background:linear-gradient(135deg,#fff3e0,#ffe0b2);padding:15px 20px;border-radius:10px;margin-bottom:15px;border-left:4px solid #ff9800;">
                                <p style="margin:0;"><span style="color:#e65100 !important;font-weight:bold;font-size:1.1em;">📑 PDF智能解析</span></p>
                                <p style="margin:5px 0 0 0;color:#555;font-size:0.9em;">自动提取PDF内容并转换为Markdown格式</p>
                            </div>
                            '''
                        )
                        pdf_file = gr.File(
                            label="📤 上传PDF文件",
                            file_types=[".pdf"],
                            type="filepath",
                        )
                        pdf_dpi = gr.Slider(
                            minimum=72,
                            maximum=288,
                            step=12,
                            value=144,
                            label="🔍 渲染精度（DPI）",
                            info="数值越高精度越好，但速度越慢"
                        )
                        export_layout_pdf = gr.Checkbox(
                            label="📐 导出布局分析PDF",
                            value=False,
                            info="生成带标注的布局分析文档（处理较慢）"
                        )
                        process_btn_pdf = gr.Button("🚀 开始解析PDF", variant="primary")
                    with gr.Column(scale=1):
                        pdf_mmd_text = gr.Textbox(
                            label="📄 Markdown输出",
                            lines=20,
                            max_lines=30,
                        )
                        pdf_det_text = gr.Textbox(
                            label="🔍 详细检测结果",
                            lines=20,
                            max_lines=30,
                        )
                        pdf_layouts_file = gr.File(
                            label="📥 下载完整结果（ZIP压缩包）",
                            interactive=False,
                        )

        def update_prompt_visibility(choice):
            return gr.update(visible=(choice == "自定义"))

        prompt_type.change(
            fn=update_prompt_visibility,
            inputs=[prompt_type],
            outputs=[custom_prompt],
        )

        # 当选择尺寸预设时，更新裁剪推荐值
        def apply_size_preset(choice):
            preset = size_configs.get(choice, size_configs["高达模式（推荐）"])
            return gr.update(value=preset["crop_mode"]) 

        model_size.change(
            fn=apply_size_preset,
            inputs=[model_size],
            outputs=[crop_mode],
        )

        # 触发服务级重启（watch 模式下通过写入信号文件触发）
        def trigger_service_restart():
            try:
                sig_path = os.path.join(ROOT_DIR, "watch_restart.json")
                import json
                ts = time.time()
                with open(sig_path, "w", encoding="utf-8") as f:
                    json.dump({"restart_at": ts}, f)
                return "已触发服务重启（watch），请稍后刷新页面。"
            except Exception as e:
                return f"触发失败：{e}"

        # 基于 GPU 显存与 max_tokens 的并发估算
        def estimate_concurrency_action(gmu: float, max_toks: int):
            try:
                if torch.cuda.is_available():
                    free_b, total_b = torch.cuda.mem_get_info()
                    total_gb = total_b / (1024 ** 3)
                    free_gb = free_b / (1024 ** 3)
                    # 有效可用显存：考虑 slider 的 gmu，尽量不超当前空闲
                    effective_gb = max(min(total_gb * gmu, free_gb) - 1.0, 1.0)
                else:
                    effective_gb = 8.0
            except Exception:
                try:
                    props = torch.cuda.get_device_properties(0)
                    total_gb = props.total_memory / (1024 ** 3)
                    effective_gb = max(total_gb * gmu - 1.0, 1.0)
                except Exception:
                    effective_gb = 8.0

            # 经验估算：8192 tokens 时每并发约 ~800MB；线性随 max_tokens 变化
            per_seq_mb = 800.0 * max(1.0, float(max_toks) / 8192.0)
            est = int(max(1, (effective_gb * 1024.0) / per_seq_mb))
            try:
                cfg_max = int(MAX_CONCURRENCY) if MAX_CONCURRENCY else 16
            except Exception:
                cfg_max = 16
            new_max = max(16, cfg_max, est)
            est = min(est, new_max)
            return gr.update(value=est, maximum=new_max)

        estimate_btn.click(
            fn=estimate_concurrency_action,
            inputs=[gpu_memory_utilization, max_tokens],
            outputs=[max_concurrency],
        )

        # 强制重启引擎（清理显存并按当前参数重建）
        def restart_engine_action(max_conc: int, gmu: float):
            try:
                _ensure_offline_env()
                _setup_cuda_env()
                # force_reinit 触发清理逻辑
                _ = init_llm(
                    max_concurrency=max_conc,
                    gpu_memory_utilization=gmu,
                    max_model_len=8192,
                    force_reinit=True,
                )
                return "引擎已重启：并发=%d，显存利用率=%.2f，max_len=8192" % (max_conc, gmu)
            except Exception as e:
                import traceback
                return f"重启失败：{str(e)}\n\n{traceback.format_exc()}"

        restart_btn.click(
            fn=restart_engine_action,
            inputs=[max_concurrency, gpu_memory_utilization],
            outputs=[engine_status],
        )

        restart_service_btn.click(
            fn=trigger_service_restart,
            inputs=[],
            outputs=[engine_status],
        )

        process_btn_single.click(
            fn=process_image,
            inputs=[
                image_input,
                prompt_type,
                custom_prompt,
                model_size,
                crop_mode,
                max_concurrency,
                gpu_memory_utilization,
                max_tokens,
            ],
            outputs=[output_text_single],
        )

        process_btn_batch.click(
            fn=process_batch_upload,
            inputs=[
                batch_files,
                prompt_type,
                custom_prompt,
                model_size,
                crop_mode,
                max_concurrency,
                gpu_memory_utilization,
                max_tokens,
            ],
            outputs=[batch_outdir_text, batch_preview_text, batch_download],
        )

        process_btn_pdf.click(
            fn=process_pdf,
            inputs=[
                pdf_file,
                pdf_dpi,
                prompt_type,
                custom_prompt,
                model_size,
                crop_mode,
                max_concurrency,
                gpu_memory_utilization,
                max_tokens,
                export_layout_pdf,
            ],
            outputs=[pdf_mmd_text, pdf_det_text, pdf_layouts_file],
        )
        
        # 页脚版权信息
        gr.HTML(
            """
            <div class="footer">
                <p style="font-size:1.1em;font-weight:600;">🏥 长小养照护智能资源数字化平台</p>
                <p class="copyright">© 2025 海南长小养智能科技 版权所有</p>
                <p class="tech-info">技术支持: DeepSeek-OCR · vLLM 高性能推理引擎</p>
            </div>
            """
        )

    return demo, custom_css


if __name__ == "__main__":
    _ensure_offline_env()
    _setup_cuda_env()
    # 可选启动预热：设置环境变量 WARMUP_ON_START=1 启用
    def warmup_engine_on_start():
        if os.environ.get("WARMUP_ON_START", "0") != "1":
            print("[INFO] 跳过模型预热（设置 WARMUP_ON_START=1 可启用）")
            return
        
        print("=" * 50)
        print("🚀 正在预热模型，请稍候...")
        print("=" * 50)
        
        try:
            _default_concurrency = int(MAX_CONCURRENCY) if MAX_CONCURRENCY else 8
        except Exception:
            _default_concurrency = 8
        
        gmu = 0.85
        print(f"[INFO] 加载配置: 并发={_default_concurrency}, 显存利用率={gmu}, max_len=8192")
        
        try:
            llm_local = init_llm(
                max_concurrency=_default_concurrency,
                gpu_memory_utilization=gmu,
                max_model_len=8192,
            )
            print("[INFO] ✅ 模型加载完成")
            
            # 构造极小图像进行一次轻量生成以触发图捕获与缓存
            print("[INFO] 正在预热推理引擎...")
            from PIL import Image as _Image
            img = _Image.new("RGB", (64, 64), color=(255, 255, 255))
            proc = DeepseekOCRProcessor(image_size=640, base_size=1024)
            image_features = proc.tokenize_with_images(images=[img], bos=True, eos=True, cropping=False)
            cache_item = {"prompt": "<image>\nWarmup.", "multi_modal_data": {"image": image_features}}
            sp = SamplingParams(temperature=0.0, max_tokens=16, skip_special_tokens=True)
            llm_local.generate([cache_item], sampling_params=sp)
            
            print("=" * 50)
            print("✅ 模型预热完成，服务即将启动！")
            print("=" * 50)
        except Exception as e:
            print(f"[WARN] ⚠️ 模型预热失败: {e}")
            print("[INFO] 服务将继续启动，首次推理时会加载模型")

    warmup_engine_on_start()
    demo, custom_css = create_demo()
    port = int(os.environ.get("DEMO_PORT", "7860"))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        debug=True,
        css=custom_css,
    )

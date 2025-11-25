import os
import sys
import time
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
    "Tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
    "Small": {"base_size": 640, "image_size": 640, "crop_mode": False},
    "Base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
    "Large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
    "Gundam (Recommended)": {"base_size": 1024, "image_size": 640, "crop_mode": True},
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
        llm_local = init_llm(
            max_concurrency=max_concurrency,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=8192,
        )

        if prompt_type == "Free OCR":
            prompt = "<image>\nFree OCR. "
        elif prompt_type == "Markdown Conversion":
            prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        elif prompt_type == "Custom":
            prompt = f"<image>\n{custom_prompt}"
        else:
            prompt = "<image>\nFree OCR. "

        # Apply size preset
        preset = size_configs.get(model_size, size_configs["Gundam (Recommended)"])
        base_size = preset["base_size"]
        image_size = preset["image_size"]
        # Use current checkbox for cropping (updated by preset change)
        image = image.convert("RGB")
        proc = DeepseekOCRProcessor(image_size=image_size, base_size=base_size)
        image_features = proc.tokenize_with_images(
            images=[image], bos=True, eos=True, cropping=crop_mode
        )

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

        outputs = llm_local.generate(
            [cache_item], sampling_params=sampling_params
        )

        content = outputs[0].outputs[0].text
        return content

    except Exception as e:
        import traceback
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
            return "请先上传图片文件", ""
        
        llm_local = init_llm(
            max_concurrency=max_concurrency,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=8192,
        )

        images = []
        valid_paths = []
        for file_path in uploaded_files:
            try:
                image = Image.open(file_path).convert("RGB")
                images.append(image)
                valid_paths.append(file_path)
            except Exception as e:
                print(f"skip file: {file_path} due to error: {e}")

        if not images:
            return "没有可处理的有效图片文件", ""

        if prompt_type == "Free OCR":
            prompt = "<image>\nFree OCR. "
        elif prompt_type == "Markdown Conversion":
            prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        elif prompt_type == "Custom":
            prompt = f"<image>\n{custom_prompt}"
        else:
            prompt = "<image>\nFree OCR. "

        preset = size_configs.get(model_size, size_configs["Gundam (Recommended)"])
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

        return f"已处理 {len(images)} 张图片，结果保存到: {out_dir}", "\n\n".join(preview_texts), zip_path

    except Exception as e:
        import traceback
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

        if prompt_type == "Free OCR":
            prompt = "<image>\nFree OCR. "
        elif prompt_type == "Markdown Conversion":
            prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        elif prompt_type == "Custom":
            prompt = f"<image>\n{custom_prompt}"
        else:
            prompt = "<image>\nFree OCR. "

        preset = size_configs.get(model_size, size_configs["Gundam (Recommended)"])
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
        llm_local = init_llm(
            max_concurrency=max_concurrency,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=8192,
        )

        if prompt_type == "Free OCR":
            prompt = "<image>\nFree OCR. "
        elif prompt_type == "Markdown Conversion":
            prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        elif prompt_type == "Custom":
            prompt = f"<image>\n{custom_prompt}"
        else:
            prompt = "<image>\nFree OCR. "

        images = pdf_to_images_high_quality(pdf_path, dpi=dpi)
        if not images:
            return "PDF 中无可处理页面", "", None

        preset = size_configs.get(model_size, size_configs["Gundam (Recommended)"])
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

        outputs_list = llm_local.generate(batch_inputs, sampling_params=sampling_params)

        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("outputs", "vllm_gradio_pdf", ts)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

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
            pil_to_pdf_img2pdf(draw_images, pdf_out_path)
        
        # 打包所有结果到 zip
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(out_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, out_dir)
                    zf.write(file_path, arcname)
        
        return contents, contents_det, zip_path

    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}", "", None


def create_demo():
    # Older Gradio versions may not support the theme kwarg; keep compatibility by omitting it.
    with gr.Blocks(title="DeepSeek-OCR vLLM Demo") as demo:
        gr.Markdown(
            """
            > 🛈 引擎重建 vs 服务重启
            - 点击“♻️ 重启引擎”仅清理显存并重建 vLLM 引擎；不会应用代码改动
            - 修改 Python 源码或 UI 布局后请使用 `run_demo.sh` 重启服务
            - 遇到 CUDA/显存异常：先尝试“重启引擎”，仍异常再重启服务
            - 需要自动重启服务：使用 `./run_demo.sh --watch` 启用文件监听
            """
        )
        gr.Markdown(
            """
            # 🔍 DeepSeek-OCR vLLM Demo

            使用 vLLM 引擎进行离线 OCR 推理（本地 MODEL/TOKENIZER）。
            - 支持并发与显存参数配置
            - 适配 4090D，KV Cache 高并发
            - 提供 单图 / 批量 / PDF 三种模式
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ 设置（通用）")
                prompt_type = gr.Radio(
                    choices=["Free OCR", "Markdown Conversion", "Custom"],
                    value="Markdown Conversion",
                    label="Prompt 类型",
                )
                custom_prompt = gr.Textbox(
                    label="自定义 Prompt（选择 Custom 时生效）",
                    placeholder="输入自定义指令...",
                    lines=2,
                    visible=False,
                )
                crop_mode = gr.Checkbox(
                    label="启用裁剪（CROP_MODE）",
                    value=bool(CROP_MODE),
                )
                model_size = gr.Radio(
                    choices=[
                        "Tiny",
                        "Small",
                        "Base",
                        "Large",
                        "Gundam (Recommended)",
                    ],
                    value="Base",
                    label="模型尺寸预设",
                )
                with gr.Accordion("高级参数", open=False):
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
                        label="并发（max_num_seqs）",
                    )
                    gpu_memory_utilization = gr.Slider(
                        minimum=0.5,
                        maximum=0.98,
                        step=0.01,
                        value=0.85,
                        label="显存利用率（gpu_memory_utilization）",
                    )
                    max_tokens = gr.Slider(
                        minimum=256,
                        maximum=16384,
                        step=512,
                        value=16384,
                        label="生成长度（max_tokens）",
                    )
                    restart_btn = gr.Button("♻️ 重启引擎（清理显存）")
                    restart_service_btn = gr.Button("🔄 重启服务（watch）")
                    engine_status = gr.Textbox(
                        label="引擎状态",
                        value="已就绪",
                        lines=2,
                        interactive=False,
                    )
                    estimate_btn = gr.Button("🧮 根据显存估算并发")

            with gr.Column(scale=1):
                gr.Markdown(
                    f"""
                    ### 🔧 当前模型
                    - MODEL_PATH: `{MODEL_PATH}`
                    - TOKENIZER_PATH: `{TOKENIZER_PATH}`
                    """
                )

        with gr.Tabs():
            with gr.Tab("单图"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="上传图片",
                            type="pil",
                            sources=["upload", "clipboard"],
                        )
                        process_btn_single = gr.Button("🚀 开始处理（单图）", variant="primary")
                    with gr.Column(scale=1):
                        output_text_single = gr.Textbox(
                            label="提取文本",
                            lines=20,
                            max_lines=30,
                        )

            with gr.Tab("批量"):
                gr.Markdown("**上传多张图片进行批量 OCR 处理**（支持 jpg/png/webp/bmp/tiff）")
                with gr.Row():
                    with gr.Column(scale=1):
                        batch_files = gr.File(
                            label="上传图片（可多选）",
                            file_count="multiple",
                            file_types=["image"],
                            type="filepath",
                        )
                        process_btn_batch = gr.Button("🚀 开始处理（批量）", variant="primary")
                    with gr.Column(scale=1):
                        batch_outdir_text = gr.Textbox(
                            label="处理状态",
                            lines=2,
                            interactive=False,
                        )
                        batch_download = gr.File(
                            label="📥 下载结果（ZIP）",
                            interactive=False,
                        )
                        batch_preview_text = gr.Textbox(
                            label="预览（前3项节选）",
                            lines=15,
                            max_lines=25,
                        )

            with gr.Tab("PDF"):
                with gr.Row():
                    with gr.Column(scale=1):
                        pdf_file = gr.File(
                            label="上传 PDF",
                            file_types=[".pdf"],
                            type="filepath",
                        )
                        pdf_dpi = gr.Slider(
                            minimum=72,
                            maximum=288,
                            step=12,
                            value=144,
                            label="PDF DPI（渲染）",
                        )
                        export_layout_pdf = gr.Checkbox(
                            label="导出布局 PDF（较慢，默认关闭）",
                            value=False,
                        )
                        process_btn_pdf = gr.Button("🚀 开始处理（PDF）", variant="primary")
                    with gr.Column(scale=1):
                        pdf_mmd_text = gr.Textbox(
                            label="Markdown 输出（合并）",
                            lines=20,
                            max_lines=30,
                        )
                        pdf_det_text = gr.Textbox(
                            label="检测输出（合并）",
                            lines=20,
                            max_lines=30,
                        )
                        pdf_layouts_file = gr.File(
                            label="📥 下载结果（ZIP，含 Markdown + 图片）",
                            interactive=False,
                        )

        def update_prompt_visibility(choice):
            return gr.update(visible=(choice == "Custom"))

        prompt_type.change(
            fn=update_prompt_visibility,
            inputs=[prompt_type],
            outputs=[custom_prompt],
        )

        # 当选择尺寸预设时，更新裁剪推荐值
        def apply_size_preset(choice):
            preset = size_configs.get(choice, size_configs["Gundam (Recommended)"])
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

    return demo


if __name__ == "__main__":
    _ensure_offline_env()
    _setup_cuda_env()
    # 可选启动预热：设置环境变量 WARMUP_ON_START=1 启用
    def warmup_engine_on_start():
        try:
            if os.environ.get("WARMUP_ON_START", "0") != "1":
                return
            try:
                _default_concurrency = int(MAX_CONCURRENCY) if MAX_CONCURRENCY else 8
            except Exception:
                _default_concurrency = 8
            gmu = 0.85
            llm_local = init_llm(
                max_concurrency=_default_concurrency,
                gpu_memory_utilization=gmu,
                max_model_len=8192,
            )
            # 构造极小图像进行一次轻量生成以触发图捕获与缓存
            from PIL import Image as _Image
            img = _Image.new("RGB", (64, 64), color=(255, 255, 255))
            proc = DeepseekOCRProcessor(image_size=640, base_size=1024)
            image_features = proc.tokenize_with_images(images=[img], bos=True, eos=True, cropping=False)
            cache_item = {"prompt": "<image>\nWarmup.", "multi_modal_data": {"image": image_features}}
            sp = SamplingParams(temperature=0.0, max_tokens=16, skip_special_tokens=True)
            llm_local.generate([cache_item], sampling_params=sp)
        except Exception:
            # 预热失败不影响正常启动
            pass

    warmup_engine_on_start()
    demo = create_demo()
    port = int(os.environ.get("DEMO_PORT", "7860"))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        debug=True,
    )

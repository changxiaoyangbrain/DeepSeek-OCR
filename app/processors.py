"""
处理模块 - 图片处理、PDF处理、边界框绘制等
"""
import os
import io
import time
import zipfile
import traceback
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF

from vllm import SamplingParams
from process.image_process import DeepseekOCRProcessor
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor

from .config import SIZE_CONFIGS, get_prompt
from .utils import (
    log_info, log_success, log_warning, log_error, log_progress,
    extract_grounding_references, clean_output_text, embed_images_in_markdown,
    re_match, re_match_pdf, clean_formula, is_image_file
)
from .engine import init_llm


# ============================================
# 边界框绘制
# ============================================
def draw_bounding_boxes_on_image(
    image: Image.Image, 
    refs: List[Tuple[str, str, str]], 
    extract_images: bool = False
) -> Tuple[Image.Image, List[Image.Image]]:
    """
    在图片上绘制边界框
    返回: (标注后的图片, 裁剪的图片列表)
    """
    img_w, img_h = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 24)
        except Exception:
            font = ImageFont.load_default()
    
    crops = []
    color_map = {}
    np.random.seed(42)
    
    for ref in refs:
        label = ref[1]
        if label not in color_map:
            color_map[label] = (
                np.random.randint(50, 255), 
                np.random.randint(50, 255), 
                np.random.randint(50, 255)
            )
        color = color_map[label]
        
        try:
            coords = eval(ref[2])
        except Exception:
            continue
            
        color_a = color + (60,)
        
        for box in coords:
            try:
                x1 = int(box[0] / 999 * img_w)
                y1 = int(box[1] / 999 * img_h)
                x2 = int(box[2] / 999 * img_w)
                y2 = int(box[3] / 999 * img_h)
                
                if extract_images and label == 'image':
                    try:
                        cropped = image.crop((x1, y1, x2, y2))
                        crops.append(cropped)
                    except Exception:
                        pass
                
                width = 5 if label == 'title' else 3
                draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
                draw2.rectangle([x1, y1, x2, y2], fill=color_a)
                
                text_bbox = draw.textbbox((0, 0), label, font=font)
                tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                ty = max(0, y1 - th - 4)
                draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 4], fill=color)
                draw.text((x1 + 2, ty + 2), label, font=font, fill=(255, 255, 255))
            except Exception:
                continue
    
    img_draw = img_draw.convert('RGBA')
    img_draw = Image.alpha_composite(img_draw, overlay)
    img_draw = img_draw.convert('RGB')
    
    return img_draw, crops


# ============================================
# PDF 工具函数
# ============================================
def pdf_to_images_high_quality(pdf_path: str, dpi: int = 144) -> List[Image.Image]:
    """将 PDF 转换为高质量图片列表"""
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


def get_pdf_page_count(pdf_path: str) -> int:
    """获取 PDF 页数"""
    if not pdf_path or not os.path.exists(pdf_path):
        return 0
    try:
        doc = fitz.open(pdf_path)
        count = doc.page_count
        doc.close()
        return count
    except Exception:
        return 0


def pdf_page_to_image(pdf_path: str, page_num: int, dpi: int = 144) -> Optional[Image.Image]:
    """将 PDF 指定页面转换为图片"""
    try:
        doc = fitz.open(pdf_path)
        if page_num < 1 or page_num > doc.page_count:
            doc.close()
            return None
        
        page = doc.load_page(page_num - 1)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.open(io.BytesIO(pixmap.tobytes("png")))
        doc.close()
        return img
    except Exception:
        return None


def pil_to_pdf(pil_images: List[Image.Image], output_path: str):
    """将 PIL 图片列表保存为 PDF"""
    import img2pdf
    if not pil_images:
        return
    
    image_bytes_list = []
    for img in pil_images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="JPEG", quality=95)
        image_bytes_list.append(img_buffer.getvalue())
    
    try:
        pdf_bytes = img2pdf.convert(image_bytes_list)
        if pdf_bytes is None:
            log_error("PDF 生成失败: img2pdf.convert 返回 None")
            return
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)
    except Exception as e:
        log_error(f"PDF 生成失败: {e}")


# ============================================
# 单图处理
# ============================================
def process_single_image(
    image: Image.Image,
    prompt_type: str,
    custom_prompt: str,
    model_size: str,
    crop_mode: bool,
    max_concurrency: int,
    gpu_memory_utilization: float,
    max_tokens: int,
) -> Tuple[str, str, str, Optional[Image.Image], List[Image.Image]]:
    """
    处理单张图片
    返回: (清理后文本, Markdown渲染, 原始输出, 标注图片, 裁剪图片列表)
    """
    try:
        if image is None:
            return "未检测到图片，请先上传图片后再点击处理。", "", "", None, []
        
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

        # 获取 prompt
        prompt, has_grounding = get_prompt(prompt_type, custom_prompt)
        
        if prompt_type == "定位识别" and not custom_prompt.strip():
            return "请输入要定位的文字", "", "", None, []
        
        log_info(f"   Prompt: {prompt[:60]}...")
        log_info(f"   Grounding: {'是' if has_grounding else '否'}")

        # 获取尺寸配置
        preset = SIZE_CONFIGS.get(model_size, SIZE_CONFIGS["高达模式（推荐）"])
        base_size = preset["base_size"]
        image_size = preset["image_size"]
        
        original_image = image.copy()
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
        outputs = llm_local.generate([cache_item], sampling_params=sampling_params)  # type: ignore[arg-type]
        inference_time = time.time() - inference_start
        log_success(f"   推理完成, 耗时 {inference_time:.2f} 秒")

        raw_content = outputs[0].outputs[0].text
        
        total_time = time.time() - single_start_time
        log_info("=" * 50)
        log_success(f"📷 单图识别完成！")
        log_info(f"   总耗时: {total_time:.2f} 秒")
        log_info(f"   输出长度: {len(raw_content)} 字符")
        log_info("=" * 50)
        
        # 处理输出
        cleaned_text = clean_output_text(raw_content, include_images=False, remove_labels=False)
        markdown_text = clean_output_text(raw_content, include_images=True, remove_labels=True)
        
        annotated_image = None
        cropped_images = []
        
        if has_grounding and '<|ref|>' in raw_content:
            refs = extract_grounding_references(raw_content)
            if refs:
                annotated_image, cropped_images = draw_bounding_boxes_on_image(
                    original_image, refs, extract_images=True
                )
                markdown_text = embed_images_in_markdown(markdown_text, cropped_images)
        
        return cleaned_text, markdown_text, raw_content, annotated_image, cropped_images

    except Exception as e:
        log_error(f"单图识别失败: {str(e)}")
        error_msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return error_msg, "", "", None, []


# ============================================
# 批量图片处理
# ============================================
def process_batch_images(
    uploaded_files: List[str],
    prompt_type: str,
    custom_prompt: str,
    model_size: str,
    crop_mode: bool,
    max_concurrency: int,
    gpu_memory_utilization: float,
    max_tokens: int,
) -> Tuple[str, str, List[Image.Image], List[Image.Image], str, str, Optional[str]]:
    """
    处理批量上传的图片
    返回: (纯文本, Markdown渲染, 边界框图列表, 裁剪图列表, 原始输出, 状态信息, ZIP文件路径)
    """
    try:
        if not uploaded_files:
            return "请先上传图片文件", "", [], [], "", "⚠️ 请先上传图片文件", None
        
        batch_start_time = time.time()
        log_info("=" * 50)
        log_info(f"📚 开始批量处理任务")
        log_info(f"   文件数量: {len(uploaded_files)}")
        log_info(f"   识别模式: {prompt_type}")
        log_info(f"   模型精度: {model_size}")
        log_info("=" * 50)
        
        llm_local = init_llm(
            max_concurrency=max_concurrency,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=8192,
        )

        # 加载图片
        images = []
        valid_paths = []
        original_images = []
        for idx, file_path in enumerate(uploaded_files):
            try:
                image = Image.open(file_path).convert("RGB")
                images.append(image)
                original_images.append(image.copy())
                valid_paths.append(file_path)
                log_progress(idx + 1, len(uploaded_files), "加载图片")
            except Exception as e:
                log_warning(f"跳过文件: {os.path.basename(file_path)} - {e}")

        if not images:
            return "没有可处理的有效图片文件", "", [], [], "", "⚠️ 没有可处理的有效图片文件", None
        
        log_success(f"成功加载 {len(images)} 张图片")

        # 获取 prompt
        prompt, has_grounding = get_prompt(prompt_type, custom_prompt)
        
        if prompt_type == "定位识别" and not custom_prompt.strip():
            return "定位识别模式需要输入要查找的文字", "", [], [], "", "⚠️ 定位识别模式需要输入要查找的文字", None

        preset = SIZE_CONFIGS.get(model_size, SIZE_CONFIGS["高达模式（推荐）"])
        base_size = preset["base_size"]
        image_size = preset["image_size"]
        
        log_info(f"正在预处理图片...")
        preprocess_start = time.time()
        proc = DeepseekOCRProcessor(image_size=image_size, base_size=base_size)
        batch_inputs = []
        for idx, img in enumerate(images):
            image_features = proc.tokenize_with_images(
                images=[img], bos=True, eos=True, cropping=crop_mode
            )
            batch_inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": image_features},
            })
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

        log_info(f"🚀 开始批量推理...")
        inference_start = time.time()
        outputs_list = llm_local.generate(batch_inputs, sampling_params=sampling_params)
        inference_time = time.time() - inference_start
        avg_time = inference_time / len(images) if images else 0
        log_success(f"推理完成, 总耗时 {inference_time:.2f} 秒, 平均 {avg_time:.2f} 秒/张")

        # 保存结果
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("outputs", "vllm_gradio_batch", ts)
        os.makedirs(out_dir, exist_ok=True)

        all_text = []
        all_markdown = []
        all_raw = []
        all_boxes: List[Image.Image] = []
        all_cropped: List[Image.Image] = []
        
        for idx, (output, file_path, orig_img) in enumerate(zip(outputs_list, valid_paths, original_images)):
            content = output.outputs[0].text
            base_name = os.path.basename(file_path)
            name_no_ext = os.path.splitext(base_name)[0]
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name_no_ext)

            # 保存原始输出
            with open(os.path.join(out_dir, f"{safe_name}_det.md"), "w", encoding="utf-8") as f:
                f.write(content)

            # 保存清理后的输出
            content_clean = clean_output_text(content, include_images=False, remove_labels=False)
            content_markdown = clean_output_text(content, include_images=True, remove_labels=True)
            with open(os.path.join(out_dir, f"{safe_name}.md"), "w", encoding="utf-8") as f:
                f.write(content_clean)

            # 收集文本
            all_text.append(f"## 📄 {base_name}\n\n{content_clean}")
            all_markdown.append(f"## 📄 {base_name}\n\n{content_markdown}")
            all_raw.append(f"## 📄 {base_name}\n\n{content}")
            
            # 处理边界框
            if has_grounding and '<|ref|>' in content:
                refs = extract_grounding_references(content)
                if refs:
                    annotated_img, cropped_imgs = draw_bounding_boxes_on_image(
                        orig_img, refs, extract_images=True
                    )
                    all_boxes.append(annotated_img)
                    all_cropped.extend(cropped_imgs)

        # 创建 ZIP
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
        log_info("=" * 50)
        
        text_output = "\n\n---\n\n".join(all_text)
        markdown_output = "\n\n---\n\n".join(all_markdown)
        raw_output = "\n\n---\n\n".join(all_raw)
        status = f"✅ 已处理 {len(images)} 张图片 | ⏱️ 总耗时: {total_time:.1f}秒 | ⚡ 平均: {avg_time:.2f}秒/张"
        
        return text_output, markdown_output, all_boxes, all_cropped, raw_output, status, zip_path

    except Exception as e:
        log_error(f"批量处理失败: {str(e)}")
        error_msg = f"处理出错: {str(e)}\n{traceback.format_exc()}"
        return error_msg, "", [], [], "", f"❌ 处理失败: {str(e)}", None


# ============================================
# PDF 处理
# ============================================
def process_pdf_document(
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
    page_start: int = 1,
    page_end: int = -1,
) -> Tuple[str, str, List[Image.Image], List[Image.Image], str, str, Optional[str]]:
    """
    处理 PDF 文档
    返回: (纯文本, Markdown渲染, 边界框图列表, 裁剪图列表, 原始输出, 状态信息, ZIP文件路径)
    """
    try:
        if not pdf_path:
            return "未检测到 PDF 文件，请先上传后再点击处理。", "", [], [], "", "⚠️ 请先上传PDF文件", None
        if not os.path.exists(pdf_path):
            return f"文件不存在：{pdf_path}", "", [], [], "", f"⚠️ 文件不存在", None
        
        pdf_start_time = time.time()
        pdf_name = os.path.basename(pdf_path)
        total_pages = get_pdf_page_count(pdf_path)
        
        # 处理页面范围
        if page_end <= 0 or page_end > total_pages:
            page_end = total_pages
        if page_start < 1:
            page_start = 1
        if page_start > page_end:
            page_start = page_end
        
        log_info("=" * 60)
        log_info(f"📄 开始处理 PDF: {pdf_name}")
        log_info("=" * 60)
        log_info(f"   总页数: {total_pages}")
        log_info(f"   处理范围: 第 {page_start} - {page_end} 页")
        log_info(f"   DPI: {dpi}")
        log_info(f"   识别模式: {prompt_type}")
        log_info(f"   模型档位: {model_size}")
        
        llm_local = init_llm(
            max_concurrency=max_concurrency,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=8192,
        )

        # 获取 prompt
        prompt, has_grounding = get_prompt(prompt_type, custom_prompt)
        
        if prompt_type == "定位识别" and not custom_prompt.strip():
            return "定位识别模式需要输入要查找的文字", "", [], [], "", "⚠️ 定位识别模式需要输入要查找的文字", None

        log_info(f"📖 正在转换 PDF 为图片 (DPI={dpi})...")
        convert_start = time.time()
        
        # 只转换指定页面范围
        all_images = pdf_to_images_high_quality(pdf_path, dpi=dpi)
        images = all_images[page_start - 1 : page_end]
        original_images = [img.copy() for img in images]
        
        convert_time = time.time() - convert_start
        if not images:
            return "PDF 中无可处理页面", "", [], [], "", "⚠️ PDF 中无可处理页面", None
        log_success(f"   PDF转换完成: {len(images)} 页, 耗时 {convert_time:.2f} 秒")

        preset = SIZE_CONFIGS.get(model_size, SIZE_CONFIGS["高达模式（推荐）"])
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
            batch_inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": image_features},
            })
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
        
        all_text = []
        all_markdown = []
        all_raw = []
        all_boxes: List[Image.Image] = []
        all_cropped: List[Image.Image] = []

        for jdx, (output, img, orig_img) in enumerate(zip(outputs_list, images, original_images)):
            content = output.outputs[0].text
            if "<｜end▁of▁sentence｜>" in content:
                content = content.replace("<｜end▁of▁sentence｜>", "")

            page_label = f"\n\n--- 📄 第 {page_start + jdx} 页 ---\n\n"
            
            # 原始输出
            all_raw.append(f"{page_label}{content}")

            # 处理边界框和裁剪图
            if has_grounding:
                refs = extract_grounding_references(content)
                if refs:
                    result_image, cropped_imgs = draw_bounding_boxes_on_image(orig_img, refs, extract_images=True)
                    all_boxes.append(result_image)
                    all_cropped.extend(cropped_imgs)
                
                # 替换图片标记
                _, matches_images, _ = re_match_pdf(content)
                for idx, match in enumerate(matches_images):
                    content = content.replace(
                        match,
                        f"![](images/{jdx}_{idx}.jpg)\n",
                    )
            
            # 清理后的文本
            content_clean = clean_output_text(content, include_images=False, remove_labels=False)
            content_markdown = clean_output_text(content, include_images=True, remove_labels=True)
            
            all_text.append(f"{page_label}{content_clean}")
            all_markdown.append(f"{page_label}{content_markdown}")

        # 保存文件
        base_name = os.path.basename(pdf_path)
        text_content = "".join(all_text)
        raw_content = "".join(all_raw)
        
        mmd_det_path = os.path.join(out_dir, base_name.replace(".pdf", "_det.mmd"))
        mmd_path = os.path.join(out_dir, base_name.replace(".pdf", ".mmd"))
        pdf_out_path = os.path.join(out_dir, base_name.replace(".pdf", "_layouts.pdf"))

        with open(mmd_det_path, "w", encoding="utf-8") as f:
            f.write(raw_content)
        with open(mmd_path, "w", encoding="utf-8") as f:
            f.write(text_content)

        zip_path = os.path.join("outputs", "vllm_gradio_pdf", f"pdf_result_{ts}.zip")
        
        if export_layout_pdf and all_boxes:
            log_info(f"📊 正在生成布局PDF...")
            pil_to_pdf(all_boxes, pdf_out_path)
        
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
        log_info("=" * 60)
        
        text_output = "".join(all_text)
        markdown_output = "".join(all_markdown)
        raw_output = "".join(all_raw)
        status = f"✅ PDF处理完成 | 📄 {pdf_name} | 📑 {len(images)}页 | ⏱️ {total_time:.1f}秒 | ⚡ {avg_time:.2f}秒/页"
        
        return text_output, markdown_output, all_boxes, all_cropped, raw_output, status, zip_path

    except Exception as e:
        log_error(f"PDF处理失败: {str(e)}")
        error_msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return error_msg, "", [], [], "", f"❌ PDF处理失败: {str(e)}", None

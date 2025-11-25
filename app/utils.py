"""
工具模块 - 日志、文本处理、正则匹配等通用函数
"""
import os
import re
import io
import base64
from datetime import datetime
from typing import List, Tuple
from PIL import Image


# ============================================
# 日志函数
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


# ============================================
# 正则匹配函数
# ============================================
def extract_grounding_references(text: str) -> List[Tuple[str, str, str]]:
    """提取所有 grounding 标记"""
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    return re.findall(pattern, text, re.DOTALL)


def re_match(text: str) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    """匹配文本中的 ref/det 标记"""
    pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
    matches = re.findall(pattern, text, re.DOTALL)
    matches_other = [m[0] for m in matches]
    return matches, matches_other


def re_match_pdf(text: str) -> Tuple[List[Tuple[str, str, str]], List[str], List[str]]:
    """匹配 PDF 文本中的 ref/det 标记，区分图片和其他"""
    pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
    matches = re.findall(pattern, text, re.DOTALL)
    matches_image = []
    matches_other = []
    for m in matches:
        if '<|ref|>image<|/ref|>' in m[0]:
            matches_image.append(m[0])
        else:
            matches_other.append(m[0])
    return matches, matches_image, matches_other


# ============================================
# 文本清理函数
# ============================================
def clean_formula(text: str) -> str:
    """清理公式文本"""
    formula_pattern = r"\\\[(.*?)\\\]"

    def process_formula(match):
        formula = match.group(1)
        formula = re.sub(r"\\quad\s*\([^)]*\)", "", formula)
        formula = formula.strip()
        return r"\[" + formula + r"\]"

    return re.sub(formula_pattern, process_formula, text)


def clean_output_text(text: str, include_images: bool = False, remove_labels: bool = False) -> str:
    """
    清理输出文本，处理 grounding 标记
    - include_images: 是否用 [图片 X] 替换图片标记
    - remove_labels: 是否移除所有标签（只保留文本内容）
    """
    if not text:
        return ""
    
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)
    img_num = 0
    
    for match in matches:
        if '<|ref|>image<|/ref|>' in match[0]:
            if include_images:
                text = text.replace(match[0], f'\n\n**[图片 {img_num + 1}]**\n\n', 1)
                img_num += 1
            else:
                text = text.replace(match[0], '', 1)
        else:
            if remove_labels:
                text = text.replace(match[0], '', 1)
            else:
                text = text.replace(match[0], match[1], 1)
    
    # 移除结束标记
    text = text.replace("<｜end▁of▁sentence｜>", "")
    
    return text.strip()


def embed_images_in_markdown(markdown: str, crops: List[Image.Image]) -> str:
    """将裁剪的图片嵌入到 Markdown 中（Base64 编码）"""
    if not crops:
        return markdown
    
    for i, img in enumerate(crops):
        try:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            markdown = markdown.replace(
                f'**[图片 {i + 1}]**', 
                f'\n\n![图片 {i + 1}](data:image/png;base64,{b64})\n\n', 
                1
            )
        except Exception:
            pass
    return markdown


# ============================================
# 文件工具函数
# ============================================
def is_image_file(path: str) -> bool:
    """检查是否为支持的图片格式"""
    return os.path.splitext(path)[1].lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def list_images_in_dir(dir_path: str) -> list:
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
            if os.path.isfile(full_path) and is_image_file(full_path):
                images.append(full_path)
        return sorted(images)
    except Exception:
        return []

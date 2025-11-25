"""
配置模块 - 常量、模型配置、环境设置
"""
import os
import sys

# ============================================
# 路径配置
# ============================================
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
VLLM_DIR = os.path.join(ROOT_DIR, "DeepSeek-OCR-master", "DeepSeek-OCR-vllm")

# 添加 vLLM 模块目录到导入路径
if VLLM_DIR not in sys.path:
    sys.path.append(VLLM_DIR)

# 从原始 config 导入模型路径
from config import (
    MODEL_PATH,
    TOKENIZER_PATH,
    PROMPT,
    CROP_MODE,
    MAX_CONCURRENCY,
)

# ============================================
# 模型尺寸预设
# ============================================
SIZE_CONFIGS = {
    "极速（Tiny）": {"base_size": 512, "image_size": 512, "crop_mode": False},
    "快速（Small）": {"base_size": 640, "image_size": 640, "crop_mode": False},
    "标准（Base）": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
    "精细（Large）": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
    "高达模式（推荐）": {"base_size": 1024, "image_size": 640, "crop_mode": True},
}

# ============================================
# Prompt 模板
# ============================================
TASK_PROMPTS = {
    "Markdown转换": {
        "prompt": "<image>\n<|grounding|>Convert the document to markdown. ",
        "has_grounding": True,
    },
    "自由识别": {
        "prompt": "<image>\nFree OCR. ",
        "has_grounding": False,
    },
    "定位识别": {
        "prompt_template": "<image>\nLocate <|ref|>{text}<|/ref|> in the image. ",
        "has_grounding": True,
        "requires_input": True,
    },
    "图片OCR": {
        "prompt": "<image>\n<|grounding|>OCR this image. ",
        "has_grounding": True,
    },
    "图表解析": {
        "prompt": "<image>\nParse the figure. ",
        "has_grounding": False,
    },
    "图像描述": {
        "prompt": "<image>\nDescribe this image in detail. ",
        "has_grounding": False,
    },
    "自定义": {
        "prompt_template": "<image>\n{text}",
        "has_grounding": None,  # 根据用户输入判断
        "requires_input": True,
    },
}

# ============================================
# 环境设置函数
# ============================================
def ensure_offline_env():
    """设置离线环境变量"""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def setup_cuda_env():
    """设置 CUDA 环境"""
    import torch
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


def get_prompt(prompt_type: str, custom_text: str = "") -> tuple:
    """
    根据识别模式获取 prompt
    返回: (prompt, has_grounding)
    """
    config = TASK_PROMPTS.get(prompt_type)
    if not config:
        return "<image>\nFree OCR. ", False
    
    if config.get("requires_input"):
        if prompt_type == "定位识别":
            prompt = config["prompt_template"].format(text=custom_text.strip())
            return prompt, True
        elif prompt_type == "自定义":
            prompt = config["prompt_template"].format(text=custom_text)
            has_grounding = '<|grounding|>' in custom_text
            return prompt, has_grounding
    
    return config["prompt"], config["has_grounding"]

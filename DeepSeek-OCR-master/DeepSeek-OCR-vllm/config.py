# TODO: change modes
# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False
# Gundam: base_size = 1024, image_size = 640, crop_mode = True

BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True
MIN_CROPS= 2
MAX_CROPS= 6 # max:9; If your GPU memory is small, it is recommended to set it to 6.
MAX_CONCURRENCY = 4 # Recommended 1-4 to save VRAM.
NUM_WORKERS = 64 # image pre-process (resize/padding) workers 
PRINT_NUM_VIS_TOKENS = False
SKIP_REPEAT = True
# Prefer local cache to avoid repeated remote downloads.
# Ensure weights are downloaded under this directory before running.
MODEL_PATH = '/root/DeepSeek-OCR/models/DeepSeek-OCR'

# TODO: change INPUT_PATH
# .pdf: run_dpsk_ocr_pdf.py; 
# .jpg, .png, .jpeg: run_dpsk_ocr_image.py; 
# Omnidocbench images path: run_dpsk_ocr_eval_batch.py

INPUT_PATH = '/root/DeepSeek-OCR/assets/show1.jpg'
OUTPUT_PATH = '/root/DeepSeek-OCR/outputs/vllm_image_test'

PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'
# PROMPT = '<image>\nFree OCR.'
# TODO commonly used prompts
# document: <image>\n<|grounding|>Convert the document to markdown.
# other image: <image>\n<|grounding|>OCR this image.
# without layouts: <image>\nFree OCR.
# figures in document: <image>\nParse the figure.
# general: <image>\nDescribe this image in detail.
# rec: <image>\nLocate <|ref|>xxxx<|/ref|> in the image.
# '先天下之忧而忧'
# .......


import os
from transformers import AutoTokenizer


def _resolve_local_model_dir():
    """Resolve local HF snapshot directory for deepseek-ai/DeepSeek-OCR.

    Returns a path string if found, otherwise None.
    """
    snapshot_root = os.path.expanduser("~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR/snapshots")
    if os.path.isdir(snapshot_root):
        candidates = [os.path.join(snapshot_root, name)
                      for name in os.listdir(snapshot_root)
                      if os.path.isdir(os.path.join(snapshot_root, name))]
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return candidates[0]
    return None


def _ensure_offline_env():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


# Prefer local tokenizer path; fall back to HF repo if snapshot not found.
_LOCAL_SNAPSHOT = _resolve_local_model_dir()
TOKENIZER_PATH = _LOCAL_SNAPSHOT if _LOCAL_SNAPSHOT else 'deepseek-ai/DeepSeek-OCR'

if _LOCAL_SNAPSHOT:
    _ensure_offline_env()
TOKENIZER = AutoTokenizer.from_pretrained(
    TOKENIZER_PATH,
    trust_remote_code=True,
    local_files_only=bool(_LOCAL_SNAPSHOT),
)

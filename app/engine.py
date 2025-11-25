"""
推理引擎模块 - LLM 初始化和管理
"""
import time
import gc
import torch
from typing import Optional

from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry

from .config import MODEL_PATH, TOKENIZER_PATH, ensure_offline_env, setup_cuda_env
from .utils import log_info, log_success, log_warning

# 导入自定义模型
from deepseek_ocr import DeepseekOCRForCausalLM

# 全局 LLM 实例
_llm: Optional[LLM] = None
_current_engine_cfg = {
    "max_concurrency": None,
    "gpu_memory_utilization": None,
    "max_model_len": None,
}


def get_llm() -> Optional[LLM]:
    """获取当前 LLM 实例"""
    global _llm
    return _llm


def init_llm(
    max_concurrency: int,
    gpu_memory_utilization: float,
    max_model_len: int = 8192,
    force_reinit: bool = False,
) -> LLM:
    """
    初始化或复用 LLM 引擎
    """
    global _llm, _current_engine_cfg

    if (
        _llm is not None
        and not force_reinit
        and _current_engine_cfg.get("max_concurrency") == max_concurrency
        and _current_engine_cfg.get("gpu_memory_utilization") == gpu_memory_utilization
        and _current_engine_cfg.get("max_model_len") == max_model_len
    ):
        return _llm

    # 清理旧引擎
    if _llm is not None or force_reinit:
        log_info("🔄 正在清理旧引擎...")
        try:
            del _llm
        except Exception:
            pass
        _llm = None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        except Exception:
            pass
        gc.collect()
        time.sleep(0.3)

    ensure_offline_env()
    setup_cuda_env()

    # 注册自定义模型
    if "DeepseekOCRForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "DeepseekOCRForCausalLM", DeepseekOCRForCausalLM
        )

    log_info(f"🚀 正在初始化推理引擎...")
    log_info(f"   模型路径: {MODEL_PATH}")
    log_info(f"   并发数: {max_concurrency}")
    log_info(f"   显存利用率: {gpu_memory_utilization:.0%}")
    log_info(f"   最大序列长度: {max_model_len}")

    try:
        _llm = LLM(
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
    except AssertionError:
        # 重试一次
        log_warning("首次初始化失败，正在重试...")
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(0.5)
        _llm = LLM(
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

    _current_engine_cfg = {
        "max_concurrency": max_concurrency,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
    }
    
    log_success("✅ 推理引擎初始化完成")
    return _llm


def restart_engine(max_concurrency: int, gpu_memory_utilization: float) -> str:
    """强制重启引擎"""
    try:
        ensure_offline_env()
        setup_cuda_env()
        init_llm(
            max_concurrency=max_concurrency,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=8192,
            force_reinit=True,
        )
        return "✅ 引擎重启成功"
    except Exception as e:
        return f"❌ 引擎重启失败: {e}"


def warmup_engine(
    max_concurrency: int = 12,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 8192,
) -> bool:
    """
    预热模型引擎 - 启动时加载模型到显存，减少用户首次请求等待时间
    返回: True 表示预热成功，False 表示跳过或失败
    """
    import os
    from PIL import Image
    import numpy as np
    
    # 检查是否启用预热
    warmup_flag = os.environ.get("WARMUP_ON_START", "1")
    if warmup_flag not in ("1", "true", "True", "yes", "YES"):
        log_info("⏭️  预热已禁用 (WARMUP_ON_START != 1)")
        return False
    
    log_info("=" * 60)
    log_info("🔥 开始模型预热...")
    log_info("=" * 60)
    
    warmup_start = time.time()
    
    try:
        # 1. 初始化引擎（加载模型到显存）
        log_info("📦 正在加载模型到显存...")
        llm = init_llm(
            max_concurrency=max_concurrency,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        
        # 2. 创建测试图片（小尺寸白色图片）
        log_info("🖼️  正在准备预热图片...")
        test_image = Image.new("RGB", (256, 256), color=(255, 255, 255))
        
        # 3. 导入处理器
        import sys
        ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
        VLLM_DIR = os.path.join(ROOT_DIR, "DeepSeek-OCR-master", "DeepSeek-OCR-vllm")
        if VLLM_DIR not in sys.path:
            sys.path.append(VLLM_DIR)
        
        from process.image_process import DeepseekOCRProcessor
        from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
        
        # 4. 预处理测试图片
        log_info("⚙️  正在预处理测试图片...")
        proc = DeepseekOCRProcessor(image_size=640, base_size=1024)
        image_features = proc.tokenize_with_images(
            images=[test_image], bos=True, eos=True, cropping=False
        )
        
        # 5. 执行一次推理（真正的预热）
        log_info("🚀 正在执行预热推理...")
        
        logits_processors = [
            NoRepeatNGramLogitsProcessor(
                ngram_size=40, window_size=90, whitelist_token_ids={128821, 128822}
            )
        ]
        
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=64,  # 预热时只生成少量 token
            logits_processors=logits_processors,
            skip_special_tokens=False,
        )
        
        warmup_prompt = "<|im_start|>User:<image>\nOCR this image<|im_end|>\n<|im_start|>Assistant:<｜det▁of▁image｜>"
        
        cache_item = {
            "prompt": warmup_prompt,
            "multi_modal_data": {"image": image_features},
        }
        
        # 执行预热推理
        _ = llm.generate([cache_item], sampling_params=sampling_params)
        
        warmup_time = time.time() - warmup_start
        
        # 显示显存使用情况
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            log_info(f"💾 显存使用: 已分配 {allocated:.2f} GB / 已预留 {reserved:.2f} GB")
        
        log_info("=" * 60)
        log_success(f"🔥 模型预热完成！耗时 {warmup_time:.2f} 秒")
        log_info("   ✅ 模型已加载到显存，用户首次请求将无需等待加载")
        log_info("=" * 60)
        
        return True
        
    except Exception as e:
        log_warning(f"⚠️  模型预热失败: {e}")
        log_warning("   服务将继续启动，但首次请求可能需要等待模型加载")
        return False

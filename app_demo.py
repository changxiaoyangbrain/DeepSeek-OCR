#!/usr/bin/env python3
"""
DeepSeek-OCR Web Demo - 模块化重构版
长小养照护智能资源数字化平台

主入口文件 - 组装 Gradio 界面并启动服务
"""
import os
import sys
import time
import json

import gradio as gr
import torch

# 添加模块路径
ROOT_DIR = os.path.dirname(__file__)
sys.path.insert(0, ROOT_DIR)

# 导入应用模块
from app.config import (
    ROOT_DIR, SIZE_CONFIGS, CROP_MODE, MAX_CONCURRENCY,
    ensure_offline_env, setup_cuda_env
)
from app.utils import log_info, log_success
from app.engine import init_llm, restart_engine, warmup_engine
from app.processors import (
    process_single_image,
    process_batch_images,
    process_pdf_document,
    get_pdf_page_count,
)
from app.ui_components import (
    CUSTOM_CSS,
    HEADER_HTML, TIPS_HTML, FOOTER_HTML,
    BATCH_INFO_HTML, PDF_INFO_HTML,
    PROMPT_CHOICES, MODEL_SIZE_CHOICES,
)


def create_demo():
    """创建 Gradio Demo 界面"""
    
    with gr.Blocks(title="长小养照护智能资源数字化平台") as demo:
        
        # ============================================
        # 页面头部
        # ============================================
        gr.HTML(HEADER_HTML)
        
        # ============================================
        # 高级设置区域（折叠）
        # ============================================
        with gr.Accordion("⚙️ 高级设置", open=False, elem_classes=["dark-panel"]):
            with gr.Row(elem_classes=["dark-panel"]):
                # 识别模式（高级用户）
                with gr.Column(scale=1, elem_classes=["dark-panel"]):
                    gr.HTML("<p style='color:#ffffff !important;font-weight:bold;margin-bottom:10px;'>📝 识别模式（默认Markdown，适合大多数场景）</p>")
                    prompt_type = gr.Radio(
                        choices=PROMPT_CHOICES,
                        value="Markdown转换",
                        label="选择模式",
                        info="Markdown转换：保留格式 | 自由识别：纯文本 | 定位识别：查找文字位置",
                        elem_classes=["dark-panel"]
                    )
                    custom_prompt = gr.Textbox(
                        label="输入内容",
                        placeholder="定位模式：输入要查找的文字\n自定义模式：输入完整指令",
                        lines=2,
                        visible=False,
                        elem_classes=["dark-panel"]
                    )
                
                # 图片处理选项
                with gr.Column(scale=1, elem_classes=["dark-panel"]):
                    gr.HTML("<p style='color:#ffffff !important;font-weight:bold;margin-bottom:10px;'>🖼️ 图片处理</p>")
                    crop_mode = gr.Checkbox(
                        label="📐 启用智能裁剪",
                        value=bool(CROP_MODE),
                        info="适用于大尺寸文档图片",
                        elem_classes=["dark-panel"]
                    )
                    model_size = gr.Radio(
                        choices=MODEL_SIZE_CHOICES,
                        value="高达模式（推荐）",
                        label="🎯 模型精度",
                        info="高达模式平衡速度与精度",
                        elem_classes=["dark-panel"]
                    )
                
                # 引擎参数
                with gr.Column(scale=1, elem_classes=["dark-panel"]):
                    gr.HTML("<p style='color:#ffffff !important;font-weight:bold;margin-bottom:10px;'>⚡ 引擎参数</p>")
                    try:
                        _default_concurrency = int(MAX_CONCURRENCY) if MAX_CONCURRENCY else 12
                    except Exception:
                        _default_concurrency = 12
                    _concurrency_max = max(16, _default_concurrency)
                    
                    max_concurrency = gr.Slider(
                        minimum=1,
                        maximum=_concurrency_max,
                        step=1,
                        value=min(_default_concurrency, _concurrency_max),
                        label="并发数量",
                        info="建议根据显存大小调整",
                        elem_classes=["dark-panel"]
                    )
                    gpu_memory_utilization = gr.Slider(
                        minimum=0.5,
                        maximum=0.98,
                        step=0.01,
                        value=0.85,
                        label="显存利用率",
                        info="建议保持在0.85左右",
                        elem_classes=["dark-panel"]
                    )
                    max_tokens = gr.Slider(
                        minimum=256,
                        maximum=16384,
                        step=512,
                        value=16384,
                        label="最大生成长度",
                        info="文档较长时请增大此值",
                        elem_classes=["dark-panel"]
                    )
                    with gr.Row():
                        estimate_btn = gr.Button("📊 估算并发", size="sm", elem_classes=["dark-panel-btn"])
                        restart_btn = gr.Button("🔄 重启引擎", variant="secondary", size="sm", elem_classes=["dark-panel-btn"])
                    engine_status = gr.Textbox(
                        label="引擎状态",
                        value="✅ 已就绪",
                        lines=1,
                        interactive=False,
                        elem_classes=["dark-panel"]
                    )
        
        # ============================================
        # 功能选项卡
        # ============================================
        with gr.Tabs():
            
            # --------------------------------------------
            # 单图识别 Tab
            # --------------------------------------------
            with gr.Tab("📷 单图识别"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="📤 上传图片（支持拖拽/粘贴）",
                            type="pil",
                            sources=["upload", "clipboard"],
                            height=350,
                        )
                        process_btn_single = gr.Button("🚀 开始识别", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.Tab("📄 文本"):
                                output_text = gr.Textbox(
                                    label="纯文本输出",
                                    lines=18,
                                    max_lines=30,
                                    info="清理后的文本，适合复制使用"
                                )
                            with gr.Tab("🎨 渲染"):
                                output_markdown = gr.Textbox(
                                    label="Markdown格式输出",
                                    lines=18,
                                    max_lines=30,
                                    info="保留版面结构的Markdown格式"
                                )
                            with gr.Tab("🖼️ 标注"):
                                output_annotated = gr.Image(
                                    label="边界框标注图",
                                    type="pil",
                                    height=450,
                                )
                            with gr.Tab("✂️ 裁剪"):
                                output_gallery = gr.Gallery(
                                    label="裁剪出的图片区域",
                                    columns=3,
                                    height=400,
                                )
                            with gr.Tab("🔍 原始"):
                                output_raw = gr.Textbox(
                                    label="模型原始输出（含坐标标记）",
                                    lines=18,
                                    max_lines=30,
                                )
            
            # --------------------------------------------
            # 批量处理 Tab
            # --------------------------------------------
            with gr.Tab("📚 批量处理"):
                gr.HTML(BATCH_INFO_HTML)
                with gr.Row():
                    # 左侧：上传和控制
                    with gr.Column(scale=1):
                        batch_files = gr.File(
                            label="📤 上传图片（可多选，支持拖拽）",
                            file_count="multiple",
                            file_types=["image"],
                            type="filepath",
                        )
                        batch_file_info = gr.Textbox(
                            label="📊 文件信息",
                            value="请上传图片文件",
                            lines=2,
                            interactive=False,
                        )
                        process_btn_batch = gr.Button("🚀 开始批量识别", variant="primary", size="lg")
                    
                    # 右侧：5个Tab输出（与官方Demo一致）
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.Tab("📄 文本"):
                                batch_text = gr.Textbox(
                                    label="合并文本输出",
                                    lines=18,
                                    max_lines=30,
                                    info="所有图片识别结果的合并文本"
                                )
                            with gr.Tab("🎨 渲染"):
                                batch_markdown = gr.Textbox(
                                    label="Markdown格式输出",
                                    lines=18,
                                    max_lines=30,
                                    info="保留版面结构的Markdown格式"
                                )
                            with gr.Tab("🖼️ 标注"):
                                batch_boxes = gr.Gallery(
                                    label="边界框标注图集",
                                    columns=2,
                                    height=400,
                                )
                            with gr.Tab("✂️ 裁剪"):
                                batch_cropped = gr.Gallery(
                                    label="裁剪出的图片区域",
                                    columns=3,
                                    height=400,
                                )
                            with gr.Tab("🔍 原始"):
                                batch_raw = gr.Textbox(
                                    label="模型原始输出（含坐标标记）",
                                    lines=18,
                                    max_lines=30,
                                )
                        
                        # 下载和状态
                        with gr.Row():
                            batch_status = gr.Textbox(
                                label="📊 处理状态",
                                lines=2,
                                max_lines=3,
                                interactive=False,
                                scale=3,
                                elem_classes=["status-box"],
                            )
                            batch_download = gr.File(
                                label="📥 下载结果",
                                interactive=False,
                                scale=1,
                            )
            
            # --------------------------------------------
            # PDF 解析 Tab
            # --------------------------------------------
            with gr.Tab("📑 PDF解析"):
                gr.HTML(PDF_INFO_HTML)
                with gr.Row():
                    # 左侧：上传和控制
                    with gr.Column(scale=1):
                        pdf_file = gr.File(
                            label="📤 上传PDF文件",
                            file_types=[".pdf"],
                            type="filepath",
                        )
                        with gr.Row():
                            pdf_page_start = gr.Number(
                                label="起始页",
                                value=1,
                                minimum=1,
                                step=1,
                                scale=2,
                            )
                            pdf_page_end = gr.Number(
                                label="结束页（-1=全部）",
                                value=-1,
                                minimum=-1,
                                step=1,
                                scale=2,
                            )
                            pdf_page_info = gr.Number(
                                label="总页数",
                                value=0,
                                interactive=False,
                                scale=1,
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
                        )
                        process_btn_pdf = gr.Button("🚀 开始解析PDF", variant="primary", size="lg")
                    
                    # 右侧：5个Tab输出（与官方Demo一致）
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.Tab("📄 文本"):
                                pdf_text = gr.Textbox(
                                    label="纯文本输出",
                                    lines=18,
                                    max_lines=30,
                                    info="清理后的文本，适合复制使用"
                                )
                            with gr.Tab("🎨 渲染"):
                                pdf_markdown = gr.Textbox(
                                    label="Markdown格式输出",
                                    lines=18,
                                    max_lines=30,
                                    info="保留版面结构的Markdown格式"
                                )
                            with gr.Tab("🖼️ 标注"):
                                pdf_boxes = gr.Gallery(
                                    label="边界框标注图（每页）",
                                    columns=2,
                                    height=400,
                                )
                            with gr.Tab("✂️ 裁剪"):
                                pdf_cropped = gr.Gallery(
                                    label="裁剪出的图片区域",
                                    columns=3,
                                    height=400,
                                )
                            with gr.Tab("🔍 原始"):
                                pdf_raw = gr.Textbox(
                                    label="模型原始输出（含坐标标记）",
                                    lines=18,
                                    max_lines=30,
                                )
                        
                        # 下载和状态
                        with gr.Row():
                            pdf_status = gr.Textbox(
                                label="📊 处理状态",
                                lines=2,
                                max_lines=3,
                                interactive=False,
                                scale=3,
                                elem_classes=["status-box"],
                            )
                            pdf_download = gr.File(
                                label="📥 下载结果",
                                interactive=False,
                                scale=1,
                            )
        
        # ============================================
        # 页脚
        # ============================================
        gr.HTML(FOOTER_HTML)
        
        # ============================================
        # 事件绑定
        # ============================================
        
        # 更新 prompt 输入框可见性
        def update_prompt_visibility(choice):
            if choice == "定位识别":
                return gr.update(
                    visible=True, 
                    label="🔍 要查找的文字",
                    placeholder="输入要在图片中定位的文字"
                )
            elif choice == "自定义":
                return gr.update(
                    visible=True, 
                    label="✏️ 自定义指令",
                    placeholder="输入完整指令，可添加 <|grounding|>"
                )
            return gr.update(visible=False)
        
        prompt_type.change(
            fn=update_prompt_visibility,
            inputs=[prompt_type],
            outputs=[custom_prompt],
        )
        
        # 模型尺寸预设更新裁剪模式
        def apply_size_preset(choice):
            preset = SIZE_CONFIGS.get(choice, SIZE_CONFIGS["高达模式（推荐）"])
            return gr.update(value=preset["crop_mode"])
        
        model_size.change(
            fn=apply_size_preset,
            inputs=[model_size],
            outputs=[crop_mode],
        )
        
        # 估算并发数
        def estimate_concurrency(gmu: float, max_toks: int):
            try:
                if torch.cuda.is_available():
                    free_b, total_b = torch.cuda.mem_get_info()
                    total_gb = total_b / (1024 ** 3)
                    free_gb = free_b / (1024 ** 3)
                    effective_gb = max(min(total_gb * gmu, free_gb) - 1.0, 1.0)
                else:
                    effective_gb = 8.0
            except Exception:
                effective_gb = 8.0

            per_seq_mb = 800.0 * max(1.0, float(max_toks) / 8192.0)
            est = int(max(1, (effective_gb * 1024.0) / per_seq_mb))
            return gr.update(value=est)
        
        estimate_btn.click(
            fn=estimate_concurrency,
            inputs=[gpu_memory_utilization, max_tokens],
            outputs=[max_concurrency],
        )
        
        # 重启引擎
        restart_btn.click(
            fn=restart_engine,
            inputs=[max_concurrency, gpu_memory_utilization],
            outputs=[engine_status],
        )
        
        # PDF 文件上传时更新页数信息
        def update_pdf_page_info(pdf_path):
            if not pdf_path:
                return 0
            count = get_pdf_page_count(pdf_path)
            return count
        
        pdf_file.change(
            fn=update_pdf_page_info,
            inputs=[pdf_file],
            outputs=[pdf_page_info],
        )
        
        # 单图识别
        process_btn_single.click(
            fn=process_single_image,
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
            outputs=[
                output_text,
                output_markdown,
                output_raw,
                output_annotated,
                output_gallery,
            ],
        )
        
        # 批量处理
        process_btn_batch.click(
            fn=process_batch_images,
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
            outputs=[batch_text, batch_markdown, batch_boxes, batch_cropped, batch_raw, batch_status, batch_download],
        )
        
        # PDF 处理
        process_btn_pdf.click(
            fn=process_pdf_document,
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
                pdf_page_start,
                pdf_page_end,
            ],
            outputs=[pdf_text, pdf_markdown, pdf_boxes, pdf_cropped, pdf_raw, pdf_status, pdf_download],
        )
    
    return demo


def main():
    """主函数"""
    ensure_offline_env()
    setup_cuda_env()
    
    log_info("=" * 60)
    log_info("🚀 启动 DeepSeek-OCR Web Demo")
    log_info("=" * 60)
    
    # 获取默认并发数（与 UI 默认值保持一致）
    try:
        default_concurrency = int(MAX_CONCURRENCY) if MAX_CONCURRENCY else 4
    except Exception:
        default_concurrency = 4
    
    # 模型预热 - 启动时加载模型到显存
    # 设置 WARMUP_ON_START=0 可禁用预热
    # 注意：预热参数必须与 UI 默认值一致，否则首次请求会重新初始化引擎
    warmup_engine(
        max_concurrency=default_concurrency,
        gpu_memory_utilization=0.85,
        max_model_len=8192,
    )
    
    demo = create_demo()
    
    port = int(os.environ.get("DEMO_PORT", 7860))
    log_info(f"🌐 服务端口: {port}")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()

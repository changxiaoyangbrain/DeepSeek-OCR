#!/usr/bin/env python3
"""
DeepSeek-OCR UI 组件模块 - 样式与界面组件

长小养照护智能资源数字化平台
"""

# ============================================
# 常量定义 - 选项列表
# ============================================

PROMPT_CHOICES = [
    "Markdown转换",
    "自由识别",
    "定位识别",
    "图片OCR",
    "图表解析",
    "图像描述",
    "自定义",
]

MODEL_SIZE_CHOICES = [
    "极速（Tiny）",
    "快速（Small）",
    "标准（Base）",
    "精细（Large）",
    "高达模式（推荐）",
]

# ============================================
# HTML 模板 - 页面头部
# ============================================

HEADER_HTML = """
<div class="header-banner">
    <h1>🏥 长小养照护智能资源数字化平台</h1>
    <p>智能文档识别 · 高效数字化转换 · 专业照护知识管理</p>
    <div class="subtitle">📄 上传图片或PDF，自动识别并转换为Markdown格式</div>
</div>
"""

TIPS_HTML = """
<div class="tips-box">
    <p class="tips-title">💡 <strong>使用提示</strong></p>
    <p class="tips-content">• <span class="tips-label">Markdown转换</span>：文档/论文识别，保留版面结构、表格、公式（推荐）</p>
    <p class="tips-content">• <span class="tips-label">自由识别</span>：纯文字提取，不含布局信息</p>
    <p class="tips-content">• <span class="tips-label">定位识别</span>：在图片中查找并标注特定文字的位置</p>
    <p class="tips-content">• <span class="tips-label">图片OCR</span>：通用图片中的文字识别</p>
    <p class="tips-content">• <span class="tips-label">图表解析</span>：专门解析图表、流程图等</p>
    <p class="tips-content">• <span class="tips-label">图像描述</span>：获取图片的详细描述</p>
</div>
"""

FOOTER_HTML = """
<div class="footer">
    <p style="color:#ffffff !important;">© 2025 海南长小养智能科技 版权所有</p>
    <p style="margin-top:5px;color:rgba(255,255,255,0.9) !important;">
        基于 <a href="https://github.com/deepseek-ai/DeepSeek-VL2" target="_blank" style="color:#93c5fd !important;">DeepSeek-OCR</a> 构建
    </p>
</div>
"""

# ============================================
# HTML 模板 - 功能区提示
# ============================================

BATCH_INFO_HTML = """
<div style="background:linear-gradient(135deg,#e8f4fd,#d4e9f7);padding:15px 20px;border-radius:10px;margin-bottom:15px;border-left:4px solid #1e3c72;">
    <p style="margin:0;"><span style="color:#1e3c72 !important;font-weight:bold;font-size:1.1em;">📂 批量识别模式</span> <span style="color:#333;">- 支持同时上传多张图片进行批处理</span></p>
    <p style="margin:5px 0 0 0;color:#555;font-size:0.9em;">支持格式: JPG, PNG, WebP, BMP, TIFF</p>
</div>
"""

PDF_INFO_HTML = """
<div style="background:linear-gradient(135deg,#fff3e0,#ffe0b2);padding:15px 20px;border-radius:10px;margin-bottom:15px;border-left:4px solid #ff9800;">
    <p style="margin:0;"><span style="color:#e65100 !important;font-weight:bold;font-size:1.1em;">📑 PDF智能解析</span></p>
    <p style="margin:5px 0 0 0;color:#555;font-size:0.9em;">自动提取PDF内容并转换为Markdown格式，支持选择页面范围</p>
</div>
"""

# ============================================
# CSS 样式定义
# ============================================

CUSTOM_CSS = """
/* 全局样式 */
.gradio-container {
    font-family: 'Microsoft YaHei', 'PingFang SC', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    min-height: 100vh;
}

/* 全局文字颜色 - 确保所有文字在浅色背景上可见 */
.gradio-container .main,
.gradio-container .main * {
    color: #1f2937;
}

/* 确保标签和说明文字可见 */
.gradio-container label,
.gradio-container .label-wrap,
.gradio-container .info {
    color: #374151 !important;
}

/* ================================================
   深色面板内白色文字 - 使用自定义类
   ================================================ */
.dark-panel,
.dark-panel > div,
.dark-panel > .form,
.dark-panel .row,
.dark-panel .column,
.dark-panel .block,
.dark-panel .wrap,
.dark-panel .container {
    background: transparent !important;
    background-color: transparent !important;
}

.dark-panel,
.dark-panel *,
.dark-panel label,
.dark-panel span,
.dark-panel p,
.dark-panel div {
    color: #ffffff !important;
}

.dark-panel .block > label,
.dark-panel .label-wrap,
.dark-panel .label-wrap span,
.dark-panel .info {
    color: rgba(255,255,255,0.85) !important;
}

/* 深色面板内输入框 */
.dark-panel input[type="text"],
.dark-panel input[type="number"],
.dark-panel textarea {
    color: #1f2937 !important;
    background: #ffffff !important;
}

/* 深色面板内按钮 */
.dark-panel button,
.dark-panel-btn,
.dark-panel-btn button {
    color: #1f2937 !important;
    background: #f3f4f6 !important;
    border: 1px solid #d1d5db !important;
}

.dark-panel-btn:hover,
.dark-panel-btn button:hover {
    background: #e5e7eb !important;
}

/* ================================================
   Accordion 折叠面板样式 - 强制移除所有白色背景
   ================================================ */
.gradio-container .accordion,
.gradio-container .accordion *,
.gradio-container .accordion > div,
.gradio-container .accordion > div > div,
.gradio-container .accordion .form,
.gradio-container .accordion .block,
.gradio-container .accordion .wrap,
.gradio-container .accordion .gap,
.gradio-container [class*="accordion"],
.gradio-container [class*="accordion"] > *,
.dark-panel,
.dark-panel *,
.dark-panel > div,
.dark-panel .svelte-1ed2p3z,
.dark-panel .padding {
    background: transparent !important;
    background-color: transparent !important;
}

/* Accordion 容器本身深色背景 */
.gradio-container .accordion {
    background: #1f2937 !important;
    border-radius: 12px !important;
}

/* Accordion 内的 label 和 info 文字 */
.accordion .label-wrap,
.accordion .label-wrap span,
.accordion .label-wrap label,
.accordion .block > label,
.accordion .block > .label-wrap,
.accordion .info,
.accordion .block .info,
.accordion small,
.accordion .caption {
    color: #ffffff !important;
    opacity: 1 !important;
}

/* 处理状态输入框 - 深色样式 */
.status-box,
.status-box textarea,
.status-box input {
    background: linear-gradient(145deg, #0f172a 0%, #111827 100%) !important;
    color: #10b981 !important;
    border: 1px solid #1f2a3d !important;
    border-radius: 12px !important;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35) !important;
}

.status-box textarea {
    padding: 12px 14px !important;
    font-size: 15px !important;
    line-height: 1.55 !important;
}

.status-box .wrap {
    background: transparent !important;
}

/* Accordion 内的 Radio/Checkbox 选项文字 */
.accordion .group span,
.accordion input + span,
.accordion input + label,
.accordion .choice span,
.accordion [role="radiogroup"] span,
.accordion [role="group"] span,
.accordion .svelte-1p9xokt,
.accordion .options span {
    color: #ffffff !important;
}

/* Accordion 内输入框 - 白色背景深色文字 */
.accordion input[type="text"],
.accordion input[type="number"],
.accordion textarea,
.accordion .input-container input {
    color: #1f2937 !important;
    background: #ffffff !important;
}

/* Accordion 内按钮 - 确保可见 */
.accordion button,
.accordion button span,
.accordion .btn,
.accordion [role="button"] {
    color: #1f2937 !important;
    background: #f3f4f6 !important;
    border: 1px solid #d1d5db !important;
}

.accordion button:hover {
    background: #e5e7eb !important;
}

/* Slider 相关 */
.accordion .range-slider *,
.accordion .slider *,
.accordion input[type="range"] ~ *,
.accordion .number-input span {
    color: #ffffff !important;
}

/* ================================================
   Accordion 外部 - 深色文字
   ================================================ */
/* Markdown 文字 - 在 Accordion 外部 */
.gradio-container .prose,
.gradio-container .prose * {
    color: #1f2937 !important;
}

/* Radio 和 Checkbox 文字 - 在 Accordion 外部 */
.gradio-container .wrap span,
.gradio-container input[type="radio"] + span,
.gradio-container input[type="checkbox"] + span {
    color: #374151 !important;
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
    top: 0; left: 0; right: 0; bottom: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Ccircle cx='30' cy='30' r='4'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    pointer-events: none;
}

.header-banner h1 {
    color: #ffffff !important;
    font-size: 2.5em !important;
    font-weight: 700 !important;
    margin: 0 0 15px 0 !important;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    position: relative;
    z-index: 1;
}

.header-banner p {
    color: rgba(255, 255, 255, 0.9) !important;
    font-size: 1.2em !important;
    margin: 0 !important;
    position: relative;
    z-index: 1;
}

.header-banner .subtitle {
    color: rgba(255, 255, 255, 0.8) !important;
    font-size: 0.95em !important;
    margin-top: 10px !important;
    padding: 8px 20px;
    background: rgba(255, 255, 255, 0.15);
    border-radius: 20px;
    display: inline-block;
}

/* 提示信息框 */
.tips-box {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    padding: 20px 25px;
    border-radius: 12px;
    margin-bottom: 20px;
    border-left: 4px solid #0ea5e9;
}

.tips-box .tips-title {
    font-size: 1.1em;
    color: #0369a1 !important;
    margin-bottom: 10px;
}

.tips-box .tips-content {
    color: #334155 !important;
    font-size: 0.95em;
    margin: 6px 0;
    line-height: 1.6;
}

.tips-box .tips-label {
    color: #0369a1 !important;
    font-weight: bold;
}

/* 按钮样式 */
.primary {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(30, 60, 114, 0.3) !important;
    transition: all 0.3s ease !important;
}

.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(30, 60, 114, 0.4) !important;
}

.secondary {
    background: linear-gradient(135deg, #f8fafc, #e2e8f0) !important;
    border: 1px solid #cbd5e1 !important;
}

/* Tab 样式 */
.tabs {
    border-radius: 12px !important;
    overflow: hidden !important;
}

.tab-nav {
    background: #f1f5f9 !important;
    padding: 8px !important;
    border-radius: 10px !important;
    margin-bottom: 15px !important;
}

.tab-nav button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, #1e3c72, #2a5298) !important;
    color: white !important;
}

/* 输入区域 */
.input-box {
    border: 2px solid #e2e8f0 !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}

.input-box:focus-within {
    border-color: #1e3c72 !important;
    box-shadow: 0 0 0 3px rgba(30, 60, 114, 0.1) !important;
}

/* 图片上传区域 */
.image-upload {
    border: 2px dashed #cbd5e1 !important;
    border-radius: 12px !important;
    background: #fafbfc !important;
    transition: all 0.3s ease !important;
}

.image-upload:hover {
    border-color: #1e3c72 !important;
    background: #f0f4ff !important;
}

/* 进度条 */
.progress-bar {
    background: linear-gradient(135deg, #1e3c72, #2a5298) !important;
    border-radius: 10px !important;
}

/* Accordion 样式 */
.accordion {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

.accordion-header {
    background: #f8fafc !important;
    font-weight: 500 !important;
}

/* 页脚样式 */
.footer {
    text-align: center;
    padding: 25px 20px;
    margin-top: 30px;
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    border-radius: 12px;
    color: white;
}

.footer p {
    margin: 0;
    font-size: 0.9em;
    color: rgba(255, 255, 255, 0.9);
}

.footer a {
    color: #93c5fd !important;
    text-decoration: none;
    transition: color 0.2s ease;
}

.footer a:hover {
    color: #bfdbfe !important;
    text-decoration: underline;
}

/* 代码块和输出框 */
textarea, .output-text {
    font-family: 'Consolas', 'Monaco', monospace !important;
}

/* Markdown 渲染预览容器 */
.markdown-preview-container {
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 16px !important;
    background: #fafbfc !important;
    min-height: 400px !important;
    max-height: 500px !important;
    overflow-y: auto !important;
}

.markdown-preview {
    color: #1f2937 !important;
    font-size: 0.95em !important;
    line-height: 1.7 !important;
}

.markdown-preview h1, .markdown-preview h2, .markdown-preview h3,
.markdown-preview h4, .markdown-preview h5, .markdown-preview h6 {
    color: #111827 !important;
    margin-top: 1em !important;
    margin-bottom: 0.5em !important;
}

.markdown-preview p {
    color: #374151 !important;
    margin: 0.5em 0 !important;
}

.markdown-preview code {
    background: #e5e7eb !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-family: 'Consolas', 'Monaco', monospace !important;
    color: #1f2937 !important;
}

.markdown-preview pre {
    background: #1f2937 !important;
    padding: 12px !important;
    border-radius: 8px !important;
    overflow-x: auto !important;
}

.markdown-preview pre code {
    background: transparent !important;
    color: #f3f4f6 !important;
}

.markdown-preview table {
    border-collapse: collapse !important;
    width: 100% !important;
    margin: 1em 0 !important;
}

.markdown-preview th, .markdown-preview td {
    border: 1px solid #d1d5db !important;
    padding: 8px 12px !important;
    text-align: left !important;
    color: #374151 !important;
}

.markdown-preview th {
    background: #f3f4f6 !important;
    font-weight: 600 !important;
    color: #111827 !important;
}

.markdown-preview blockquote {
    border-left: 4px solid #1e3c72 !important;
    padding-left: 16px !important;
    margin: 1em 0 !important;
    color: #4b5563 !important;
    background: #f9fafb !important;
}

.markdown-preview ul, .markdown-preview ol {
    padding-left: 1.5em !important;
    color: #374151 !important;
}

.markdown-preview li {
    margin: 0.25em 0 !important;
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

/* 隐藏 Gradio 默认页脚 */
footer {
    display: none !important;
}

.gradio-container footer,
.gradio-container .footer-links,
footer.svelte-1rjryqp,
.built-with {
    display: none !important;
    visibility: hidden !important;
}

/* 强制文本框显示滚动条 */
.gradio-container textarea {
    overflow-y: auto !important;
    max-height: 500px !important;
    resize: vertical !important;
}

/* 输出区域文本框固定高度并启用滚动 */
.gradio-container .output-class textarea,
.gradio-container .tabs textarea {
    min-height: 400px !important;
    max-height: 500px !important;
    overflow-y: scroll !important;
}

/* 处理状态框 - 压缩高度，避免占用过多空间 */
.gradio-container .status-box textarea {
    min-height: 110px !important;
    max-height: 200px !important;
    height: auto !important;
    overflow-y: auto !important;
}
"""

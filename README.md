# 🎤 智能会议纪要生成器

基于AI的会议转录、纠错和纪要生成工具，集成Whisper、OpenAI GPT和Ollama模型。

## ✨ 主要特性

### 🆕 升级功能

- **🎯 Whisper Medium模型**：更高精度的语音转录，支持中英文双语
- **🤖 AI智能纠错**：集成OpenAI GPT和本地Ollama模型
- **📝 智能文本纠正**：结合参考文档对转录结果进行纠错
- **📊 详细会议纪要**：包含会议重点、决议、进度、计划等
- **🔄 实时进度显示**：前端和后端都有详细的处理步骤显示
- **📄 多格式文档支持**：PDF、DOCX、Markdown、TXT
- **💾 多文件输出**：纠正后的转录文本和完整会议报告

### 🌟 核心功能

1. **智能语音转录**
   - 使用Whisper Medium模型
   - 支持中英双语自动识别
   - 高精度转录效果

2. **AI文本纠错**
   - 结合参考文档纠正转录错误
   - 改进语法和专业术语
   - 保持原意的同时提升可读性

3. **结构化会议纪要**
   - 会议概要和重点
   - 工作决议和决定事项
   - 当前进度和后续计划
   - 具体的待办事项

4. **实时进度追踪**
   - WebSocket实时通信
   - 详细的处理步骤显示
   - 错误处理和重试机制

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 系统依赖

安装FFmpeg（必需）：
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Windows
# 下载并安装 FFmpeg: https://ffmpeg.org/download.html
```

### 3. 配置环境变量

创建 `.env` 文件：
```env
# OpenAI API 配置 (可选)
OPENAI_API_KEY=your_openai_api_key_here

# Ollama 配置 (可选)
OLLAMA_BASE_URL=http://localhost:11434
```

### 4. 启动应用

```bash
python app.py
```

访问 `http://localhost:5000`

## 📖 使用说明

### 处理流程

1. **上传会议视频**（支持MP4、MOV、AVI）
2. **添加参考文档**（PDF、DOCX、MD、TXT，可选）
3. **选择AI服务**（OpenAI GPT 或 本地Ollama）
4. **实时查看处理进度**
5. **下载生成的文件**：
   - 纠正后的转录文本
   - 完整的会议报告

### 处理步骤

- 🎬 视频上传
- 🎵 音频提取  
- 🎤 语音转录
- 📄 文档处理
- ✏️ 转录纠错
- 📋 生成纪要
- 📊 创建报告

## 🛠️ 技术栈

- **前端**：HTML5, CSS3, JavaScript, Socket.IO
- **后端**：Flask, Flask-SocketIO
- **AI模型**：Whisper, OpenAI GPT, Ollama
- **文档处理**：PyMuPDF, python-docx
- **音视频处理**：FFmpeg

## 📁 项目结构

```
meeting_summary/
├── app.py              # 主应用程序
├── requirements.txt    # 依赖包列表
├── CONFIG.md          # 详细配置说明
├── templates/
│   └── index.html     # 前端页面
├── static/
│   └── style.css      # 样式文件
└── meeting/           # 生成的会议文件
```

## 🔧 配置选项

### AI服务提供商

1. **OpenAI GPT**（推荐）
   - 更好的纠错和纪要生成效果
   - 需要API密钥
   - 按使用量计费

2. **本地Ollama**（免费）
   - 完全本地运行
   - 隐私性更好
   - 需要预先安装模型

### 支持的文件格式

- **视频**：MP4, MOV, AVI
- **文档**：PDF, DOCX, Markdown, TXT
- **输出**：Markdown格式报告

## 🎯 使用场景

- 📝 会议记录整理
- 🎓 学术讲座转录
- 💼 商务会议纪要
- 🎙️ 访谈内容整理
- 📺 视频内容总结

## 🔍 性能优化

### 推荐配置

- **内存**：至少8GB RAM
- **存储**：至少2GB可用空间
- **处理器**：支持AVX指令集的CPU
- **GPU**：可选，支持CUDA的GPU可加速转录

### 大文件处理

- 建议视频文件 < 500MB
- 超过1小时的会议考虑分段处理
- 可以使用GPU加速（如果可用）

## 🚨 注意事项

1. **API密钥安全**：请妥善保管OpenAI API密钥
2. **文件隐私**：敏感会议建议使用本地Ollama
3. **网络要求**：使用OpenAI需要稳定的网络连接
4. **系统要求**：确保有足够的磁盘空间和内存

## 📚 详细文档

查看 [CONFIG.md](CONFIG.md) 获取完整的配置说明和故障排除指南。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

请遵守所有依赖项的许可证要求，包括：
- OpenAI API使用条款
- Whisper模型许可证
- 其他开源库许可证

---

**开始使用智能会议纪要生成器，让AI帮助您提高会议效率！** 🚀 
# 智能会议纪要生成器 - 配置说明

## 功能特点

✨ **全新升级的功能**：
- 🎯 **Whisper Medium模型**：更高精度的语音转录，支持中英文双语
- 🤖 **AI智能纠错**：集成OpenAI GPT和本地Ollama模型
- 📝 **智能文本纠正**：结合参考文档对转录结果进行纠错
- 📊 **详细会议纪要**：包含会议重点、决议、进度、计划等
- 🔄 **实时进度显示**：前端和后端都有详细的处理步骤显示
- 📄 **多格式文档支持**：PDF、DOCX、Markdown、TXT
- 💾 **多文件输出**：纠正后的转录文本和完整会议报告

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件：

```env
# OpenAI API 配置
OPENAI_API_KEY=your_openai_api_key_here

# Ollama 配置 (本地模型服务)
OLLAMA_BASE_URL=http://localhost:11434

# Flask 配置
FLASK_ENV=development
FLASK_DEBUG=True
```

### 3. 系统依赖

#### FFmpeg (必需)
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# Windows
# 下载并安装 FFmpeg: https://ffmpeg.org/download.html
```

#### Tesseract (可选，用于图片文字识别)
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt install tesseract-ocr

# Windows
# 下载并安装 Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
```

## AI服务配置

### OpenAI GPT (推荐)

1. **获取API密钥**：
   - 访问 [OpenAI Platform](https://platform.openai.com/api-keys)
   - 创建新的API密钥
   - 将密钥添加到 `.env` 文件中

2. **模型选择**：
   - 默认使用 `gpt-3.5-turbo`
   - 可在代码中修改为 `gpt-4` 获得更好效果

### 本地Ollama (免费选项)

1. **安装Ollama**：
   ```bash
   # macOS/Linux
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Windows
   # 下载并安装: https://ollama.com/download
   ```

2. **下载模型**：
   ```bash
   # 推荐模型
   ollama pull llama2
   ollama pull codellama
   ollama pull mistral
   ```

3. **启动服务**：
   ```bash
   ollama serve
   ```

## 使用说明

### 启动应用

```bash
python app.py
```

访问 `http://localhost:5000`

### 使用流程

1. **上传会议视频**：
   - 支持格式：MP4、MOV、AVI
   - 建议文件大小：< 500MB

2. **添加参考文档**（可选）：
   - PDF文档
   - Word文档（DOCX）
   - Markdown文件
   - 纯文本文件
   - 或直接输入文本内容

3. **选择AI服务**：
   - OpenAI GPT（需要API密钥）
   - 本地Ollama（需要预先安装）

4. **处理过程**：
   - 视频上传
   - 音频提取
   - 语音转录
   - 文档处理
   - 转录纠错
   - 生成纪要
   - 创建报告

5. **下载结果**：
   - 纠正后的转录文本
   - 完整的会议报告

### 处理步骤说明

1. **视频上传**：将视频文件上传到服务器
2. **音频提取**：使用FFmpeg从视频中提取音频
3. **语音转录**：使用Whisper Medium模型进行转录
4. **文档处理**：解析上传的参考文档
5. **转录纠错**：使用AI模型结合参考文档纠正转录错误
6. **生成纪要**：创建结构化的会议纪要
7. **创建报告**：生成最终的Markdown报告文件

## 故障排除

### 常见问题

1. **FFmpeg not found**：
   - 确保已安装FFmpeg并添加到PATH环境变量

2. **OpenAI API错误**：
   - 检查API密钥是否正确
   - 确认账户有足够的余额
   - 检查网络连接

3. **Ollama连接失败**：
   - 确认Ollama服务正在运行
   - 检查服务端口（默认11434）
   - 验证模型是否已下载

4. **视频处理失败**：
   - 检查视频文件格式
   - 确认文件没有损坏
   - 尝试较小的文件进行测试

5. **内存不足**：
   - 对于大视频文件，可能需要更多内存
   - 尝试使用smaller Whisper模型（修改代码中的'medium'为'small'）

## 性能优化

### 推荐配置

- **内存**：至少8GB RAM
- **存储**：至少2GB可用空间
- **处理器**：支持AVX指令集的CPU
- **GPU**：可选，支持CUDA的GPU可加速Whisper转录

### 大文件处理

对于超过1小时的会议视频：
1. 考虑先进行视频压缩
2. 使用GPU加速（如果可用）
3. 分段处理长视频

## 安全注意事项

1. **API密钥安全**：
   - 不要在代码中硬编码API密钥
   - 使用环境变量存储敏感信息
   - 定期轮换API密钥

2. **文件安全**：
   - 上传的文件存储在临时目录
   - 建议定期清理临时文件
   - 对于敏感会议，考虑使用本地Ollama

3. **网络安全**：
   - 在生产环境中使用HTTPS
   - 考虑设置访问控制
   - 监控API使用情况

## 扩展功能

### 自定义提示词

可以修改 `app.py` 中的提示词来适应特定的会议场景：

```python
# 纠错提示词
correction_prompt = """
请根据以下参考文档，对会议转录文本进行纠正...
"""

# 纪要生成提示词
summary_prompt = """
请根据以下会议转录文本和参考文档，生成详细的会议纪要...
"""
```

### 添加新的AI模型

可以集成其他AI服务，如：
- Claude API
- 本地运行的其他模型
- 自定义的微调模型

## 技术架构

- **前端**：HTML5, CSS3, JavaScript, Socket.IO
- **后端**：Flask, Flask-SocketIO
- **AI模型**：Whisper, OpenAI GPT, Ollama
- **文档处理**：PyMuPDF, python-docx
- **音视频处理**：FFmpeg

## 许可证

请确保遵守所有依赖项的许可证要求，特别是：
- OpenAI API的使用条款
- Whisper模型的许可证
- 其他开源库的许可证 
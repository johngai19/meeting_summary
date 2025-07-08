# 🎤 智能会议纪要生成器

基于AI的会议转录、纠错和纪要生成工具，集成Whisper、OpenAI GPT、OpenRouter和Ollama模型。

## ✨ 主要特性

### 🆕 最新升级功能 (v2.1)

- **🚀 GPU加速支持**：支持Apple Silicon MPS和NVIDIA CUDA加速
- **🌐 OpenRouter集成**：支持多种AI模型（GPT-4o、Claude、Llama、DeepSeek等）
- **🎯 智能模型选择**：自动检测可用模型，提供最佳推荐
- **📊 实时进度追踪**：详细的步骤进度和时间估算
- **🔄 高级纠错算法**：两阶段AI纠错，结合上下文优化
- **💪 增强错误处理**：自动重试机制和优雅降级
- **📱 响应式界面**：全新的现代化用户界面
- **🔐 服务器端配置**：API密钥等敏感信息在服务器端管理
- **🔍 实时调试监控**：WebSocket实时通信和调试信息
- **📁 多文件下载**：原始转录、AI纠错、完整报告分别下载

### 🌟 核心功能

1. **智能语音转录**
   - 使用Whisper Medium模型
   - 支持中英双语自动识别
   - GPU加速处理（MPS/CUDA/CPU自适应）
   - 实时转录进度显示

2. **多平台AI支持**
   - **OpenAI GPT**：GPT-4o、GPT-4 Turbo、GPT-3.5
   - **OpenRouter**：Claude、Llama、DeepSeek、Gemini等
   - **本地Ollama**：Mistral、Qwen、Gemma等

3. **高级文本纠错**
   - 两阶段纠错算法
   - 结合参考文档的上下文优化
   - 专业术语识别和修正
   - 语法和流畅性改进

4. **结构化会议纪要**
   - 会议概要和参会人员
   - 讨论重点和工作决议
   - 当前进度和后续计划
   - 具体的待办事项和关键数据

5. **实时进度追踪**
   - WebSocket实时通信
   - 详细的处理步骤显示
   - 时间估算和剩余时间预测
   - 错误处理和重试机制

6. **智能用户界面**
   - 连接状态实时监控
   - 会话ID追踪和管理
   - 调试信息和错误详情显示
   - 多文件下载管理
   - 响应式设计和移动端适配

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
# OpenAI API 配置（必需，如果使用OpenAI服务）
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini

# OpenRouter 配置（可选，替代OpenAI）
# OPENAI_API_KEY=your_openrouter_api_key_here
# OPENAI_BASE_URL=https://openrouter.ai/api/v1
# OPENAI_MODEL=gpt-4o-mini

# Ollama 配置（本地模型）
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
```

**重要说明**：
- **所有AI配置现在都在服务器端**，用户无需在界面中输入API密钥
- 至少需要配置一种AI服务（OpenAI或Ollama）
- 如果使用OpenRouter，请将`OPENAI_BASE_URL`设置为`https://openrouter.ai/api/v1`

### 4. 启动应用

```bash
python app.py
```

访问 `http://localhost:5000`

## 📖 使用说明

### 处理流程

1. **上传会议视频**（支持MP4、MOV、AVI）
2. **添加参考文档**（PDF、DOCX、MD、TXT，可选）
3. **选择AI服务**：
   - OpenAI GPT（推荐GPT-4o Mini）
   - OpenRouter（支持多种模型）
   - 本地Ollama（推荐Mistral）
4. **实时查看详细进度**
5. **下载生成的文件**：
   - 纠正后的转录文本
   - 完整的会议报告

### 处理步骤详情

- 🎬 **视频验证**：检查文件格式和系统资源
- 🎵 **音频提取**：使用FFmpeg提取高质量音频
- 🎤 **语音转录**：Whisper模型智能转录（支持GPU加速）
- 📄 **文档处理**：解析参考文档，提取关键信息
- ✏️ **AI智能纠错**：两阶段纠错，结合上下文优化
- 📋 **生成纪要**：结构化会议纪要生成
- 📊 **创建报告**：生成完整的会议报告文件

### 模型选择建议

#### OpenAI/OpenRouter模型
- **GPT-4o Mini** (推荐)：性价比最高，速度快
- **GPT-4o**：最高质量，适合重要会议
- **DeepSeek Chat**：成本低，中文支持好
- **Claude 3 Haiku**：平衡性能和成本
- **Llama 3.1 70B**：开源模型，性能优秀

#### 本地Ollama模型
- **Mistral** (推荐)：综合性能最佳
- **DeepSeek Coder**：代码相关会议
- **Qwen2**：中文优化
- **Llama 3.1**：通用能力强

## 🛠️ 技术栈

- **前端**：HTML5, CSS3, JavaScript, Socket.IO
- **后端**：Flask, Flask-SocketIO, PyTorch
- **AI模型**：Whisper, OpenAI GPT, OpenRouter, Ollama
- **文档处理**：PyMuPDF, python-docx
- **音视频处理**：FFmpeg
- **GPU加速**：MPS (Apple Silicon), CUDA (NVIDIA)

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

## 🔧 高级配置

### GPU加速配置

应用程序会自动检测并使用最佳的计算设备：

1. **Apple Silicon (MPS)**：M1/M2 Mac自动启用
2. **NVIDIA CUDA**：支持CUDA的GPU自动启用
3. **CPU模式**：作为后备选项

### OpenRouter配置

使用OpenRouter访问多种AI模型：

```env
OPENAI_API_KEY=your_openrouter_key
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

支持的模型：
- GPT-4o, GPT-4 Turbo
- Claude 3 Haiku
- Llama 3.1 70B
- DeepSeek Chat
- Gemini Pro

### Ollama本地部署

安装并运行Ollama：

```bash
# 安装Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 下载推荐模型
ollama pull mistral
ollama pull deepseek-coder
ollama pull qwen2
ollama pull llama3.1
```

## 📊 性能优化

### 系统要求

- **内存**：至少8GB RAM（推荐16GB）
- **存储**：至少2GB可用空间
- **处理器**：支持AVX指令集的CPU
- **GPU**：可选，Apple Silicon或NVIDIA GPU

### 性能建议

1. **GPU加速**：使用MPS或CUDA显著加快转录速度
2. **模型选择**：根据质量要求选择合适模型
3. **文件大小**：建议视频文件 < 1GB
4. **内存管理**：大文件处理时监控内存使用

### 处理时间参考

| 音频时长 | CPU模式 | GPU加速 | 纠错时间 |
|---------|---------|---------|---------|
| 10分钟  | 3-5分钟 | 1-2分钟 | 30-60秒 |
| 30分钟  | 8-12分钟| 3-5分钟 | 1-2分钟 |
| 1小时   | 15-25分钟| 6-10分钟| 2-4分钟 |

## 🖥️ 用户界面功能

### 连接状态监控
- **实时连接指示器**：显示WebSocket连接状态
- **自动重连机制**：连接断开时自动尝试重连
- **状态颜色编码**：绿色（已连接）、红色（断开/错误）

### 进度追踪界面
- **会话ID显示**：每次处理分配唯一会话标识
- **步骤式进度条**：7个主要处理步骤的详细显示
- **实时时间估算**：已用时间和预计剩余时间
- **子步骤进度**：每个步骤内部的详细进度显示

### 文件下载管理
- **分类下载**：原始转录、AI纠错、完整报告三种文件
- **智能显示**：根据处理结果动态显示可用下载
- **文件信息**：显示文件大小、处理时间等统计信息

### 调试和监控
- **实时日志**：显示处理过程的详细日志信息
- **错误诊断**：详细的错误信息和解决建议
- **性能统计**：处理时间、音频时长、检测语言等信息
- **一键调试**：快速开启/关闭调试模式

## 🔍 使用场景

- 📝 **企业会议**：董事会、项目讨论、团队会议
- 🎓 **学术会议**：研讨会、讲座、答辩
- 💼 **商务谈判**：客户沟通、合作讨论
- 🎙️ **访谈节目**：专访、调研、采访
- 📺 **培训课程**：内部培训、在线课程

## 🚨 注意事项

1. **API密钥安全**：请妥善保管API密钥，不要提交到版本控制
2. **数据隐私**：敏感会议建议使用本地Ollama模型
3. **网络要求**：使用在线AI服务需要稳定的网络连接
4. **资源监控**：大文件处理时注意系统资源使用情况

## 🔧 故障排除

### 常见问题

1. **Whisper初始化失败**
   - 检查PyTorch安装
   - 确保有足够内存
   - 尝试CPU模式

2. **API调用失败**
   - 验证.env文件中的API密钥有效性
   - 检查网络连接
   - 确认API配额和BASE_URL设置

3. **前端连接问题**
   - 检查WebSocket连接状态（页面顶部指示器）
   - 刷新页面重新建立连接
   - 检查浏览器控制台是否有错误信息
   - 确认端口5000未被其他程序占用

4. **进度不更新**
   - 确认WebSocket连接正常（绿色指示器）
   - 检查会话ID是否正确显示
   - 开启调试模式查看详细日志
   - 检查浏览器网络面板中的WebSocket连接

5. **下载链接不显示**
   - 等待处理完全完成（显示"处理完成"页面）
   - 检查浏览器控制台是否有JavaScript错误
   - 刷新页面重新获取结果
   - 确认服务器生成了对应的文件

6. **文档处理错误**
   - 检查文件格式支持（PDF、DOCX、MD、TXT）
   - 确认文件未损坏
   - 验证文件权限和大小限制

7. **GPU加速不工作**
   - 检查PyTorch GPU支持
   - 更新显卡驱动
   - 确认CUDA版本兼容

### 调试和监控功能

1. **启用调试模式**
   - 在处理完成页面点击"显示调试信息"
   - 查看详细的处理日志和时间戳
   - 监控WebSocket通信状态

2. **连接状态监控**
   - 页面顶部实时显示连接状态
   - 绿色表示正常连接
   - 红色表示连接断开或错误

3. **错误诊断**
   - 错误发生时点击"显示错误详情"
   - 查看完整的错误堆栈信息
   - 复制错误信息用于问题报告

4. **性能监控**
   - 实时显示处理进度和剩余时间
   - 查看各步骤的执行时间
   - 监控音频时长和检测语言

### 性能优化建议

1. **定期清理**：删除临时文件释放空间
2. **监控资源**：使用系统监控工具和内置调试功能
3. **模型选择**：根据需要平衡质量和速度
4. **批量处理**：避免同时处理多个大文件
5. **网络优化**：确保稳定的网络连接用于AI API调用

## 📚 API文档

### REST API端点

- `GET /` - 主页面
- `POST /process` - 处理会议视频
- `GET /download/<filename>` - 下载生成的文件
- `GET /get_models` - 获取可用模型列表

### WebSocket事件

- `progress_update` - 处理进度更新
- `processing_complete` - 处理完成
- `error` - 错误信息

## 🤝 贡献指南

1. Fork本项目
2. 创建功能分支
3. 提交更改
4. 发起Pull Request

## 📄 许可证

请遵守所有依赖项的许可证要求，包括：
- OpenAI API使用条款
- OpenRouter服务条款
- Whisper模型许可证
- 其他开源库许可证

## 📞 支持

如有问题或建议，请：
1. 查看[CONFIG.md](CONFIG.md)详细配置说明
2. 提交GitHub Issue
3. 参考故障排除指南

---

**🎯 开始使用智能会议纪要生成器，让AI助力您的会议效率提升！**

*v2.1 - 全面升级版本，支持服务器端配置、实时调试监控、多文件下载管理* 🚀

## 🔄 更新日志

### v2.1 (最新版本)
- ✨ 服务器端配置管理，API密钥安全保护
- 🔍 实时WebSocket通信和调试监控
- 📁 多文件下载管理（原始、纠错、报告）
- 🐛 修复前端进度更新和下载链接问题
- 🎨 增强的用户界面和错误处理

### v2.0
- 🚀 GPU加速支持（MPS/CUDA）
- 🌐 OpenRouter多模型集成
- 📊 实时进度追踪和时间估算
- 🔄 两阶段AI纠错算法
- 💪 增强的错误处理和重试机制 
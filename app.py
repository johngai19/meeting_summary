from flask import Flask, render_template, request, send_file, redirect, url_for, jsonify
from flask_socketio import SocketIO, emit
import os
import subprocess
import whisper
import torch
try:
    import openai
except ImportError:
    openai = None
import requests
import json
from datetime import datetime
import fitz
from fitz import Document
import pytesseract
from PIL import Image
import pandas as pd
import tempfile
from typing import cast, Optional, Dict, List, Tuple
import logging
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import uuid
import threading
import time
import re
from functools import wraps
import psutil

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
socketio = SocketIO(app, cors_allowed_origins="*")

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 从环境变量读取配置
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'mistral')

# 支持的模型列表
OPENAI_MODELS = [
    {'id': 'gpt-4o-mini', 'name': 'GPT-4o Mini (推荐)', 'provider': 'openai'},
    {'id': 'gpt-4o', 'name': 'GPT-4o', 'provider': 'openai'},
    {'id': 'gpt-4-turbo', 'name': 'GPT-4 Turbo', 'provider': 'openai'},
    {'id': 'gpt-3.5-turbo', 'name': 'GPT-3.5 Turbo', 'provider': 'openai'},
    # OpenRouter models
    {'id': 'deepseek/deepseek-chat', 'name': 'DeepSeek Chat', 'provider': 'openrouter'},
    {'id': 'anthropic/claude-3-haiku', 'name': 'Claude 3 Haiku', 'provider': 'openrouter'},
    {'id': 'meta-llama/llama-3.1-70b-instruct', 'name': 'Llama 3.1 70B', 'provider': 'openrouter'},
    {'id': 'mistralai/mistral-7b-instruct', 'name': 'Mistral 7B Instruct', 'provider': 'openrouter'},
    {'id': 'google/gemini-pro', 'name': 'Gemini Pro', 'provider': 'openrouter'},
]

OLLAMA_DEFAULT_MODELS = [
    {'id': 'mistral', 'name': 'Mistral (推荐)', 'size': '7B'},
    {'id': 'deepseek-coder', 'name': 'DeepSeek Coder', 'size': '7B'},
    {'id': 'llama3.1', 'name': 'Llama 3.1', 'size': '8B'},
    {'id': 'qwen2', 'name': 'Qwen2', 'size': '7B'},
    {'id': 'gemma2', 'name': 'Gemma 2', 'size': '9B'},
]

# 检查GPU可用性
def check_gpu_availability():
    """检查GPU可用性"""
    try:
        # 检查是否支持MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        # 检查CUDA
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    except Exception as e:
        logger.warning(f"GPU检查失败: {e}")
        return 'cpu'

# 初始化Whisper模型
def initialize_whisper_model():
    """初始化Whisper模型"""
    try:
        device = check_gpu_availability()
        logger.info(f"正在初始化Whisper模型... 使用设备: {device}")
        
        if device == 'mps':
            # Apple Silicon MPS
            model = whisper.load_model('medium', device='mps')
        elif device == 'cuda':
            # NVIDIA GPU
            model = whisper.load_model('medium', device='cuda')
        else:
            # CPU
            model = whisper.load_model('medium', device='cpu')
            
        logger.info(f"Whisper模型初始化完成，设备: {device}")
        return model, device
    except Exception as e:
        logger.error(f"Whisper模型初始化失败: {e}")
        # 降级到CPU
        model = whisper.load_model('medium', device='cpu')
        return model, 'cpu'

whisper_model, whisper_device = initialize_whisper_model()

# 错误处理装饰器
def handle_errors(max_retries=3):
    """错误处理装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}")
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(1)  # 等待1秒后重试
            return None
        return wrapper
    return decorator

# 获取可用的Ollama模型
@handle_errors(max_retries=2)
def get_available_ollama_models():
    """获取可用的Ollama模型"""
    try:
        logger.info(f"正在检测Ollama模型，URL: {OLLAMA_BASE_URL}")
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            available_models = []
            
            if 'models' in data and data['models']:
                # 获取已安装的模型名称
                installed_names = set()
                for model in data['models']:
                    model_name = model['name'].split(':')[0]
                    installed_names.add(model_name)
                    logger.info(f"检测到已安装的Ollama模型: {model_name}")
                
                # 匹配默认模型列表
                for default_model in OLLAMA_DEFAULT_MODELS:
                    if default_model['id'] in installed_names:
                        available_models.append(default_model)
                        logger.info(f"添加可用模型: {default_model['name']}")
                
                # 如果没有匹配到默认模型，添加所有已安装的模型
                if not available_models:
                    for model in data['models']:
                        model_name = model['name'].split(':')[0]
                        available_models.append({
                            'id': model_name,
                            'name': model_name.title(),
                            'size': 'Unknown'
                        })
                        logger.info(f"添加检测到的模型: {model_name}")
            
            logger.info(f"共检测到 {len(available_models)} 个可用Ollama模型")
            return available_models
        else:
            logger.warning(f"无法连接到Ollama服务: HTTP {response.status_code}")
            return []
    except requests.exceptions.ConnectionError:
        logger.warning("无法连接到Ollama服务，请确保Ollama正在运行")
        return []
    except Exception as e:
        logger.error(f"获取Ollama模型失败: {e}")
        return []

class ProcessingProgress:
    def __init__(self):
        self.steps = []
        self.current_step = 0
        self.total_steps = 0
        self.session_id: Optional[str] = None
        self.start_time: Optional[float] = None
        self.estimated_total_time: Optional[float] = None
        
    def add_step(self, step_name: str, description: str = "", estimated_duration: float = 0):
        self.steps.append({
            'name': step_name,
            'description': description,
            'status': 'pending',
            'start_time': None,
            'end_time': None,
            'estimated_duration': estimated_duration,
            'progress_percentage': 0
        })
        self.total_steps = len(self.steps)
        
    def start_step(self, step_index: int):
        if 0 <= step_index < len(self.steps):
            self.current_step = step_index
            self.steps[step_index]['status'] = 'processing'
            self.steps[step_index]['start_time'] = time.time()
            
            if self.start_time is None:
                self.start_time = time.time()
                
            step_name = self.steps[step_index]['name']
            logger.info(f"🚀 [{self.session_id[:8] if self.session_id else 'LOCAL'}] 开始: {step_name}")
            self.emit_progress()
            
    def update_step_progress(self, step_index: int, progress: float):
        """更新步骤内部进度"""
        if 0 <= step_index < len(self.steps):
            self.steps[step_index]['progress_percentage'] = min(100, max(0, progress))
            self.emit_progress()
            
    def complete_step(self, step_index: int):
        if 0 <= step_index < len(self.steps):
            self.steps[step_index]['status'] = 'completed'
            self.steps[step_index]['end_time'] = time.time()
            self.steps[step_index]['progress_percentage'] = 100
            
            step_name = self.steps[step_index]['name']
            duration = self.steps[step_index]['end_time'] - self.steps[step_index]['start_time']
            logger.info(f"✅ [{self.session_id[:8] if self.session_id else 'LOCAL'}] 完成: {step_name} (耗时 {duration:.1f}秒)")
            self.emit_progress()
            
    def estimate_remaining_time(self):
        """估算剩余时间"""
        if not self.start_time:
            return None
            
        elapsed_time = time.time() - self.start_time
        completed_steps = sum(1 for step in self.steps if step['status'] == 'completed')
        
        if completed_steps == 0:
            return None
            
        avg_time_per_step = elapsed_time / completed_steps
        remaining_steps = self.total_steps - completed_steps
        
        return remaining_steps * avg_time_per_step
            
    def emit_progress(self):
        if self.session_id:
            # 计算总体进度
            total_progress = 0
            for i, step in enumerate(self.steps):
                if step['status'] == 'completed':
                    total_progress += 100
                elif step['status'] == 'processing':
                    total_progress += step['progress_percentage']
            
            overall_percentage = total_progress / self.total_steps if self.total_steps > 0 else 0
            
            # 估算剩余时间
            remaining_time = self.estimate_remaining_time()
            
            # 处理datetime序列化问题
            serialized_steps = []
            for step in self.steps:
                serialized_step = {
                    'name': step['name'],
                    'description': step['description'],
                    'status': step['status'],
                    'progress_percentage': step['progress_percentage'],
                    'start_time': step['start_time'],
                    'end_time': step['end_time']
                }
                serialized_steps.append(serialized_step)
            
            progress_data = {
                'current_step': self.current_step,
                'total_steps': self.total_steps,
                'steps': serialized_steps,
                'percentage': overall_percentage,
                'estimated_remaining_time': remaining_time
            }
            
            # 命令行监控输出
            current_step_name = self.steps[self.current_step]['name'] if self.current_step < len(self.steps) else "完成"
            remaining_str = f", 预计剩余 {remaining_time:.1f}秒" if remaining_time else ""
            logger.info(f"🔄 [{self.session_id[:8]}] 进度: {overall_percentage:.1f}% - {current_step_name}{remaining_str}")
            
            socketio.emit('progress_update', progress_data, to=self.session_id)

@handle_errors(max_retries=3)
def call_openai_api(prompt: str, model: Optional[str] = None, max_tokens: int = 2000) -> Optional[str]:
    """调用OpenAI API或OpenRouter API"""
    try:
        if not openai:
            logger.warning("🚫 OpenAI library not available")
            return None
            
        if not OPENAI_API_KEY:
            logger.warning("🔑 OPENAI_API_KEY not configured in .env file")
            return None
            
        used_model = model or OPENAI_MODEL
        logger.info(f"🤖 调用OpenAI API - 模型: {used_model}, 基础URL: {OPENAI_BASE_URL}")
        
        # 创建客户端
        client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )
        
        response = client.chat.completions.create(
            model=used_model,
            messages=[
                {"role": "system", "content": "You are a professional meeting assistant specialized in transcription correction and meeting summary generation. Please respond in Chinese."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        result = content.strip() if content else ""
        logger.info(f"✅ OpenAI API 调用成功 - 输出长度: {len(result)} 字符")
        return result
        
    except Exception as e:
        logger.error(f"❌ OpenAI API call failed: {str(e)}")
        return None

@handle_errors(max_retries=3)
def call_ollama_api(prompt: str, model: Optional[str] = None, max_tokens: int = 2000) -> Optional[str]:
    """调用本地ollama模型"""
    try:
        used_model = model or OLLAMA_MODEL
        logger.info(f"🦙 调用Ollama API - 模型: {used_model}, 输入长度: {len(prompt)} 字符")
        
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": used_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7
                }
            },
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json().get('response', '').strip()
            logger.info(f"✅ Ollama API 调用成功 - 输出长度: {len(result)} 字符")
            return result
        else:
            logger.error(f"❌ Ollama API call failed with status {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Ollama API call failed: {str(e)}")
        return None

def advanced_transcription_correction(transcript: str, reference_docs: List[str], progress: ProcessingProgress, step_index: int, ai_provider: str) -> str:
    """高级转录文本纠错"""
    progress.start_step(step_index)
    
    try:
        # 处理参考文档，分类显示
        reference_content = ""
        if reference_docs:
            reference_content = f"""
## 参考背景资料：
本次会议提供了以下参考资料，请在修正过程中参考这些信息：

"""
            for i, doc in enumerate(reference_docs, 1):
                doc_preview = doc[:800] + "..." if len(doc) > 800 else doc
                reference_content += f"""
### 参考资料 {i}：
{doc_preview}

"""
        
        # 构建专业的提示词
        correction_prompt = f"""
## 角色定位
你是一名**专业的会议记录修订专家**，负责将语音转写的原始文本清理、润色并结构化成可读性强、逻辑清晰的会议纪要。

---

## 输入变量
- `transcript`：原始语音转文字识别结果（纯文本）
- `reference_docs`：与会议主题相关的参考背景资料

---

## 修正与优化步骤

### 1. 语言清理
- **去除语气词**：如"嗯"、"啊"、"呃"、"那个"、"这个"、"就是说"、"然后呢"等
- **删除重复与冗余**：去掉重复词语、废话、卡顿或识别噪音
- **消除识别错误**：修正同音字、近音字及乱码

### 2. 语法与表达
- **校正语法错误**：调整主谓宾、标点使用、断句位置
- **提升专业性**：丰富词汇，避免口语化表达，句式简洁凝练
- **统一时态与语态**：保持全文一致

### 3. 内容结构化
- **分段与标题**：根据议题或发言人分段，并为每一部分添加合适的二级标题
- **添加列表与要点**：将重要决策、行动项、问题与结论，用无序或有序列表呈现
- **标注关键信息**：如"决策"、"待办事项"、"问题"、"负责人"等，用加粗突出

### 4. 专业术语标准化
- **对照参考资料**：将术语与参考文档中的标准保持一致，修正错误用词
- **保持一致性**：确保同一概念在全文中使用统一的术语表达

### 5. 上下文与逻辑
- **连贯性检查**：确保前后逻辑流畅，必要时补充过渡语句
- **背景补充**：结合参考资料中的关键信息，对容易误解或遗漏的部分进行注释或补充

---

{reference_content}

## 原始语音转文字内容：
{transcript}

---

## 输出要求
请按照以下格式输出修正后的内容：

```markdown
# 会议内容（修正版）

## 主要讨论内容
### 议题一：[根据实际内容确定]
- **发言要点**：
  1. ...
  2. ...

### 议题二：[根据实际内容确定]
- **关键决策**：...
- **行动项目**：...

## 重要信息汇总
- **决策事项**：...
- **待办任务**：...
- **关键数据**：...
```

请提供修正后的会议内容：
"""
        
        progress.update_step_progress(step_index, 30)
        
        # 根据AI提供商调用相应的API
        corrected_text = None
        if ai_provider == 'openai' and OPENAI_API_KEY:
            corrected_text = call_openai_api(correction_prompt, max_tokens=4000)
        elif ai_provider == 'ollama':
            corrected_text = call_ollama_api(correction_prompt, max_tokens=4000)
        
        progress.update_step_progress(step_index, 80)
        
        if not corrected_text:
            logger.warning("⚠️ AI纠错失败，返回原始转录文本")
            corrected_text = transcript
        
        progress.complete_step(step_index)
        return corrected_text
        
    except Exception as e:
        logger.error(f"❌ 转录纠错失败: {str(e)}")
        progress.complete_step(step_index)
        return transcript

def generate_meeting_summary(corrected_transcript: str, reference_docs: List[str], progress: ProcessingProgress, step_index: int, ai_provider: str) -> Dict[str, str]:
    """生成详细的会议纪要"""
    progress.start_step(step_index)
    
    try:
        # 构建参考文档展示
        reference_content = ""
        if reference_docs:
            reference_content = f"""
## 参考背景资料：
本次会议基于以下参考资料进行讨论，请在生成纪要时体现这些背景信息的影响：

"""
            for i, doc in enumerate(reference_docs, 1):
                doc_preview = doc[:600] + "..." if len(doc) > 600 else doc
                reference_content += f"""
### 参考资料 {i}：
{doc_preview}

"""
        
        summary_prompt = f"""
## 角色定位
你是一名**资深的会议纪要专家**，负责将修正后的会议转录内容转化为专业、结构化、具有实用价值的会议纪要。

---

## 输入变量
- `corrected_transcript`：已修正的会议转录内容
- `reference_docs`：与会议主题相关的参考背景资料

---

## 分析与生成步骤

### 1. 内容深度分析
- **识别核心主题**：确定会议的主要目标和讨论重点
- **分析讨论层次**：理解议题的逻辑关系和重要性排序
- **提取关键观点**：识别重要意见、争议点和达成的共识
- **发现隐含信息**：注意暗示的问题、机会和未明确表达的需求

### 2. 信息提取与整理
- **事实数据收集**：提取具体的数字、时间、地点、人员信息
- **决策识别**：区分已确定的决策和待定的倾向
- **行动项梳理**：整理具体的任务、责任人和时间要求
- **优先级判断**：按重要性和紧急性对内容进行排序

### 3. 结构化处理
- **逻辑组织**：按照议题相关性和重要性重新组织内容
- **分类标注**：明确区分决策、行动项、问题、结论等不同类型信息
- **层次建立**：创建清晰的信息层次结构

---

{reference_content}

## 会议转录内容（已修正）：
{corrected_transcript}

---

## 输出格式要求
请按照以下专业格式生成会议纪要：

```markdown
# 会议纪要

## 1. 会议背景
> 基于实际内容分析的会议目的、背景和参与情况

## 2. 主要讨论内容
### 2.1 议题一：[根据实际内容确定标题]
- **发言要点**：
  1. ...
  2. ...
- **关键观点**：...
- **争议点**：...

### 2.2 议题二：[根据实际内容确定标题]
- **讨论要点**：...
- **技术细节**：...

## 3. 决策与共识
| 决策项 | 具体内容 | 影响范围 |
| ------ | -------- | -------- |
| ...    | ...      | ...      |

## 4. 行动计划
| 行动项 | 负责人 | 完成时间 | 优先级 |
| ------ | ------ | -------- | ------ |
| ...    | ...    | ...      | ...    |

## 5. 重要信息汇总
- **关键数据**：...
- **时间节点**：...
- **相关人员**：...
- **技术要点**：...

## 6. 待解决问题
1. ...
2. ...

## 7. 跟进事项
- **下次会议**：...
- **需要确认**：...
- **后续沟通**：...

## 8. 术语说明（如有必要）
| 术语 | 含义 |
| ---- | ---- |
| ...  | ...  |

## 9. 附录
- **参考文档**：基于提供的参考资料
- **会议时长**：...
- **参与人数**：...
```

---

## 特别要求
1. **避免简单复制**：不要直接复制转录内容，而是基于理解进行提炼和重组
2. **突出实用价值**：重点关注会议的实际成果、决策和行动计划
3. **体现背景关联**：如果有参考资料，要体现其与会议讨论的关联性
4. **保持专业性**：使用商务语言，避免口语化表达
5. **确保逻辑性**：内容组织要有清晰的逻辑结构
6. **注重可操作性**：行动项要具体明确，便于后续执行

请生成专业的会议纪要：
"""
        
        progress.update_step_progress(step_index, 30)
        
        # 根据AI提供商调用相应的API
        summary = None
        if ai_provider == 'openai' and OPENAI_API_KEY:
            summary = call_openai_api(summary_prompt, max_tokens=4000)
        elif ai_provider == 'ollama':
            summary = call_ollama_api(summary_prompt, max_tokens=4000)
        
        progress.update_step_progress(step_index, 80)
        
        if not summary:
            logger.warning("⚠️ AI会议纪要生成失败，使用智能分析格式")
            # 智能分析转录内容，提取关键信息
            lines = corrected_transcript.split('\n')
            content_lines = [line.strip() for line in lines if line.strip()]
            
            # 简单的关键词提取
            keywords = []
            for line in content_lines:
                if any(keyword in line for keyword in ['决定', '确定', '计划', '安排', '负责', '完成']):
                    keywords.append(line)
            
            summary = f"""## 会议概要
本次会议主要围绕相关议题进行了深入讨论，形成了多项重要共识。

## 主要讨论内容
{chr(10).join(content_lines[:10])}
{'...' if len(content_lines) > 10 else ''}

## 重要决议与行动计划
{chr(10).join(keywords[:5]) if keywords else '会议讨论了相关议题，形成了初步共识。'}

## 完整转录内容
{corrected_transcript}

---
_注：此纪要为自动生成的简化版本，建议配置AI服务获得更专业的会议纪要分析。_"""
        
        progress.complete_step(step_index)
        return {"summary": summary}
        
    except Exception as e:
        logger.error(f"❌ 会议纪要生成失败: {str(e)}")
        progress.complete_step(step_index)
        return {"summary": f"会议纪要生成失败: {str(e)}"}

def extract_text_from_file(file_path: str) -> str:
    """从文件中提取文本"""
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            doc = fitz.open(file_path)
            text = ""
            for page_num in range(len(doc)):
                page = doc[page_num]
                text += page.get_text()  # type: ignore
            doc.close()
            return text
        elif file_ext in ['.md', '.txt']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            logger.warning(f"Unsupported file type: {file_ext}")
            return ""
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {str(e)}")
        return ""

def estimate_audio_duration(audio_path: str) -> float:
    """估算音频时长"""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', audio_path],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())
    except Exception as e:
        logger.error(f"无法获取音频时长: {e}")
        return 0

@app.route('/')
def index():
    # 获取可用的Ollama模型
    available_ollama_models = get_available_ollama_models()
    
    # 检查配置状态
    config_status = {
        'openai_configured': bool(OPENAI_API_KEY),
        'ollama_available': len(available_ollama_models) > 0,
        'whisper_device': whisper_device
    }
    
    return render_template('index.html', 
                         openai_models=OPENAI_MODELS,
                         ollama_models=available_ollama_models,
                         config_status=config_status,
                         whisper_device=whisper_device)

@app.route('/get_models')
def get_models():
    """获取可用模型列表"""
    available_ollama_models = get_available_ollama_models()
    config_status = {
        'openai_configured': bool(OPENAI_API_KEY),
        'ollama_available': len(available_ollama_models) > 0
    }
    
    return jsonify({
        'openai_models': OPENAI_MODELS,
        'ollama_models': available_ollama_models,
        'config_status': config_status,
        'whisper_device': whisper_device
    })

@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")  # type: ignore

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")  # type: ignore

@socketio.on('join')
def handle_join(session_id):
    """客户端加入会话房间"""
    from flask_socketio import join_room
    join_room(session_id)
    logger.info(f"Client {request.sid} joined session: {session_id}")  # type: ignore
    emit('joined', {'session_id': session_id})

@app.route('/process', methods=['POST'])
def process_meeting():
    session_id = str(uuid.uuid4())
    
    try:
        # 获取表单数据
        video = request.files.get('video')
        docs = request.files.getlist('docs')
        text_input = request.form.get('docsText', '')
        ai_provider = request.form.get('aiProvider', 'openai')
        
        if not video or not video.filename:
            return jsonify({"error": "请选择视频文件"}), 400
        
        # 检查AI服务配置
        if ai_provider == 'openai' and not OPENAI_API_KEY:
            return jsonify({"error": "OpenAI API Key未在服务器端配置，请联系管理员"}), 400
        
        if ai_provider == 'ollama':
            available_models = get_available_ollama_models()
            if not available_models:
                return jsonify({"error": "未检测到可用的Ollama模型，请确保Ollama服务正在运行"}), 400
        
        # 在主线程中保存文件
        video_filename = video.filename
        base_name = os.path.splitext(video_filename)[0]
        mov_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        video.save(mov_path)
        
        # 处理文档文件
        doc_files = []
        for doc in docs:
            if doc and doc.filename:
                doc_path = os.path.join(app.config['UPLOAD_FOLDER'], doc.filename)
                doc.save(doc_path)
                doc_files.append({'filename': doc.filename, 'path': doc_path})
        
        # 在单独的线程中处理
        thread = threading.Thread(
            target=process_meeting_async, 
            args=(session_id, mov_path, video_filename, doc_files, text_input, ai_provider)
        )
        thread.start()
        
        return jsonify({"session_id": session_id, "status": "processing"})
    
    except Exception as e:
        logger.error(f"❌ 处理请求错误: {str(e)}")
        return jsonify({"error": f"处理请求错误: {str(e)}"}), 500

def process_meeting_async(session_id: str, mov_path: str, video_filename: str, doc_files: list, text_input: str, ai_provider: str):
    """异步处理会议"""
    progress = ProcessingProgress()
    progress.session_id = session_id
    
    logger.info(f"🎬 [{session_id[:8]}] 开始处理会议 - 视频: {video_filename}, AI: {ai_provider}")
    start_time = time.time()
    
    try:
        # 设置处理步骤（带时间估算）
        progress.add_step("video_validation", "验证视频文件", 2)
        progress.add_step("audio_extraction", "提取音频", 10)
        progress.add_step("speech_transcription", "语音转文字", 60)
        progress.add_step("document_processing", "处理参考文档", 5)
        progress.add_step("ai_correction", "AI智能纠错", 30)
        progress.add_step("summary_generation", "生成会议纪要", 25)
        progress.add_step("file_generation", "生成下载文件", 5)
        
        # 步骤1：验证视频文件
        progress.start_step(0)
        base_name = os.path.splitext(video_filename)[0]
        
        if not os.path.exists(mov_path):
            logger.error(f"❌ [{session_id[:8]}] 视频文件不存在: {mov_path}")
            socketio.emit('error', {'message': '视频文件不存在'}, to=session_id)
            return
            
        file_size = os.path.getsize(mov_path)
        logger.info(f"📁 [{session_id[:8]}] 视频文件大小: {file_size / (1024*1024):.1f} MB")
        
        # 检查系统资源
        memory_usage = psutil.virtual_memory().percent
        if memory_usage > 85:
            logger.warning(f"⚠️ 系统内存使用率较高: {memory_usage}%")
            
        progress.complete_step(0)
        
        # 步骤2：提取音频
        progress.start_step(1)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{base_name}_audio.wav')
        
        try:
            subprocess.run([
                'ffmpeg', '-i', mov_path, '-vn', '-acodec', 'pcm_s16le', 
                '-ar', '16000', '-ac', '1', audio_path, '-y'
            ], check=True, capture_output=True, text=True)
            
            audio_duration = estimate_audio_duration(audio_path)
            logger.info(f"🎵 [{session_id[:8]}] 音频提取完成 - 时长: {audio_duration:.1f}秒")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ 音频提取失败: {e}")
            socketio.emit('error', {'message': '音频提取失败，请检查视频文件格式'}, to=session_id)
            return
            
        progress.complete_step(1)
        
        # 步骤3：语音转文字
        progress.start_step(2)
        
        try:
            logger.info(f"🎤 [{session_id[:8]}] 开始语音转录 - 设备: {whisper_device}")
            
            # 更新进度
            for i in range(0, 91, 10):
                progress.update_step_progress(2, i)
                time.sleep(0.1)
            
            result = whisper_model.transcribe(audio_path, language=None)
            transcript = str(result["text"])
            
            detected_lang = result.get("language", "unknown")
            logger.info(f"🎤 [{session_id[:8]}] 转录完成 - 语言: {detected_lang}, 字符数: {len(transcript)}")
            
        except Exception as e:
            logger.error(f"❌ 语音转录失败: {e}")
            socketio.emit('error', {'message': f'语音转录失败: {str(e)}'}, to=session_id)
            return
            
        progress.complete_step(2)
        
        # 步骤4：处理参考文档
        progress.start_step(3)
        doc_texts = []
        
        # 处理文本输入（作为背景信息）
        if text_input and text_input.strip():
            formatted_text = f"## 会议背景信息\n\n{text_input.strip()}"
            doc_texts.append(formatted_text)
            logger.info(f"📝 [{session_id[:8]}] 添加背景信息: {len(text_input.strip())} 字符")
        
        # 处理上传的文档文件
        for doc_info in doc_files:
            try:
                text = extract_text_from_file(doc_info['path'])
                if text:
                    # 为每个文档添加标识
                    file_ext = os.path.splitext(doc_info['filename'])[1].lower()
                    doc_type = {
                        '.pdf': 'PDF文档',
                        '.doc': 'Word文档',
                        '.docx': 'Word文档',
                        '.md': 'Markdown文档',
                        '.txt': '文本文档'
                    }.get(file_ext, '文档')
                    
                    formatted_text = f"## 参考文档：{doc_info['filename']} ({doc_type})\n\n{text}"
                    doc_texts.append(formatted_text)
                    logger.info(f"📄 [{session_id[:8]}] 处理{doc_type}: {doc_info['filename']}, 提取 {len(text)} 字符")
            except Exception as e:
                logger.warning(f"⚠️ [{session_id[:8]}] 文档处理失败: {doc_info['filename']}, 错误: {str(e)}")
        
        logger.info(f"📚 [{session_id[:8]}] 参考文档处理完成 - 共 {len(doc_texts)} 个资料")
        progress.complete_step(3)
        
        # 步骤5：AI智能纠错
        corrected_transcript = advanced_transcription_correction(transcript, doc_texts, progress, 4, ai_provider)
        
        # 步骤6：生成会议纪要
        meeting_summary = generate_meeting_summary(corrected_transcript, doc_texts, progress, 5, ai_provider)
        
        # 步骤7：生成文件
        progress.start_step(6)
        
        # 生成原始转录文件
        raw_transcript_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{base_name}_raw_transcript.md')
        with open(raw_transcript_path, 'w', encoding='utf-8') as f:
            f.write(f'# 原始语音转录\n\n')
            f.write(f'**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'**处理设备**: {whisper_device.upper()}\n')
            f.write(f'**检测语言**: {detected_lang}\n')
            f.write(f'**音频时长**: {audio_duration:.1f}秒\n\n')
            f.write(f'## 转录内容\n\n{transcript}\n')
        
        # 生成纠正后的转录文件
        corrected_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{base_name}_corrected_transcript.md')
        with open(corrected_path, 'w', encoding='utf-8') as f:
            f.write(f'# AI纠正后的会议转录\n\n')
            f.write(f'**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'**处理设备**: {whisper_device.upper()}\n')
            f.write(f'**检测语言**: {detected_lang}\n')
            f.write(f'**AI服务**: {ai_provider.upper()}\n')
            f.write(f'**音频时长**: {audio_duration:.1f}秒\n\n')
            f.write(f'## 纠正后的转录内容\n\n{corrected_transcript}\n')
        
        # 生成完整的会议报告
        report_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{base_name}_meeting_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f'# 会议纪要报告\n\n')
            f.write(f'**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'**处理设备**: {whisper_device.upper()}\n')
            f.write(f'**检测语言**: {detected_lang}\n')
            f.write(f'**AI服务**: {ai_provider.upper()}\n')
            f.write(f'**音频时长**: {audio_duration:.1f}秒\n')
            f.write(f'**参考资料**: {len(doc_texts)} 个文档/背景信息\n\n')
            
            # 如果有参考文档，显示概览
            if doc_texts:
                f.write(f'## 参考资料概览\n\n')
                for i, doc in enumerate(doc_texts, 1):
                    # 提取文档标题（第一行）
                    first_line = doc.split('\n')[0] if doc else '未知文档'
                    f.write(f'{i}. {first_line.replace("##", "").strip()}\n')
                f.write(f'\n---\n\n')
            
            f.write(f'{meeting_summary["summary"]}\n\n')
            f.write(f'---\n\n## 附录：完整转录内容\n\n{corrected_transcript}\n')
        
        progress.complete_step(6)
        
        # 通知处理完成
        total_time = time.time() - start_time
        logger.info(f"🎉 [{session_id[:8]}] 处理完成! 总耗时: {total_time:.1f}秒")
        
        socketio.emit('processing_complete', {
            'raw_transcript_path': raw_transcript_path,
            'corrected_transcript_path': corrected_path,
            'report_path': report_path,
            'processing_time': total_time,
            'audio_duration': audio_duration,
            'detected_language': detected_lang,
            'ai_provider': ai_provider
        }, to=session_id)
        
    except Exception as e:
        logger.error(f"💥 [{session_id[:8]}] 处理错误: {str(e)}")
        socketio.emit('error', {'message': f'处理错误: {str(e)}'}, to=session_id)

@app.route('/download/<filename>')
def download_file(filename):
    """下载生成的文件"""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({"error": "文件不存在"}), 404
    except Exception as e:
        logger.error(f"文件下载错误: {e}")
        return jsonify({"error": f"文件下载错误: {str(e)}"}), 500

if __name__ == '__main__':
    # 启动时显示配置信息
    logger.info("=" * 50)
    logger.info("🎤 智能会议纪要生成器启动")
    logger.info(f"🖥️  Whisper设备: {whisper_device.upper()}")
    logger.info(f"🤖 OpenAI配置: {'✅ 已配置' if OPENAI_API_KEY else '❌ 未配置'}")
    
    # 检查Ollama
    ollama_models = get_available_ollama_models()
    logger.info(f"🦙 Ollama模型: {len(ollama_models)} 个可用")
    for model in ollama_models:
        logger.info(f"   - {model['name']}")
    
    logger.info("=" * 50)
    
    socketio.run(app, debug=True, port=5000)
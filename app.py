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
        # 构建专业的提示词
        correction_prompt = f"""
你是一个专业的会议记录助手。请对以下语音转文字识别的在线会议内容进行修正和优化。

任务要求：
1. 修正语音识别的错误（同音字、近音字、语法错误）
2. 改善语句的流畅性和可读性
3. 保持原意和会议的逻辑结构
4. 标准化专业术语和表达方式
5. 适当添加标点符号，提高可读性

{f'''
参考背景信息：
{chr(10).join(reference_docs[:3000])}
''' if reference_docs else ''}

原始语音转文字内容：
{transcript}

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
        summary_prompt = f"""
你是一个专业的会议纪要助手。请根据以下已修正的会议转录内容生成结构化的会议纪要。

任务要求：
1. 提取关键讨论点和决议
2. 整理重要信息和数据
3. 列出具体的行动项目
4. 保持专业和简洁的表达

请按以下格式生成会议纪要：

## 会议概要
[简要概述会议目的、时间、主要议题]

## 参会人员
[如果提到参会人员，请列出]

## 主要讨论内容
[按重要性列出讨论的关键点，使用项目符号]

## 重要决议
[明确的决策和决定事项]

## 行动计划
[具体的后续行动，包括负责人和时间（如有提及）]

## 关键数据和信息
[会议中提到的重要数据、时间、地点等]

## 待跟进事项
[需要进一步确认或处理的事项]

{f'''
参考背景信息：
{chr(10).join(reference_docs[:2000])}
''' if reference_docs else ''}

会议转录内容：
{corrected_transcript}

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
            logger.warning("⚠️ AI会议纪要生成失败，使用基础格式")
            summary = f"""## 会议概要
本次会议的主要内容如下。

## 主要讨论内容
{corrected_transcript[:1000]}{'...' if len(corrected_transcript) > 1000 else ''}

## 重要决议
根据会议讨论形成相关决议。

## 行动计划
会议确定了后续工作安排。

_注：此纪要为基础格式，建议配置AI服务获得更详细的会议纪要。_"""
        
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
        
        if text_input and text_input.strip():
            doc_texts.append(text_input.strip())
            logger.info(f"📝 [{session_id[:8]}] 添加文本输入: {len(text_input.strip())} 字符")
        
        for doc_info in doc_files:
            try:
                text = extract_text_from_file(doc_info['path'])
                if text:
                    doc_texts.append(text)
                    logger.info(f"📄 [{session_id[:8]}] 处理文档: {doc_info['filename']}, 提取 {len(text)} 字符")
            except Exception as e:
                logger.warning(f"⚠️ [{session_id[:8]}] 文档处理失败: {doc_info['filename']}, 错误: {str(e)}")
        
        logger.info(f"📚 [{session_id[:8]}] 参考文档处理完成 - 共 {len(doc_texts)} 个文档")
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
            f.write(f'**音频时长**: {audio_duration:.1f}秒\n\n')
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
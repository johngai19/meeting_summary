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
from typing import cast, Optional, Dict, List, Tuple, Any
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
import base64
from io import BytesIO
from openai.types.chat import ChatCompletionMessageParam
from werkzeug.utils import secure_filename

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
def call_openai_api(prompt: str, model: Optional[str] = None, max_tokens: int = 2000, image_base64: Optional[str] = None) -> Optional[str]:
    """调用OpenAI API或OpenRouter API，支持多模态"""
    try:
        if not openai:
            logger.warning("🚫 OpenAI library not available")
            return None
            
        if not OPENAI_API_KEY:
            logger.warning("🔑 OPENAI_API_KEY not configured in .env file")
            return None
            
        used_model = model or OPENAI_MODEL
        logger.info(f"🤖 调用OpenAI API - 模型: {used_model}, 基础URL: {OPENAI_BASE_URL}")
        
        client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are a professional meeting assistant specialized in transcription correction and meeting summary generation. Please respond in Chinese."},
        ]

        if image_base64:
            # 多模态调用
            used_model = 'gpt-4o' # 强制使用支持多模态的模型
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            })
        else:
            # 纯文本调用
            messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=used_model,
            messages=messages,
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

@handle_errors(max_retries=3)
def advanced_transcription_correction(transcript: str, context: str, reference_docs: List[str], progress: ProcessingProgress, step_index: int, ai_provider: str, model: str) -> str:
    """Corrects transcript using AI, context, and reference docs."""
    progress.start_step(step_index)
    reference_content = "\n\n---\n\n".join(reference_docs)
    prompt = correction_prompt_template.replace("{{transcript}}", transcript) \
                                       .replace("{{context}}", context) \
                                       .replace("{{reference_docs}}", reference_content)
    progress.update_step_progress(step_index, 30)
    corrected_text = None
    if ai_provider == 'openai' and OPENAI_API_KEY:
        corrected_text = call_openai_api(prompt, model=model, max_tokens=4000)
    elif ai_provider == 'ollama':
        corrected_text = call_ollama_api(prompt, model=model, max_tokens=4000)
    progress.update_step_progress(step_index, 80)
    if not corrected_text:
        logger.warning("AI correction failed, returning original transcript.")
        corrected_text = transcript
    progress.complete_step(step_index)
    return corrected_text

@handle_errors(max_retries=3)
def generate_meeting_summary(corrected_transcript: str, context: str, reference_docs: List[str], progress: ProcessingProgress, step_index: int, ai_provider: str, model: str) -> Dict[str, str]:
    """Generates a meeting summary using AI."""
    progress.start_step(step_index)
    reference_content = "\n\n---\n\n".join(reference_docs)
    prompt = summary_prompt_template.replace("{{corrected_transcript}}", corrected_transcript) \
                                    .replace("{{context}}", context) \
                                    .replace("{{reference_docs}}", reference_content)
    progress.update_step_progress(step_index, 30)
    summary = None
    if ai_provider == 'openai' and OPENAI_API_KEY:
        summary = call_openai_api(prompt, model=model, max_tokens=4000)
    elif ai_provider == 'ollama':
        summary = call_ollama_api(prompt, model=model, max_tokens=4000)
    progress.update_step_progress(step_index, 80)
    if not summary:
        logger.warning("AI summary generation failed, creating a basic summary.")
        summary = f"# Meeting Summary\n\n## Key Points\n- AI summary generation failed. This is a fallback summary.\n\n## Full Transcript\n{corrected_transcript}"
    progress.complete_step(step_index)
    return {"summary": summary}

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
        video = request.files.get('video')
        docs = request.files.getlist('docs[]')
        context_input = request.form.get('context', '')
        ai_provider = request.form.get('aiProvider', 'openai')
        
        # Determine the model to use
        if ai_provider == 'openai':
            model = request.form.get('model', OPENAI_MODELS[0]['id'])
        else:
            # Fallback for ollama if no model is selected or available
            available_ollama = get_available_ollama_models()
            model = request.form.get('model', available_ollama[0]['id'] if available_ollama else OLLAMA_DEFAULT_MODELS[0]['id'])

        logger.info(f"收到新的处理请求 - Session: {session_id[:8]}, AI: {ai_provider}, Model: {model}")
        if video and video.filename:
            logger.info(f"  - 视频文件: {video.filename}")
        
        # 增强日志，记录所有收到的文档文件名
        if docs:
            file_names = [doc.filename for doc in docs if doc.filename]
            logger.info(f"  - 参考文档: {len(file_names)} 个 -> {file_names}")

        if not video or not video.filename:
            return jsonify({"error": "请选择视频文件"}), 400
        
        if ai_provider == 'openai' and not OPENAI_API_KEY:
            return jsonify({"error": "OpenAI API Key未在服务器端配置，请联系管理员"}), 400
        
        if ai_provider == 'ollama':
            available_models = get_available_ollama_models()
            if not available_models:
                return jsonify({"error": "未检测到可用的Ollama模型，请确保Ollama服务正在运行"}), 400
        
        # 保存视频文件，确保文件名安全
        video_filename = secure_filename(video.filename) if video and video.filename else f"video-{session_id}.mov"
        mov_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        video.save(mov_path)
        
        # 处理文档文件，确保文件名唯一
        doc_files = []
        for doc in docs:
            if doc and doc.filename:
                original_filename = secure_filename(doc.filename)
                # 创建唯一文件名来避免覆盖
                unique_filename = f"{session_id[:8]}-{uuid.uuid4().hex[:8]}-{original_filename}"
                saved_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                doc.save(saved_path)
                doc_files.append({'original_filename': original_filename, 'saved_path': saved_path})
        
        # 在单独的线程中处理
        thread = threading.Thread(
            target=process_meeting_async, 
            args=(session_id, mov_path, video_filename, doc_files, context_input, ai_provider, model)
        )
        thread.start()
        
        return jsonify({"session_id": session_id, "status": "processing"})
    except Exception as e:
        logger.error(f"❌ 处理请求错误: {str(e)}")
        return jsonify({"error": f"处理请求错误: {str(e)}"}), 500

def process_meeting_async(session_id: str, mov_path: str, video_filename: str, doc_files: list, context_input: str, ai_provider: str, model: str):
    """Asynchronously processes the meeting from video to summary."""
    progress = ProcessingProgress()
    progress.session_id = session_id
    logger.info(f"🎬 [{session_id[:8]}] Starting meeting processing...")
    start_time = time.time()
    
    try:
        # Define processing steps
        progress.add_step("video_validation", "Validating video file", 2)
        progress.add_step("audio_extraction", "Extracting audio", 10)
        progress.add_step("speech_transcription", "Transcribing audio to text", 60)
        progress.add_step("document_processing", "Processing reference documents", 5)
        progress.add_step("image_analysis", "Analyzing images in PDFs", 20)
        progress.add_step("ai_correction", "Correcting transcript with AI", 30)
        progress.add_step("summary_generation", "Generating meeting summary", 25)
        progress.add_step("file_generation", "Creating downloadable files", 5)

        # Step 1: Validate Video
        progress.start_step(0)
        base_name = os.path.splitext(video_filename)[0]
        if not os.path.exists(mov_path):
            raise FileNotFoundError(f"Video file not found: {mov_path}")
        progress.complete_step(0)

        # Step 2: Extract Audio
        progress.start_step(1)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{base_name}_audio.wav')
        subprocess.run(['ffmpeg', '-i', mov_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_path, '-y'], check=True, capture_output=True, text=True)
        audio_duration = estimate_audio_duration(audio_path)
        progress.complete_step(1)

        # Step 3: Transcribe Speech
        progress.start_step(2)
        result = whisper_model.transcribe(audio_path, language=None)
        transcript = str(result["text"])
        detected_lang = result.get("language", "unknown")
        progress.complete_step(2)

        # Step 4: Process Reference Documents
        progress.start_step(3)
        doc_texts = []
        pdf_files_for_image_analysis = []
        total_docs = len(doc_files)
        for i, doc_info in enumerate(doc_files):
            original_filename, saved_path = doc_info['original_filename'], doc_info['saved_path']
            progress.steps[3]['description'] = f"Processing {i+1}/{total_docs}: {original_filename}"
            progress.emit_progress()
            
            text = extract_text_from_file(saved_path)
            if text:
                doc_texts.append(f"## Reference: {original_filename}\n\n{text}")
            if original_filename.lower().endswith('.pdf'):
                pdf_files_for_image_analysis.append(saved_path)
        progress.complete_step(3)

        # Step 5: Analyze Images in PDFs
        progress.start_step(4)
        if ai_provider == 'openai':
            image_analysis_results = []
            for pdf_path in pdf_files_for_image_analysis:
                image_analysis_results.extend(analyze_pdf_images(pdf_path, ai_provider))
            if image_analysis_results:
                doc_texts.extend(image_analysis_results)
        progress.complete_step(4)

        # Step 6: AI Correction
        corrected_transcript = advanced_transcription_correction(transcript, context_input, doc_texts, progress, 5, ai_provider, model)
        
        # Step 7: Generate Summary
        meeting_summary = generate_meeting_summary(corrected_transcript, context_input, doc_texts, progress, 6, ai_provider, model)
        
        # Step 8: Generate Files
        progress.start_step(7)
        
        base_name = os.path.splitext(video_filename)[0]
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Generate raw transcript file
        raw_transcript_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{base_name}_raw_transcript.md')
        with open(raw_transcript_path, 'w', encoding='utf-8') as f:
            f.write(f'# 原始语音转录\n\n')
            f.write(f'**生成时间**: {now_str}\n')
            f.write(f'**处理设备**: {whisper_device.upper()}\n')
            f.write(f'**检测语言**: {detected_lang}\n')
            f.write(f'**音频时长**: {audio_duration:.1f}秒\n\n')
            f.write(f'## 转录内容\n\n{transcript}\n')

        # Generate corrected transcript file
        corrected_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{base_name}_corrected_transcript.md')
        with open(corrected_path, 'w', encoding='utf-8') as f:
            f.write(f'# AI修正后的会议转录\n\n')
            f.write(f'**生成时间**: {now_str}\n')
            f.write(f'**AI模型**: {model}\n')
            f.write(f'**音频时长**: {audio_duration:.1f}秒\n\n')
            f.write(f'## 修正后的转录内容\n\n{corrected_transcript}\n')

        # Generate full meeting report
        report_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{base_name}_meeting_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f'# 会议纪要报告\n\n')
            f.write(f'**生成时间**: {now_str}\n')
            f.write(f'**AI模型**: {model}\n')
            f.write(f'**音频时长**: {audio_duration:.1f}秒\n')
            f.write(f'**参考资料**: {len(doc_texts)} 个\n\n')
            
            if doc_texts:
                f.write(f'## 参考资料概览\n\n')
                for i, doc in enumerate(doc_texts, 1):
                    first_line = doc.split('\n')[0] if doc else '未知文档'
                    f.write(f'{i}. {first_line.replace("##", "").strip()}\n')
                f.write(f'\n---\n\n')
            
            f.write(f'{meeting_summary["summary"]}\n\n')
            f.write(f'---\n\n## 附录：AI修正后完整记录\n\n{corrected_transcript}\n')

        progress.complete_step(7)

        # Final step: Notify client
        total_time = time.time() - start_time
        logger.info(f"🎉 [{session_id[:8]}] Processing complete! Total time: {total_time:.1f}s")
        
        socketio.emit('processing_complete', {
            'raw_transcript_path': raw_transcript_path,
            'corrected_transcript_path': corrected_path,
            'report_path': report_path,
            'processing_time': total_time,
            'audio_duration': audio_duration,
            'detected_language': detected_lang,
            'ai_provider': ai_provider,
            'model': model
        }, to=session_id)

    except Exception as e:
        logger.error(f"💥 [{session_id[:8]}] Processing error: {e}", exc_info=True)
        socketio.emit('error', {'message': f'An error occurred: {str(e)}'}, to=session_id)

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

def analyze_pdf_images(file_path: str, ai_provider: str) -> List[str]:
    """从PDF中提取图片并使用AI进行分析"""
    if ai_provider != 'openai' or not OPENAI_API_KEY:
        logger.info("非OpenAI提供商或未配置Key，跳过图片分析")
        return []

    logger.info(f"🖼️ 开始从PDF中提取和分析图片: {os.path.basename(file_path)}")
    image_analyses = []
    try:
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            image_list = doc.get_page_images(page_num)
            for img_index, img in enumerate(image_list, 1):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # 将图片转换为可发送的格式
                image = Image.open(BytesIO(image_bytes))
                
                # 确保是RGB格式
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                    
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                # AI分析图片
                prompt = "请详细分析这张图片的内容。如果它是一个图表，请解读其数据、趋势和关键信息。如果是一个流程图，请解释其步骤和逻辑。如果是一张截图，请描述其界面和功能。"
                analysis = call_openai_api(prompt, image_base64=img_base64, max_tokens=500)
                
                if analysis:
                    analysis_text = f"## 来自PDF '{os.path.basename(file_path)}' (第{page_num+1}页, 图{img_index})的图表分析\n\n{analysis}\n"
                    image_analyses.append(analysis_text)
                    logger.info(f"✅ 成功分析了PDF '{os.path.basename(file_path)}' 中的一张图片")
                else:
                    logger.warning(f"⚠️ 未能分析PDF '{os.path.basename(file_path)}' 中的一张图片")

        doc.close()
    except Exception as e:
        logger.error(f"❌ 分析PDF图片时出错: {e}")
    
    return image_analyses

# 从文件加载提示词
def load_prompt(filename: str) -> str:
    """从文件加载提示词"""
    try:
        with open(os.path.join('prompts', filename), 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"提示词文件未找到: {filename}")
        return ""

correction_prompt_template = load_prompt('correction_prompt.txt')
summary_prompt_template = load_prompt('summary_prompt.txt')

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
    
    # 检查提示词文件
    logger.info("=" * 50)
    logger.info("📝 加载AI提示词:")
    logger.info(f"   - 转录修正: {'✅ 已加载' if correction_prompt_template else '❌ 加载失败'}")
    logger.info(f"   - 纪要生成: {'✅ 已加载' if summary_prompt_template else '❌ 加载失败'}")
    logger.info("=" * 50)
    
    socketio.run(app, debug=True, port=5000)
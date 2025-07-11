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
        if not self.start_time or self.total_steps == 0:
            return None
        
        # 使用预设的估算时长来计算
        total_estimated_duration = sum(step.get('estimated_duration', 0) for step in self.steps)
        if total_estimated_duration == 0:
            return None # 无法预估
            
        completed_duration = 0
        for step in self.steps:
            if step['status'] == 'completed':
                completed_duration += step.get('estimated_duration', 0)
        
        remaining_duration = total_estimated_duration - completed_duration
        
        # 对于正在处理的步骤，可以加入更精细的估算
        if 0 <= self.current_step < len(self.steps) and self.steps[self.current_step]['status'] == 'processing':
            current_step_progress = self.steps[self.current_step].get('progress_percentage', 0) / 100
            current_step_estimated_duration = self.steps[self.current_step].get('estimated_duration', 0)
            remaining_duration -= current_step_progress * current_step_estimated_duration

        return max(0, remaining_duration)
            
    def emit_progress(self):
        if self.session_id:
            # 计算总体进度
            total_estimated_duration = sum(step.get('estimated_duration', 0) for step in self.steps)
            completed_duration = 0
            
            for step in self.steps:
                if step['status'] == 'completed':
                    completed_duration += step.get('estimated_duration', 0)
                elif step['status'] == 'processing':
                    progress_percentage = step.get('progress_percentage', 0)
                    estimated_duration = step.get('estimated_duration', 0)
                    completed_duration += (progress_percentage / 100) * estimated_duration
            
            overall_percentage = (completed_duration / total_estimated_duration) * 100 if total_estimated_duration > 0 else 0
            
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
            logger.error("🚫 OpenAI library not available - please install: pip install openai")
            return None
            
        if not OPENAI_API_KEY:
            logger.error("🔑 OPENAI_API_KEY not configured in .env file")
            return None
            
        used_model = model or OPENAI_MODEL
        logger.info(f"🤖 调用OpenAI API - 模型: {used_model}, 基础URL: {OPENAI_BASE_URL}")
        logger.debug(f"📝 提示词长度: {len(prompt)} 字符")
        
        # 检查提示词长度
        if len(prompt) > 100000:  # 约100k字符上限
            logger.warning(f"⚠️ 提示词过长 ({len(prompt)} 字符)，可能影响处理效果")
            # 截断提示词
            prompt = prompt[:100000] + "\n\n[内容过长，已截断]"
        
        # 验证提示词内容
        if not prompt.strip():
            logger.error("❌ 提示词为空")
            return None
        
        client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": "你是专业的会议记录处理专家，擅长转录纠错和会议纪要生成。请用中文回复。"},
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
            temperature=0.3,  # 降低温度提高一致性
            timeout=120  # 增加超时时间
        )
        
        content = response.choices[0].message.content
        result = content.strip() if content else ""
        
        # 验证返回结果
        if not result:
            logger.error("❌ OpenAI API 返回空结果")
            return None
            
        if len(result) < 20:  # 结果太短可能有问题
            logger.warning(f"⚠️ OpenAI API 返回结果较短: {len(result)} 字符")
            
        # 检查是否包含错误信息
        if "Error" in result or "error" in result or "错误" in result:
            logger.warning(f"⚠️ OpenAI返回内容可能包含错误信息: {result[:200]}...")
            
        logger.info(f"✅ OpenAI API 调用成功 - 输出长度: {len(result)} 字符")
        return result
        
    except Exception as e:
        logger.error(f"❌ OpenAI API 调用失败: {str(e)}")
        # 记录更详细的错误信息
        response = getattr(e, 'response', None)
        if response:
            logger.error(f"HTTP状态码: {getattr(response, 'status_code', 'unknown')}")
        return None

@handle_errors(max_retries=3)
def call_ollama_api(prompt: str, model: Optional[str] = None, max_tokens: int = 2000) -> Optional[str]:
    """调用本地ollama模型"""
    try:
        used_model = model or OLLAMA_MODEL
        logger.info(f"🦙 调用Ollama API - 模型: {used_model}, 输入长度: {len(prompt)} 字符")
        
        # 验证提示词内容
        if not prompt.strip():
            logger.error("❌ 提示词为空")
            return None
        
        # 检查提示词长度
        if len(prompt) > 50000:  # Ollama通常支持的上下文更小
            logger.warning(f"⚠️ 提示词过长 ({len(prompt)} 字符)，截断处理")
            prompt = prompt[:50000] + "\n\n[内容过长，已截断]"
        
        # 检查Ollama服务是否可用
        try:
            health_response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if health_response.status_code != 200:
                logger.error(f"❌ Ollama服务不可用，状态码: {health_response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ 无法连接到Ollama服务: {e}")
            return None
        
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": used_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.3,  # 降低温度提高一致性
                    "top_p": 0.9,
                    "stop": ["<|im_end|>", "<|endoftext|>", "###", "---"]  # 添加停止词
                }
            },
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json().get('response', '').strip()
            
            # 验证返回结果
            if not result:
                logger.error("❌ Ollama API 返回空结果")
                return None
                
            if len(result) < 20:  # 结果太短可能有问题
                logger.warning(f"⚠️ Ollama API 返回结果较短: {len(result)} 字符")
                
            # 检查是否包含错误信息
            if "Error" in result or "error" in result or "错误" in result:
                logger.warning(f"⚠️ Ollama返回内容可能包含错误信息: {result[:200]}...")
                
            logger.info(f"✅ Ollama API 调用成功 - 输出长度: {len(result)} 字符")
            return result
        else:
            error_msg = response.json().get('error', '未知错误') if response.content else '无响应内容'
            logger.error(f"❌ Ollama API 调用失败，状态码: {response.status_code}, 错误: {error_msg}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Ollama API 调用异常: {str(e)}")
        return None

def chunk_transcript(transcript: str, max_chunk_size: int = 8000) -> List[str]:
    """将长转录文本分割成适合AI处理的块"""
    if len(transcript) <= max_chunk_size:
        return [transcript]
    
    logger.info(f"📄 开始分割长文本: {len(transcript)} 字符 -> 目标大小: {max_chunk_size}")
    
    # 尝试按段落分割
    paragraphs = transcript.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # 检查单个段落是否过长
        if len(paragraph) > max_chunk_size:
            # 如果当前chunk不为空，先保存
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            
            # 按句子分割过长的段落
            sentences = paragraph.split('。')
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
                    if current_chunk:
                        current_chunk += '。' + sentence
                    else:
                        current_chunk = sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk + '。')
                    current_chunk = sentence
        else:
            # 检查是否可以加入当前chunk
            if len(current_chunk) + len(paragraph) + 2 <= max_chunk_size:
                if current_chunk:
                    current_chunk += '\n\n' + paragraph
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph
    
    # 保存最后一个chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    # 如果仍有过长的块，进行强制分割
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chunk_size:
            final_chunks.append(chunk)
        else:
            # 强制按字符分割
            logger.warning(f"⚠️ 强制分割过长块: {len(chunk)} 字符")
            for i in range(0, len(chunk), max_chunk_size):
                sub_chunk = chunk[i:i + max_chunk_size]
                final_chunks.append(sub_chunk)
    
    logger.info(f"✅ 长文本分割完成: 原长度 {len(transcript)}, 分成 {len(final_chunks)} 块")
    for i, chunk in enumerate(final_chunks):
        logger.debug(f"   块 {i+1}: {len(chunk)} 字符")
    
    return final_chunks

def process_long_transcript_correction(transcript: str, context: str, reference_docs: List[str], 
                                     progress: ProcessingProgress, step_index: int, 
                                     ai_provider: str, model: str) -> str:
    """处理长转录文本的纠错"""
    # 如果文本不长，直接处理
    if len(transcript) <= 12000:
        # 直接调用底层函数，避免递归
        return _single_transcript_correction(transcript, context, reference_docs, 
                                           progress, step_index, ai_provider, model)
    
    logger.info(f"📄 检测到长文本 ({len(transcript)} 字符)，使用分块处理")
    progress.start_step(step_index)
    
    # 分块处理
    chunks = chunk_transcript(transcript, max_chunk_size=10000)
    corrected_chunks = []
    
    # 准备简化的参考文档（避免每次都发送完整文档）
    simplified_reference = reference_docs[:3] if reference_docs else []  # 只取前3个文档
    
    for i, chunk in enumerate(chunks):
        logger.info(f"📝 处理第 {i+1}/{len(chunks)} 块 ({len(chunk)} 字符)")
        progress.update_step_progress(step_index, 20 + (i * 60 // len(chunks)))
        
        # 为每个块创建简化的上下文
        chunk_context = f"这是会议转录的第{i+1}部分，共{len(chunks)}部分。{context}"
        
        # 处理单个块 - 使用内部函数避免递归
        try:
            corrected_chunk = _single_transcript_correction(
                chunk, chunk_context, simplified_reference, 
                ProcessingProgress(), 0,  # 使用临时进度对象
                ai_provider, model
            )
            corrected_chunks.append(corrected_chunk)
        except Exception as e:
            logger.error(f"❌ 处理第 {i+1} 块时出错: {str(e)}")
            corrected_chunks.append(chunk)  # 出错时使用原始块
    
    # 合并处理结果
    try:
        # 智能合并，保持段落结构
        final_result = ""
        for i, chunk in enumerate(corrected_chunks):
            if i > 0:
                # 检查是否需要添加分隔符
                if not chunk.startswith('\n') and not final_result.endswith('\n'):
                    final_result += '\n\n'
            final_result += chunk
        
        progress.update_step_progress(step_index, 90)
        
        # 基本的后处理
        final_result = final_result.strip()
        
        # 去除可能的重复内容
        lines = final_result.split('\n')
        seen_lines = set()
        unique_lines = []
        for line in lines:
            if line.strip() and line not in seen_lines:
                seen_lines.add(line)
                unique_lines.append(line)
            elif not line.strip():  # 保留空行
                unique_lines.append(line)
        
        final_result = '\n'.join(unique_lines)
        
    except Exception as e:
        logger.error(f"❌ 合并处理结果时出错: {str(e)}")
        final_result = '\n\n'.join(corrected_chunks)
    
    progress.complete_step(step_index)
    
    logger.info(f"✅ 长文本分块处理完成: {len(chunks)} 块 -> {len(final_result)} 字符")
    return final_result

def _single_transcript_correction(transcript: str, context: str, reference_docs: List[str], 
                                progress: ProcessingProgress, step_index: int, 
                                ai_provider: str, model: str) -> str:
    """单次转录纠错（内部函数，避免递归）"""
    progress.start_step(step_index)
    
    try:
        # 验证输入
        if not transcript or len(transcript.strip()) < 10:
            logger.warning("⚠️ 转录文本过短或为空，跳过AI纠错")
            progress.complete_step(step_index)
            return transcript
        
        # 验证提示词模板
        if not correction_prompt_template:
            logger.error("❌ 转录纠错提示词未加载，返回原始转录")
            progress.complete_step(step_index)
            return transcript
        
        # 准备参考内容
        reference_content = "\n\n---\n\n".join(reference_docs) if reference_docs else "无参考文档"
        context_content = context if context and context.strip() else "无特定背景信息"
        
        # 构建提示词
        try:
            prompt = correction_prompt_template.replace("{{transcript}}", transcript) \
                                               .replace("{{context}}", context_content) \
                                               .replace("{{reference_docs}}", reference_content)
        except Exception as e:
            logger.error(f"❌ 构建提示词失败: {str(e)}")
            progress.complete_step(step_index)
            return transcript
        
        # 验证提示词完整性
        if "{{" in prompt or "}}" in prompt:
            logger.warning("⚠️ 提示词变量替换不完整，可能影响效果")
        
        logger.info(f"📝 开始AI纠错 - 转录长度: {len(transcript)}, 提示词长度: {len(prompt)}")
        progress.update_step_progress(step_index, 30)
        
        corrected_text = None
        
        # 尝试调用AI服务
        if ai_provider == 'openai' and OPENAI_API_KEY:
            logger.info("🤖 使用OpenAI进行转录纠错")
            corrected_text = call_openai_api(prompt, model=model, max_tokens=6000)
        elif ai_provider == 'ollama':
            logger.info("🦙 使用Ollama进行转录纠错")
            corrected_text = call_ollama_api(prompt, model=model, max_tokens=6000)
        else:
            logger.error(f"❌ 未配置有效的AI服务: {ai_provider}")
        
        progress.update_step_progress(step_index, 80)
        
        # 验证AI返回结果
        if not corrected_text:
            logger.warning("⚠️ AI纠错失败，返回原始转录")
            corrected_text = transcript
        elif len(corrected_text.strip()) < len(transcript) * 0.2:  # 纠错后文本过短
            logger.warning("⚠️ AI纠错结果异常（过短），返回原始转录")
            corrected_text = transcript
        else:
            # 基本的质量检查
            improvement_ratio = len(corrected_text) / len(transcript)
            if improvement_ratio > 5:  # 纠错后文本过长可能有问题
                logger.warning(f"⚠️ AI纠错结果异常（过长 {improvement_ratio:.1f}x），返回原始转录")
                corrected_text = transcript
            else:
                # 检查是否仍然是有意义的文本
                if corrected_text.count('\n') > len(transcript) * 0.1:  # 换行过多
                    logger.warning("⚠️ AI纠错结果格式异常，返回原始转录")
                    corrected_text = transcript
                else:
                    logger.info(f"✅ AI纠错完成 - 原文: {len(transcript)} 字符, 纠错后: {len(corrected_text)} 字符")
        
        progress.complete_step(step_index)
        return corrected_text
        
    except Exception as e:
        logger.error(f"❌ 转录纠错过程出错: {str(e)}")
        progress.complete_step(step_index)
        return transcript  # 发生错误时返回原始转录

@handle_errors(max_retries=3)
def advanced_transcription_correction(transcript: str, context: str, reference_docs: List[str], progress: ProcessingProgress, step_index: int, ai_provider: str, model: str) -> str:
    """使用AI纠正转录文本，包含上下文和参考文档"""
    # 对于长文本，使用分块处理
    if len(transcript) > 15000:
        return process_long_transcript_correction(transcript, context, reference_docs, 
                                                progress, step_index, ai_provider, model)
    else:
        return _single_transcript_correction(transcript, context, reference_docs, 
                                           progress, step_index, ai_provider, model)

@handle_errors(max_retries=3)
def generate_meeting_summary(corrected_transcript: str, context: str, reference_docs: List[str], progress: ProcessingProgress, step_index: int, ai_provider: str, model: str) -> Dict[str, str]:
    """使用AI生成会议纪要"""
    progress.start_step(step_index)
    
    try:
        # 验证输入
        if not corrected_transcript or len(corrected_transcript.strip()) < 50:
            logger.warning("⚠️ 纠错后转录文本过短，无法生成有效纪要")
            progress.complete_step(step_index)
            return {"summary": create_fallback_summary(corrected_transcript, context)}
        
        # 验证提示词模板
        if not summary_prompt_template:
            logger.error("❌ 会议纪要提示词未加载，创建基础纪要")
            progress.complete_step(step_index)
            return {"summary": create_fallback_summary(corrected_transcript, context)}
        
        # 准备参考内容
        reference_content = "\n\n---\n\n".join(reference_docs) if reference_docs else "无参考文档"
        context_content = context if context and context.strip() else "无特定背景信息"
        
        # 构建提示词
        try:
            prompt = summary_prompt_template.replace("{{corrected_transcript}}", corrected_transcript) \
                                            .replace("{{context}}", context_content) \
                                            .replace("{{reference_docs}}", reference_content)
        except Exception as e:
            logger.error(f"❌ 构建纪要提示词失败: {str(e)}")
            progress.complete_step(step_index)
            return {"summary": create_fallback_summary(corrected_transcript, context)}
        
        # 验证提示词完整性
        if "{{" in prompt or "}}" in prompt:
            logger.warning("⚠️ 纪要提示词变量替换不完整，可能影响效果")
        
        logger.info(f"📋 开始生成会议纪要 - 转录长度: {len(corrected_transcript)}, 提示词长度: {len(prompt)}")
        progress.update_step_progress(step_index, 30)
        
        summary = None
        
        # 尝试调用AI服务
        if ai_provider == 'openai' and OPENAI_API_KEY:
            logger.info("🤖 使用OpenAI生成会议纪要")
            summary = call_openai_api(prompt, model=model, max_tokens=5000)
        elif ai_provider == 'ollama':
            logger.info("🦙 使用Ollama生成会议纪要")
            summary = call_ollama_api(prompt, model=model, max_tokens=5000)
        else:
            logger.error(f"❌ 未配置有效的AI服务: {ai_provider}")
        
        progress.update_step_progress(step_index, 80)
        
        # 验证AI返回结果
        if not summary:
            logger.warning("⚠️ AI纪要生成失败，创建基础纪要")
            summary = create_fallback_summary(corrected_transcript, context)
        elif len(summary.strip()) < 100:  # 纪要过短
            logger.warning("⚠️ AI生成的纪要过短，创建基础纪要")
            summary = create_fallback_summary(corrected_transcript, context)
        else:
            # 检查是否包含基本的纪要结构
            required_keywords = ["会议", "议题", "讨论", "决策", "行动", "概览", "要点"]
            if not any(keyword in summary for keyword in required_keywords):
                logger.warning("⚠️ AI生成的纪要缺少关键结构，创建基础纪要")
                summary = create_fallback_summary(corrected_transcript, context)
            else:
                # 检查是否包含markdown格式
                if "##" not in summary and "**" not in summary:
                    logger.warning("⚠️ AI生成的纪要缺少格式化，但内容可用")
                
                # 检查长度是否合理
                if len(summary) > len(corrected_transcript) * 2:
                    logger.warning("⚠️ AI生成的纪要过长，可能有重复内容")
                
                logger.info(f"✅ 会议纪要生成完成 - 长度: {len(summary)} 字符")
        
        progress.complete_step(step_index)
        return {"summary": summary}
        
    except Exception as e:
        logger.error(f"❌ 会议纪要生成过程出错: {str(e)}")
        progress.complete_step(step_index)
        return {"summary": create_fallback_summary(corrected_transcript, context)}

def create_fallback_summary(transcript: str, context: str) -> str:
    """创建基础会议纪要（当AI生成失败时使用）"""
    try:
        # 统计基本信息
        word_count = len(transcript)
        estimated_duration = word_count // 200  # 估算会议时长（按200字/分钟）
        
        # 简单的关键词提取
        keywords = []
        common_meeting_terms = ["项目", "任务", "计划", "目标", "问题", "方案", "建议", "决定", "安排", "时间", "讨论", "会议", "团队", "进度", "完成", "需要", "考虑", "确认"]
        for term in common_meeting_terms:
            if term in transcript:
                keywords.append(term)
        
        # 尝试提取可能的参与者
        participants = []
        import re
        # 简单的姓名模式匹配
        name_patterns = [
            r'[张李王刘陈杨赵黄周吴徐孙朱马胡郭林何高罗郑梁谢][A-Za-z\u4e00-\u9fff]{1,3}',
            r'[A-Z][a-z]{2,8}',
        ]
        for pattern in name_patterns:
            matches = re.findall(pattern, transcript)
            participants.extend(matches[:5])  # 最多5个
        
        # 去重
        participants = list(set(participants))
        
        # 提取可能的时间信息
        time_patterns = [
            r'\d{1,2}[月]\d{1,2}[日]',
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',
            r'[本上下][周月年]',
            r'[明今昨][天日]',
            r'\d{1,2}[点时]',
        ]
        time_mentions = []
        for pattern in time_patterns:
            matches = re.findall(pattern, transcript)
            time_mentions.extend(matches[:3])  # 最多3个
        
        # 构建基础纪要
        summary = f"""# 会议纪要

## 会议概览
- **时间**: {', '.join(time_mentions) if time_mentions else '待补充'}
- **预计时长**: 约 {estimated_duration} 分钟
- **主要内容**: {context if context else '工作会议讨论'}
- **参与人员**: {', '.join(participants) if participants else '待补充'}

## 讨论要点
{transcript[:800]}{'...' if len(transcript) > 800 else ''}

## 关键信息
- **涉及关键词**: {', '.join(keywords[:10]) if keywords else '待分析'}
- **内容长度**: {word_count} 字符
- **处理状态**: 自动生成基础版本

## 后续跟进
- 本纪要为系统自动生成的基础版本
- 建议人工进一步完善和补充
- 如需详细分析，请重新尝试AI生成功能

---
*本纪要由AI会议助手自动生成于 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*"""
        
        logger.info(f"✅ 创建基础纪要完成 - 长度: {len(summary)} 字符")
        return summary
        
    except Exception as e:
        logger.error(f"❌ 创建基础纪要失败: {str(e)}")
        return f"""# 会议纪要

## 会议内容
{transcript[:1000]}{'...' if len(transcript) > 1000 else ''}

## 处理说明
- 本纪要为原始转录内容
- 由于系统处理异常，未能生成结构化纪要
- 建议人工整理或重新处理

---
*生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*"""

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
    
    # 动态预估初始时长
    audio_duration_estimate = estimate_audio_duration(mov_path) # 先对视频估算
    transcription_estimate = audio_duration_estimate * 0.2 if audio_duration_estimate > 0 else 60 # 估算为音频时长的20%
    
    try:
        # 定义处理步骤和预估时长(秒)
        progress.add_step("video_validation", "验证视频文件", 2)
        progress.add_step("audio_extraction", "提取音频", audio_duration_estimate * 0.05 + 5)
        progress.add_step("speech_transcription", "语音转文字", transcription_estimate)
        progress.add_step("document_processing", "处理参考文档", 10)
        progress.add_step("image_analysis", "分析文档图片", 30)
        progress.add_step("ai_correction", "AI转录纠错", 45)
        progress.add_step("summary_generation", "生成会议纪要", 35)
        progress.add_step("file_generation", "生成下载文件", 5)

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
        
        # 更新转录步骤的预估时长
        progress.steps[2]['estimated_duration'] = audio_duration * 0.2 
        progress.emit_progress() # 重新广播一下进度
        
        progress.complete_step(1)

        # Step 3: Transcribe Speech
        progress.start_step(2)
        # 使用更详细的参数调用，并尝试启用词级时间戳
        result = whisper_model.transcribe(audio_path, language=None, word_timestamps=True, verbose=False)
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
        image_analysis_results = []
        if ai_provider == 'openai':
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
        prompt_path = os.path.join('prompts', filename)
        if not os.path.exists(prompt_path):
            logger.error(f"❌ 提示词文件不存在: {prompt_path}")
            return ""
            
        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        if not content:
            logger.error(f"❌ 提示词文件为空: {filename}")
            return ""
            
        if len(content) < 50:  # 提示词太短可能有问题
            logger.warning(f"⚠️ 提示词文件过短: {filename} ({len(content)} 字符)")
            
        # 检查是否包含必要的占位符
        if filename == 'correction_prompt.txt':
            required_placeholders = ['{{transcript}}', '{{context}}', '{{reference_docs}}']
        elif filename == 'summary_prompt.txt':
            required_placeholders = ['{{corrected_transcript}}', '{{context}}', '{{reference_docs}}']
        else:
            required_placeholders = []
            
        missing_placeholders = []
        for placeholder in required_placeholders:
            if placeholder not in content:
                missing_placeholders.append(placeholder)
        
        if missing_placeholders:
            logger.error(f"❌ 提示词文件缺少必要占位符: {filename} - 缺少: {missing_placeholders}")
            return ""
            
        logger.info(f"✅ 提示词加载成功: {filename} ({len(content)} 字符)")
        return content
        
    except Exception as e:
        logger.error(f"❌ 加载提示词文件失败 {filename}: {str(e)}")
        return ""

# 加载提示词模板
correction_prompt_template = load_prompt('correction_prompt.txt')
summary_prompt_template = load_prompt('summary_prompt.txt')

# 验证提示词是否加载成功
if not correction_prompt_template:
    logger.error("❌ 转录纠错提示词加载失败，AI纠错功能将不可用")
if not summary_prompt_template:
    logger.error("❌ 纪要生成提示词加载失败，AI纪要功能将不可用")

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
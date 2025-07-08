from flask import Flask, render_template, request, send_file, redirect, url_for, jsonify
from flask_socketio import SocketIO, emit
import os
import subprocess
import whisper
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
from typing import cast, Optional, Dict, List
import logging
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import uuid
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
socketio = SocketIO(app, cors_allowed_origins="*")

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化模型
logger.info("正在初始化Whisper模型...")
whisper_model = whisper.load_model('medium')  # 升级为medium模型
logger.info("Whisper模型初始化完成")

# OpenAI配置
if openai:
    openai.api_key = os.getenv('OPENAI_API_KEY')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')

class ProcessingProgress:
    def __init__(self):
        self.steps = []
        self.current_step = 0
        self.total_steps = 0
        self.session_id: Optional[str] = None
        
    def add_step(self, step_name: str, description: str = ""):
        self.steps.append({
            'name': step_name,
            'description': description,
            'status': 'pending',
            'start_time': None,
            'end_time': None
        })
        self.total_steps = len(self.steps)
        
    def start_step(self, step_index: int):
        if 0 <= step_index < len(self.steps):
            self.current_step = step_index
            self.steps[step_index]['status'] = 'processing'
            self.steps[step_index]['start_time'] = datetime.now()
            step_name = self.steps[step_index]['name']
            logger.info(f"🚀 [{self.session_id[:8] if self.session_id else 'LOCAL'}] 开始: {step_name}")
            self.emit_progress()
            
    def complete_step(self, step_index: int):
        if 0 <= step_index < len(self.steps):
            self.steps[step_index]['status'] = 'completed'
            self.steps[step_index]['end_time'] = datetime.now()
            step_name = self.steps[step_index]['name']
            duration = (self.steps[step_index]['end_time'] - self.steps[step_index]['start_time']).total_seconds()
            logger.info(f"✅ [{self.session_id[:8] if self.session_id else 'LOCAL'}] 完成: {step_name} (耗时 {duration:.1f}秒)")
            self.emit_progress()
            
    def emit_progress(self):
        if self.session_id:
            # 处理datetime序列化问题
            serialized_steps = []
            for step in self.steps:
                serialized_step = {
                    'name': step['name'],
                    'description': step['description'],
                    'status': step['status'],
                    'start_time': step['start_time'].isoformat() if step['start_time'] else None,
                    'end_time': step['end_time'].isoformat() if step['end_time'] else None
                }
                serialized_steps.append(serialized_step)
            
            progress_data = {
                'current_step': self.current_step,
                'total_steps': self.total_steps,
                'steps': serialized_steps,
                'percentage': (self.current_step / self.total_steps * 100) if self.total_steps > 0 else 0
            }
            
            # 命令行监控输出
            current_step_name = self.steps[self.current_step]['name'] if self.current_step < len(self.steps) else "完成"
            logger.info(f"🔄 [{self.session_id[:8]}] 进度: {progress_data['percentage']:.1f}% - {current_step_name}")
            
            socketio.emit('progress_update', progress_data, to=self.session_id)

def call_openai_api(prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 2000) -> Optional[str]:
    """调用OpenAI API"""
    try:
        if not openai or not hasattr(openai, 'api_key'):
            logger.warning("🚫 OpenAI not available")
            return None
            
        if not openai.api_key:
            logger.warning("🔑 OpenAI API key not configured")
            return None
            
        logger.info(f"🤖 调用OpenAI API - 模型: {model}, 输入长度: {len(prompt)} 字符")
        
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional meeting assistant specialized in transcription correction and meeting summary generation."},
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

def call_ollama_api(prompt: str, model: str = "llama2", max_tokens: int = 2000) -> Optional[str]:
    """调用本地ollama模型"""
    try:
        logger.info(f"🦙 调用Ollama API - 模型: {model}, 输入长度: {len(prompt)} 字符")
        
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
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

def correct_transcription(transcript: str, reference_docs: List[str], progress: ProcessingProgress, step_index: int) -> str:
    """使用AI模型纠正转录文本"""
    progress.start_step(step_index)
    
    # 如果没有参考文档，直接返回原始转录
    if not reference_docs:
        logger.info("📝 没有参考文档，跳过AI纠错")
        progress.complete_step(step_index)
        return transcript
    
    # 构建提示词
    prompt = f"""
请根据以下参考文档，对会议转录文本进行纠正和改进。主要关注：
1. 修正语音转文字的错误
2. 改进语法和表达
3. 补充专业术语
4. 保持原意不变

参考文档：
{chr(10).join(reference_docs)}

原始转录文本：
{transcript}

请提供纠正后的转录文本：
"""
    
    # 优先使用OpenAI，如果失败则使用ollama
    corrected_text = call_openai_api(prompt)
    if not corrected_text:
        logger.info("🔄 OpenAI不可用，尝试Ollama")
        corrected_text = call_ollama_api(prompt)
    
    if not corrected_text:
        logger.warning("⚠️ 所有AI服务不可用，返回原始文本")
        corrected_text = transcript
    
    progress.complete_step(step_index)
    return corrected_text

def generate_meeting_summary(corrected_transcript: str, reference_docs: List[str], progress: ProcessingProgress, step_index: int) -> Dict[str, str]:
    """生成详细的会议纪要"""
    progress.start_step(step_index)
    
    prompt = f"""
请根据以下会议转录文本和参考文档，生成详细的会议纪要。请按以下格式输出：

## 会议概要
[简要概述会议目的和主要议题]

## 会议重点
[列出会议的关键讨论点和重要信息]

## 工作决议
[明确的决策和决定事项]

## 当前进度
[已完成的工作和当前状态]

## 后续计划
[未来的工作计划和时间安排]

## 待办事项
[具体的行动项目，包括负责人和截止日期]

参考文档：
{chr(10).join(reference_docs) if reference_docs else "无参考文档"}

会议转录文本：
{corrected_transcript}

请生成详细的会议纪要：
"""
    
    # 调用AI生成纪要
    summary = call_openai_api(prompt, max_tokens=3000)
    if not summary:
        logger.info("🔄 OpenAI不可用，尝试Ollama生成纪要")
        summary = call_ollama_api(prompt, max_tokens=3000)
    
    if not summary:
        logger.warning("⚠️ 所有AI服务不可用，使用基础格式生成纪要")
        # 创建基础的会议纪要
        transcript_preview = corrected_transcript[:800] + "..." if len(corrected_transcript) > 800 else corrected_transcript
        
        summary = f"""## 会议概要
本次会议进行了相关工作讨论和安排。

## 会议重点
{transcript_preview}

## 工作决议
根据会议讨论内容，形成相关决议。

## 当前进度
会议中汇报了当前工作进展。

## 后续计划
制定了后续工作计划和安排。

## 待办事项
会议确定了相关待办事项。

_注：此纪要为基础格式，建议配置AI服务以获得更详细的会议纪要。_"""
    
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

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")  # type: ignore

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")  # type: ignore

@app.route('/process', methods=['POST'])
def process_meeting():
    session_id = str(uuid.uuid4())
    
    try:
        # 获取表单数据
        video = request.files.get('video')
        docs = request.files.getlist('docs')
        text_input = request.form.get('docsText', '')
        
        if not video or not video.filename:
            return jsonify({"error": "请选择视频文件"}), 400
        
        # 在主线程中保存文件，避免异步线程中的文件访问问题
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
        
        # 在单独的线程中处理，避免阻塞
        thread = threading.Thread(target=process_meeting_async, args=(session_id, mov_path, video_filename, doc_files, text_input))
        thread.start()
        
        return jsonify({"session_id": session_id, "status": "processing"})
    
    except Exception as e:
        logger.error(f"❌ 处理请求错误: {str(e)}")
        return jsonify({"error": f"处理请求错误: {str(e)}"}), 500

def process_meeting_async(session_id: str, mov_path: str, video_filename: str, doc_files: list, text_input: str):
    """异步处理会议"""
    progress = ProcessingProgress()
    progress.session_id = session_id
    
    logger.info(f"🎬 [{session_id[:8]}] 开始处理会议 - 视频: {video_filename}")
    start_time = time.time()  # 记录开始时间
    
    try:
        # 设置处理步骤
        progress.add_step("video_upload", "上传视频文件")
        progress.add_step("audio_extraction", "提取音频")
        progress.add_step("transcription", "语音转文字")
        progress.add_step("doc_processing", "处理参考文档")
        progress.add_step("text_correction", "纠正转录文本")
        progress.add_step("summary_generation", "生成会议纪要")
        progress.add_step("report_creation", "创建报告文件")
        
        # 步骤1：处理视频文件（已经在主线程中完成）
        progress.start_step(0)
        base_name = os.path.splitext(video_filename)[0]
        
        # 获取文件大小
        if os.path.exists(mov_path):
            file_size = os.path.getsize(mov_path)
            logger.info(f"📁 [{session_id[:8]}] 视频文件大小: {file_size / (1024*1024):.1f} MB")
        else:
            logger.error(f"❌ [{session_id[:8]}] 视频文件不存在: {mov_path}")
            socketio.emit('error', {'message': '视频文件不存在'}, to=session_id)
            return
        progress.complete_step(0)
        
        # 步骤2：提取音频
        progress.start_step(1)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{base_name}_audio.wav')
        subprocess.run(['ffmpeg', '-i', mov_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_path], 
                      check=True, capture_output=True)
        progress.complete_step(1)
        
        # 步骤3：语音转文字
        progress.start_step(2)
        result = whisper_model.transcribe(audio_path, language=None)  # 自动检测语言
        transcript = str(result["text"])
        
        # 检测语言和转录长度
        detected_lang = result.get("language", "unknown")
        logger.info(f"🎤 [{session_id[:8]}] 转录完成 - 语言: {detected_lang}, 字符数: {len(transcript)}")
        progress.complete_step(2)
        
        # 步骤4：处理参考文档
        progress.start_step(3)
        doc_texts = []
        
        # 处理文本输入
        if text_input and text_input.strip():
            doc_texts.append(text_input.strip())
            logger.info(f"📝 [{session_id[:8]}] 添加文本输入: {len(text_input.strip())} 字符")
        
        # 处理上传的文档
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
        
        # 步骤5：纠正转录文本
        corrected_transcript = correct_transcription(transcript, doc_texts, progress, 4)
        
        # 步骤6：生成会议纪要
        meeting_summary = generate_meeting_summary(corrected_transcript, doc_texts, progress, 5)
        
        # 步骤7：创建报告文件
        progress.start_step(6)
        
        # 生成纠正后的转录文件
        corrected_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{base_name}_corrected_transcript.md')
        with open(corrected_path, 'w', encoding='utf-8') as f:
            f.write(f'# 纠正后的会议转录\n\n')
            f.write(f'**会议日期**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
            f.write(f'## 原始转录\n{transcript}\n\n')
            f.write(f'## 纠正后转录\n{corrected_transcript}\n\n')
        
        # 生成完整的会议报告
        report_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{base_name}_meeting_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f'# 会议报告\n\n')
            f.write(f'**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
            f.write(f'## 纠正后的转录文本\n{corrected_transcript}\n\n')
            f.write(f'{meeting_summary["summary"]}\n\n')
        
        progress.complete_step(6)
        
        # 通知处理完成
        total_time = time.time() - start_time
        logger.info(f"🎉 [{session_id[:8]}] 处理完成! 总耗时: {total_time:.1f}秒")
        logger.info(f"📄 [{session_id[:8]}] 生成文件: {os.path.basename(corrected_path)}, {os.path.basename(report_path)}")
        
        socketio.emit('processing_complete', {
            'corrected_transcript_path': corrected_path,
            'report_path': report_path
        }, to=session_id)
        
    except Exception as e:
        logger.error(f"💥 [{session_id[:8]}] 处理错误: {str(e)}")
        socketio.emit('error', {'message': f'处理错误: {str(e)}'}, to=session_id)

@app.route('/download/<filename>')
def download_file(filename):
    """下载生成的文件"""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "文件不存在", 404

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)
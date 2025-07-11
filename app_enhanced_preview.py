
# 在原有app.py基础上添加知识库功能
import os
import sys

# 添加知识库模块路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from knowledge_base.core import EnhancedKnowledgeBase
    KB_AVAILABLE = True
    # 初始化全局知识库实例
    knowledge_base = EnhancedKnowledgeBase()
    print("✅ 知识库模块加载成功")
except ImportError as e:
    KB_AVAILABLE = False
    knowledge_base = None
    print(f"⚠️ 知识库模块不可用: {e}")

# 在现有的process_meeting_async函数中添加知识库功能
def process_meeting_async_with_kb(session_id: str, mov_path: str, video_filename: str, 
                                 doc_files: list, context_input: str, ai_provider: str, 
                                 model: str, organization: str = None):
    """带知识库的会议处理函数"""
    global knowledge_base
    
    # 原有的处理逻辑...
    # (视频转音频、语音转录等步骤保持不变)
    
    # 新增：如果有组织信息且知识库可用，生成智能上下文
    enhanced_context = context_input
    
    if KB_AVAILABLE and knowledge_base and organization:
        try:
            # 假设这里已经有了transcript变量
            intelligent_context = knowledge_base.get_intelligent_context(
                meeting_content=transcript[:500],  # 使用转录的前500字符
                organization=organization
            )
            
            if intelligent_context and intelligent_context != "未找到相关的组织背景信息。":
                enhanced_context = f"{context_input}\n\n## 智能背景信息\n{intelligent_context}"
                logger.info("✅ 生成智能背景上下文成功")
        except Exception as e:
            logger.error(f"生成智能上下文失败: {e}")
    
    # 使用增强的上下文继续原有流程
    # corrected_transcript = advanced_transcription_correction(...)
    # meeting_summary = generate_meeting_summary(...)
    
    return enhanced_context

# 新增路由：知识库管理
@app.route('/kb/upload', methods=['POST'])
def upload_to_kb():
    """上传文档到知识库"""
    if not KB_AVAILABLE:
        return jsonify({"error": "知识库功能不可用"}), 500
    
    try:
        files = request.files.getlist('files')
        organization = request.form.get('organization', 'default')
        doc_type = request.form.get('doc_type', 'reference')
        
        uploaded_count = 0
        errors = []
        
        for file in files:
            if file and file.filename:
                # 保存临时文件
                temp_path = os.path.join(tempfile.gettempdir(), file.filename)
                file.save(temp_path)
                
                # 添加到知识库
                success = knowledge_base.add_document(
                    file_path=temp_path,
                    organization=organization,
                    doc_type=doc_type,
                    title=file.filename
                )
                
                if success:
                    uploaded_count += 1
                else:
                    errors.append(f"上传失败: {file.filename}")
                
                # 清理临时文件
                try:
                    os.unlink(temp_path)
                except:
                    pass
        
        return jsonify({
            "success": True,
            "uploaded_count": uploaded_count,
            "total_files": len(files),
            "errors": errors
        })
        
    except Exception as e:
        logger.error(f"上传到知识库失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/kb/stats')
def get_kb_stats():
    """获取知识库统计"""
    if not KB_AVAILABLE:
        return jsonify({"error": "知识库功能不可用"}), 500
    
    try:
        stats = knowledge_base.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

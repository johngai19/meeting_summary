#!/usr/bin/env python3
"""
Test script to verify AI correction and summary fixes
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import (
    load_prompt, 
    correction_prompt_template, 
    summary_prompt_template,
    chunk_transcript,
    create_fallback_summary
)

def test_prompt_loading():
    """测试提示词加载"""
    print("🧪 测试提示词加载...")
    
    # 测试纠错提示词
    if correction_prompt_template:
        print(f"✅ 纠错提示词加载成功 ({len(correction_prompt_template)} 字符)")
        if "{{transcript}}" in correction_prompt_template:
            print("✅ 纠错提示词包含必要的占位符")
        else:
            print("❌ 纠错提示词缺少占位符")
    else:
        print("❌ 纠错提示词加载失败")
    
    # 测试纪要提示词
    if summary_prompt_template:
        print(f"✅ 纪要提示词加载成功 ({len(summary_prompt_template)} 字符)")
        if "{{corrected_transcript}}" in summary_prompt_template:
            print("✅ 纪要提示词包含必要的占位符")
        else:
            print("❌ 纪要提示词缺少占位符")
    else:
        print("❌ 纪要提示词加载失败")

def test_chunking():
    """测试文本分块功能"""
    print("\n🧪 测试文本分块功能...")
    
    # 短文本测试
    short_text = "这是一个短文本测试。"
    chunks = chunk_transcript(short_text)
    print(f"短文本分块: {len(chunks)} 块")
    assert len(chunks) == 1, "短文本应该不分块"
    
    # 长文本测试
    long_text = "这是第一段。" * 1000 + "\n\n这是第二段。" * 1000
    chunks = chunk_transcript(long_text, max_chunk_size=5000)
    print(f"长文本分块: 原长度 {len(long_text)}, 分成 {len(chunks)} 块")
    assert len(chunks) > 1, "长文本应该被分块"
    
    # 验证块的大小
    for i, chunk in enumerate(chunks):
        if len(chunk) > 5000:
            print(f"⚠️ 第{i+1}块过长: {len(chunk)} 字符")
        else:
            print(f"✅ 第{i+1}块大小合适: {len(chunk)} 字符")

def test_fallback_summary():
    """测试后备纪要生成"""
    print("\n🧪 测试后备纪要生成...")
    
    test_transcript = """
    今天我们讨论了项目进展。主要议题包括：
    1. 任务分配问题
    2. 时间计划调整
    3. 资源需求分析
    
    决定下周开始新的开发阶段。
    """
    
    summary = create_fallback_summary(test_transcript, "项目会议")
    print(f"✅ 后备纪要生成成功 ({len(summary)} 字符)")
    
    # 检查纪要是否包含基本结构
    required_sections = ["会议概览", "讨论要点", "关键信息"]
    for section in required_sections:
        if section in summary:
            print(f"✅ 包含 {section} 部分")
        else:
            print(f"❌ 缺少 {section} 部分")

def test_prompt_replacement():
    """测试提示词变量替换"""
    print("\n🧪 测试提示词变量替换...")
    
    if correction_prompt_template:
        test_transcript = "这是测试转录文本"
        test_context = "测试背景"
        test_refs = "测试参考文档"
        
        # 替换变量
        prompt = correction_prompt_template.replace("{{transcript}}", test_transcript) \
                                           .replace("{{context}}", test_context) \
                                           .replace("{{reference_docs}}", test_refs)
        
        # 检查是否还有未替换的变量
        if "{{" in prompt:
            print("⚠️ 提示词中仍有未替换的变量")
        else:
            print("✅ 所有变量替换成功")
            
        # 检查替换后的内容
        if test_transcript in prompt:
            print("✅ 转录文本已插入提示词")
        else:
            print("❌ 转录文本插入失败")

if __name__ == "__main__":
    print("🚀 开始测试AI修复功能...")
    
    try:
        test_prompt_loading()
        test_chunking() 
        test_fallback_summary()
        test_prompt_replacement()
        
        print("\n🎉 所有测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

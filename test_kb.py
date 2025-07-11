#!/usr/bin/env python3
"""
知识库测试脚本
"""
import os
import sys
import tempfile
import shutil

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def create_test_documents():
    """创建测试文档"""
    test_docs = []
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="kb_test_")
    
    # 测试文档1：会议纪要
    meeting_content = """# 项目启动会议纪要

**时间**: 2024-01-15 14:00
**参与人员**: 张经理、李工程师、王设计师

## 讨论内容
1. 项目目标确认
2. 技术方案讨论
3. 时间节点安排

## 决策事项
- 采用前后端分离架构
- 使用Python Flask后端
- 预计开发周期3个月

## 行动计划
- 李工程师负责后端开发
- 王设计师负责UI设计
- 下周一提交详细设计方案
"""
    
    meeting_file = os.path.join(temp_dir, "meeting_20240115.md")
    with open(meeting_file, 'w', encoding='utf-8') as f:
        f.write(meeting_content)
    test_docs.append(('meeting_minutes', meeting_file))
    
    # 测试文档2：技术标准
    standard_content = """# 开发规范文档

## 代码规范
1. 使用Python PEP8编码标准
2. 函数命名采用snake_case
3. 类命名采用PascalCase

## 文档规范
- 所有函数必须有docstring
- 重要逻辑需要添加注释
- README文档必须包含安装和使用说明

## 测试规范
- 单元测试覆盖率不低于80%
- 关键功能必须有集成测试
"""
    
    standard_file = os.path.join(temp_dir, "dev_standards.md")
    with open(standard_file, 'w', encoding='utf-8') as f:
        f.write(standard_content)
    test_docs.append(('standards', standard_file))
    
    # 测试文档3：工作计划
    plan_content = """# Q1工作计划

## 1月目标
- 完成需求分析
- 搭建开发环境
- 制定技术方案

## 2月目标  
- 完成核心功能开发
- 进行单元测试
- 完成前端界面

## 3月目标
- 系统集成测试
- 性能优化
- 用户验收测试

## 风险控制
- 技术难点提前攻关
- 定期进度review
- 及时调整计划
"""
    
    plan_file = os.path.join(temp_dir, "q1_plan.md")
    with open(plan_file, 'w', encoding='utf-8') as f:
        f.write(plan_content)
    test_docs.append(('plans', plan_file))
    
    print(f"测试文档创建在: {temp_dir}")
    return test_docs, temp_dir

def test_basic_functionality():
    """测试基本功能"""
    print("=" * 50)
    print("开始测试知识库基本功能")
    print("=" * 50)
    
    temp_dir = None
    try:
        # 导入知识库
        from knowledge_base.core import EnhancedKnowledgeBase
        
        print("✅ 成功导入知识库模块")
        
        # 初始化知识库（测试模式）
        print("\n正在初始化知识库（测试模式）...")
        kb = EnhancedKnowledgeBase(use_simple_embeddings=True)
        print("✅ 知识库初始化成功")
        
        # 创建测试文档
        print("\n正在创建测试文档...")
        test_docs, temp_dir = create_test_documents()
        print("✅ 测试文档创建成功")
        
        # 添加文档到知识库
        print("\n正在添加文档到知识库...")
        org_name = "测试组织"
        
        success_count = 0
        for doc_type, file_path in test_docs:
            success = kb.add_document(
                file_path=file_path,
                organization=org_name,
                doc_type=doc_type,
                title=os.path.basename(file_path)
            )
            if success:
                print(f"✅ 成功添加: {os.path.basename(file_path)}")
                success_count += 1
            else:
                print(f"❌ 添加失败: {os.path.basename(file_path)}")
        
        if success_count == 0:
            print("❌ 没有成功添加任何文档")
            return False
        
        # 测试搜索功能
        print("\n正在测试搜索功能...")
        test_query = "项目开发计划和技术方案"
        results = kb.search_relevant_documents(
            query=test_query,
            organization=org_name,
            k=3
        )
        
        print(f"搜索查询: {test_query}")
        print(f"找到 {len(results)} 个相关文档:")
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get('source_file', '未知')
            doc_type = doc.metadata.get('doc_type', '未知')
            print(f"  {i}. {source} ({doc_type}) - 相似度: {score:.3f}")
            print(f"     内容预览: {doc.page_content[:100]}...")
        
        # 测试智能上下文生成
        print("\n正在测试智能上下文生成...")
        meeting_content = "我们需要讨论项目的技术架构和开发规范"
        context = kb.get_intelligent_context(
            meeting_content=meeting_content,
            organization=org_name
        )
        
        print("生成的智能上下文:")
        print(context[:500] + "..." if len(context) > 500 else context)
        
        # 显示统计信息
        print("\n知识库统计信息:")
        stats = kb.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n🎉 知识库测试完成!")
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保已安装所需依赖:")
        print("  pip install langchain langchain-community langchain-openai langchain-text-splitters chromadb")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理临时文件
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"清理临时目录: {temp_dir}")
            except:
                pass

if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\n✅ 所有测试通过! 知识库功能正常")
    else:
        print("\n❌ 测试失败! 请检查错误信息")
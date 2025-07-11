import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import hashlib

# 使用新的LangChain v0.2导入方式
try:
    # 新的导入方式（LangChain v0.2+）
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    import chromadb
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"警告: 新版LangChain依赖未安装 - {e}")
    try:
        # 备用：旧版导入方式
        from langchain.vectorstores import Chroma
        from langchain.embeddings.openai import OpenAIEmbeddings
        from langchain.document_loaders import PyPDFLoader, TextLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.schema import Document
        import chromadb
        LANGCHAIN_AVAILABLE = True
        print("使用旧版LangChain导入")
    except ImportError:
        print("请安装LangChain相关依赖")
        LANGCHAIN_AVAILABLE = False

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """文档元数据"""
    doc_id: str
    title: str
    organization: str
    doc_type: str  # meeting_minutes, standards, plans, templates, reference
    upload_time: str
    file_path: str
    file_size: int
    tags: List[str]
    priority: float = 1.0
    usage_count: int = 0
    last_accessed: Optional[str] = None

class SimpleEmbeddings(Embeddings):
    """简单的嵌入实现，用于测试"""
    def __init__(self):
        self.dimension = 384
        import random
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """生成随机向量用于测试"""
        import random
        return [[random.random() for _ in range(self.dimension)] for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """生成随机向量用于测试"""
        import random
        return [random.random() for _ in range(self.dimension)]

class EnhancedKnowledgeBase:
    """增强型知识库管理器"""
    
    def __init__(self, config: Optional[Dict] = None, use_simple_embeddings: bool = False):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("请先安装LangChain相关依赖")
            
        # 设置默认配置
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        # 确保目录存在
        os.makedirs(self.config['vector_db_path'], exist_ok=True)
        os.makedirs(self.config['persist_directory'], exist_ok=True)
        os.makedirs(self.config['organizations_path'], exist_ok=True)
        
        # 初始化嵌入模型
        self.use_simple_embeddings = use_simple_embeddings
        self.embeddings = self._init_embeddings()
        
        # 初始化向量存储
        self.vectorstore = self._init_vectorstore()
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config['chunk_size'],
            chunk_overlap=self.config['chunk_overlap'],
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        # 文档元数据存储
        self.metadata_file = os.path.join(
            self.config['vector_db_path'], "documents_metadata.json"
        )
        self.documents_metadata = self._load_metadata()
        
        logger.info("知识库系统初始化完成")
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'vector_db_path': "./knowledge_base/vector_db",
            'persist_directory': "./knowledge_base/chroma_persist",
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'max_documents_per_query': 5,
            'similarity_threshold': 0.7,
            'organizations_path': "./knowledge_base/organizations",
            'embedding_model': "text-embedding-3-small",  # 更新为可用的模型
            'document_type_weights': {
                'meeting_minutes': 0.8,
                'standards': 1.0,
                'plans': 0.9,
                'templates': 0.6,
                'reference': 0.7
            }
        }
    
    def _init_embeddings(self):
        """初始化嵌入模型"""
        if self.use_simple_embeddings:
            logger.info("使用简单嵌入模型（测试用）")
            return SimpleEmbeddings()
        
        openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if openai_api_key:
            try:
                embeddings = OpenAIEmbeddings(
                    model=self.config['embedding_model']
                )
                # 测试连接
                test_result = embeddings.embed_query("test")
                logger.info(f"使用OpenAI嵌入模型: {self.config['embedding_model']}")
                return embeddings
            except Exception as e:
                logger.warning(f"OpenAI嵌入初始化失败: {e}")
        
        # 降级到简单模型
        logger.info("降级到简单嵌入模型")
        return SimpleEmbeddings()
    
    def _init_vectorstore(self):
        """初始化向量存储"""
        try:
            vectorstore = Chroma(
                persist_directory=self.config['persist_directory'],
                embedding_function=self.embeddings,
                collection_metadata={"hnsw:space": "cosine"}
            )
            logger.info("Chroma向量数据库初始化成功")
            return vectorstore
        except Exception as e:
            logger.error(f"Chroma初始化失败: {e}")
            raise
    
    def _load_metadata(self) -> Dict[str, DocumentMetadata]:
        """加载文档元数据"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return {
                        doc_id: DocumentMetadata(**metadata) 
                        for doc_id, metadata in data.items()
                    }
            except Exception as e:
                logger.error(f"加载元数据失败: {e}")
        return {}
    
    def _save_metadata(self):
        """保存文档元数据"""
        try:
            data = {
                doc_id: asdict(metadata) 
                for doc_id, metadata in self.documents_metadata.items()
            }
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存元数据失败: {e}")
    
    def add_document(self, 
                    file_path: str, 
                    organization: str,
                    doc_type: str,
                    title: Optional[str] = None,
                    tags: Optional[List[str]] = None) -> bool:
        """添加单个文档到知识库"""
        try:
            if not os.path.exists(file_path):
                logger.error(f"文件不存在: {file_path}")
                return False
            
            # 生成文档ID
            doc_id = self._generate_doc_id(file_path, organization)
            
            # 检查是否已存在
            if doc_id in self.documents_metadata:
                logger.info(f"文档已存在: {file_path}")
                return True
            
            # 加载文档
            documents = self._load_document_file(file_path)
            if not documents:
                logger.error(f"无法加载文档: {file_path}")
                return False
            
            # 处理文档内容
            processed_docs = []
            for doc in documents:
                # 添加元数据
                doc.metadata.update({
                    'doc_id': doc_id,
                    'organization': organization,
                    'doc_type': doc_type,
                    'source_file': os.path.basename(file_path),
                    'upload_time': datetime.now().isoformat()
                })
                processed_docs.append(doc)
            
            # 分割文档
            splits = self.text_splitter.split_documents(processed_docs)
            
            if not splits:
                logger.error(f"文档分割后为空: {file_path}")
                return False
            
            # 添加到向量存储
            self.vectorstore.add_documents(splits)
            self.vectorstore.persist()
            
            # 保存元数据
            metadata = DocumentMetadata(
                doc_id=doc_id,
                title=title or os.path.basename(file_path),
                organization=organization,
                doc_type=doc_type,
                upload_time=datetime.now().isoformat(),
                file_path=file_path,
                file_size=os.path.getsize(file_path),
                tags=tags or [],
                priority=self.config['document_type_weights'].get(doc_type, 1.0)
            )
            
            self.documents_metadata[doc_id] = metadata
            self._save_metadata()
            
            logger.info(f"文档添加成功: {title or os.path.basename(file_path)} ({len(splits)} 个片段)")
            return True
            
        except Exception as e:
            logger.error(f"添加文档失败 {file_path}: {e}", exc_info=True)
            return False
    
    def _generate_doc_id(self, file_path: str, organization: str) -> str:
        """生成文档ID"""
        try:
            mtime = os.path.getmtime(file_path)
        except:
            mtime = 0
        content = f"{file_path}{organization}{mtime}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _load_document_file(self, file_path: str) -> List[Document]:
        """根据文件类型加载文档"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_ext in ['.txt', '.md']:
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                logger.warning(f"不支持的文件类型: {file_ext}")
                # 尝试作为文本文件读取
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    return [Document(page_content=content, metadata={'source': file_path})]
                except:
                    return []
            
            return loader.load()
            
        except Exception as e:
            logger.error(f"加载文档失败 {file_path}: {e}")
            return []
    
    def search_relevant_documents(self, 
                                 query: str,
                                 organization: Optional[str] = None,
                                 doc_types: Optional[List[str]] = None,
                                 k: int = 5) -> List[Tuple[Document, float]]:
        """搜索相关文档"""
        try:
            # 构建过滤条件
            filter_dict = {}
            if organization:
                filter_dict['organization'] = organization
            if doc_types and len(doc_types) == 1:
                filter_dict['doc_type'] = doc_types[0]
            
            # 执行相似性搜索
            if filter_dict:
                results = self.vectorstore.similarity_search_with_score(
                    query=query,
                    k=k * 2,
                    filter=filter_dict
                )
            else:
                results = self.vectorstore.similarity_search_with_score(
                    query=query,
                    k=k * 2
                )
            
            # 应用额外的过滤和排序
            filtered_results = self._post_process_search_results(
                results, query, organization, doc_types
            )
            
            return filtered_results[:k]
            
        except Exception as e:
            logger.error(f"搜索文档失败: {e}")
            return []
    
    def _post_process_search_results(self, 
                                   results: List[Tuple[Document, float]],
                                   query: str,
                                   organization: Optional[str] = None,
                                   doc_types: Optional[List[str]] = None) -> List[Tuple[Document, float]]:
        """后处理搜索结果"""
        processed_results = []
        
        for doc, score in results:
            # 过滤掉相似度太低的结果（注意：分数越低越相似）
            if not self.use_simple_embeddings and score > self.config['similarity_threshold']:
                continue
            
            # 额外的类型过滤
            if doc_types and len(doc_types) > 1:
                if doc.metadata.get('doc_type') not in doc_types:
                    continue
            
            # 调整分数
            adjusted_score = self._calculate_adjusted_score(doc, score)
            processed_results.append((doc, adjusted_score))
            
            # 更新使用统计
            self._update_document_usage(doc.metadata.get('doc_id'))
        
        # 按调整后的分数排序
        processed_results.sort(key=lambda x: x[1])
        
        return processed_results
    
    def _calculate_adjusted_score(self, doc: Document, base_score: float) -> float:
        """计算调整后的相似度分数"""
        doc_id = doc.metadata.get('doc_id')
        if not doc_id or doc_id not in self.documents_metadata:
            return base_score
        
        metadata = self.documents_metadata[doc_id]
        
        # 文档类型权重
        type_weight = metadata.priority
        
        # 时间衰减因子
        try:
            upload_time = datetime.fromisoformat(metadata.upload_time)
            days_old = (datetime.now() - upload_time).days
            time_factor = max(0.5, 1 - days_old / 365)
        except:
            time_factor = 1.0
        
        # 使用频率加成
        usage_factor = min(1.2, 1 + metadata.usage_count * 0.01)
        
        # 综合调整
        if self.use_simple_embeddings:
            # 简单模式：分数越小越好
            adjusted_score = base_score / (type_weight * time_factor * usage_factor)
        else:
            # OpenAI模式：分数越小越好
            adjusted_score = base_score / (type_weight * time_factor * usage_factor)
        
        return adjusted_score
    
    def _update_document_usage(self, doc_id: Optional[str]):
        """更新文档使用统计"""
        if doc_id and doc_id in self.documents_metadata:
            self.documents_metadata[doc_id].usage_count += 1
            self.documents_metadata[doc_id].last_accessed = datetime.now().isoformat()
    
    def get_intelligent_context(self, 
                               meeting_content: str,
                               organization: str,
                               max_context_length: int = 4000) -> str:
        """为会议生成智能上下文"""
        try:
            # 构建查询
            search_query = meeting_content[:500]
            
            # 搜索相关文档
            relevant_docs = self.search_relevant_documents(
                query=search_query,
                organization=organization,
                k=8
            )
            
            if not relevant_docs:
                return "未找到相关的组织背景信息。"
            
            # 构建上下文
            context_parts = []
            context_length = 0
            
            # 按文档类型分组
            docs_by_type = {}
            for doc, score in relevant_docs:
                doc_type = doc.metadata.get('doc_type', 'unknown')
                if doc_type not in docs_by_type:
                    docs_by_type[doc_type] = []
                docs_by_type[doc_type].append((doc, score))
            
            # 按优先级顺序添加上下文
            type_priorities = ['standards', 'plans', 'meeting_minutes', 'templates', 'reference']
            
            for doc_type in type_priorities:
                if doc_type in docs_by_type and context_length < max_context_length:
                    type_name = {
                        'standards': '相关标准规范',
                        'plans': '相关工作计划',
                        'meeting_minutes': '相关历史会议',
                        'templates': '相关模板文档',
                        'reference': '相关参考资料'
                    }.get(doc_type, doc_type)
                    
                    context_parts.append(f"\n## {type_name}:")
                    
                    for doc, score in docs_by_type[doc_type][:2]:
                        content_preview = doc.page_content[:300]
                        source_file = doc.metadata.get('source_file', '未知来源')
                        
                        context_part = f"\n- 来源: {source_file}\n  内容: {content_preview}..."
                        
                        if context_length + len(context_part) > max_context_length:
                            break
                        
                        context_parts.append(context_part)
                        context_length += len(context_part)
            
            return "\n".join(context_parts) if context_parts else "未找到相关背景信息。"
            
        except Exception as e:
            logger.error(f"生成智能上下文失败: {e}")
            return "生成背景信息时出现错误。"
    
    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        total_docs = len(self.documents_metadata)
        
        # 按组织统计
        orgs = {}
        doc_types = {}
        
        for metadata in self.documents_metadata.values():
            org = metadata.organization
            doc_type = metadata.doc_type
            
            orgs[org] = orgs.get(org, 0) + 1
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        # 获取向量数量
        vector_count = 0
        try:
            if hasattr(self.vectorstore, '_collection'):
                vector_count = self.vectorstore._collection.count()
            elif hasattr(self.vectorstore, '_client'):
                collections = self.vectorstore._client.list_collections()
                if collections:
                    vector_count = len(collections)
        except:
            pass
        
        return {
            'total_documents': total_docs,
            'organizations': orgs,
            'document_types': doc_types,
            'vector_count': vector_count,
            'embedding_type': 'simple' if self.use_simple_embeddings else 'openai'
        }

# 测试函数
def test_knowledge_base():
    """测试知识库功能"""
    try:
        print("正在初始化知识库（测试模式）...")
        kb = EnhancedKnowledgeBase(use_simple_embeddings=True)
        
        print("知识库初始化成功！")
        print(f"统计信息: {kb.get_stats()}")
        
        return kb
        
    except Exception as e:
        print(f"知识库测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_knowledge_base()
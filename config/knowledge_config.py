import os
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class KnowledgeBaseConfig:
    """知识库配置"""
    # 向量数据库配置
    vector_db_path: str = "./knowledge_base/vector_db"
    persist_directory: str = "./knowledge_base/chroma_persist"
    
    # 文档处理配置
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_documents_per_query: int = 5
    similarity_threshold: float = 0.7
    
    # 组织管理配置
    organizations_path: str = "./knowledge_base/organizations"
    max_orgs: int = 50
    max_docs_per_org: int = 1000
    
    # 嵌入模型配置
    embedding_model: str = "text-embedding-ada-002"  # OpenAI
    # embedding_model: str = "all-MiniLM-L6-v2"  # 本地备选
    
    # 文档类型权重
    document_type_weights: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.document_type_weights is None:
            self.document_type_weights = {
                'meeting_minutes': 0.8,
                'standards': 1.0,
                'plans': 0.9,
                'templates': 0.6,
                'reference': 0.7
            }
        
        # 确保目录存在
        os.makedirs(self.vector_db_path, exist_ok=True)
        os.makedirs(self.persist_directory, exist_ok=True)
        os.makedirs(self.organizations_path, exist_ok=True)

# 全局配置实例
KB_CONFIG = KnowledgeBaseConfig()
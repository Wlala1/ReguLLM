import re
import json
import pickle
import numpy as np
from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import networkx as nx
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import requests
import time
import os
import getpass
from dotenv import load_dotenv

load_dotenv()


class JurisdictionLevel(Enum):
    """法律管辖层级"""
    INTERNATIONAL = "international"  
    FEDERAL = "federal"  
    STATE = "state"
    LOCAL = "local"
    REFERENCE = "reference"  


@dataclass
class JurisdictionNode:
    """管辖区节点"""
    id: str
    name: str
    level: JurisdictionLevel
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    document_ids: List[str] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []
        if self.document_ids is None:
            self.document_ids = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'level': self.level.value,
            'parent_id': self.parent_id,
            'children_ids': self.children_ids,
            'document_ids': self.document_ids,
            'metadata': self.metadata
        }


@dataclass
class LegalDocument:
    """法律文档节点"""
    id: str
    title: str
    content: str
    jurisdiction_id: str
    document_type: str
    chunks: List[str] = None
    chunk_embeddings: np.ndarray = None
    metadata: Dict = None
    related_document_ids: List[str] = None
    
    def __post_init__(self):
        if self.chunks is None:
            self.chunks = []
        if self.metadata is None:
            self.metadata = {}
        if self.related_document_ids is None:
            self.related_document_ids = []
    
    def to_dict(self):
        """转换为可JSON序列化的字典"""
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'jurisdiction_id': self.jurisdiction_id,
            'document_type': self.document_type,
            'chunks': self.chunks,
            'metadata': self.metadata,
            'related_document_ids': self.related_document_ids
        }


class DynamicQwenClassifier:
    """支持动态管辖区识别的分类器"""
    
    def __init__(self, api_url: str = None, api_key: str = None):
        self.api_url = (api_url or os.getenv("DASHSCOPE_API_BASE"))
        self.model_name = "qwen-max"
        
        if api_key:
            self.api_key = api_key
        elif os.getenv("DASHSCOPE_API_KEY"):
            self.api_key = os.getenv("DASHSCOPE_API_KEY")
        else:
            print("未找到API密钥，将尝试免费方案或提示输入...")
            self.api_key = self._get_api_key_interactively()
        
        self.headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
    
    def _get_api_key_interactively(self) -> Optional[str]:
        """交互式获取API密钥"""
        print("请输入您的千问API密钥（如使用免费方案可直接回车）：")
        api_key = getpass.getpass("API Key: ").strip()
        return api_key if api_key else None
    
    def classify_document(self, document_content: str, file_path: str = "") -> Dict[str, Any]:
        """使用Qwen分类文档 - 支持动态管辖区识别"""
        content_sample = document_content[:800] if len(document_content) > 800 else document_content
        
        prompt = f"""请分析以下法律文档，判断其管辖区和文档类型，并以JSON格式返回结果。

文件路径: {file_path}
文档内容: {content_sample}

请返回JSON格式，包含以下字段：
{{
    "jurisdiction_code": "管辖区代码（如：california, new_jersey, texas等）",
    "jurisdiction_name": "管辖区全名（如：California, New Jersey, Texas等）", 
    "jurisdiction_level": "管辖层级（international/federal/state/local/reference）",
    "parent_jurisdiction": "父管辖区代码（如州的父级是usa，国家的父级是eu等）",
    "document_type": "文档类型",
    "confidence": "置信度(0-1)",
    "title": "文档标题",
    "year": "年份（如果能识别）",
    "bill_number": "法案编号（如果有）"
}}

管辖层级说明：
- "international": 国际组织（如EU）
- "federal": 国家级（如USA, Germany, France等）
- "state": 州/省级（如California, Texas, New Jersey等）
- "reference": 参考文档（术语表、定义等）

父管辖区推断规则：
- 美国各州的父级是"usa"
- 欧盟国家的父级是"eu" 
- 独立国家的父级可以是null

请识别任何管辖区，不要局限于预设选项。如果是新的管辖区，请准确标识其层级和父级关系。
请只返回JSON，不要其他说明。"""

        try:
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个专业的法律文档分析助手，擅长识别任何国家和地区的法律文档管辖区和类型。你可以识别全世界任何管辖区，不局限于预设选项。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 800
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                try:
                    # 清理markdown格式
                    content = content.strip()
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()
                    
                    classification = json.loads(content)
                    
                    # 验证并补充必要字段
                    required_fields = {
                        "jurisdiction_code": "unknown",
                        "jurisdiction_name": "Unknown Jurisdiction", 
                        "jurisdiction_level": "reference",
                        "parent_jurisdiction": None,
                        "document_type": "Unknown Document",
                        "confidence": 0.5
                    }
                    
                    for field, default in required_fields.items():
                        if field not in classification:
                            classification[field] = default
                    
                    return classification
                    
                except json.JSONDecodeError as e:
                    print(f"JSON解析失败: {e}")
                    print(f"原始内容: {content}")
            else:
                print(f"API调用失败: {response.status_code}")
                
        except Exception as e:
            print(f"分类器调用异常: {e}")
        
        # 返回默认分类
        return {
            "jurisdiction_code": "unknown",
            "jurisdiction_name": "Unknown Jurisdiction",
            "jurisdiction_level": "reference", 
            "parent_jurisdiction": None,
            "document_type": "Unknown Document",
            "confidence": 0.1,
            "title": Path(file_path).stem if file_path else "Unknown"
        }


class DynamicLegalGraphRAG:
    """支持动态管辖区扩展的Legal Graph RAG"""
    
    def __init__(self, 
                 embedding_model_name: str = "BAAI/bge-base-en-v1.5",
                 max_chunk_size: int = 800,
                 overlap_size: int = 100,
                 qwen_api_url: str = None,
                 qwen_api_key: str = None):
        
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        
        # 使用动态分类器
        self.classifier = DynamicQwenClassifier(qwen_api_url, qwen_api_key)
        
        # 图数据结构
        self.graph = nx.DiGraph()
        self.jurisdictions: Dict[str, JurisdictionNode] = {}
        self.documents: Dict[str, LegalDocument] = {}
        
        # 向量存储
        self.document_embeddings: Dict[str, np.ndarray] = {}
        
        # 初始化RecursiveCharacterTextSplitter用于统一分块
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.overlap_size,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", " ", ""]
        )
        
        # 初始化基础管辖区结构
        self._initialize_base_jurisdictions()
    
    def _initialize_base_jurisdictions(self):
        """初始化最小化基础管辖区结构"""
        base_jurisdictions = [
            # 参考文档类别
            ("reference", "Reference Documents", JurisdictionLevel.REFERENCE, None),
            
            # 主要根节点
            ("eu", "European Union", JurisdictionLevel.INTERNATIONAL, None),
            ("usa", "United States", JurisdictionLevel.FEDERAL, None),
            ("canada", "Canada", JurisdictionLevel.FEDERAL, None),
            ("uk", "United Kingdom", JurisdictionLevel.FEDERAL, None),
            ("china", "China", JurisdictionLevel.FEDERAL, None),
        ]
        
        print(f"初始化 {len(base_jurisdictions)} 个基础管辖区根节点...")
        
        # 创建管辖区节点
        for jur_id, name, level, parent_id in base_jurisdictions:
            jurisdiction = JurisdictionNode(
                id=jur_id,
                name=name,
                level=level,
                parent_id=parent_id,
                metadata={
                    "is_root_node": True,
                    "initialization_type": "base"
                }
            )
            self.jurisdictions[jur_id] = jurisdiction
            print(f"  创建根节点: {name} [{level.value}]")
        
        # 建立父子关系
        for jur_id, jurisdiction in self.jurisdictions.items():
            if jurisdiction.parent_id and jurisdiction.parent_id in self.jurisdictions:
                self.jurisdictions[jurisdiction.parent_id].children_ids.append(jur_id)
        
        # 添加到图中
        for jur_id, jurisdiction in self.jurisdictions.items():
            self.graph.add_node(jur_id, node_type="jurisdiction", data=jurisdiction)
            if jurisdiction.parent_id and jurisdiction.parent_id in self.jurisdictions:
                self.graph.add_edge(jurisdiction.parent_id, jur_id, relationship="governs")
        
        print("基础管辖区结构初始化完成")

    def _create_default_parent_jurisdiction(self, parent_code: str):
        """创建默认的父管辖区"""
        default_jurisdictions = {
            "usa": ("United States", JurisdictionLevel.FEDERAL, None),
            "eu": ("European Union", JurisdictionLevel.INTERNATIONAL, None),
            "canada": ("Canada", JurisdictionLevel.FEDERAL, None),
            "uk": ("United Kingdom", JurisdictionLevel.FEDERAL, None),
            "china": ("China", JurisdictionLevel.FEDERAL, None)
        }
        
        if parent_code in default_jurisdictions:
            name, level, grandparent = default_jurisdictions[parent_code]
            
            # 确保祖父级管辖区也存在
            if grandparent and grandparent not in self.jurisdictions:
                self._create_default_parent_jurisdiction(grandparent)
            
            parent_jurisdiction = JurisdictionNode(
                id=parent_code,
                name=name,
                level=level,
                parent_id=grandparent,
                metadata={
                    "auto_created": True, 
                    "type": "default_parent",
                    "creation_reason": "missing_parent_jurisdiction"
                }
            )
            
            self.jurisdictions[parent_code] = parent_jurisdiction
            self.graph.add_node(parent_code, node_type="jurisdiction", data=parent_jurisdiction)
            
            # 建立与祖父级的关系
            if grandparent and grandparent in self.jurisdictions:
                self.jurisdictions[grandparent].children_ids.append(parent_code)
                self.graph.add_edge(grandparent, parent_code, relationship="governs")
            
            print(f"自动创建父管辖区: {name} -> 父级: {grandparent or 'None'}")
        else:
            # 创建通用管辖区
            print(f"警告: 未知父管辖区 {parent_code}，创建为通用管辖区")
            
            generic_jurisdiction = JurisdictionNode(
                id=parent_code,
                name=parent_code.replace("_", " ").title(),
                level=JurisdictionLevel.FEDERAL,
                parent_id=None,
                metadata={
                    "auto_created": True,
                    "type": "generic_unknown",
                    "creation_reason": "unknown_parent_jurisdiction",
                    "needs_manual_review": True
                }
            )
            
            self.jurisdictions[parent_code] = generic_jurisdiction
            self.graph.add_node(parent_code, node_type="jurisdiction", data=generic_jurisdiction)

    def ensure_jurisdiction_exists(self, classification: Dict[str, Any]) -> str:
        """确保管辖区节点存在，如果不存在则动态创建"""
        jurisdiction_code = classification.get("jurisdiction_code", "unknown")
        jurisdiction_name = classification.get("jurisdiction_name", "Unknown")
        jurisdiction_level_str = classification.get("jurisdiction_level", "reference")
        parent_code = classification.get("parent_jurisdiction")
        
        # 如果管辖区已存在，直接返回
        if jurisdiction_code in self.jurisdictions:
            return jurisdiction_code
        
        print(f"发现新管辖区: {jurisdiction_code} ({jurisdiction_name})")
        
        # 确定管辖层级
        try:
            jurisdiction_level = JurisdictionLevel(jurisdiction_level_str)
        except ValueError:
            print(f"未知管辖层级: {jurisdiction_level_str}，使用默认值 reference")
            jurisdiction_level = JurisdictionLevel.REFERENCE
        
        # 确保父管辖区存在
        if parent_code and parent_code not in self.jurisdictions:
            print(f"父管辖区 {parent_code} 不存在，尝试创建...")
            self._create_default_parent_jurisdiction(parent_code)
        
        # 创建新的管辖区节点
        new_jurisdiction = JurisdictionNode(
            id=jurisdiction_code,
            name=jurisdiction_name,
            level=jurisdiction_level,
            parent_id=parent_code,
            metadata={
                "auto_created": True,
                "creation_timestamp": time.time(),
                "source_classification": classification
            }
        )
        
        # 添加到系统中
        self.jurisdictions[jurisdiction_code] = new_jurisdiction
        
        # 更新父子关系
        if parent_code and parent_code in self.jurisdictions:
            self.jurisdictions[parent_code].children_ids.append(jurisdiction_code)
        
        # 添加到图中
        self.graph.add_node(jurisdiction_code, node_type="jurisdiction", data=new_jurisdiction)
        if parent_code and parent_code in self.jurisdictions:
            self.graph.add_edge(parent_code, jurisdiction_code, relationship="governs")
        
        print(f"成功创建管辖区: {jurisdiction_name} -> 父级: {parent_code or 'None'}")
        return jurisdiction_code

    def clean_text(self, text: str) -> str:
        """清洗文本内容"""
        print("开始文本清洗...")
        
        # 移除页眉页脚
        text = re.sub(r'-{5,}\s*Page\s+\d+\s*-{5,}', '', text, flags=re.IGNORECASE)
        text = re.sub(r'HB\s+\d+\s*\n\s*\d{4}', '', text)
        text = re.sub(r'CODING:\s*Words stricken.*?additions\.', '', text)
        
        # 修复被空格分割的单词
        text = re.sub(r'\b([a-zA-Z])\s+([a-zA-Z])\s+([a-zA-Z])\b', r'\1\2\3', text)
        text = re.sub(r'\b([a-zA-Z])\s+([a-zA-Z])\b', r'\1\2', text)
        
        # 合并被分行的句子
        text = re.sub(r'([a-z,;:])\s*\n\s*([a-z])', r'\1 \2', text)
        
        # 标准化空白字符
        text = re.sub(r'\t', ' ', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s+$', '', text, flags=re.MULTILINE)
        
        print("文本清洗完成")
        return text.strip()

    def create_uniform_chunks(self, text: str) -> List[str]:
        """使用统一的RecursiveCharacterTextSplitter进行分块"""
        print("开始统一分块...")
        chunks = self.text_splitter.split_text(text)
        valid_chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 50]
        print(f"分块完成，生成 {len(valid_chunks)} 个有效块")
        return valid_chunks
    
    def process_document(self, text_content: str, file_path: str) -> str:
        """处理文档的完整流程：清洗 → 分类 → 建图 → 分块 → 向量化"""
        print(f"\n开始处理文档: {file_path}")
        
        # 步骤1: 清洗文本
        cleaned_text = self.clean_text(text_content)
        
        # 步骤2: 使用大模型识别管辖区和文档类型
        print("使用大模型进行文档分类...")
        classification = self.classifier.classify_document(cleaned_text, file_path)
        
        print(f"分类结果:")
        print(f"  - 管辖区: {classification.get('jurisdiction_code')} ({classification.get('jurisdiction_name')})")
        print(f"  - 层级: {classification.get('jurisdiction_level')}")
        print(f"  - 父级: {classification.get('parent_jurisdiction')}")
        print(f"  - 类型: {classification.get('document_type')}")
        print(f"  - 置信度: {classification.get('confidence', 0):.2f}")
        
        # 步骤3: 确保管辖区节点存在（动态创建）
        jurisdiction_id = self.ensure_jurisdiction_exists(classification)
        
        # 步骤4: 生成文档ID和创建文档对象
        doc_id = hashlib.md5(f"{file_path}_{cleaned_text[:100]}".encode()).hexdigest()[:12]
        
        title = classification.get("title", Path(file_path).stem)
        if classification.get("bill_number"):
            title = f"{classification['bill_number']} - {title}"
        
        document = LegalDocument(
            id=doc_id,
            title=title,
            content=cleaned_text,
            jurisdiction_id=jurisdiction_id,
            document_type=classification.get("document_type", "Unknown"),
            metadata={
                "source_path": file_path,
                "classification_confidence": classification.get("confidence", 0.5),
                "full_classification": classification
            }
        )
        
        # 步骤5: 统一分块
        document.chunks = self.create_uniform_chunks(cleaned_text)
        print(f"生成 {len(document.chunks)} 个文档块")
        
        # 步骤6: 向量化
        print("开始向量化...")
        if document.chunks:
            chunk_embeddings = []
            for i, chunk in enumerate(document.chunks):
                embedding = self.embedding_model.embed_query(chunk)
                chunk_embeddings.append(embedding)
                
                if (i + 1) % 10 == 0:
                    print(f"已完成 {i + 1}/{len(document.chunks)} 个块的向量化")
            
            document.chunk_embeddings = np.array(chunk_embeddings)
            self.document_embeddings[doc_id] = document.chunk_embeddings
        
        # 步骤7: 添加到图数据结构
        self.documents[doc_id] = document
        self.jurisdictions[jurisdiction_id].document_ids.append(doc_id)
        
        self.graph.add_node(doc_id, node_type="document", data=document)
        self.graph.add_edge(jurisdiction_id, doc_id, relationship="contains")
        
        print(f"文档处理完成: {title} -> 管辖区: {jurisdiction_id}")
        return doc_id
    
    def build_from_directory(self, directory_path: str):
        """从目录批量处理文档"""
        print(f"开始从目录加载文档: {directory_path}")
        
        loader = DirectoryLoader(
            directory_path,
            glob="**/*.txt",
            show_progress=True,
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True}
        )
        
        documents = loader.load()
        print(f"找到 {len(documents)} 个文档文件")
        
        processed_docs = []
        for i, doc in enumerate(documents, 1):
            print(f"\n处理进度: {i}/{len(documents)}")
            try:
                doc_id = self.process_document(doc.page_content, doc.metadata.get('source', ''))
                processed_docs.append(doc_id)
                time.sleep(1)  # 避免API调用过快
                
            except Exception as e:
                print(f"处理文档失败: {doc.metadata.get('source', 'Unknown')}")
                print(f"错误: {e}")
                continue
        
        print(f"\n批量处理完成，成功处理 {len(processed_docs)} 个文档")
        
        # 构建文档间关系
        print("构建文档关系...")
        self.build_document_relationships()
        
        return processed_docs
    
    def build_document_relationships(self):
        """构建文档间的关系"""
        doc_list = list(self.documents.values())
        
        for doc in doc_list:
            related_ids = []
            
            for other_doc in doc_list:
                if other_doc.id == doc.id:
                    continue
                
                # 检查标题引用
                if doc.title in other_doc.content or other_doc.title in doc.content:
                    related_ids.append(other_doc.id)
                
                # 检查法案编号引用
                if (doc.metadata.get('bill_number') and 
                    doc.metadata['bill_number'] in other_doc.content):
                    related_ids.append(other_doc.id)
            
            doc.related_document_ids = related_ids
            
            # 在图中添加关系边
            for related_id in related_ids:
                self.graph.add_edge(doc.id, related_id, relationship="references")

    def search(self, query: str, jurisdiction_id: Optional[str] = None, 
               top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """搜索相关法律文档"""
        query_embedding = self.embedding_model.embed_query(query)
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        # 确定搜索范围
        if jurisdiction_id:
            applicable_doc_ids = self.get_applicable_laws(jurisdiction_id)
        else:
            applicable_doc_ids = list(self.documents.keys())
        
        # 计算相似度
        results = []
        for doc_id in applicable_doc_ids:
            if doc_id not in self.document_embeddings:
                continue
            
            doc = self.documents[doc_id]
            embeddings = self.document_embeddings[doc_id]
            
            similarities = cosine_similarity(query_embedding, embeddings)[0]
            
            for i, (chunk, sim) in enumerate(zip(doc.chunks, similarities)):
                results.append((
                    chunk,
                    float(sim),
                    {
                        'document_id': doc_id,
                        'document_title': doc.title,
                        'jurisdiction': self.jurisdictions[doc.jurisdiction_id].name,
                        'document_type': doc.document_type,
                        'chunk_index': i,
                        **doc.metadata
                    }
                ))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_applicable_laws(self, jurisdiction_id: str) -> List[str]:
        """获取适用于特定管辖区的所有法律"""
        applicable_doc_ids = set()
        
        if jurisdiction_id not in self.jurisdictions:
            return []
        
        # 获取当前管辖区的文档
        applicable_doc_ids.update(self.jurisdictions[jurisdiction_id].document_ids)
        
        # 递归获取父管辖区的文档
        current_id = jurisdiction_id
        visited = set()
        
        while current_id in self.jurisdictions and current_id not in visited:
            visited.add(current_id)
            parent_id = self.jurisdictions[current_id].parent_id
            
            if parent_id and parent_id in self.jurisdictions:
                applicable_doc_ids.update(self.jurisdictions[parent_id].document_ids)
                current_id = parent_id
            else:
                break
        
        return list(applicable_doc_ids)

    def visualize_jurisdiction_creation_stats(self):
        """显示管辖区创建统计"""
        base_nodes = [j for j in self.jurisdictions.values() 
                     if j.metadata.get("is_root_node", False)]
        auto_created = [j for j in self.jurisdictions.values() 
                       if j.metadata.get("auto_created", False)]
        
        print(f"\n=== 管辖区创建统计 ===")
        print(f"基础根节点: {len(base_nodes)} 个")
        for node in base_nodes:
            child_count = len(node.children_ids)
            doc_count = len(node.document_ids)
            print(f"  - {node.name}: {child_count} 个子级, {doc_count} 个文档")
        
        print(f"\n动态创建节点: {len(auto_created)} 个")
        creation_types = {}
        for node in auto_created:
            node_type = node.metadata.get("type", "unknown")
            creation_types[node_type] = creation_types.get(node_type, 0) + 1
            
            parent_name = (self.jurisdictions[node.parent_id].name 
                          if node.parent_id and node.parent_id in self.jurisdictions 
                          else "None")
            print(f"  - {node.name} [{node.level.value}] -> {parent_name}")
        
        if creation_types:
            print(f"\n创建类型分布:")
            for creation_type, count in creation_types.items():
                print(f"  - {creation_type}: {count} 个")

    def save(self, base_path: str = "./legal_graph_db"):
        """保存图谱数据 - 仅使用JSON和Pickle格式"""
        base_path = Path(base_path)
        base_path.mkdir(exist_ok=True)
        
        # 保存管辖区结构
        jurisdictions_data = {
            jur_id: jur.to_dict() 
            for jur_id, jur in self.jurisdictions.items()
        }
        with open(base_path / "jurisdictions.json", 'w', encoding='utf-8') as f:
            json.dump(jurisdictions_data, f, ensure_ascii=False, indent=2)
        
        # 保存文档数据
        documents_data = {
            doc_id: doc.to_dict()
            for doc_id, doc in self.documents.items()
        }
        with open(base_path / "documents.json", 'w', encoding='utf-8') as f:
            json.dump(documents_data, f, ensure_ascii=False, indent=2)
        
        # 保存图结构
        graph_data = {
            'nodes': list(self.graph.nodes()),
            'edges': [(u, v, data) for u, v, data in self.graph.edges(data=True)]
        }
        with open(base_path / "graph_structure.json", 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        
        # 保存向量嵌入 - 使用Pickle
        with open(base_path / "embeddings.pkl", 'wb') as f:
            pickle.dump(self.document_embeddings, f)
        
        # 保存配置信息
        config_data = {
            "embedding_model": "BAAI/bge-base-en-v1.5",
            "max_chunk_size": self.max_chunk_size,
            "overlap_size": self.overlap_size,
            "version": "dynamic_v1.0"
        }
        with open(base_path / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"图谱数据已保存到: {base_path}")
        print("存储格式: JSON (元数据) + Pickle (向量嵌入)")

    def load(self, base_path: str = "./legal_graph_db"):
        """加载图谱数据"""
        base_path = Path(base_path)
        
        # 加载配置
        config_path = base_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.max_chunk_size = config.get("max_chunk_size", 800)
            self.overlap_size = config.get("overlap_size", 100)
        
        # 加载管辖区结构
        with open(base_path / "jurisdictions.json", 'r', encoding='utf-8') as f:
            jurisdictions_data = json.load(f)
        
        self.jurisdictions = {}
        for jur_id, jur_data in jurisdictions_data.items():
            jur = JurisdictionNode(
                id=jur_data['id'],
                name=jur_data['name'],
                level=JurisdictionLevel(jur_data['level']),
                parent_id=jur_data.get('parent_id'),
                children_ids=jur_data.get('children_ids', []),
                document_ids=jur_data.get('document_ids', []),
                metadata=jur_data.get('metadata', {})
            )
            self.jurisdictions[jur_id] = jur
        
        # 加载文档数据
        with open(base_path / "documents.json", 'r', encoding='utf-8') as f:
            documents_data = json.load(f)
        
        self.documents = {}
        for doc_id, doc_data in documents_data.items():
            doc = LegalDocument(
                id=doc_id,
                title=doc_data['title'],
                content=doc_data['content'],
                jurisdiction_id=doc_data['jurisdiction_id'],
                document_type=doc_data['document_type'],
                chunks=doc_data.get('chunks', []),
                metadata=doc_data.get('metadata', {}),
                related_document_ids=doc_data.get('related_document_ids', [])
            )
            self.documents[doc_id] = doc
        
        # 加载图结构
        with open(base_path / "graph_structure.json", 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        self.graph = nx.DiGraph()
        for node in graph_data['nodes']:
            if node in self.jurisdictions:
                self.graph.add_node(node, node_type="jurisdiction", 
                                  data=self.jurisdictions[node])
            elif node in self.documents:
                self.graph.add_node(node, node_type="document",
                                  data=self.documents[node])
        
        for u, v, data in graph_data['edges']:
            self.graph.add_edge(u, v, **data)
        
        # 加载向量嵌入
        with open(base_path / "embeddings.pkl", 'rb') as f:
            self.document_embeddings = pickle.load(f)
        
        print(f"图谱数据已从 {base_path} 加载")

    def visualize_graph_stats(self):
        """可视化图谱统计信息"""
        print("\n=== 法律图谱统计信息 ===")
        print(f"管辖区节点数: {len(self.jurisdictions)}")
        print(f"文档节点数: {len(self.documents)}")
        print(f"图中总节点数: {self.graph.number_of_nodes()}")
        print(f"图中总边数: {self.graph.number_of_edges()}")
        print("存储格式: JSON (元数据) + Pickle (向量嵌入)")
        
        print("\n=== 法律体系结构 ===")
        # 显示各体系结构
        systems = {"reference": [], "eu": [], "usa": [], "other": []}
        
        for jur_id, jur in self.jurisdictions.items():
            if jur_id == "reference":
                systems["reference"].append(jur)
            elif jur.parent_id == "eu" or jur_id == "eu":
                systems["eu"].append(jur)
            elif jur.parent_id == "usa" or jur_id == "usa":
                systems["usa"].append(jur)
            else:
                systems["other"].append(jur)
        
        for system_name, jurs in systems.items():
            if not jurs:
                continue
                
            system_display_names = {
                "reference": "参考文档体系",
                "eu": "欧盟法律体系", 
                "usa": "美国法律体系",
                "other": "其他管辖区体系"
            }
            
            print(f"\n{system_display_names.get(system_name, system_name)}:")
            
            # 构建树形显示
            def print_tree(jur_id, level=1):
                if jur_id not in self.jurisdictions:
                    return
                jur = self.jurisdictions[jur_id]
                doc_count = len(jur.document_ids)
                indent = "  " * level
                doc_info = f" ({doc_count} 个文档)" if doc_count > 0 else ""
                auto_flag = " [自动创建]" if jur.metadata.get("auto_created") else ""
                print(f"{indent}├─ {jur.name} [{jur.level.value}]{doc_info}{auto_flag}")
                
                for child_id in jur.children_ids:
                    print_tree(child_id, level + 1)
            
            # 找出该体系的根节点
            if system_name in ["reference", "eu", "usa"]:
                print_tree(system_name)
            else:
                # 显示其他体系的根节点
                root_nodes = [j for j in jurs if j.parent_id is None or j.parent_id not in self.jurisdictions]
                for root in root_nodes:
                    print_tree(root.id)
        
        print("\n=== 文档类型分布 ===")
        doc_types = {}
        for doc in self.documents.values():
            doc_type = doc.document_type
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        for doc_type, count in sorted(doc_types.items(), key=lambda x: x[1], reverse=True):
            print(f"{doc_type}: {count} 个文档")


def build_dynamic_legal_graph_rag(knowledge_dir: str = "knowledge",
                                 qwen_api_url: str = None,
                                 qwen_api_key: str = None):
    """构建支持动态管辖区扩展的法律图谱RAG系统"""
    
    print("=== 构建动态扩展法律图谱RAG系统 ===")
    print("特性: 自动识别和创建新管辖区")
    
    # 使用动态扩展版本
    graph_rag = DynamicLegalGraphRAG(
        embedding_model_name="BAAI/bge-base-en-v1.5",
        max_chunk_size=300,
        overlap_size=50,
        qwen_api_url=qwen_api_url,
        qwen_api_key=qwen_api_key
    )
    
    processed_docs = graph_rag.build_from_directory(knowledge_dir)
    
    # 显示统计信息
    graph_rag.visualize_graph_stats()
    graph_rag.visualize_jurisdiction_creation_stats()
    
    # 保存图谱
    graph_rag.save("./dynamic_legal_graph_db")
    
    print(f"\n=== 动态法律图谱RAG系统构建完成！处理了 {len(processed_docs)} 个文档 ===")
    
    return graph_rag


def demo_dynamic_search():
    """演示动态法律图谱搜索功能"""
    print("\n=== 动态法律图谱搜索演示 ===")
    
    # 加载已保存的图谱
    graph_rag = DynamicLegalGraphRAG()
    try:
        graph_rag.load("./dynamic_legal_graph_db")
    except FileNotFoundError:
        print("未找到已保存的图谱数据，请先运行构建过程")
        return
    
    # 演示搜索功能
    test_queries = [
        {
            "query": "social media age verification requirements",
            "jurisdiction": None,
            "description": "全局搜索社交媒体年龄验证要求"
        },
        {
            "query": "data protection privacy user consent",
            "jurisdiction": "eu",
            "description": "在欧盟搜索数据保护和用户同意相关法规"
        }
    ]
    
    for test in test_queries:
        print(f"\n{'-'*60}")
        print(f"测试: {test['description']}")
        print(f"查询: {test['query']}")
        
        if test['jurisdiction']:
            print(f"管辖区: {test['jurisdiction']}")
        
        print("\n--- 搜索结果 ---")
        results = graph_rag.search(
            test['query'], 
            test['jurisdiction'],
            top_k=3
        )
        
        if not results:
            print("  未找到相关结果")
            continue
            
        for i, (chunk, score, metadata) in enumerate(results, 1):
            print(f"\n  [{i}] 相似度: {score:.4f}")
            print(f"      文档: {metadata.get('document_title', 'Unknown')}")
            print(f"      类型: {metadata.get('document_type', 'Unknown')}")
            print(f"      管辖区: {metadata.get('jurisdiction', 'Unknown')}")
            print(f"      内容片段: {chunk[:200]}...")


def main():
    """主函数 - 构建动态Graph RAG知识库"""
    
    # 配置参数
    knowledge_dir = "knowledge"
    qwen_api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    qwen_api_key = None
    
    print("=== 动态Graph RAG 知识库构建工具 ===")
    print(f"知识库目录: {knowledge_dir}")
    print(f"API地址: {qwen_api_url}")
    print("存储格式: JSON + Pickle")
    print("特性: 动态管辖区扩展")
    
    # 检查知识库目录是否存在
    if not Path(knowledge_dir).exists():
        print(f"错误: 知识库目录 '{knowledge_dir}' 不存在")
        print(f"请创建目录并放入txt文档文件")
        return
    
    # 构建动态Graph RAG知识库
    try:
        graph_rag = build_dynamic_legal_graph_rag(
            knowledge_dir=knowledge_dir,
            qwen_api_url=qwen_api_url,
            qwen_api_key=qwen_api_key
        )
        
        print("\n=== 动态知识库构建完成 ===")
        print("文件保存位置: ./dynamic_legal_graph_db/")
        print("包含文件:")
        print("  - jurisdictions.json (管辖区信息)")
        print("  - documents.json (文档元数据)")  
        print("  - graph_structure.json (图结构)")
        print("  - embeddings.pkl (向量嵌入)")
        print("  - config.json (配置信息)")
        
        # 可选：运行搜索演示
        print("\n是否运行搜索演示？(y/n): ", end="")
        if input().lower().strip() == 'y':
            demo_dynamic_search()
        
    except Exception as e:
        print(f"构建过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
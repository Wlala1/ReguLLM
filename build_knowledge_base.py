
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
    REFERENCE = "reference"  # 参考文档类型（如术语表）


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


class QwenJurisdictionClassifier:
    """基于Qwen大模型的管辖区分类器"""
    
    def __init__(self, api_url: str = None, api_key: str = None):
        """
        初始化Qwen分类器
        
        Args:
            api_url: Qwen API地址
            api_key: API密钥
        """
        self.api_url = (api_url or os.getenv("DASHSCOPE_API_BASE"))
        self.model_name = "qwen-max"
        
        # 安全地获取API密钥
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
        """
        使用Qwen分类文档的管辖区和类型
        
        Args:
            document_content: 文档内容
            file_path: 文件路径
            
        Returns:
            包含管辖区、文档类型等信息的字典
        """
        # 限制输入长度
        content_sample = document_content[:1000] if len(document_content) > 1000 else document_content
        
        prompt = f"""请分析以下法律文档，判断其管辖区和文档类型，并以JSON格式返回结果。

文件路径: {file_path}
文档内容: {content_sample}

请返回JSON格式，包含以下字段：
{{
    "jurisdiction": "管辖区代码",
    "document_type": "文档类型",
    "confidence": "置信度(0-1)",
    "title": "文档标题",
    "year": "年份（如果能识别）",
    "bill_number": "法案编号（如果有）"
}}

管辖区代码选项：
- "eu": 欧盟法规
- "usa": 美国联邦法律
- "california": 加利福尼亚州
- "utah": 犹他州
- "florida": 佛罗里达州
- "texas": 德克萨斯州
- "germany": 德国
- "france": 法国
- "italy": 意大利
- "spain": 西班牙
- "netherlands": 荷兰
- "reference": 参考文档（术语表、定义等）

文档类型选项：
- "EU Regulation": 欧盟法规
- "Federal Code": 美国联邦法典
- "State Law": 州法律
- "Terminology Table": 术语表
- "Reference Document": 参考文档

请只返回JSON，不要其他说明。"""

        try:
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个专业的法律文档分析助手，擅长识别法律文档的管辖区和类型。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 500
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
                
                # 尝试解析JSON
                try:
                    # 清理可能的markdown格式
                    content = content.strip()
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()
                    
                    classification = json.loads(content)
                    
                    # 验证必要字段
                    if "jurisdiction" not in classification:
                        classification["jurisdiction"] = "usa"  # 默认值
                    if "document_type" not in classification:
                        classification["document_type"] = "Unknown"
                    if "confidence" not in classification:
                        classification["confidence"] = 0.5
                    
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
            "jurisdiction": "usa",
            "document_type": "Unknown",
            "confidence": 0.1,
            "title": Path(file_path).stem if file_path else "Unknown"
        }


class SimplifiedLegalGraphRAG:
    """简化的法律图谱RAG系统 - 仅使用Pickle和JSON格式"""
    
    def __init__(self, 
                 embedding_model_name: str = "BAAI/bge-base-en-v1.5",
                 max_chunk_size: int = 800,
                 overlap_size: int = 100,
                 qwen_api_url: str = None,
                 qwen_api_key: str = None):
        
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        
        # 安全地处理API密钥
        if qwen_api_key is None:
            qwen_api_key = self._get_api_key_safely()
        
        # 初始化Qwen分类器
        self.classifier = QwenJurisdictionClassifier(qwen_api_url, qwen_api_key)
        
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
    
    def _get_api_key_safely(self) -> Optional[str]:
        """安全地获取API密钥"""
        # 首先检查环境变量
        api_key = os.getenv("DASHSCOPE_API_KEY")
        
        if not api_key:
            print("未找到环境变量 DASHSCOPE_API_KEY")
            print("请输入您的千问API密钥（如使用免费方案可直接回车）：")
            api_key = getpass.getpass("API Key: ").strip()
            if not api_key:
                api_key = None
                print("未输入API密钥，将尝试使用免费方案")
        
        return api_key
    
    def _initialize_base_jurisdictions(self):
        """初始化基础管辖区结构"""
        base_jurisdictions = [
            # 参考文档类别
            ("reference", "Reference Documents", JurisdictionLevel.REFERENCE, None),
            
            # 欧盟体系
            ("eu", "European Union", JurisdictionLevel.INTERNATIONAL, None),
            ("germany", "Germany", JurisdictionLevel.FEDERAL, "eu"),
            ("france", "France", JurisdictionLevel.FEDERAL, "eu"),
            ("italy", "Italy", JurisdictionLevel.FEDERAL, "eu"),
            ("spain", "Spain", JurisdictionLevel.FEDERAL, "eu"),
            ("netherlands", "Netherlands", JurisdictionLevel.FEDERAL, "eu"),
            
            # 美国体系
            ("usa", "United States", JurisdictionLevel.FEDERAL, None),
            ("california", "California", JurisdictionLevel.STATE, "usa"),
            ("utah", "Utah", JurisdictionLevel.STATE, "usa"),
            ("florida", "Florida", JurisdictionLevel.STATE, "usa"),
            ("texas", "Texas", JurisdictionLevel.STATE, "usa"),
        ]
        
        # 创建管辖区节点
        for jur_id, name, level, parent_id in base_jurisdictions:
            jurisdiction = JurisdictionNode(
                id=jur_id,
                name=name,
                level=level,
                parent_id=parent_id
            )
            self.jurisdictions[jur_id] = jurisdiction
        
        # 建立父子关系
        for jur_id, jurisdiction in self.jurisdictions.items():
            if jurisdiction.parent_id and jurisdiction.parent_id in self.jurisdictions:
                self.jurisdictions[jurisdiction.parent_id].children_ids.append(jur_id)
        
        # 添加到图中
        for jur_id, jurisdiction in self.jurisdictions.items():
            self.graph.add_node(jur_id, node_type="jurisdiction", data=jurisdiction)
            if jurisdiction.parent_id and jurisdiction.parent_id in self.jurisdictions:
                self.graph.add_edge(jurisdiction.parent_id, jur_id, relationship="governs")

    def clean_text(self, text: str) -> str:
        """清洗文本内容"""
        print("开始文本清洗...")
        
        # 1. 移除页眉页脚
        text = re.sub(r'-{5,}\s*Page\s+\d+\s*-{5,}', '', text, flags=re.IGNORECASE)
        text = re.sub(r'HB\s+\d+\s*\n\s*\d{4}', '', text)
        text = re.sub(r'CODING:\s*Words stricken.*?additions\.', '', text)
        text = re.sub(r'hb\d+-\d+\s*Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
        
        # 2. 修复被空格分割的单词
        text = re.sub(r'\b([a-zA-Z])\s+([a-zA-Z])\s+([a-zA-Z])\b', r'\1\2\3', text)
        text = re.sub(r'\b([a-zA-Z])\s+([a-zA-Z])\b', r'\1\2', text)
        
        # 3. 合并被分行的句子
        text = re.sub(r'([a-z,;:])\s*\n\s*([a-z])', r'\1 \2', text)
        
        # 4. 标准化条文编号
        text = re.sub(r'\(\s*(\d+)\s*\)', r'(\1)', text)
        text = re.sub(r'\(\s*([a-z])\s*\)', r'(\1)', text)
        text = re.sub(r'(\d+)\s*\.', r'\1.', text)
        
        # 5. 标准化空白字符
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
        
        # 使用RecursiveCharacterTextSplitter进行分块
        chunks = self.text_splitter.split_text(text)
        
        # 过滤掉过短的chunks
        valid_chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 50]
        
        print(f"分块完成，生成 {len(valid_chunks)} 个有效块")
        return valid_chunks
    
    def process_document(self, text_content: str, file_path: str) -> str:
        """
        处理文档的完整流程：清洗 → 分类 → 建图 → 分块 → 向量化
        
        Args:
            text_content: 原始文本内容
            file_path: 文件路径
            
        Returns:
            文档ID
        """
        print(f"\n开始处理文档: {file_path}")
        
        # 步骤1: 清洗文本
        cleaned_text = self.clean_text(text_content)
        
        # 步骤2: 使用大模型识别管辖区和文档类型
        print("使用大模型进行文档分类...")
        classification = self.classifier.classify_document(cleaned_text, file_path)
        
        jurisdiction_id = classification["jurisdiction"]
        document_type = classification["document_type"]
        confidence = classification.get("confidence", 0.5)
        
        print(f"分类结果: 管辖区={jurisdiction_id}, 类型={document_type}, 置信度={confidence:.2f}")
        
        # 步骤3: 确保管辖区节点存在
        if jurisdiction_id not in self.jurisdictions:
            print(f"警告: 管辖区 {jurisdiction_id} 不在预定义列表中，使用默认分类")
            jurisdiction_id = "usa"  # 默认归类到美国联邦
        
        # 步骤4: 生成文档ID和创建文档对象
        doc_id = hashlib.md5(f"{file_path}_{cleaned_text[:100]}".encode()).hexdigest()[:12]
        
        # 提取标题
        title = classification.get("title", Path(file_path).stem)
        if classification.get("bill_number"):
            title = classification["bill_number"]
        
        # 创建文档对象
        document = LegalDocument(
            id=doc_id,
            title=title,
            content=cleaned_text,
            jurisdiction_id=jurisdiction_id,
            document_type=document_type,
            metadata={
                "source_path": file_path,
                "classification_confidence": confidence,
                **{k: v for k, v in classification.items() 
                   if k not in ["jurisdiction", "document_type", "confidence", "title"]}
            }
        )
        
        # 步骤5: 统一分块
        print("开始统一分块...")
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
        
        # 添加到图中
        self.graph.add_node(doc_id, node_type="document", data=document)
        self.graph.add_edge(jurisdiction_id, doc_id, relationship="contains")
        
        print(f"文档处理完成: {title} -> 管辖区: {jurisdiction_id}")
        return doc_id
    
    def build_from_directory(self, directory_path: str):
        """从目录批量处理文档"""
        print(f"开始从目录加载文档: {directory_path}")
        
        # 加载所有txt文件
        loader = DirectoryLoader(
            directory_path,
            glob="**/*.txt",
            show_progress=True,
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True}
        )
        
        documents = loader.load()
        print(f"找到 {len(documents)} 个文档文件")
        
        # 逐个处理文档
        processed_docs = []
        for i, doc in enumerate(documents, 1):
            print(f"\n处理进度: {i}/{len(documents)}")
            try:
                doc_id = self.process_document(doc.page_content, doc.metadata.get('source', ''))
                processed_docs.append(doc_id)
                
                # 添加延迟避免API调用过快
                time.sleep(1)
                
            except Exception as e:
                print(f"处理文档失败: {doc.metadata.get('source', 'Unknown')}")
                print(f"错误: {e}")
                continue  # 继续处理下一个文档
        
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
            
            # 基于引用查找相关文档
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
            "version": "simplified_v1.0"
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
        systems = {"reference": [], "eu": [], "usa": []}
        
        for jur_id, jur in self.jurisdictions.items():
            if jur_id == "reference":
                systems["reference"].append(jur)
            elif jur.parent_id == "eu" or jur_id == "eu":
                systems["eu"].append(jur)
            elif jur.parent_id == "usa" or jur_id == "usa":
                systems["usa"].append(jur)
        
        for system_name, jurs in systems.items():
            if not jurs:
                continue
                
            system_display_name = {
                "reference": "参考文档体系",
                "eu": "欧盟法律体系", 
                "usa": "美国法律体系"
            }[system_name]
            
            print(f"\n{system_display_name}:")
            
            # 构建树形显示
            def print_tree(jur_id, level=1):
                if jur_id not in self.jurisdictions:
                    return
                jur = self.jurisdictions[jur_id]
                doc_count = len(jur.document_ids)
                indent = "  " * level
                doc_info = f" ({doc_count} 个文档)" if doc_count > 0 else ""
                print(f"{indent}├─ {jur.name} [{jur.level.value}]{doc_info}")
                
                for child_id in jur.children_ids:
                    print_tree(child_id, level + 1)
            
            # 找出该体系的根节点
            if system_name == "reference":
                print_tree("reference")
            elif system_name == "eu":
                print_tree("eu")
            elif system_name == "usa":
                print_tree("usa")
        
        print("\n=== 文档类型分布 ===")
        doc_types = {}
        for doc in self.documents.values():
            doc_type = doc.document_type
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        for doc_type, count in sorted(doc_types.items(), key=lambda x: x[1], reverse=True):
            print(f"{doc_type}: {count} 个文档")


def build_simplified_legal_graph_rag(knowledge_dir: str = "knowledge",
                                    qwen_api_url: str = None,
                                    qwen_api_key: str = None):
    """构建简化版法律图谱RAG系统的主函数"""
    
    print("=== 开始构建简化版法律图谱RAG系统 ===")
    print("流程: 文本清洗 → 大模型分类 → 图节点构建 → 统一分块 → 向量化 → JSON+Pickle存储")
    
    # 初始化图谱系统
    graph_rag = SimplifiedLegalGraphRAG(
        embedding_model_name="BAAI/bge-base-en-v1.5",
        max_chunk_size=800,
        overlap_size=100,
        qwen_api_url=qwen_api_url,
        qwen_api_key=qwen_api_key
    )
    
    print(f"\n正在从目录加载文档: {knowledge_dir}")
    
    # 从目录批量处理文档
    processed_docs = graph_rag.build_from_directory(knowledge_dir)
    
    # 显示统计信息
    graph_rag.visualize_graph_stats()
    
    # 保存图谱
    print("\n正在保存图谱数据...")
    graph_rag.save("./legal_graph_db")
    
    print(f"\n=== 法律图谱RAG系统构建完成！处理了 {len(processed_docs)} 个文档 ===")
    print("使用JSON + Pickle存储格式")
    
    return graph_rag


def demo_simplified_search():
    """演示简化版搜索功能"""
    print("\n=== 简化版法律图谱搜索演示 ===")
    
    # 加载已保存的图谱
    graph_rag = SimplifiedLegalGraphRAG()
    try:
        graph_rag.load("./legal_graph_db")
    except FileNotFoundError:
        print("未找到已保存的图谱数据，请先运行构建过程")
        return
    
    # 演示不同类型的搜索
    test_queries = [
        {
            "query": "social media age verification requirements",
            "jurisdiction": "utah",
            "description": "在犹他州搜索社交媒体年龄验证要求"
        },
        {
            "query": "data protection privacy user consent",
            "jurisdiction": "eu",
            "description": "在欧盟搜索数据保护和用户同意相关法规"
        },
        {
            "query": "content moderation platform responsibilities",
            "jurisdiction": "california", 
            "description": "在加州搜索内容审核平台责任"
        },
        {
            "query": "terminology definitions",
            "jurisdiction": "reference",
            "description": "在参考文档中搜索术语定义"
        },
        {
            "query": "digital services obligations",
            "jurisdiction": None,
            "description": "全局搜索数字服务义务"
        }
    ]
    
    for test in test_queries:
        print(f"\n{'-'*60}")
        print(f"测试: {test['description']}")
        print(f"查询: {test['query']}")
        
        if test['jurisdiction']:
            print(f"管辖区: {test['jurisdiction']}")
            
            # 显示适用法律数量
            applicable_laws = graph_rag.get_applicable_laws(test['jurisdiction'])
            print(f"适用法律数量: {len(applicable_laws)}")
        
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
            if 'classification_confidence' in metadata:
                print(f"      分类置信度: {metadata['classification_confidence']:.2f}")
            print(f"      内容片段: {chunk[:200]}...")


def main():
    """主函数 - 直接构建Graph RAG知识库"""
    
    # 配置参数
    knowledge_dir = "knowledge"  # 知识库目录
    qwen_api_url = "http://localhost:8000/v1/chat/completions"  # 千问API地址
    qwen_api_key = None  # API密钥，从环境变量获取
    
    print("=== Graph RAG 知识库构建工具 ===")
    print(f"知识库目录: {knowledge_dir}")
    print(f"API地址: {qwen_api_url}")
    print("存储格式: JSON + Pickle")
    
    # 检查知识库目录是否存在
    if not Path(knowledge_dir).exists():
        print(f"错误: 知识库目录 '{knowledge_dir}' 不存在")
        print(f"请创建目录并放入txt文档文件")
        return
    
    # 构建Graph RAG知识库
    try:
        graph_rag = build_simplified_legal_graph_rag(
            knowledge_dir=knowledge_dir,
            qwen_api_url=qwen_api_url,
            qwen_api_key=qwen_api_key
        )
        
        print("\n=== 知识库构建完成 ===")
        print("文件保存位置: ./legal_graph_db/")
        print("包含文件:")
        print("  - jurisdictions.json (管辖区信息)")
        print("  - documents.json (文档元数据)")  
        print("  - graph_structure.json (图结构)")
        print("  - embeddings.pkl (向量嵌入)")
        print("  - config.json (配置信息)")
        
    except Exception as e:
        print(f"构建过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
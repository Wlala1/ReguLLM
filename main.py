import re
import json
from typing import Dict, List, Tuple, Optional, Any
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# 加载环境变量
load_dotenv()


class JargonTranslator:
    """黑话翻译器 - 从Graph RAG知识库中加载术语表"""
    
    def __init__(self, graph_db_path: str = None):
        self.graph_db_path = graph_db_path
        self.jargon_dict = {}
        if graph_db_path:
            self._load_jargon_from_graph_db()
    
    def _load_jargon_from_graph_db(self):
        """从Graph RAG的JSON文件加载术语表"""
        try:
            print("从Graph RAG数据库加载术语表...")
            
            # 加载documents.json
            documents_path = Path(self.graph_db_path) / "documents.json"
            if not documents_path.exists():
                print(f"警告: 文档文件不存在: {documents_path}")
                return
            
            with open(documents_path, 'r', encoding='utf-8') as f:
                documents_data = json.load(f)
            
            print(f"找到 {len(documents_data)} 个文档，开始搜索术语表...")
            
            # 查找术语表文档
            terminology_found = False
            for doc_id, doc_data in documents_data.items():
                content = doc_data.get("content", "")
                doc_type = doc_data.get("document_type", "")
                title = doc_data.get("title", "")
                
                # 检查是否是术语表
                if (doc_type == "Terminology Table" or 
                    "terminology" in title.lower() or
                    "术语" in content or
                    self._looks_like_terminology(content)):
                    
                    print(f"找到术语表文档: {title}")
                    print(f"内容预览: {content[:200]}...")
                    
                    # 提取术语
                    extracted_jargon = self._extract_jargon_comprehensive(content)
                    
                    if extracted_jargon:
                        print(f"成功提取 {len(extracted_jargon)} 个术语:")
                        for jargon, definition in extracted_jargon.items():
                            self.jargon_dict[jargon] = definition
                            print(f"  {jargon}: {definition}")
                        terminology_found = True
                        break
            
            if not terminology_found:
                print("警告: 未找到术语表，使用默认术语")
                
            else:
                print(f"术语表加载完成，共 {len(self.jargon_dict)} 个术语")
                        
        except Exception as e:
            print(f"术语表加载失败: {e}")
            
    
    def _looks_like_terminology(self, content: str) -> bool:
        """判断内容是否看起来像术语表"""
        # 检查是否有多个大写缩写和冒号定义的模式
        abbreviation_pattern = r'\b[A-Z]{2,}\s*[:-]'
        matches = re.findall(abbreviation_pattern, content)
        return len(matches) >= 3  # 至少有3个缩写定义
    
    
    def _extract_jargon_comprehensive(self, content: str) -> Dict[str, str]:
        """综合提取黑话术语"""
        jargon_dict = {}
        
        # 多种匹配模式
        patterns = [
            # 标准格式: ASL: Age-sensitive logic
            r'([A-Z]{2,}|[A-Z][a-z]+(?:[A-Z][a-z]*)*)\s*[:-]\s*([^\n\r]+)',
            # 括号格式: ASL (Age-sensitive logic)
            r'([A-Z]{2,})\s*\(([^)]+)\)',
            # 释义格式: ASL means Age-sensitive logic  
            r'([A-Z]{2,})\s+(?:means?|is|stands for|refers to)\s+([^\n\r\.]+)',
            # 带引号: "ASL": Age-sensitive logic
            r'["\']([A-Z]{2,})["\']?\s*[:-]\s*([^\n\r]+)',
            # 连字符: ASL - Age-sensitive logic
            r'([A-Z]{2,})\s*[-–—]\s*([^\n\r]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
            for jargon, definition in matches:
                jargon = jargon.strip()
                definition = definition.strip().rstrip('.,;:|')
                
                # 质量过滤
                if (len(jargon) >= 2 and len(definition) > 5 and 
                    len(definition) < 200 and
                    not definition.isupper() and
                    jargon.isupper()):  # 确保术语是大写
                    jargon_dict[jargon] = definition
        
        return jargon_dict
    
    def translate_jargon(self, text: str) -> Tuple[str, List[str]]:
        """翻译文本中的黑话"""
        if not self.jargon_dict:
            return text, []
            
        translated_text = text
        found_jargons = []
        
        # 按长度排序，避免部分匹配问题
        sorted_jargon = sorted(self.jargon_dict.items(), key=lambda x: len(x[0]), reverse=True)
        
        for jargon, definition in sorted_jargon:
            pattern = r'\b' + re.escape(jargon) + r'\b'
            if re.search(pattern, translated_text, re.IGNORECASE):
                found_jargons.append(jargon)
                replacement = f"{jargon} ({definition})"
                translated_text = re.sub(pattern, replacement, translated_text, flags=re.IGNORECASE)
        
        return translated_text, found_jargons


class OptimizedGraphRAGRetriever:
    """优化的Graph RAG检索器"""
    
    def __init__(self, graph_db_path: str):
        self.graph_db_path = Path(graph_db_path)
        self.documents = {}
        self.jurisdictions = {}
        self.document_embeddings = {}
        self.embedding_model = None
        self._load_graph_data()
    
    def _load_graph_data(self):
        """加载Graph RAG数据"""
        try:
            print(f"加载Graph RAG数据从: {self.graph_db_path}")
            
            # 加载文档数据
            documents_path = self.graph_db_path / "documents.json"
            if documents_path.exists():
                with open(documents_path, 'r', encoding='utf-8') as f:
                    documents_data = json.load(f)
                self.documents = documents_data
                print(f"✓ 加载了 {len(documents_data)} 个文档")
            else:
                raise FileNotFoundError(f"文档文件不存在: {documents_path}")
            
            # 加载管辖区数据
            jurisdictions_path = self.graph_db_path / "jurisdictions.json"  
            if jurisdictions_path.exists():
                with open(jurisdictions_path, 'r', encoding='utf-8') as f:
                    self.jurisdictions = json.load(f)
                print(f"✓ 加载了 {len(self.jurisdictions)} 个管辖区")
            
            # 加载向量嵌入
            embeddings_path = self.graph_db_path / "embeddings.pkl"
            if embeddings_path.exists():
                with open(embeddings_path, 'rb') as f:
                    self.document_embeddings = pickle.load(f)
                print(f"✓ 加载了 {len(self.document_embeddings)} 个文档的嵌入向量")
                
                # 初始化嵌入模型
                self.embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
                print("✓ 嵌入模型初始化完成")
            else:
                print("⚠️ 未找到嵌入向量文件，将使用关键词搜索")
            
        except Exception as e:
            print(f"加载Graph RAG数据失败: {e}")
            raise e
    
    def similarity_search(self, query: str, k: int = 5, jurisdictions: List[str] = None) -> List[Dict]:
        """优化的语义相似性搜索"""
        if not self.embedding_model or not self.document_embeddings:
            print("使用关键词搜索模式")
            return self._keyword_search(query, k, jurisdictions)
        
        try:
            print(f"执行向量搜索: {query}")
            
            # 查询向量化
            query_embedding = self.embedding_model.embed_query(query)
            query_embedding = np.array(query_embedding).reshape(1, -1)
            
            results = []
            
            # 筛选目标文档
            if jurisdictions:
                target_doc_ids = self._filter_documents_by_jurisdiction(jurisdictions)
                print(f"基于管辖区 {jurisdictions} 筛选到 {len(target_doc_ids)} 个文档")
            else:
                target_doc_ids = list(self.documents.keys())
                print(f"搜索全部 {len(target_doc_ids)} 个文档")
            
            # 计算相似度
            for doc_id in target_doc_ids:
                if doc_id not in self.document_embeddings:
                    continue
                
                doc_data = self.documents[doc_id]
                doc_embeddings = self.document_embeddings[doc_id]
                chunks = doc_data.get('chunks', [])
                
                if len(chunks) == 0 or doc_embeddings.shape[0] == 0:
                    continue
                
                # 批量计算相似度
                similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
                
                # 收集高质量结果
                for i, (chunk, similarity) in enumerate(zip(chunks, similarities)):
                    if similarity > 0.1:  # 相似度阈值
                        results.append({
                            'page_content': chunk,
                            'similarity_score': float(similarity),
                            'metadata': {
                                'document_id': doc_id,
                                'document_title': doc_data.get('title', 'Unknown'),
                                'document_type': doc_data.get('document_type', 'Unknown'),
                                'jurisdiction_id': doc_data.get('jurisdiction_id', 'Unknown'),
                                'chunk_index': i,
                                'source': doc_data.get('metadata', {}).get('source_path', 'Graph RAG DB'),
                                'classification_confidence': doc_data.get('metadata', {}).get('classification_confidence', 0.5)
                            }
                        })
            
            # 按相似度排序
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            final_results = results[:k]
            
            print(f"找到 {len(final_results)} 个高质量结果")
            return final_results
            
        except Exception as e:
            print(f"向量搜索失败，切换到关键词搜索: {e}")
            return self._keyword_search(query, k, jurisdictions)
    
    def _filter_documents_by_jurisdiction(self, jurisdictions: List[str]) -> List[str]:
        """根据管辖区筛选文档（包含层级关系）"""
        filtered_doc_ids = set()
        
        for jurisdiction in jurisdictions:
            # 直接包含该管辖区的文档
            if jurisdiction in self.jurisdictions:
                jur_data = self.jurisdictions[jurisdiction]
                doc_ids = jur_data.get('document_ids', [])
                filtered_doc_ids.update(doc_ids)
                
                # 包含父管辖区的文档（法律层级继承）
                parent_id = jur_data.get('parent_id')
                if parent_id and parent_id in self.jurisdictions:
                    parent_doc_ids = self.jurisdictions[parent_id].get('document_ids', [])
                    filtered_doc_ids.update(parent_doc_ids)
        
        return list(filtered_doc_ids)
    
    def _keyword_search(self, query: str, k: int = 5, jurisdictions: List[str] = None) -> List[Dict]:
        """优化的关键词搜索"""
        print(f"执行关键词搜索: {query}")
        
        # 预处理查询词
        query_words = [word.lower() for word in query.split() if len(word) > 2]
        results = []
        
        # 筛选文档
        if jurisdictions:
            target_doc_ids = self._filter_documents_by_jurisdiction(jurisdictions)
        else:
            target_doc_ids = list(self.documents.keys())
        
        print(f"在 {len(target_doc_ids)} 个文档中搜索")
        
        for doc_id in target_doc_ids:
            doc_data = self.documents[doc_id]
            chunks = doc_data.get('chunks', [])
            
            # 在chunks中搜索
            if chunks:
                for i, chunk in enumerate(chunks):
                    chunk_lower = chunk.lower()
                    
                    # 计算关键词匹配分数
                    exact_matches = sum(1 for word in query_words if word in chunk_lower)
                    partial_matches = sum(0.5 for word in query_words 
                                        if any(word in token for token in chunk_lower.split()))
                    
                    total_score = exact_matches + partial_matches
                    
                    if total_score > 0:
                        results.append({
                            'page_content': chunk,
                            'keyword_score': total_score,
                            'metadata': {
                                'document_id': doc_id,
                                'document_title': doc_data.get('title', 'Unknown'),
                                'document_type': doc_data.get('document_type', 'Unknown'), 
                                'jurisdiction_id': doc_data.get('jurisdiction_id', 'Unknown'),
                                'chunk_index': i,
                                'source': doc_data.get('metadata', {}).get('source_path', 'Graph RAG DB'),
                                'classification_confidence': doc_data.get('metadata', {}).get('classification_confidence', 0.5)
                            }
                        })
            else:
                # 在整个文档中搜索
                content = doc_data.get('content', '').lower()
                exact_matches = sum(1 for word in query_words if word in content)
                
                if exact_matches > 0:
                    # 提取包含关键词的片段
                    content_excerpt = self._extract_relevant_excerpt(
                        doc_data.get('content', ''), query_words[0] if query_words else query
                    )
                    
                    results.append({
                        'page_content': content_excerpt,
                        'keyword_score': exact_matches,
                        'metadata': {
                            'document_id': doc_id,
                            'document_title': doc_data.get('title', 'Unknown'),
                            'document_type': doc_data.get('document_type', 'Unknown'),
                            'jurisdiction_id': doc_data.get('jurisdiction_id', 'Unknown'),
                            'source': doc_data.get('metadata', {}).get('source_path', 'Graph RAG DB'),
                            'classification_confidence': doc_data.get('metadata', {}).get('classification_confidence', 0.5)
                        }
                    })
        
        # 排序并返回
        score_key = 'keyword_score' if results and 'keyword_score' in results[0] else 'similarity_score'
        results.sort(key=lambda x: x.get(score_key, 0), reverse=True)
        
        final_results = results[:k]
        print(f"关键词搜索找到 {len(final_results)} 个结果")
        return final_results
    
    def _extract_relevant_excerpt(self, content: str, keyword: str, max_length: int = 800) -> str:
        """提取包含关键词的相关片段"""
        keyword_lower = keyword.lower()
        content_lower = content.lower()
        
        # 找到关键词位置
        pos = content_lower.find(keyword_lower)
        if pos == -1:
            return content[:max_length]
        
        # 提取关键词周围的内容
        start = max(0, pos - max_length // 2)
        end = min(len(content), pos + max_length // 2)
        
        excerpt = content[start:end]
        
        # 清理截断的句子
        if start > 0:
            first_period = excerpt.find('.')
            if first_period > 0:
                excerpt = excerpt[first_period + 1:]
        
        if end < len(content):
            last_period = excerpt.rfind('.')
            if last_period > 0:
                excerpt = excerpt[:last_period + 1]
        
        return excerpt.strip()


class GeographicJurisdictionDetector:
    """地理管辖区检测器"""
    
    def __init__(self):
        self.location_patterns = self._build_location_patterns()
    
    def _build_location_patterns(self) -> Dict[str, List[str]]:
        """构建地理位置匹配模式"""
        patterns = {
            # 美国各州
            "california": ["california", "ca ", "calif", "golden state"],
            "utah": ["utah", "ut ", "beehive state"],
            "florida": ["florida", "fl ", "fla", "sunshine state"],
            "texas": ["texas", "tx ", "tex", "lone star"],
            "usa": ["usa", "united states", "us ", "america", "federal", "u.s."],
            
            # 欧盟及成员国
            "eu": ["eu ", "european union", "europe", "eea", "european economic area"],
            "germany": ["germany", "german", "deutschland", "de "],
            "france": ["france", "french", "fr "],
            "italy": ["italy", "italian", "it "],
            "spain": ["spain", "spanish", "es "],
            "netherlands": ["netherlands", "dutch", "holland", "nl "],
            
            # 参考文档
            "reference": ["terminology", "definitions", "glossary", "reference"]
        }
        return patterns
    
    def detect_jurisdictions(self, text: str) -> List[str]:
        """检测文本中涉及的管辖区"""
        text_lower = text.lower()
        detected_jurisdictions = []
        
        for jurisdiction_id, patterns in self.location_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    if jurisdiction_id not in detected_jurisdictions:
                        detected_jurisdictions.append(jurisdiction_id)
                    break
        
        # 处理层级关系
        state_to_country = {
            "california": "usa", "utah": "usa", "florida": "usa", "texas": "usa",
            "germany": "eu", "france": "eu", "italy": "eu", "spain": "eu", "netherlands": "eu"
        }
        
        for jurisdiction in detected_jurisdictions.copy():
            if jurisdiction in state_to_country:
                parent = state_to_country[jurisdiction]
                if parent not in detected_jurisdictions:
                    detected_jurisdictions.append(parent)
        
        return detected_jurisdictions


class OptimizedLegalClassifier:
    """优化的法律分类器 - 纯Graph RAG版本"""
    
    def __init__(self, 
                 graph_db_path: str = "./legal_graph_db",
                 model_name: str = "qwen-max"):
        
        print("=== 初始化优化法律分类器 (纯Graph RAG版本) ===")
        
        # 检查数据库路径
        if not Path(graph_db_path).exists():
            raise FileNotFoundError(f"Graph RAG数据库不存在: {graph_db_path}")
        
        # 初始化各组件
        print("1. 初始化大语言模型...")
        self.llm = ChatTongyi(model=model_name, temperature=0.1)
        
        print("2. 加载Graph RAG检索器...")
        self.retriever = OptimizedGraphRAGRetriever(graph_db_path)
        
        print("3. 初始化黑话翻译器...")
        self.jargon_translator = JargonTranslator(graph_db_path)
        
        print("4. 初始化地理管辖区检测器...")
        self.geo_detector = GeographicJurisdictionDetector()
        
        print("5. 初始化提示模板...")
        self.parser = JsonOutputParser()
        self.prompt = self._create_enhanced_prompt()
        
        print("✓ 初始化完成！\n")
    
    def _create_enhanced_prompt(self) -> PromptTemplate:
        """创建增强的提示模板"""
        template = """
你是一位资深的法律合规分析师，具有Graph RAG法律知识图谱搜索能力。

任务：将功能特性准确分类为以下三类之一：
- "LegalRequirement"      (法律/法规/监管要求强制执行)
- "BusinessDriven"        (产品策略/实验/安全选择；非法律强制要求)
- "UnspecifiedNeedsHuman" (意图不明确或证据缺失/冲突)

分析输入：
原始功能: {original_feature}

翻译后功能(含黑话解释): {translated_feature}

检测到的管辖区: {detected_jurisdictions}

发现的黑话术语: {found_jargons}

Graph RAG搜索的法律证据:
{context}

决策框架：
LegalRequirement (得分 0-1)：
+0.40 上下文引用特定法律条文 + 管辖区匹配
+0.20 功能行为明确对应法律要求(如年龄限制↔儿童保护法)
+0.20 地理限制符合法律管辖边界
+0.20 至少有一个可信的法律条文引用

BusinessDriven (得分 0-1)：
+0.50 明确商业动机：A/B测试、实验、性能优化、增长
+0.30 检测到管辖区但缺乏法律证据支持
+0.20 地理差异主要用于产品推出策略

UnspecifiedNeedsHuman (得分 0-1)：
+0.50 管辖区检测但法律证据不足或冲突
+0.30 功能意图不明确，存在混合信号
+0.20 地区限制无明确法律或商业理由

输出要求：
{{
  "assessment": "LegalRequirement" | "BusinessDriven" | "UnspecifiedNeedsHuman",
  "needs_compliance_logic": true | false | null,
  "reasoning": "基于证据的详细分析 ≤200字",
  "detected_jurisdictions": {detected_jurisdictions},
  "translated_jargon": {found_jargons},
  "jurisdictions": ["确定的适用管辖区"],
  "regulations": [
    {{
      "id": "法规ID或null",
      "title": "法规名称", 
      "jurisdiction": "管辖区",
      "relevance": 0.0-1.0,
      "passages": [
        {{"quote": "≤200字符的关键条文", "source_id": "文档ID"}}
      ],
      "decision": "Constrained" | "NotConstrained" | "Unclear",
      "reason": "该法规如何约束此功能"
    }}
  ],
  "triggers": {{
    "legal": ["识别的法律关键词"],
    "business": ["识别的商业关键词"],
    "ambiguity": ["模糊或冲突的表述"]
  }},
  "scores": {{
    "LegalRequirement": 0.0-1.0,
    "BusinessDriven": 0.0-1.0,
    "UnspecifiedNeedsHuman": 0.0-1.0
  }},
  "citations": ["引用的来源文档ID"],
  "confidence": 0.10-0.99
}}

约束：
- 仅使用上下文中的实际法律证据
- assessment ≠ "LegalRequirement" 时，regulations=[]
- 不得编造法律引用
- 管辖区代码必须与数据库匹配
"""
        
        return PromptTemplate(
            template=template,
            input_variables=[
                "original_feature", "translated_feature", "detected_jurisdictions", 
                "found_jargons", "context"
            ]
        )
    
    def _search_legal_evidence(self, query: str, jurisdictions: List[str] = None) -> List[str]:
        """搜索法律证据"""
        try:
            print(f"开始Graph RAG法律证据搜索...")
            
            # 构建多样化的搜索查询
            search_queries = [query]
            
            # 基于管辖区的特定查询
            if jurisdictions:
                jurisdiction_specific = {
                    "utah": ["Utah Social Media Act", "minor curfew", "parental consent"],
                    "california": ["California SB976", "addictive feed", "teen protection"],
                    "florida": ["Florida minor protection", "harmful content"],
                    "eu": ["Digital Services Act", "DSA", "content moderation"],
                    "usa": ["NCMEC", "COPPA", "child protection", "federal reporting"]
                }
                
                for jurisdiction in jurisdictions:
                    if jurisdiction in jurisdiction_specific:
                        for term in jurisdiction_specific[jurisdiction]:
                            search_queries.append(f"{term}")
                            search_queries.append(f"{term} {query}")
            
            # 主题相关查询
            query_lower = query.lower()
            if any(term in query_lower for term in ["minor", "child", "underage", "youth"]):
                search_queries.extend(["child protection law", "minor safety", "parental consent"])
            
            if "feed" in query_lower or "personalization" in query_lower:
                search_queries.extend(["algorithmic feed", "personalized content", "addictive design"])
                
            if "curfew" in query_lower:
                search_queries.extend(["time restrictions", "access limitations"])
            
            # 执行搜索
            all_results = []
            seen_docs = set()
            
            print(f"执行 {len(search_queries)} 个搜索查询")
            
            for i, search_query in enumerate(search_queries[:8]):  # 限制查询数量
                try:
                    print(f"  查询 {i+1}: {search_query}")
                    docs = self.retriever.similarity_search(
                        search_query, k=3, jurisdictions=jurisdictions
                    )
                    
                    for doc in docs:
                        doc_id = doc['metadata'].get('document_id')
                        chunk_id = f"{doc_id}_{doc['metadata'].get('chunk_index', 0)}"
                        
                        # 避免重复
                        if chunk_id in seen_docs:
                            continue
                        seen_docs.add(chunk_id)
                        
                        # 构建结果信息
                        score_info = ""
                        if 'similarity_score' in doc:
                            score_info = f"Similarity: {doc['similarity_score']:.3f}"
                        elif 'keyword_score' in doc:
                            score_info = f"KeywordScore: {doc['keyword_score']}"
                        
                        source_info = f"[GraphRAG-{len(all_results)+1}] "
                        source_info += f"Query: {search_query[:30]}... | "
                        source_info += f"{score_info} | "
                        source_info += f"Doc: {doc['metadata'].get('document_title', 'Unknown')} | "
                        source_info += f"Type: {doc['metadata'].get('document_type', 'Unknown')} | "
                        source_info += f"Jurisdiction: {doc['metadata'].get('jurisdiction_id', 'Unknown')}"
                        
                        content = f"{source_info}\nContent: {doc['page_content'][:1000]}"
                        all_results.append(content)
                        
                        print(f"    ✓ 添加结果: {doc['metadata'].get('document_title', 'Unknown')[:30]}")
                        
                        if len(all_results) >= 8:
                            break
                    
                except Exception as e:
                    print(f"    查询失败: {e}")
                
                if len(all_results) >= 8:
                    break
            
            print(f"Graph RAG搜索完成，共找到 {len(all_results)} 条法律证据")
            
            # 如果结果不足，进行补充搜索
            if len(all_results) < 3:
                print("结果不足，执行补充搜索...")
                fallback_queries = ["law", "regulation", "requirement", "compliance"]
                for fallback_query in fallback_queries:
                    try:
                        docs = self.retriever.similarity_search(fallback_query, k=2)
                        for doc in docs[:1]:  # 只取一个
                            content = f"[GraphRAG-Fallback] {fallback_query}\n{doc['page_content'][:800]}"
                            all_results.append(content)
                            print(f"  添加补充结果: {doc['metadata'].get('document_title', 'Unknown')}")
                            if len(all_results) >= 5:
                                break
                    except:
                        continue
                    if len(all_results) >= 5:
                        break
            
            return all_results
            
        except Exception as e:
            print(f"Graph RAG搜索失败: {e}")
            import traceback
            traceback.print_exc()
            return [f"搜索失败: {str(e)}"]
    
    def classify_feature(self, feature_description: str) -> Dict[str, Any]:
        """
        完整的功能特性分类流程
        """
        print(f"\n{'='*60}")
        print(f"开始分析功能: {feature_description[:80]}...")
        print(f"{'='*60}")
        
        # 步骤1: 黑话识别与翻译
        print("步骤1: 黑话识别与翻译")
        translated_text, found_jargons = self.jargon_translator.translate_jargon(feature_description)
        
        if found_jargons:
            print(f"  ✓ 发现黑话: {found_jargons}")
            print(f"  ✓ 翻译结果: {translated_text}")
        else:
            print("  - 未发现黑话术语")
        
        # 步骤2: 地理管辖区检测
        print("步骤2: 地理管辖区检测")
        detected_jurisdictions = self.geo_detector.detect_jurisdictions(feature_description)
        if detected_jurisdictions:
            print(f"  ✓ 检测到管辖区: {detected_jurisdictions}")
        else:
            print("  - 未检测到特定管辖区")
        
        # 步骤3: 法律证据搜索
        print("步骤3: Graph RAG法律证据搜索")
        search_query = f"{feature_description} legal requirements compliance regulation"
        
        legal_evidence = self._search_legal_evidence(search_query, detected_jurisdictions)
        print(f"  ✓ 找到 {len(legal_evidence)} 条法律证据")
        
        # 步骤4: 构建上下文
        context = "\n\n".join(legal_evidence) if legal_evidence else "NO_LEGAL_EVIDENCE_FOUND"
        
        # 步骤5: LLM分析与分类
        print("步骤5: 大模型分析与分类")
        
        try:
            inputs = {
                "original_feature": feature_description,
                "translated_feature": translated_text,
                "detected_jurisdictions": detected_jurisdictions,
                "found_jargons": found_jargons,
                "context": context[:5000]  # 控制上下文长度
            }
            
            formatted_prompt = self.prompt.format(**inputs)
            response = self.llm.invoke(formatted_prompt)
            result = self.parser.parse(response.content)
            
            # 添加处理元数据
            result["processing_metadata"] = {
                "workflow_completed": True,
                "data_source": "Pure Graph RAG",
                "jargons_found": found_jargons,
                "jargons_translated": bool(found_jargons),
                "jurisdictions_detected": detected_jurisdictions,
                "legal_evidence_count": len(legal_evidence),
                "context_length": len(context),
                "embedding_search_available": bool(self.retriever.embedding_model),
                "search_method": "Vector" if self.retriever.embedding_model else "Keyword"
            }
            
            print(f"  ✓ 分类完成: {result.get('assessment', 'Unknown')}")
            print(f"  ✓ 置信度: {result.get('confidence', 'Unknown')}")
            if result.get('needs_compliance_logic') is not None:
                print(f"  ✓ 需要合规逻辑: {result.get('needs_compliance_logic')}")
            
            return result
            
        except Exception as e:
            print(f"  ✗ 分类失败: {e}")
            return {
                "assessment": "UnspecifiedNeedsHuman",
                "needs_compliance_logic": None,
                "reasoning": f"分类过程出错: {str(e)}",
                "detected_jurisdictions": detected_jurisdictions,
                "translated_jargon": found_jargons,
                "error": str(e),
                "processing_metadata": {
                    "workflow_completed": False,
                    "data_source": "Pure Graph RAG",
                    "error": str(e),
                    "jargons_found": found_jargons,
                    "jurisdictions_detected": detected_jurisdictions
                }
            }
    
    def batch_classify(self, feature_list: List[str]) -> List[Dict[str, Any]]:
        """批量分类功能"""
        results = []
        total = len(feature_list)
        
        print(f"开始批量分类 {total} 个功能特性")
        
        for i, feature in enumerate(feature_list, 1):
            print(f"\n批量进度: {i}/{total}")
            try:
                result = self.classify_feature(feature)
                results.append({
                    'index': i,
                    'input': feature,
                    'output': result,
                    'success': True
                })
            except Exception as e:
                print(f"功能 {i} 分类失败: {e}")
                results.append({
                    'index': i,
                    'input': feature,
                    'output': {'error': str(e)},
                    'success': False
                })
        
        success_count = sum(1 for r in results if r['success'])
        print(f"\n批量分类完成: {success_count}/{total} 成功")
        
        return results


def main():
    """主函数演示"""
    print("=== 优化法律分类系统 - 纯Graph RAG版本 ===\n")
    
    # 初始化分类器
    try:
        # 修改为你的实际路径
        graph_db_path = "./legal_graph_db"
        
        classifier = OptimizedLegalClassifier(graph_db_path=graph_db_path)
        print("✓ 分类器初始化成功\n")
        
    except Exception as e:
        print(f"✗ 分类器初始化失败: {e}")
        print("\n请检查:")
        print("1. Graph RAG数据库路径是否正确")
        print("2. 是否已运行第一份代码生成数据库")
        print("3. 数据库是否包含必要文件: documents.json, jurisdictions.json, embeddings.pkl")
        return
    
    # 测试数据
    test_features = [
        {
            'name': 'Curfew login blocker with ASL and GH for Utah minors',
            'description': 'To comply with the Utah Social Media Regulation Act, we are implementing a curfew-based login restriction for users under 18. The system uses ASL to detect minor accounts and routes enforcement through GH to apply only within Utah boundaries. The feature activates during restricted night hours and logs activity using EchoTrace for auditability. This allows parental control to be enacted without user-facing alerts, operating in ShadowMode during initial rollout.'
        }
    ]
    
    # 执行分类测试
    results = []
    for i, test_case in enumerate(test_features):
        print(f"\n{'='*80}")
        print(f"测试案例 {i+1}/{len(test_features)}: {test_case['name']}")
        print(f"{'='*80}")
        
        # 执行分类
        result = classifier.classify_feature(test_case['description'])
        
        # 保存结果
        test_result = {
            'case_name': test_case['name'],
            'input': test_case['description'],
            'output': result
        }
        results.append(test_result)
        
        # 显示关键结果
        print(f"\n📋 分类结果摘要:")
        print(f"  🏷️  分类: {result.get('assessment', 'Unknown')}")
        print(f"  🔧 需要合规逻辑: {result.get('needs_compliance_logic', 'Unknown')}")
        print(f"  📊 置信度: {result.get('confidence', 'Unknown')}")
        
        if result.get('detected_jurisdictions'):
            print(f"  🌍 检测管辖区: {result.get('detected_jurisdictions')}")
        
        if result.get('translated_jargon'):
            print(f"  🔤 发现黑话: {result.get('translated_jargon')}")
        
        reasoning = result.get('reasoning', '')
        if reasoning:
            print(f"  💭 推理: {reasoning[:100]}{'...' if len(reasoning) > 100 else ''}")
        
        # 显示处理元数据
        if result.get('processing_metadata'):
            metadata = result['processing_metadata']
            print(f"\n🔍 处理统计:")
            print(f"  ✅ 工作流完成: {metadata.get('workflow_completed', False)}")
            print(f"  🔍 搜索方法: {metadata.get('search_method', 'Unknown')}")
            print(f"  📄 法律证据数: {metadata.get('legal_evidence_count', 0)}")
            print(f"  📝 上下文长度: {metadata.get('context_length', 0)}")
        
        print(f"\n{'-'*60}")
    
    # 保存结果到文件
    try:
        output_file = 'optimized_graph_rag_classification_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False, default=str)
        
        print(f"\n💾 结果已保存到: {output_file}")
        print(f"📊 总计处理: {len(results)} 个测试案例")
        
        # 统计分类结果
        assessments = [r['output'].get('assessment', 'Error') for r in results]
        from collections import Counter
        assessment_counts = Counter(assessments)
        
        print(f"\n📈 分类统计:")
        for assessment, count in assessment_counts.items():
            print(f"  {assessment}: {count} 个")
        
    except Exception as e:
        print(f"\n❌ 保存结果失败: {e}")


def interactive_mode():
    """交互模式"""
    try:
        classifier = OptimizedLegalClassifier()
        print("\n🔍 交互式法律功能分类器")
        print("💡 输入 'quit' 退出程序")
        print("📍 可用管辖区: usa, california, utah, florida, texas, eu, germany, france, italy, spain, netherlands, reference")
        
        while True:
            print(f"\n{'-'*50}")
            try:
                feature_input = input("🎯 请输入功能描述: ").strip()
                if feature_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 再见!")
                    break
                
                if not feature_input:
                    print("⚠️ 请输入有效的功能描述")
                    continue
                
                # 执行分类
                result = classifier.classify_feature(feature_input)
                
                # 显示结果
                print(f"\n📋 分类结果:")
                print(f"  🏷️  {result.get('assessment', 'Unknown')}")
                print(f"  📊 置信度: {result.get('confidence', 'N/A')}")
                print(f"  💭 推理: {result.get('reasoning', 'N/A')}")
                
                if result.get('regulations'):
                    print(f"  ⚖️  相关法规: {len(result['regulations'])} 个")
                
            except KeyboardInterrupt:
                print("\n\n👋 程序被中断，再见!")
                break
            except Exception as e:
                print(f"\n❌ 处理出错: {e}")
                
    except Exception as e:
        print(f"❌ 无法启动交互模式: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        main()
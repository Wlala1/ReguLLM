import re
import json
from typing import Dict, List, Tuple, Optional, Any
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema import Document
from dotenv import load_dotenv
import os
from pathlib import Path

# 加载环境变量
load_dotenv()


class JargonTranslator:
    """黑话翻译器 - 从向量知识库中加载术语表"""
    
    def __init__(self, vector_store: Optional[Chroma] = None):
        self.vector_store = vector_store
        self.jargon_dict = {}
        self._load_jargon_from_vector_store()
    
    def _load_jargon_from_vector_store(self):
        """使用直接提取方法从数据库加载术语表"""
        if not self.vector_store:
            print("警告: 向量存储未初始化，无法加载术语表")
            return
            
        try:
            print("使用直接提取方法加载术语表...")
            
            # 直接从数据库提取所有文档
            documents = self._extract_all_documents()
            if not documents:
                print("未找到任何文档")
                return
            
            print(f"找到 {len(documents)} 个文档，开始搜索术语表...")
            
            # 查找术语表文档（第一个文档通常是术语表）
            for i, doc in enumerate(documents):
                content = doc.get("content", "")
                if not content:
                    continue
                
                # 检查是否是术语表
                if ("Terminology Table" in content or 
                    (content.count("ASL:") > 0 and content.count("GH:") > 0 and content.count("PF:") > 0)):
                    
                    print(f"找到术语表文档 {i+1}")
                    print(f"内容预览: {content[:200]}...")
                    
                    # 使用多种方法提取术语
                    extracted_jargon = self._extract_jargon_comprehensive(content)
                    
                    print(f"成功提取 {len(extracted_jargon)} 个术语:")
                    for jargon, definition in extracted_jargon.items():
                        self.jargon_dict[jargon] = definition
                        print(f"  {jargon}: {definition}")
                    
                    break
            
            if not self.jargon_dict:
                print("警告: 未找到术语表或提取失败")
            else:
                print(f"术语表加载完成，共 {len(self.jargon_dict)} 个术语")
                        
        except Exception as e:
            print(f"术语表加载失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _extract_all_documents(self) -> List[Dict]:
        """直接从数据库提取所有文档"""
        documents = []
        
        # 方法1: 从JSON文件提取
        try:
            from pathlib import Path
            import json
            
            # 假设vector_store有persist_directory属性
            if hasattr(self.vector_store, '_persist_directory'):
                json_path = Path(self.vector_store._persist_directory) / "documents.json"
            else:
                # 尝试常见路径
                json_path = Path("/Users/jackwang/Desktop/比赛1/legal_graph_db/documents.json")
            
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                if isinstance(json_data, list):
                    documents = json_data
                elif isinstance(json_data, dict):
                    documents = list(json_data.values())
                    
                print(f"从JSON文件提取到 {len(documents)} 个文档")
                
        except Exception as e:
            print(f"从JSON提取失败: {e}")
            
        # 方法2: 如果JSON失败，尝试从向量存储直接搜索
        if not documents:
            try:
                # 搜索可能包含术语的文档
                search_results = self.vector_store.similarity_search("ASL GH PF NR T5", k=10)
                for doc in search_results:
                    documents.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    })
                print(f"从向量搜索提取到 {len(documents)} 个文档")
            except Exception as e:
                print(f"向量搜索提取失败: {e}")
        
        return documents
    
    def _extract_jargon_comprehensive(self, content: str) -> Dict[str, str]:
        """综合提取黑话术语"""
        jargon_dict = {}
        
        # 专门针对你的术语表格式: "ASL: Age-sensitive logic"
        patterns = [
            r'([A-Z]{2,}|[A-Z][a-z]+(?:[A-Z][a-z]*)*)\s*[:-]\s*([^\n\r]+)',  # ASL: Definition
            r'([A-Z]{2,})\s*\(([^)]+)\)',  # ASL (Definition) 
            r'([A-Z]{2,})\s+(?:means?|is|stands for)\s+([^\n\r\.]+)',  # ASL means Definition
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            for jargon, definition in matches:
                jargon = jargon.strip()
                definition = definition.strip().rstrip('.,;:|')
                
                # 质量检查
                if (len(jargon) >= 2 and len(definition) > 5 and 
                    len(definition) < 200 and
                    not definition.isupper()):
                    jargon_dict[jargon] = definition
        
        return jargon_dict
    
    def _extract_jargon_from_document(self, content: str):
        """从术语文档中提取黑话定义"""
        print("    尝试提取术语定义...")
        
        # 扩展的匹配模式
        patterns = [
            # 标准格式: ASL: Age Segmentation Logic
            r'([A-Z]{2,})\s*[:-]\s*([^\n\r]+)',
            # CamelCase: EchoTrace: Activity Logging System  
            r'([A-Z][a-z]+(?:[A-Z][a-z]*)*)\s*[:-]\s*([^\n\r]+)',
            # 带符号: • ASL: Age Segmentation Logic
            r'[•\*\-]\s*([A-Z]{2,})\s*[:-]\s*([^\n\r]+)',
            # 带引号: "ASL": Age Segmentation Logic
            r'["\']([A-Z]{2,})["\']?\s*[:-]\s*([^\n\r]+)',
            # 连字符: ASL - Age Segmentation Logic
            r'([A-Z]{2,})\s*[-–—]\s*([^\n\r]+)',
            # 括号: ASL (Age Segmentation Logic)
            r'([A-Z]{2,})\s*\(([^)]+)\)',
            # 表格格式: ASL | Age Segmentation Logic
            r'([A-Z]{2,})\s*\|\s*([^\n\r\|]+)',
            # 更宽松的格式: ASL means Age Segmentation Logic
            r'([A-Z]{2,})\s+(?:means?|is|stands for|refers to)\s+([^\n\r\.]+)',
        ]
        
        extracted_count = 0
        for i, pattern in enumerate(patterns):
            matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
            pattern_count = 0
            for jargon, definition in matches:
                jargon = jargon.strip()
                definition = definition.strip().rstrip('.,;:|')
                
                # 验证术语和定义的质量
                if (len(jargon) >= 2 and len(definition) > 3 and
                    jargon.isupper() and  # 术语应该是大写
                    not definition.isupper() and  # 定义不应该全大写
                    len(definition) < 200):  # 定义不应该太长
                    
                    if jargon not in self.jargon_dict:  # 避免重复
                        self.jargon_dict[jargon] = definition
                        print(f"      ✓ 术语: {jargon} -> {definition[:50]}...")
                        extracted_count += 1
                        pattern_count += 1
            
            if pattern_count > 0:
                print(f"    模式{i+1}提取了{pattern_count}个术语")
        
        if extracted_count == 0:
            print("    未找到符合格式的术语定义")
        
        return extracted_count
    
    
    def translate_jargon(self, text: str) -> Tuple[str, List[str]]:
        """
        翻译文本中的黑话
        
        Returns:
            (translated_text, found_jargons)
        """
        translated_text = text
        found_jargons = []
        
        # 按长度排序，先替换长的术语避免部分匹配
        sorted_jargon = sorted(self.jargon_dict.items(), key=lambda x: len(x[0]), reverse=True)
        
        for jargon, definition in sorted_jargon:
            # 使用单词边界确保精确匹配
            pattern = r'\b' + re.escape(jargon) + r'\b'
            if re.search(pattern, translated_text, re.IGNORECASE):
                found_jargons.append(jargon)
                # 替换为更清晰的形式: 黑话 (定义)
                replacement = f"{jargon} ({definition})"
                translated_text = re.sub(pattern, replacement, translated_text, flags=re.IGNORECASE)
        
        return translated_text, found_jargons


class GeographicJurisdictionDetector:
    """地理管辖区检测器"""
    
    def __init__(self):
        self.location_patterns = self._build_location_patterns()
    
    def _build_location_patterns(self) -> Dict[str, List[str]]:
        """构建地理位置匹配模式"""
        patterns = {
            # 美国各州
            "california": ["california", "ca ", "calif"],
            "utah": ["utah", "ut "],
            "florida": ["florida", "fl ", "fla"],
            "texas": ["texas", "tx ", "tex"],
            "usa": ["usa", "united states", "us ", "america", "federal"],
            
            # 欧盟及成员国
            "eu": ["eu ", "european union", "europe", "eea", "european economic area"],
            "germany": ["germany", "german", "deutschland", "de "],
            "france": ["france", "french", "fr "],
            "italy": ["italy", "italian", "it "],
            "spain": ["spain", "spanish", "es "],
            "netherlands": ["netherlands", "dutch", "holland", "nl "],
            
            # 其他地区
            "canada": ["canada", "canadian", "ca "],
            "south_korea": ["south korea", "korea", "kr "],
            "indonesia": ["indonesia", "indonesian", "id "],
            "brazil": ["brazil", "brazilian", "br "],
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
        
        # 如果检测到州，也包含对应的国家
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


class EnhancedLegalClassifier:
    """增强的法律分类器"""
    
    def __init__(self, 
                 graph_db_path: str = "/Users/jackwang/Desktop/比赛1/legal_graph_db",
                 model_name: str = "qwen-max",
                 embedding_model_name: str = "BAAI/bge-base-en-v1.5"):
        
        print("=== 初始化增强法律分类器 ===")
        
        # 步骤1: 初始化LLM
        print("1. 初始化大语言模型...")
        self.llm = ChatTongyi(model=model_name, temperature=0.1)
        
        # 步骤2: 初始化嵌入模型
        print("2. 初始化嵌入模型...")
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # 步骤3: 加载Chroma向量知识库
        print("3. 加载Chroma向量知识库...")
        self.vector_store = None
        self.retriever = None
        self._init_vector_store(graph_db_path)
        
        # 步骤4: 初始化黑话翻译器(依赖向量库)
        print("4. 初始化黑话翻译器...")
        self.jargon_translator = JargonTranslator(self.vector_store)
        
        # 步骤5: 初始化地理管辖区检测器
        print("5. 初始化地理管辖区检测器...")
        self.geo_detector = GeographicJurisdictionDetector()
        
        # 步骤6: 初始化解析器和提示模板
        print("6. 初始化提示模板...")
        self.parser = JsonOutputParser()
        self.prompt = self._create_enhanced_prompt()
        
        print("初始化完成！\n")
    
    def _init_vector_store(self, graph_db_path: str):
        """第一步：加载Graph RAG的Chroma向量知识库"""
        try:
            if Path(graph_db_path).exists():
                self.vector_store = Chroma(
                    persist_directory=graph_db_path,
                    embedding_function=self.embedding_model
                )
                self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
                print(f"   ✓ 成功加载向量知识库: {graph_db_path}")
                
                # 测试向量库连接
                test_docs = self.vector_store.similarity_search("test", k=1)
                print(f"   ✓ 向量库包含文档数量: {len(test_docs) if test_docs else '无法确定'}")
                
            else:
                print(f"   ✗ 向量库路径不存在: {graph_db_path}")
                
        except Exception as e:
            print(f"   ✗ 加载向量库失败: {e}")
    
    def _create_enhanced_prompt(self) -> PromptTemplate:
        """创建增强的提示模板"""
        template = """
你是一位资深的法律合规分析师，具有法律知识图谱和向量搜索能力。

任务：将功能特性准确分类为以下三类之一：
- "LegalRequirement"      (法律/法规/监管要求强制执行)
- "BusinessDriven"        (产品策略/实验/安全选择；非法律强制要求)
- "UnspecifiedNeedsHuman" (意图不明确或证据缺失/冲突)

增强分析流程：
1. 原始功能描述已经过预处理：
   - 黑话术语已通过术语数据库翻译
   - 地理管辖区已自动检测: {detected_jurisdictions}
   - 发现的黑话术语: {found_jargons}

2. 法律证据已通过多源搜索收集：
   - 基于检测到的管辖区进行目标搜索
   - 向量相似性搜索全法律语料库
   - 与适用法律层次结构交叉引用

输入信息：
原始功能: {original_feature}

翻译后功能(含黑话解释): {translated_feature}

检测到的管辖区: {detected_jurisdictions}

法律上下文(来自向量搜索):
{context}

决策框架：
对于 LegalRequirement (得分 0-1)：
+0.40 如果上下文引用特定法律 + 管辖区匹配检测到的位置
+0.20 如果功能行为明确符合法律要求(如：年龄门槛 ↔ 儿童保护法)
+0.20 如果地理限制匹配法律管辖边界
+0.20 如果至少有一个可引用的段落(≤180字符)包含来源

对于 BusinessDriven (得分 0-1)：
+0.50 如果有明确的商业动机：A/B测试、市场测试、增长、性能、容量
+0.30 如果尽管检测到管辖区但未找到支持的法律证据
+0.20 如果地理差异似乎仅用于推出/实验

对于 UnspecifiedNeedsHuman (得分 0-1)：
+0.50 如果检测到管辖区但法律证据不足/冲突
+0.30 如果功能意图不明确或存在混合信号
+0.20 如果"除了/仅在地区"但没有明确的法律或商业理由

分类规则：
- 选择得分最高的标签
- LegalRequirement 需要得分 ≥ 0.60 且至少有一个法律引用
- BusinessDriven 需要得分 ≥ 0.60 或明确的商业指标且无法律证据
- 否则：UnspecifiedNeedsHuman

输出格式：
仅返回有效的JSON：

{{
  "assessment": "LegalRequirement" | "BusinessDriven" | "UnspecifiedNeedsHuman",
  "needs_compliance_logic": true | false | null,
  "reasoning": "基于证据的分析 ≤150字",
  "detected_jurisdictions": {detected_jurisdictions},
  "translated_jargon": {found_jargons},
  "jurisdictions": ["最终适用管辖区"],
  "regulations": [
    {{
      "id": "法规标识符或null",
      "title": "法规名称或null", 
      "jurisdiction": "具体管辖区",
      "relevance": 0.0-1.0,
      "passages": [
        {{"quote": "≤180字符", "source_id": "文档/块ID"}}
      ],
      "decision": "Constrained" | "NotConstrained" | "Unclear",
      "reason": "此法规如何约束功能"
    }}
  ],
  "triggers": {{
    "legal": ["法律指示短语"],
    "business": ["商业指示短语"], 
    "ambiguity": ["不明确指示短语"]
  }},
  "scores": {{
    "LegalRequirement": 0.0-1.0,
    "BusinessDriven": 0.0-1.0,
    "UnspecifiedNeedsHuman": 0.0-1.0
  }},
  "citations": ["使用的来源ID"],
  "confidence": 0.10-0.99
}}

约束条件：
- 仅引用上下文中存在的来源
- 如果assessment ≠ "LegalRequirement"，设置 needs_compliance_logic=false 和 regulations=[]
- 精确说明与我们法律数据库匹配的管辖区代码
- 绝不编造法律引用或法规
"""
        
        return PromptTemplate(
            template=template,
            input_variables=[
                "original_feature", "translated_feature", "detected_jurisdictions", 
                "found_jargons", "context"
            ]
        )
    
    def _search_vector_store(self, query: str, jurisdictions: List[str] = None) -> List[str]:
        """改进的向量搜索 - 基于实际数据库内容"""
        if not self.retriever:
            print("警告: 向量检索器未初始化")
            return []
        
        try:
            print(f"开始向量搜索法律证据...")
            
            # 基于你的数据库内容构建更精准的查询
            search_queries = []
            
            # 1. 基础查询
            search_queries.append(query)
            
            # 2. 如果涉及特定管辖区，添加对应的法律文档查询
            jurisdiction_to_doc_keywords = {
                "utah": ["Utah", "social media regulation", "S.B. 152", "H.B. 311"],
                "california": ["California", "SB 976", "Senate Bill", "Skinner"],
                "florida": ["Florida", "harmful to minors", "F.S."],
                "eu": ["European Union", "Digital Services Act", "Regulation 2022/2065"],
                "usa": ["U.S. Code", "NCMEC", "2258A", "reporting requirements"]
            }
            
            if jurisdictions:
                for jurisdiction in jurisdictions:
                    if jurisdiction in jurisdiction_to_doc_keywords:
                        for keyword in jurisdiction_to_doc_keywords[jurisdiction]:
                            search_queries.append(f"{keyword}")
                            search_queries.append(f"{keyword} {query}")
            
            # 3. 添加特定的法律主题查询
            if any(term in query.lower() for term in ["minor", "child", "underage"]):
                search_queries.extend([
                    "child protection",
                    "minors harmful",
                    "youth addiction", 
                    "parental consent",
                    "age verification"
                ])
            
            if "curfew" in query.lower():
                search_queries.append("Social Media Regulation Utah")
                
            if "feed" in query.lower() or "personalization" in query.lower():
                search_queries.append("addictive feed California")
            
            # 执行搜索
            all_results = []
            seen_content = set()
            
            print(f"将执行 {len(search_queries)} 个搜索查询")
            
            for i, search_query in enumerate(search_queries[:10]):  # 限制查询数量
                try:
                    print(f"  查询 {i+1}: {search_query}")
                    docs = self.retriever.invoke(search_query)
                    
                    for doc in docs[:3]:  # 每个查询最多取3个结果
                        # 避免重复内容
                        content_hash = hash(doc.page_content[:300])
                        if content_hash in seen_content:
                            continue
                        seen_content.add(content_hash)
                        
                        # 检查文档质量和相关性
                        content_lower = doc.page_content.lower()
                        
                        # 优先选择包含法律关键词的文档
                        legal_score = 0
                        legal_keywords = ["law", "regulation", "act", "code", "requirement", "shall", "prohibited", "unlawful", "compliance"]
                        for keyword in legal_keywords:
                            if keyword in content_lower:
                                legal_score += 1
                        
                        # 检查是否与查询相关
                        query_words = search_query.lower().split()
                        relevance_score = sum(1 for word in query_words if word in content_lower)
                        
                        if legal_score > 0 or relevance_score > 0 or len(all_results) < 3:
                            source_info = f"[VectorDB-{len(all_results)+1}] "
                            source_info += f"Query: {search_query[:40]}... | "
                            source_info += f"Legal_Score: {legal_score} | Relevance: {relevance_score} | "
                            source_info += f"Source: {doc.metadata.get('source', 'Unknown')}"
                            
                            content = f"{source_info}\nContent: {doc.page_content[:1000]}..."  # 限制长度
                            all_results.append(content)
                            print(f"    ✓ 添加文档: Legal_Score={legal_score}, Relevance={relevance_score}")
                        
                        if len(all_results) >= 8:  # 总体限制
                            break
                    
                except Exception as e:
                    print(f"    查询失败: {e}")
                
                if len(all_results) >= 8:
                    break
            
            print(f"向量搜索完成，找到 {len(all_results)} 条法律证据")
            
            # 如果还是没有找到结果，尝试最简单的搜索
            if not all_results:
                print("尝试基础法律文档搜索...")
                fallback_queries = ["regulation", "law", "code", "act"]
                for fallback_query in fallback_queries:
                    try:
                        docs = self.retriever.invoke(fallback_query)
                        if docs:
                            doc = docs[0]
                            content = f"[VectorDB-Fallback] Query: {fallback_query}\n{doc.page_content[:800]}..."
                            all_results.append(content)
                            print(f"  添加回退文档: {doc.metadata.get('source', 'Unknown')}")
                            break
                    except:
                        continue
            
            return all_results
            
        except Exception as e:
            print(f"向量搜索失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def classify_feature(self, feature_description: str) -> Dict[str, Any]:
        """
        按照工作流程分类功能特性
        
        工作流程：
        1. 加载Graph RAG的Chroma向量知识库 (已在初始化时完成)
        2. 从输入的feature里找黑话，用括号加载黑话定义
        3. 输入大模型，区分功能类型
        4. 输出结果：相关法规、理由、置信度
        
        Args:
            feature_description: 功能描述
            
        Returns:
            分类结果字典
        """
        print(f"\n{'='*60}")
        print(f"开始分析功能: {feature_description[:80]}...")
        print(f"{'='*60}")
        
        # 第二步: 从输入的feature里找黑话，用括号加载黑话定义
        print("第二步: 黑话识别与翻译...")
        translated_text, found_jargons = self.jargon_translator.translate_jargon(feature_description)
        
        if found_jargons:
            print(f"  ✓ 发现黑话术语: {found_jargons}")
            print(f"  ✓ 翻译后文本: {translated_text}")
        else:
            print("  - 未发现黑话术语")
        
        # 检测地理管辖区
        print("检测地理管辖区...")
        detected_jurisdictions = self.geo_detector.detect_jurisdictions(feature_description)
        if detected_jurisdictions:
            print(f"  ✓ 检测到管辖区: {detected_jurisdictions}")
        else:
            print("  - 未检测到特定管辖区")
        
        # 第三步: 向量搜索相关法律证据
        print("第三步: 搜索相关法律证据...")
        search_query = f"{feature_description} legal requirements compliance regulation law"
        
        vector_results = self._search_vector_store(search_query, detected_jurisdictions)
        print(f"  ✓ 向量搜索找到 {len(vector_results)} 条相关证据")
        
        # 合并搜索结果
        context = "\n\n".join(vector_results) if vector_results else "NO_LEGAL_EVIDENCE_FOUND"
        
        # 第四步: 使用大模型进行分类
        print("第四步: 大模型分析与分类...")
        
        try:
            inputs = {
                "original_feature": feature_description,
                "translated_feature": translated_text,
                "detected_jurisdictions": detected_jurisdictions,
                "found_jargons": found_jargons,
                "context": context[:4000]  # 限制上下文长度
            }
            
            formatted_prompt = self.prompt.format(**inputs)
            response = self.llm.invoke(formatted_prompt)
            result = self.parser.parse(response.content)
            
            # 添加处理元数据
            result["processing_metadata"] = {
                "workflow_completed": True,
                "original_jargons_found": found_jargons,
                "auto_detected_jurisdictions": detected_jurisdictions,
                "vector_search_results": len(vector_results),
                "total_evidence_sources": len(vector_results),
                "jargon_translation_performed": bool(found_jargons),
                "context_length": len(context)
            }
            
            print(f"  ✓ 分类完成: {result.get('assessment', 'Unknown')}")
            print(f"  ✓ 置信度: {result.get('confidence', 'Unknown')}")
            
            return result
            
        except Exception as e:
            print(f"  ✗ 分类过程出错: {e}")
            return {
                "assessment": "UnspecifiedNeedsHuman",
                "needs_compliance_logic": None,
                "reasoning": f"Classification failed due to error: {str(e)}",
                "error": str(e),
                "processing_metadata": {
                    "workflow_completed": False,
                    "original_jargons_found": found_jargons,
                    "auto_detected_jurisdictions": detected_jurisdictions,
                    "error": str(e)
                }
            }


def main():
    """主函数演示完整工作流程"""
    print("=== 增强法律分类系统 - 完整工作流程演示 ===\n")
    
    # 初始化分类器（包含第一步：加载向量知识库）
    try:
        classifier = EnhancedLegalClassifier(
            graph_db_path="/Users/jackwang/Desktop/比赛1/legal_graph_db"  # 请根据实际路径修改
        )
        print("✓ 分类器初始化成功\n")
    except Exception as e:
        print(f"✗ 分类器初始化失败: {e}")
        return
    
    # 测试数据 - 你提供的30个样例
    data = [
        'Curfew login blocker with ASL and GH for Utah minors',
        'PF default toggle with NR enforcement for California teens',
        'Child abuse content scanner using T5 and CDS triggers',
        'Content visibility lock with NSP for EU DSA',
        'Jellybean-based parental notifications for Florida regulation',
        'Unified retention control via DRT & CDS',
        'NSP auto-flagging',
        'T5 tagging for sensitive reports',
        'Underage protection via Snowcap trigger',
        'Universal PF deactivation on guest mode'
    ]
    
    description = [
        "To comply with the Utah Social Media Regulation Act, we are implementing a curfew-based login restriction for users under 18. The system uses ASL to detect minor accounts and routes enforcement through GH to apply only within Utah boundaries.",
        "As part of compliance with California's SB976, the app will disable PF by default for users under 18 located in California. This default setting is considered NR to override, unless explicit parental opt-in is provided.",
        "In line with the US federal law requiring providers to report child sexual abuse content to NCMEC, this feature scans uploads and flags suspected materials tagged as T5. Once flagged, the CDS auto-generates reports.",
        "To meet the transparency expectations of the EU Digital Services Act, we are introducing a visibility lock for flagged user-generated content labeled under NSP. When such content is detected, a soft Softblock is applied.",
        "To support Florida's Online Protections for Minors law, this feature extends the Jellybean parental control framework. Notifications are dispatched to verified parent accounts when a minor attempts to access restricted features.",
        "Introduce a data retention feature using DRT thresholds, ensuring automatic log deletion across all regions. CDS will continuously audit retention violations, triggering EchoTrace as necessary.",
        "This feature will automatically detect and tag content that violates NSP policy. Once flagged, Softblock is applied and a Redline alert is generated if downstream sharing is attempted.",
        "When users report content containing high-risk information, it is tagged as T5 for internal routing. CDS then enforces escalation. The system is universal and does not rely on regional toggles.",
        "Snowcap is activated for all underage users platform-wide, applying ASL to segment accounts. Actions taken under this logic are routed to CDS and monitored using BB to identify deviations.",
        "By default, PF will be turned off for all users browsing in guest mode."
    ]
    
    # 测试前5个样例
    results = []
    for i in range(min(5, len(data))):  # 限制测试数量
        feature_input = f"{data[i]}: {description[i]}"
        
        print(f"\n测试案例 {i+1}/5")
        print(f"功能名称: {data[i]}")
        
        # 执行完整工作流程
        result = classifier.classify_feature(feature_input)
        
        # 保存结果
        test_result = {
            'input': feature_input,
            'output': result
        }
        results.append(test_result)
        
        # 显示关键结果
        print(f"\n结果摘要:")
        print(f"  分类: {result.get('assessment', 'Unknown')}")
        print(f"  需要合规逻辑: {result.get('needs_compliance_logic', 'Unknown')}")
        print(f"  置信度: {result.get('confidence', 'Unknown')}")
        print(f"  推理: {result.get('reasoning', 'No reasoning provided')[:100]}...")
        
        if result.get('processing_metadata'):
            metadata = result['processing_metadata']
            print(f"  工作流程状态: {'✓ 完成' if metadata.get('workflow_completed') else '✗ 未完成'}")
            print(f"  发现黑话: {metadata.get('original_jargons_found', [])}")
            print(f"  检测管辖区: {metadata.get('auto_detected_jurisdictions', [])}")
            print(f"  法律证据数量: {metadata.get('vector_search_results', 0)}")
        
        print(f"{'='*60}")
    
    # 保存结果到JSON文件
    try:
        with open('enhanced_classification_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False, default=str)
        print(f"\n✓ 结果已保存到 enhanced_classification_results.json")
        print(f"✓ 共处理 {len(results)} 个测试案例")
    except Exception as e:
        print(f"\n✗ 保存结果失败: {e}")


if __name__ == "__main__":
    main()
import re
from typing import List, Dict, Optional
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

class LegalDocumentSplitter: 
    def __init__(self, max_chunk_size: int = 800, overlap_size: int = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
    
    def _clean_text(self, text: str) -> str:
        """深度清理文本，处理法律文档中的格式问题"""
        # 1. 移除页眉页脚标识
        text = re.sub(r'-{5,}\s*Page\s+\d+\s*-{5,}', '', text, flags=re.IGNORECASE)
        text = re.sub(r'HB\s+\d+\s*\n\s*\d{4}', '', text)
        text = re.sub(r'CODING:\s*Words stricken.*?additions\.', '', text)
        text = re.sub(r'hb\d+-\d+\s*Page\s+\d+\s+of\s+\d+', '', text)
        
        # 2. 处理被空格截断的单词和句子
        # 移除单词内的异常空格 
        text = re.sub(r'\b([a-zA-Z])\s+([a-zA-Z])\s+([a-zA-Z])', r'\1\2\3', text)
        text = re.sub(r'\b([a-zA-Z])\s+([a-zA-Z])', r'\1\2', text)
        
        # 3. 合并被分行的句子
        # 如果一行以小写字母或特定符号结尾，且下一行以小写开始，则合并
        text = re.sub(r'([a-z,;:])\s*\n\s*([a-z])', r'\1 \2', text)
        
        # 4. 处理条文编号被分割的情况
        text = re.sub(r'\(\s*(\d+)\s*\)', r'(\1)', text)  # (1) 格式
        text = re.sub(r'\(\s*([a-z])\s*\)', r'(\1)', text)  # (a) 格式
        text = re.sub(r'(\d+)\s*\.', r'\1.', text)  # 数字后的点号
        
        # 5. 标准化空白字符
        text = re.sub(r'\t', ' ', text)  # 制表符转空格
        text = re.sub(r'[ \t]+', ' ', text)  # 多个空格合并为一个
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # 多个换行合并为两个
        text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)  # 移除行首空格
        text = re.sub(r'\s+$', '', text, flags=re.MULTILINE)  # 移除行尾空格

        return text
    
    def _reconstruct_sentences(self, text: str) -> str:
        """重建被破坏的句子结构"""
        lines = text.split('\n')
        reconstructed_lines = []
        current_sentence = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_sentence:
                    reconstructed_lines.append(current_sentence.strip())
                    current_sentence = ""
                continue
            
            # 如果行以数字开始，可能是新的条文
            if re.match(r'^\d+\s', line) or re.match(r'^Section\s+\d', line) or re.match(r'^\([a-z0-9]+\)', line):
                if current_sentence:
                    reconstructed_lines.append(current_sentence.strip())
                current_sentence = line + " "
            # 如果行以句号、分号结尾，是句子的结束
            elif line.endswith('.') or line.endswith(';'):
                current_sentence += line + " "
                reconstructed_lines.append(current_sentence.strip())
                current_sentence = ""
            # 否则继续合并到当前句子
            else:
                current_sentence += line + " "
        
        # 处理最后一个句子
        if current_sentence:
            reconstructed_lines.append(current_sentence.strip())
        
        return '\n'.join(reconstructed_lines)
    
    def _extract_document_metadata(self, text: str, source_path: str = '') -> Dict:
        """提取文档元数据，包括处理专业术语词典"""
        # 首先清理文本
        clean_text = self._clean_text(text)
        metadata = {}
        
        # 检查是否为术语词典
        if self._is_terminology_table(clean_text, source_path):
            metadata.update(self._extract_terminology_metadata(clean_text))
            return metadata
        
        # 检查是否为联邦法规（如USC）
        if self._is_federal_code(clean_text):
            metadata.update(self._extract_federal_code_metadata(clean_text))
        
        # 检查是否为欧盟法规
        elif self._is_eu_regulation(clean_text):
            metadata.update(self._extract_eu_regulation_metadata(clean_text))
            
        # 检查是否为州法
        else:
            metadata.update(self._extract_state_law_metadata(clean_text))
            
        return metadata
    
    def _is_terminology_table(self, text: str, source_path: str) -> bool:
        """判断是否为术语词典"""
        indicators = [
            'Terminology Table' in text,
            'terminology_table' in source_path.lower(),
            re.search(r'^[A-Z]+:', text, re.MULTILINE) and ':' in text[:500],
            text.count(':') > 10 and len(text.split('\n')) > 10
        ]
        return any(indicators)
    
    def _is_federal_code(self, text: str) -> bool:
        """判断是否为联邦法典"""
        indicators = [
            'U.S. Code' in text,
            'USC' in text,
            'Title 18' in text,
            'CFR' in text,
            'Federal Register' in text
        ]
        return any(indicators)
    
    def _is_eu_regulation(self, text: str) -> bool:
        """判断是否为欧盟法规"""
        indicators = [
            'REGULATION (EU)' in text,
            'European Parliament' in text,
            'Official Journal of the European Union' in text,
            'Digital Services Act' in text
        ]
        return any(indicators)
    
    def _extract_terminology_metadata(self, text: str) -> Dict:
        """提取术语词典元数据"""
        metadata = {
            'document_type': 'Terminology Table',
            'content_type': 'definitions',
        }
        
        # 统计术语数量
        term_count = len(re.findall(r'^[A-Z]+:', text, re.MULTILINE))
        metadata['term_count'] = term_count
        
        return metadata
    
    def _extract_federal_code_metadata(self, text: str) -> Dict:
        """提取联邦法典元数据"""
        metadata = {'document_type': 'Federal Code'}
        
        # 提取USC section
        usc_match = re.search(r'(\d+)\s+U\.?S\.?\s+Code\s+§\s*(\d+[A-Z]?)', text, re.IGNORECASE)
        if usc_match:
            metadata['title'] = f"Title {usc_match.group(1)}"
            metadata['section'] = f"Section {usc_match.group(2)}"
            metadata['bill_number'] = f"{usc_match.group(1)} USC {usc_match.group(2)}"
        
        metadata['jurisdiction'] = 'Federal'
        return metadata
    
    def _extract_eu_regulation_metadata(self, text: str) -> Dict:
        """提取欧盟法规元数据"""
        metadata = {'document_type': 'EU Regulation'}
        
        # 提取法规编号
        eu_reg_match = re.search(r'REGULATION\s+\(EU\)\s+(\d+/\d+)', text)
        if eu_reg_match:
            metadata['bill_number'] = f"EU {eu_reg_match.group(1)}"
        
        # 提取日期
        date_match = re.search(r'of\s+(\d+\s+\w+\s+\d{4})', text)
        if date_match:
            metadata['date'] = date_match.group(1)
            
        # 提取主题
        if 'Digital Services Act' in text:
            metadata['title'] = 'Digital Services Act'
            
        metadata['jurisdiction'] = 'European Union'
        return metadata
    
    def _extract_state_law_metadata(self, text: str) -> Dict:
        """提取州法元数据"""
        metadata = {}
        
        # 提取各种类型的法案编号
        bill_patterns = [
            (r'\b(HB\s+\d+)\b', 'House Bill'),
            (r'\b(SB\s+\d+)\b', 'Senate Bill'), 
            (r'\b(Senate\s+Bill\s+No\.\s+\d+)\b', 'Senate Bill'),
            (r'\b(H\.B\.\s+\d+)\b', 'House Bill'),
            (r'\b(S\.B\.\s+\d+)\b', 'Senate Bill'),
            (r'\b(CHAPTER\s+\d+)\b', 'Chapter')
        ]
        
        for pattern, doc_type in bill_patterns:
            bill_match = re.search(pattern, text, re.IGNORECASE)
            if bill_match:
                metadata['bill_number'] = bill_match.group(1).upper().replace('.', '')
                metadata['document_type'] = doc_type
                break
        
        # 提取标题
        title_patterns = [
            r'An act relating to (.+?)[;.]',
            r'A bill to be entitled\s+An act (.+?)[;.]',
            r'An act to add (.+?), relating to (.+?)\.',
        ]
        
        for pattern in title_patterns:
            title_match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if title_match:
                metadata['title'] = title_match.group(1).strip()
                break
        
        # 提取年份
        year_matches = re.findall(r'\b(20\d{2})\b', text)
        if year_matches:
            metadata['year'] = str(max(int(y) for y in year_matches))
        
        # 识别管辖区域
        jurisdiction_patterns = [
            (r'\bFlorida\b', 'Florida'),
            (r'\bCalifornia\b', 'California'),
            (r'\bUtah\b', 'Utah'),
            (r'\bTexas\b', 'Texas'),
            (r'\bNew York\b', 'New York'),
        ]
        
        for pattern, jurisdiction in jurisdiction_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                metadata['jurisdiction'] = jurisdiction
                break
        
        # 提取有效日期
        effective_patterns = [
            r'shall take effect (.+?)[\.\n]',
            r'effective (.+?)[\.\n]',
            r'takes effect on (.+?)[\.\n]'
        ]
        
        for pattern in effective_patterns:
            effective_match = re.search(pattern, text, re.IGNORECASE)
            if effective_match:
                metadata['effective_date'] = effective_match.group(1).strip()
                break
                
        return metadata
    
    def _identify_sections(self, text: str) -> List[Dict]:
        """识别和提取法律条文的各个部分"""
        sections = []
        
        # 匹配主要条文 (如 Section 1, Section 2)
        section_pattern = r'(Section\s+\d+\..*?)(?=Section\s+\d+\.|$)'
        main_sections = re.findall(section_pattern, text, re.DOTALL | re.IGNORECASE)
        
        for i, section in enumerate(main_sections, 1):
            sections.append({
                'type': 'main_section',
                'number': i,
                'content': section.strip(),
                'subsections': self._extract_subsections(section)
            })
        
        return sections
    
    def _extract_subsections(self, section_text: str) -> List[Dict]:
        """提取子条文"""
        subsections = []
        
        # 匹配形如 (1), (2), (a), (b) 的子条文
        subsection_patterns = [
            r'\((\d+)\)(.*?)(?=\(\d+\)|$)',  # (1), (2), etc.
            r'\(([a-z])\)(.*?)(?=\([a-z]\)|$)',  # (a), (b), etc.
            r'\b(\d+)\.(.*?)(?=\b\d+\.|$)',  # 1., 2., etc.
        ]
        
        for pattern in subsection_patterns:
            matches = re.findall(pattern, section_text, re.DOTALL)
            for match in matches:
                if len(match[1].strip()) > 20:  # 过滤掉太短的内容
                    subsections.append({
                        'identifier': match[0],
                        'content': match[1].strip()
                    })
        
        return subsections
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """按句子分割文本，保持法律条文的完整性"""
        # 法律文档的句子分割模式
        sentence_endings = r'[.;]\s+(?=[A-Z]|\(\d+\)|\([a-z]\)|\d+\.)'
        sentences = re.split(sentence_endings, text)
        
        # 清理和过滤句子
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # 过滤掉太短的句子
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _create_smart_chunks(self, text: str, metadata: Dict) -> List[str]:
        """创建智能分块 - 针对不同类型文档使用不同策略"""
        
        # 根据文档类型选择分块策略
        doc_type = metadata.get('document_type', '')
        
        if doc_type == 'Terminology Table':
            return self._chunk_terminology_table(text, metadata)
        elif doc_type == 'EU Regulation':
            return self._chunk_eu_regulation(text, metadata)
        elif doc_type == 'Federal Code':
            return self._chunk_federal_code(text, metadata)
        else:
            return self._chunk_state_law(text, metadata)
    
    def _chunk_terminology_table(self, text: str, metadata: Dict) -> List[str]:
        """处理术语词典的分块"""
        chunks = []
        
        # 按术语条目分割
        term_entries = re.split(r'\n(?=[A-Z][A-Za-z]*:)', text)
        
        current_chunk = ""
        for entry in term_entries:
            entry = entry.strip()
            if not entry:
                continue
                
            # 如果添加这个条目会超过限制，先保存当前块
            if current_chunk and len(current_chunk + entry) > self.max_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            current_chunk += entry + "\n\n"
        
        # 添加最后一个块
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks if chunks else [text[:self.max_chunk_size]]
    
    def _chunk_eu_regulation(self, text: str, metadata: Dict) -> List[str]:
        """处理欧盟法规的分块"""
        chunks = []
        
        # EU法规有特殊的结构：Articles, paragraphs等
        # 按Article分割
        article_pattern = r'(Article\s+\d+[^\n]*(?:\n(?!Article\s+\d+)[^\n]*)*)'
        articles = re.findall(article_pattern, text, re.MULTILINE | re.DOTALL)
        
        if articles:
            for article in articles:
                article = article.strip()
                if len(article) <= self.max_chunk_size:
                    chunks.append(article)
                else:
                    # Article太长，按段落分割
                    paragraphs = re.split(r'\n\s*\(\d+\)', article)
                    current_chunk = paragraphs[0] if paragraphs else ""
                    
                    for i, para in enumerate(paragraphs[1:], 1):
                        para_with_num = f"\n({i}){para}"
                        if len(current_chunk + para_with_num) <= self.max_chunk_size:
                            current_chunk += para_with_num
                        else:
                            chunks.append(current_chunk.strip())
                            current_chunk = f"Article continuation:{para_with_num}"
                    
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
        else:
            # 如果没找到Article结构，回退到标准方法
            return self._fallback_sentence_split(text)
            
        return chunks
    
    def _chunk_federal_code(self, text: str, metadata: Dict) -> List[str]:
        """处理联邦法典的分块"""
        chunks = []
        
        # 联邦法典通常有(a), (b), (c)等子节
        # 先尝试按主要subsection分割
        subsection_pattern = r'\n\s*\([a-z]\)\s*[A-Z]'
        parts = re.split(subsection_pattern, text)
        
        if len(parts) > 1:
            # 找到了subsection结构
            current_chunk = parts[0]  # 开头部分
            
            # 重新找到所有subsection标记
            subsections = re.findall(subsection_pattern, text)
            
            for i, (subsection_mark, content) in enumerate(zip(subsections, parts[1:])):
                subsection_text = subsection_mark.strip() + " " + content.strip()
                
                if len(current_chunk + "\n" + subsection_text) <= self.max_chunk_size:
                    current_chunk += "\n" + subsection_text
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = subsection_text
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
        else:
            # 没有找到subsection，按句子分割
            return self._fallback_sentence_split(text)
            
        return chunks
    
    def _chunk_state_law(self, text: str, metadata: Dict) -> List[str]:
        """处理州法的分块"""
        chunks = []
        sections = self._identify_sections(text)
        
        if not sections:
            return self._fallback_sentence_split(text)
        
        for section in sections:
            section_content = section['content']
            
            # 如果整个section小于最大块大小，作为一个完整块
            if len(section_content) <= self.max_chunk_size:
                chunks.append(section_content)
            else:
                # 尝试按子条文分割
                if section['subsections']:
                    current_chunk = f"Section {section['number']}:\n"
                    
                    for subsection in section['subsections']:
                        subsection_text = f"({subsection['identifier']}) {subsection['content']}"
                        
                        # 检查是否可以添加到当前块
                        if len(current_chunk + subsection_text) <= self.max_chunk_size:
                            current_chunk += subsection_text + "\n"
                        else:
                            # 保存当前块，开始新块
                            if len(current_chunk.strip()) > 0:
                                chunks.append(current_chunk.strip())
                            current_chunk = f"Section {section['number']} (continued):\n{subsection_text}\n"
                    
                    # 添加最后一个块
                    if len(current_chunk.strip()) > 0:
                        chunks.append(current_chunk.strip())
                else:
                    # 按句子分割长section
                    sentences = self._split_by_sentences(section_content)
                    current_chunk = ""
                    
                    for sentence in sentences:
                        if len(current_chunk + sentence) <= self.max_chunk_size:
                            current_chunk += sentence + " "
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence + " "
                    
                    if current_chunk:
                        chunks.append(current_chunk.strip())
        
        return chunks
    
    def _fallback_sentence_split(self, text: str) -> List[str]:
        """回退方案：基于句子的智能分割"""
        sentences = self._split_by_sentences(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) <= self.max_chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档列表"""
        split_docs = []
        
        for doc in documents:
            print(f"\n正在处理文档: {doc.metadata.get('source', 'Unknown')}")
            source_path = doc.metadata.get('source', '')
            
            # 显示原始文档的一些统计信息
            original_text = doc.page_content
            print(f"  原始文档长度: {len(original_text)} 字符")
            print(f"  原始文档行数: {len(original_text.splitlines())}")
            
            # 深度清理文档
            cleaned_text = self._clean_text(original_text)
            cleaned_text = self._reconstruct_sentences(cleaned_text)
            
            print(f"  清理后长度: {len(cleaned_text)} 字符")
            print(f"  清理后行数: {len(cleaned_text.splitlines())}")
            
            # 提取文档级元数据
            doc_metadata = self._extract_document_metadata(original_text, source_path)
            doc_metadata.update(doc.metadata)  # 保留原有元数据
            
            print(f"  文档类型: {doc_metadata.get('document_type', 'Unknown')}")
            
            # 创建智能分块
            chunks = self._create_smart_chunks(cleaned_text, doc_metadata)
            
            print(f"  生成分块数: {len(chunks)}")
            
            # 为每个块创建Document对象
            for i, chunk in enumerate(chunks):
                chunk_metadata = doc_metadata.copy()
                chunk_metadata.update({
                    'chunk_id': i,
                    'total_chunks': len(chunks),
                    'chunk_type': self._determine_chunk_type(chunk, doc_metadata),
                    'original_length': len(original_text),
                    'cleaned_length': len(cleaned_text)
                })
                
                split_docs.append(Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                ))
        
        return split_docs
    
    def _determine_chunk_type(self, chunk: str, doc_metadata: Dict) -> str:
        """确定分块的具体类型"""
        doc_type = doc_metadata.get('document_type', '')
        
        if doc_type == 'Terminology Table':
            return 'definition'
        elif 'Article' in chunk:
            return 'article'
        elif 'Section' in chunk:
            return 'section'
        elif re.search(r'\([a-z]\)', chunk):
            return 'subsection'
        elif re.search(r'\(\d+\)', chunk):
            return 'paragraph'
        else:
            return 'legal_text'

def build_legal_knowledge_base():
    """构建法律知识库的主函数"""
    
    print("正在加载法律文档...")
    # 1. 加载所有文档
    loader = DirectoryLoader(
        "knowledge",
        glob="**/*.txt",
        show_progress=True,
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True}
    )
    
    documents = loader.load()
    print(f"已加载 {len(documents)} 个文档")
    
    # 2. 使用智能法律文档分块器
    legal_splitter = LegalDocumentSplitter(max_chunk_size=800, overlap_size=100)
    chunks = legal_splitter.split_documents(documents)
    
    print(f"文档已分割为 {len(chunks)} 个智能块")
    
    # 3. 初始化嵌入模型
    print("\n正在初始化嵌入模型...")
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    
    # 4. 创建向量数据库
    print("正在构建向量数据库...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./legal_compliance_db1"
    )
    
    print("法律知识库构建完成！")
    return vector_store


if __name__ == "__main__":
    build_legal_knowledge_base()
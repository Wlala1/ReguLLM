#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
法规文档向量化知识库构建脚本
使用 LangChain + ChromaDB + Google Generative AI 构建本地向量数据库
"""

import os
import glob
from typing import List, Dict
import logging
from pathlib import Path

# LangChain 相关导入
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LegalDocumentVectorStore:
    """
    法规文档向量数据库构建器
    """
    
    def __init__(self, 
                 google_api_key: str,
                 knowledge_dir: str = "./knowledge",
                 vector_db_dir: str = "./vector_db",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        初始化向量数据库构建器
        
        Args:
            google_api_key: Google AI API密钥
            knowledge_dir: 存放txt法规文件的目录
            vector_db_dir: 向量数据库存储目录
            chunk_size: 文档分块大小
            chunk_overlap: 分块重叠大小
        """
        self.knowledge_dir = Path(knowledge_dir)
        self.vector_db_dir = Path(vector_db_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 初始化Google AI嵌入模型
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "!", "?", "，", " ", ""]
        )
        
        # 确保目录存在
        self.vector_db_dir.mkdir(exist_ok=True)
        
    def load_documents(self) -> List[Document]:
        """
        加载所有txt法规文件
        
        Returns:
            Document列表
        """
        documents = []
        txt_files = list(self.knowledge_dir.glob("*.txt"))
        
        if not txt_files:
            logger.warning(f"在 {self.knowledge_dir} 目录中未找到任何txt文件")
            return documents
            
        logger.info(f"找到 {len(txt_files)} 个txt文件")
        
        for txt_file in txt_files:
            try:
                logger.info(f"正在加载文件: {txt_file.name}")
                
                # 使用TextLoader加载文档，尝试不同编码
                for encoding in ['utf-8', 'gbk', 'gb2312']:
                    try:
                        loader = TextLoader(str(txt_file), encoding=encoding)
                        docs = loader.load()
                        
                        # 为每个文档添加元数据
                        for doc in docs:
                            doc.metadata.update({
                                'source': txt_file.name,
                                'file_path': str(txt_file),
                                'document_type': '法规文件'
                            })
                        
                        documents.extend(docs)
                        logger.info(f"成功加载 {txt_file.name} (编码: {encoding})")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    logger.error(f"无法加载文件 {txt_file.name}: 编码问题")
                    
            except Exception as e:
                logger.error(f"加载文件 {txt_file.name} 时出错: {str(e)}")
                
        logger.info(f"总共加载了 {len(documents)} 个文档")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        分割文档为较小的块
        
        Args:
            documents: 原始文档列表
            
        Returns:
            分割后的文档块列表
        """
        logger.info("开始分割文档...")
        
        split_docs = self.text_splitter.split_documents(documents)
        
        # 为分割后的文档添加块编号
        for i, doc in enumerate(split_docs):
            doc.metadata['chunk_id'] = i
            doc.metadata['chunk_size'] = len(doc.page_content)
            
        logger.info(f"文档分割完成，共生成 {len(split_docs)} 个文档块")
        return split_docs
    
    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """
        创建向量数据库
        
        Args:
            documents: 文档块列表
            
        Returns:
            ChromaDB向量数据库实例
        """
        logger.info("开始创建向量数据库...")
        
        try:
            # 创建ChromaDB向量数据库
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(self.vector_db_dir),
                collection_name="legal_documents"
            )
            
            # 持久化数据库
            vectorstore.persist()
            
            logger.info(f"向量数据库创建完成，存储在: {self.vector_db_dir}")
            logger.info(f"数据库包含 {len(documents)} 个文档块")
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"创建向量数据库时出错: {str(e)}")
            raise
    
    def build_knowledge_base(self) -> Chroma:
        """
        构建完整的法规知识库
        
        Returns:
            ChromaDB向量数据库实例
        """
        logger.info("开始构建法规知识库...")
        
        # 1. 加载文档
        documents = self.load_documents()
        if not documents:
            raise ValueError("没有找到任何可加载的文档")
        
        # 2. 分割文档
        split_docs = self.split_documents(documents)
        
        # 3. 创建向量数据库
        vectorstore = self.create_vector_store(split_docs)
        
        logger.info("法规知识库构建完成！")
        return vectorstore
    
    def load_existing_vector_store(self) -> Chroma:
        """
        加载已存在的向量数据库
        
        Returns:
            ChromaDB向量数据库实例
        """
        if not self.vector_db_dir.exists():
            raise FileNotFoundError(f"向量数据库目录不存在: {self.vector_db_dir}")
            
        vectorstore = Chroma(
            persist_directory=str(self.vector_db_dir),
            embedding_function=self.embeddings,
            collection_name="legal_documents"
        )
        
        logger.info(f"成功加载现有向量数据库: {self.vector_db_dir}")
        return vectorstore
    
    def search_similar_documents(self, vectorstore: Chroma, query: str, k: int = 5) -> List[Document]:
        """
        搜索相似文档
        
        Args:
            vectorstore: 向量数据库实例
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            相似文档列表
        """
        logger.info(f"搜索查询: {query}")
        
        results = vectorstore.similarity_search(query, k=k)
        
        logger.info(f"找到 {len(results)} 个相似文档")
        return results

def main():
    """
    主函数 - 示例用法
    """
    # 请在这里设置您的Google AI API密钥
    GOOGLE_API_KEY = "your_google_api_key_here"
    
    if GOOGLE_API_KEY == "your_google_api_key_here":
        print("请先设置您的Google AI API密钥！")
        print("您可以在以下地址获取API密钥: https://makersuite.google.com/app/apikey")
        return
    
    try:
        # 初始化向量数据库构建器
        builder = LegalDocumentVectorStore(
            google_api_key=GOOGLE_API_KEY,
            knowledge_dir="./knowledge",
            vector_db_dir="./vector_db",
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # 构建知识库
        vectorstore = builder.build_knowledge_base()
        
        # 示例：搜索相关法规
        print("\n=== 测试搜索功能 ===")
        test_query = "合同违约责任"
        results = builder.search_similar_documents(vectorstore, test_query, k=3)
        
        print(f"\n查询: {test_query}")
        print("搜索结果:")
        for i, doc in enumerate(results, 1):
            print(f"\n结果 {i}:")
            print(f"来源: {doc.metadata.get('source', 'Unknown')}")
            print(f"内容预览: {doc.page_content[:200]}...")
            
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()
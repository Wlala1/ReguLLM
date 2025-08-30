#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Legal Document Vectorization Knowledge Base Builder Script
Build local vector database using LangChain + ChromaDB + Google Generative AI
"""

import os
import glob
from typing import List, Dict
import logging
from pathlib import Path

# LangChain related imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LegalDocumentVectorStore:
    """
    Legal Document Vector Database Builder
    """
    
    def __init__(self, 
                 google_api_key: str,
                 knowledge_dir: str = "./knowledge",
                 vector_db_dir: str = "./vector_db",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize vector database builder
        
        Args:
            google_api_key: Google AI API key
            knowledge_dir: Directory containing txt legal files
            vector_db_dir: Vector database storage directory
            chunk_size: Document chunk size
            chunk_overlap: Chunk overlap size
        """
        self.knowledge_dir = Path(knowledge_dir)
        self.vector_db_dir = Path(vector_db_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "!", "?", "，", " ", ""]
        )
        
        # Ensure directories exist
        self.vector_db_dir.mkdir(exist_ok=True)
        
    def load_documents(self) -> List[Document]:
        """
        Load all txt legal files
        
        Returns:
            List of Documents
        """
        documents = []
        txt_files = list(self.knowledge_dir.glob("*.txt"))
        
        if not txt_files:
            logger.warning(f"No txt files found in {self.knowledge_dir} directory")
            return documents
            
        logger.info(f"Found {len(txt_files)} txt files")
        
        for txt_file in txt_files:
            try:
                logger.info(f"Loading file: {txt_file.name}")
                
                # Use TextLoader to load documents, try different encodings
                for encoding in ['utf-8', 'gbk', 'gb2312']:
                    try:
                        loader = TextLoader(str(txt_file), encoding=encoding)
                        docs = loader.load()
                        
                        # Add metadata to each document
                        for doc in docs:
                            doc.metadata.update({
                                'source': txt_file.name,
                                'file_path': str(txt_file),
                                'document_type': 'Legal Document'
                            })
                        
                        documents.extend(docs)
                        logger.info(f"Successfully loaded {txt_file.name} (encoding: {encoding})")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    logger.error(f"Unable to load file {txt_file.name}: encoding issue")
                    
            except Exception as e:
                logger.error(f"Error loading file {txt_file.name}: {str(e)}")
                
        logger.info(f"Total loaded {len(documents)} documents")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks
        
        Args:
            documents: List of original documents
            
        Returns:
            List of split document chunks
        """
        logger.info("Starting document splitting...")
        
        split_docs = self.text_splitter.split_documents(documents)
        
        # Add chunk numbering to split documents
        for i, doc in enumerate(split_docs):
            doc.metadata['chunk_id'] = i
            doc.metadata['chunk_size'] = len(doc.page_content)
            
        logger.info(f"Document splitting completed, generated {len(split_docs)} document chunks")
        return split_docs
    
    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """
        Create vector database
        
        Args:
            documents: List of document chunks
            
        Returns:
            ChromaDB vector database instance
        """
        logger.info("Starting vector database creation...")
        
        try:
            # Create ChromaDB vector database
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(self.vector_db_dir),
                collection_name="legal_documents"
            )
            
            # Persist database
            vectorstore.persist()
            
            logger.info(f"Vector database creation completed, stored in: {self.vector_db_dir}")
            logger.info(f"Database contains {len(documents)} document chunks")
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error creating vector database: {str(e)}")
            raise
    
    def build_knowledge_base(self) -> Chroma:
        """
        Build complete legal knowledge base
        
        Returns:
            ChromaDB vector database instance
        """
        logger.info("Starting legal knowledge base construction...")
        
        # 1. Load documents
        documents = self.load_documents()
        if not documents:
            raise ValueError("No loadable documents found")
        
        # 2. Split documents
        split_docs = self.split_documents(documents)
        
        # 3. Create vector database
        vectorstore = self.create_vector_store(split_docs)
        
        logger.info("Legal knowledge base construction completed!")
        return vectorstore
    
    def load_existing_vector_store(self) -> Chroma:
        """
        Load existing vector database
        
        Returns:
            ChromaDB vector database instance
        """
        if not self.vector_db_dir.exists():
            raise FileNotFoundError(f"Vector database directory does not exist: {self.vector_db_dir}")
            
        vectorstore = Chroma(
            persist_directory=str(self.vector_db_dir),
            embedding_function=self.embeddings,
            collection_name="legal_documents"
        )
        
        logger.info(f"Successfully loaded existing vector database: {self.vector_db_dir}")
        return vectorstore
    
    def search_similar_documents(self, vectorstore: Chroma, query: str, k: int = 5) -> List[Document]:
        """
        Search for similar documents
        
        Args:
            vectorstore: Vector database instance
            query: Query text
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        logger.info(f"Search query: {query}")
        
        results = vectorstore.similarity_search(query, k=k)
        
        logger.info(f"Found {len(results)} similar documents")
        return results

def main():
    """
    Main function - Example usage
    """
    # Please set your Google AI API key here
    GOOGLE_API_KEY = "your_google_api_key_here"
    
    if GOOGLE_API_KEY == "your_google_api_key_here":
        print("Please set your Google AI API key first!")
        print("You can get your API key at: https://makersuite.google.com/app/apikey")
        return
    
    try:
        # Initialize vector database builder
        builder = LegalDocumentVectorStore(
            google_api_key=GOOGLE_API_KEY,
            knowledge_dir="./knowledge",
            vector_db_dir="./vector_db",
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Build knowledge base
        vectorstore = builder.build_knowledge_base()
        
        # Example: Search for relevant regulations
        print("\n=== Testing Search Function ===")
        test_query = "contract breach liability"
        results = builder.search_similar_documents(vectorstore, test_query, k=3)
        
        print(f"\nQuery: {test_query}")
        print("Search results:")
        for i, doc in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Content preview: {doc.page_content[:200]}...")
            
    except Exception as e:
        logger.error(f"Program execution error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
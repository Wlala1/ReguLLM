#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
法规知识库测试脚本
用于测试向量数据库的功能
"""

from trans_embedding import LegalDocumentVectorStore
import os

def test_knowledge_base():
    """测试知识库功能"""
    
    # 检查API密钥
    api_key = os.getenv('GOOGLE_API_KEY', 'your_google_api_key_here')
    
    if api_key == 'your_google_api_key_here':
        print("请设置GOOGLE_API_KEY环境变量或修改config.py文件中的API密钥")
        print("获取API密钥地址: https://makersuite.google.com/app/apikey")
        return
    
    try:
        # 初始化向量数据库构建器
        builder = LegalDocumentVectorStore(
            google_api_key=api_key,
            knowledge_dir="./knowledge",
            vector_db_dir="./vector_db"
        )
        
        # 检查是否存在现有数据库
        if os.path.exists("./vector_db"):
            print("发现现有向量数据库，正在加载...")
            vectorstore = builder.load_existing_vector_store()
        else:
            print("未发现现有数据库，开始构建新的知识库...")
            vectorstore = builder.build_knowledge_base()
        
        # 测试搜索功能
        test_queries = [
            "合同违约责任",
            "行政处罚",
            "民事诉讼程序",
            "刑事责任",
            "损害赔偿"
        ]
        
        print("\n=== 测试搜索功能 ===")
        for query in test_queries:
            print(f"\n搜索查询: {query}")
            results = builder.search_similar_documents(vectorstore, query, k=2)
            
            for i, doc in enumerate(results, 1):
                print(f"  结果 {i}:")
                print(f"    来源: {doc.metadata.get('source', 'Unknown')}")
                print(f"    内容: {doc.page_content[:100]}...")
                print(f"    相关度评分可通过similarity_search_with_score获取")
        
        print("\n✅ 知识库测试完成！")
        
    except Exception as e:
        print(f"❌ 测试过程中出错: {str(e)}")

def interactive_search():
    """交互式搜索"""
    
    api_key = os.getenv('GOOGLE_API_KEY', 'your_google_api_key_here')
    
    if api_key == 'your_google_api_key_here':
        print("请设置GOOGLE_API_KEY环境变量")
        return
    
    try:
        builder = LegalDocumentVectorStore(google_api_key=api_key)
        
        if not os.path.exists("./vector_db"):
            print("未找到向量数据库，请先运行构建脚本")
            return
            
        vectorstore = builder.load_existing_vector_store()
        
        print("=== 法规知识库交互式搜索 ===")
        print("输入'quit'或'exit'退出")
        
        while True:
            query = input("\n请输入搜索关键词: ").strip()
            
            if query.lower() in ['quit', 'exit', '退出']:
                print("再见！")
                break
                
            if not query:
                continue
                
            try:
                results = builder.search_similar_documents(vectorstore, query, k=3)
                
                print(f"\n找到 {len(results)} 个相关结果:")
                for i, doc in enumerate(results, 1):
                    print(f"\n--- 结果 {i} ---")
                    print(f"来源文件: {doc.metadata.get('source', 'Unknown')}")
                    print(f"内容: {doc.page_content[:300]}...")
                    
            except Exception as e:
                print(f"搜索出错: {str(e)}")
                
    except Exception as e:
        print(f"程序初始化出错: {str(e)}")

if __name__ == "__main__":
    print("选择运行模式:")
    print("1. 自动测试")
    print("2. 交互式搜索")
    
    choice = input("请选择 (1/2): ").strip()
    
    if choice == "1":
        test_knowledge_base()
    elif choice == "2":
        interactive_search()
    else:
        print("无效选择，运行自动测试...")
        test_knowledge_base()
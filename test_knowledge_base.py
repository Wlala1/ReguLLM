#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
法规知识库测试脚本
用于测试向量数据库的功能和自动化人工反馈流程
"""

from trans_embedding import LegalDocumentVectorStore
import os
import json
from confidence_agent import ConfidenceAgent
from main import OptimizedLegalClassifier

def test_knowledge_base_and_feedback():
    """测试知识库功能和反馈系统"""
    
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

def automated_feedback_workflow():
    """
    自动化人工反馈工作流
    
    实现自动化的人工反馈流程：
    1. 初始化置信度评估Agent
    2. 处理测试案例
    3. 对需要人工审核的案例进行交互式审核
    4. 将人工反馈添加到知识库
    5. 使用更新后的知识库处理新案例
    """
    print("\n=== 自动化人工反馈工作流 ===")
    
    # 设置路径
    vector_db_path = os.path.join(os.getcwd(), "dynamic_legal_graph_db")
    knowledge_base_path = os.path.join(os.getcwd(), "feedback_knowledge_base.pkl")
    
    print(f"向量数据库路径: {vector_db_path}")
    print(f"反馈知识库路径: {knowledge_base_path}")
    
    # 初始化组件
    print("\n初始化法律分类器和置信度评估Agent...")
    legal_classifier = OptimizedLegalClassifier(graph_db_path=vector_db_path)
    confidence_agent = ConfidenceAgent(
        legal_classifier=legal_classifier,
        confidence_threshold=0.7,
        knowledge_base_path=knowledge_base_path,
        use_feedback_learning=True  # 启用反馈学习功能
    )
    print("✓ 初始化成功")
    
    # 测试案例
    test_cases = [
        {
            "name": "",
            "description": "A video filter feature is available globally except KR"
        },

        {
            "name": "",
            "description": "Geofences feature rollout in US for market testing"
        },

        {
            "name": "",
            "description": "A video filter feature is available globally except KR"
        },

        {
            "name": "",
            "description": "A picture filter feature is available globally except Indonesia"
        },
        {
            "name": "",
            "description": "A text filter feature is available globally except China"
        },
        {
            "name": "",
            "description": "A picture filter feature is available globally except Thailand"
        }
    ]
    
    # 处理测试案例
    results = []
    for i, case in enumerate(test_cases):
        print(f"\n测试案例 {i+1}/{len(test_cases)}: {case['name']}")
        print(f"描述: {case['description'][:100]}...")
        
        # 执行完整处理流程
        feature_description = case['description']
        result = confidence_agent.process_feature(feature_description)
        
        # 添加案例名称
        result['case_name'] = case['name']
        
        # 保存结果
        results.append(result)
        
        # 检查是否需要人工审核
        # 当系统标记需要人工审核或最终分类为UnspecifiedNeedsHuman时触发人工审核
        if result['final_result']['needs_human_review'] or result['final_result']['assessment'] == "UnspecifiedNeedsHuman":
            print("\n需要人工审核，启动交互式审核流程...")
            
            # 显示案例信息
            print(f"\n{'='*80}")
            print(f"人工审核界面 - 案例 #{i+1}")
            print(f"{'='*80}")
            print(f"功能描述: {feature_description}")
            print(f"原始分类: {result['final_result']['assessment']}")
            print(f"置信度: {result['final_result']['confidence']}")
            
            # 获取人工输入
            print("\n请选择正确的分类:")
            print("1. LegalRequirement")
            print("2. BusinessDriven")
            print("3. UnspecifiedNeedsHuman")
            
            while True:
                choice = input("请输入选择 (1/2): ").strip()
                if choice in ["1", "2"]:
                    break
                print("无效选择，请重新输入")
            
            # 映射选择到分类标签
            human_assessment_map = {
                "1": "LegalRequirement",
                "2": "BusinessDriven",
            }
            human_assessment = human_assessment_map[choice]
            
            # 获取审核员备注
            notes = input("请输入审核备注 (可选): ").strip()
            
            # 添加人工反馈到知识库
            metadata = {
                "reviewer": "交互式审核员",
                "confidence": "high",
                "notes": notes if notes else "无备注"
            }
            
            # 使用之前提取的feature_description
            feedback_case = confidence_agent.add_human_feedback(
                feature_description=feature_description,
                original_assessment=result['final_result']['assessment'],
                human_assessment=human_assessment,
                metadata=metadata
            )
            
            print(f"\n✓ 反馈案例已添加到知识库: #{feedback_case['id']}")
    
    # 结果摘要
    print("\n=== 测试结果摘要 ===")
    for i, result in enumerate(results):
        print(f"\n案例 #{i+1}:")
        print(f"  功能描述: {test_cases[i]['description'][:50]}...")
        print(f"  最终分类: {result['final_result']['assessment']}")
        print(f"  置信度: {result['final_result']['confidence']}")
        print(f"  需要人工审核: {result['final_result']['needs_human_review']}")
        
        # 显示相似案例信息
        similar_cases = result['confidence_evaluation'].get('similar_cases', [])
        if similar_cases:
            print(f"  使用了 {len(similar_cases)} 个相似反馈案例")
    
    # 导出知识库统计信息
    if confidence_agent.feedback_kb:
        stats = confidence_agent.feedback_kb.get_statistics()
        print(f"\n知识库统计:")
        print(f"  总案例数: {stats['total_cases']}")
        print(f"  修正率: {stats['correction_rate']:.2%}")

if __name__ == "__main__":
    print("选择运行模式:")
    print("1. 知识库测试")
    print("2. 交互式搜索")
    print("3. 自动化人工反馈工作流")
    
    choice = input("请选择 (1/2/3): ").strip()
    
    if choice == "1":
        test_knowledge_base_and_feedback()
    elif choice == "2":
        interactive_search()
    elif choice == "3":
        automated_feedback_workflow()
    else:
        print("无效选择，运行知识库测试...")
        test_knowledge_base_and_feedback()
import json
import os
from confidence_agent import ConfidenceAgent
from main import OptimizedLegalClassifier

def test_confidence_agent(use_feedback_learning=False):
    """测试置信度评估Agent的功能
    """
    print("=== 置信度评估Agent - 测试脚本 ===\n")
    
    # 设置向量数据库路径
    vector_db_path = os.path.join(os.getcwd(), "dynamic_legal_graph_db")
    knowledge_base_path = os.path.join(os.getcwd(), "feedback_knowledge_base.pkl")
    print(f"向量数据库路径: {vector_db_path}")
    
    # 初始化法律分类器和置信度评估Agent
    try:
        print("初始化法律分类器...")
        legal_classifier = OptimizedLegalClassifier(
            graph_db_path=vector_db_path
        )
        
        print("\n初始化置信度评估Agent...")
        confidence_agent = ConfidenceAgent(
            confidence_threshold=0.7,  # 设置置信度阈值
            legal_classifier=legal_classifier,
            knowledge_base_path=knowledge_base_path,
            use_feedback_learning=use_feedback_learning
        )
        print("✓ 初始化成功\n")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        return
    
    # 测试数据 - 包含不同置信度级别的案例
    test_cases = [
        # 案例1: 可能高置信度的案例 - 明确的法律要求
        {
            "name": "Curfew login blocker with ASL and GH for Utah minors",
            "description": "To comply with the Utah Social Media Regulation Act, we are implementing a curfew-based login restriction for users under 18. The system uses ASL to detect minor accounts and routes enforcement through GH to apply only within Utah boundaries."
        },
        
        # 案例2: 可能中等置信度的案例 - 部分法律依据
        {
            "name": "PF default toggle with NR enforcement for California teens",
            "description": "As part of compliance with California's SB976, the app will disable PF by default for users under 18 located in California. This default setting is considered NR to override, unless explicit parental opt-in is provided. Geo-detection is handled via GH, and rollout is monitored with FR logs. The design ensures minimal disruption while meeting the strict personalization requirements imposed by the law."
        },
        
        # 案例3: 可能低置信度的案例 - 缺乏明确法律依据
        {
            "name": "Child abuse content scanner using T5 and CDS triggers",
            "description": "In line with the US federal law requiring providers to report child sexual abuse content to NCMEC, this feature scans uploads and flags suspected materials tagged as T5. Once flagged, the CDS auto-generates reports and routes them via secure channel APIs. The logic runs in real-time, supports human validation, and logs detection metadata for internal audits. Regional thresholds are governed by LCP parameters in the backend."
        },
        
        # 案例4: 可能混合信号的案例 - 法律和商业因素混合
        {
            "name": "Feature reads user location to enforce France's copyright rules",
            "description": ""
        },
        
        # 案例5: 可能需要人工干预的案例 - 信息不足
        {
            "name": "",
            "description": "Requires age gates specific to Indonesia's Child Protection Law"
        },

        {
            "name": "",
            "description": "Geofences feature rollout in US for market testing"
        },

        {
            "name": "",
            "description": "A video filter feature is available globally except KR"
        }
    ]
    
    # 处理测试案例
    results = []
    for i, case in enumerate(test_cases):
        print(f"\n测试案例 {i+1}/{len(test_cases)}: {case['name']}")
        print(f"描述: {case['description'][:100]}...")
        
        # 执行完整处理流程
        try:
            result = confidence_agent.process_feature(case['description'])
            
            # 添加案例名称
            result['case_name'] = case['name']
            
            # 保存结果
            results.append(result)
            
            # 显示关键结果
            print(f"\n结果摘要:")
            print(f"  原始分类: {result['original_classification']['assessment']}")
            print(f"  原始置信度: {result['original_classification']['confidence']}")
            print(f"  是否需要反思: {result['confidence_evaluation']['needs_reflection']}")
            print(f"  最终分类: {result['final_result']['assessment']}")
            print(f"  最终置信度: {result['final_result']['confidence']}")
            print(f"  是否需要人工审核: {result['final_result']['needs_human_review']}")
            
            if result['confidence_evaluation']['needs_reflection'] and result['confidence_evaluation']['reflection_details']:
                reflection = result['confidence_evaluation']['reflection_details']
                print(f"\n反思分析:")
                print(f"  决策: {reflection.get('decision', 'N/A')}")
                print(f"  证据分析: {reflection.get('reflection', {}).get('evidence_analysis', 'N/A')[:100]}...")
                print(f"  逻辑分析: {reflection.get('reflection', {}).get('logic_analysis', 'N/A')[:100]}...")
                print(f"  不确定性来源: {reflection.get('reflection', {}).get('uncertainty_source', 'N/A')[:100]}...")
                
                # 显示相似案例信息
                similar_cases = result['confidence_evaluation'].get('similar_cases', [])
                if similar_cases:
                    print(f"\n参考的相似案例:")
                    for i, case in enumerate(similar_cases):
                        print(f"  案例 #{i+1}: {case['id']}")
                        print(f"    原始分类: {case['original_assessment']}")
                        print(f"    人工修正: {case['human_assessment']}")
                        print(f"    相似度: {case['similarity']:.2f}")
            
        except Exception as e:
            print(f"✗ 处理失败: {e}")
            results.append({
                "case_name": case['name'],
                "error": str(e),
                "status": "failed"
            })
        
        print(f"{('='*80)}")
    
    # 保存结果到JSON文件
    try:
        output_file = 'test_confidence_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False, default=str)
        print(f"\n✓ 结果已保存到 {output_file}")
        print(f"✓ 共处理 {len(results)} 个测试案例")
        
        # 统计结果
        needs_human_count = sum(1 for r in results if r.get('final_result', {}).get('needs_human_review', False))
        reflection_count = sum(1 for r in results if r.get('confidence_evaluation', {}).get('needs_reflection', False))
        
        print(f"\n统计信息:")
        print(f"  需要反思的案例数: {reflection_count}/{len(results)}")
        print(f"  需要人工审核的案例数: {needs_human_count}/{len(results)}")
        
    except Exception as e:
        print(f"\n✗ 保存结果失败: {e}")


def test_human_feedback():
    """测试人工反馈功能"""
    print(f"\n{'='*100}")
    print(f"测试人工反馈功能")
    print(f"{'='*100}")
    
    try:
        # 初始化带有反馈学习功能的Agent
        vector_db_path = os.path.join(os.getcwd(), "dynamic_legal_graph_db")
        knowledge_base_path = os.path.join(os.getcwd(), "feedback_knowledge_base.pkl")
        
        print(f"向量数据库路径: {vector_db_path}")
        print(f"反馈知识库路径: {knowledge_base_path}")
        
        # 初始化法律分类器
        print("\n初始化法律分类器...")
        legal_classifier = OptimizedLegalClassifier(
            graph_db_path=vector_db_path
        )
        print("✓ 法律分类器初始化成功")
        
        # 初始化置信度评估Agent（启用反馈学习）
        print("\n初始化置信度评估Agent（启用反馈学习）...")
        confidence_agent = ConfidenceAgent(
            confidence_threshold=0.7,
            legal_classifier=legal_classifier,
            knowledge_base_path=knowledge_base_path,
            use_feedback_learning=True
        )
        print("✓ 置信度评估Agent初始化成功\n")
        
        # 测试案例
        feature_description = "我们需要添加一个功能，允许用户下载自己的所有数据，包括个人信息和使用记录。这是为了遵守数据可携权的要求。"
        
        # 第一步：处理功能并获取初始结果
        print("\n第一步：处理功能并获取初始结果...")
        result = confidence_agent.process_feature(feature_description)
        
        original_assessment = result['final_result']['assessment']
        needs_human_review = result['final_result']['needs_human_review']
        
        # 第二步：模拟人工审核并添加反馈
        print("\n第二步：模拟人工审核并添加反馈...")
        if needs_human_review:
            print("需要人工审核，进行人工标注...")
            # 模拟人工审核结果
            human_assessment = "法律要求"  # 假设人工审核确定这是法律要求
            
            # 添加人工反馈到知识库
            feedback_case = confidence_agent.add_human_feedback(
                feature_description=feature_description,
                original_assessment=original_assessment,
                human_assessment=human_assessment,
                metadata={
                    "source": "test_case",
                    "reviewer": "test_user",
                    "confidence": "high"
                }
            )
            
            print(f"\n反馈案例已添加: {feedback_case['id']}")
        else:
            print("不需要人工审核，跳过反馈添加")
        
        # 第三步：测试相似案例的影响
        print("\n第三步：测试相似案例的影响...")
        similar_feature = "我们需要实现一个功能，让用户能够导出自己的所有数据，这是GDPR中数据可携权的要求。"
        
        print("\n处理相似功能描述...")
        result_with_feedback = confidence_agent.process_feature(similar_feature)
        
        # 检查是否使用了相似案例
        similar_cases = result_with_feedback['confidence_evaluation'].get('similar_cases', [])
        if similar_cases:
            print(f"\n成功！系统使用了 {len(similar_cases)} 个相似反馈案例进行决策")
        else:
            print("\n注意：系统未使用任何相似反馈案例")
            
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_confidence_agent(use_feedback_learning=True)
    
    # 测试人工反馈功能
    test_human_feedback()
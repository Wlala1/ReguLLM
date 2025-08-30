import json
import os
from confidence_agent import ConfidenceAgent
from main import EnhancedLegalClassifier

def test_confidence_agent():
    """
    测试置信度评估Agent的功能
    """
    print("=== 置信度评估Agent - 测试脚本 ===\n")
    
    # 设置向量数据库路径
    vector_db_path = os.path.join(os.getcwd(), "legal_compliance_db1")
    print(f"向量数据库路径: {vector_db_path}")
    
    # 初始化法律分类器和置信度评估Agent
    try:
        print("初始化法律分类器...")
        legal_classifier = EnhancedLegalClassifier(
            graph_db_path=vector_db_path
        )
        
        print("\n初始化置信度评估Agent...")
        confidence_agent = ConfidenceAgent(
            confidence_threshold=0.7,  # 设置置信度阈值
            legal_classifier=legal_classifier
        )
        print("✓ 初始化成功\n")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        return
    
    # 测试数据 - 包含不同置信度级别的案例
    test_cases = [
        # 案例1: 可能高置信度的案例 - 明确的法律要求
        {
            "name": "明确的法律要求案例",
            "description": "Curfew login blocker with ASL and GH for Utah minors: To comply with the Utah Social Media Regulation Act, we are implementing a curfew-based login restriction for users under 18. The system uses ASL to detect minor accounts and routes enforcement through GH to apply only within Utah boundaries."
        },
        
        # 案例2: 可能中等置信度的案例 - 部分法律依据
        {
            "name": "部分法律依据案例",
            "description": "Content visibility lock with NSP for EU DSA: To meet the transparency expectations of the EU Digital Services Act, we are introducing a visibility lock for flagged user-generated content labeled under NSP. When such content is detected, a soft Softblock is applied."
        },
        
        # 案例3: 可能低置信度的案例 - 缺乏明确法律依据
        {
            "name": "缺乏明确法律依据案例",
            "description": "Universal PF deactivation on guest mode: By default, PF will be turned off for all users browsing in guest mode."
        },
        
        # 案例4: 可能混合信号的案例 - 法律和商业因素混合
        {
            "name": "混合信号案例",
            "description": "T5 tagging for sensitive reports: When users report content containing high-risk information, it is tagged as T5 for internal routing. CDS then enforces escalation. The system is universal and does not rely on regional toggles."
        },
        
        # 案例5: 可能需要人工干预的案例 - 信息不足
        {
            "name": "信息不足案例",
            "description": "NSP auto-flagging: This feature will automatically detect and tag content that violates NSP policy. Once flagged, Softblock is applied and a Redline alert is generated if downstream sharing is attempted."
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
            
        except Exception as e:
            print(f"✗ 处理失败: {e}")
            results.append({
                "case_name": case['name'],
                "error": str(e),
                "status": "failed"
            })
        
        print(f"{'='*80}")
    
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


if __name__ == "__main__":
    test_confidence_agent()
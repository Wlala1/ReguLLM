import json
import re
from typing import Dict, List, Tuple, Optional, Any
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from main import OptimizedLegalClassifier

class ConfidenceAgent:
    """
    置信度评估Agent - 对置信度低的功能标签进行反思和判断
    
    工作流程：
    1. 接收OptimizedLegalClassifier的分类结果
    2. 评估置信度是否低于阈值
    3. 如果置信度低，进行深度反思分析
    4. 输出最终判断：确认原标签、修正标签或标记为需要人工干预
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 model_name: str = "qwen-max",
                 legal_classifier: Optional[OptimizedLegalClassifier] = None):
        """
        初始化置信度评估Agent
        
        Args:
            confidence_threshold: 置信度阈值，低于此值将触发深度反思
            model_name: 使用的大语言模型名称
            legal_classifier: 可选的法律分类器实例，如果为None则创建新实例
        """
        print("=== 初始化置信度评估Agent ===")
        
        # 设置置信度阈值
        self.confidence_threshold = confidence_threshold
        print(f"1. 设置置信度阈值: {confidence_threshold}")
        
        # 初始化大语言模型
        print("2. 初始化大语言模型...")
        self.llm = ChatTongyi(model=model_name, temperature=0.2)
        
        # 初始化或复用法律分类器
        print("3. 初始化法律分类器...")
        self.legal_classifier = legal_classifier if legal_classifier else OptimizedLegalClassifier()
        
        # 初始化输出解析器
        print("4. 初始化输出解析器...")
        self.parser = JsonOutputParser()
        
        # 创建反思提示模板
        print("5. 创建反思提示模板...")
        self.reflection_prompt = self._create_reflection_prompt()
        
        print("初始化完成！\n")
    
    def _create_reflection_prompt(self) -> PromptTemplate:
        """
        创建反思分析的提示模板
        """
        template = """
你是一位资深的法律合规分析专家，具有批判性思维和反思能力。

任务：对置信度较低的功能分类结果进行深度反思和重新评估，决定是否需要修正标签或人工干预。

原始分类结果：
```
{original_result}
```

原始功能描述：
{feature_description}

原始分类标签：{original_assessment}
原始置信度：{original_confidence}

反思分析框架：
1. 证据评估：
   - 原始分类的证据是否充分？
   - 是否存在证据冲突或不一致？
   - 是否缺少关键信息？

2. 逻辑评估：
   - 原始分类的推理是否合理？
   - 是否存在逻辑漏洞？
   - 是否考虑了所有相关因素？

3. 替代解释：
   - 是否存在其他合理的分类可能性？
   - 不同分类标签的支持证据如何？

4. 不确定性来源：
   - 置信度低的主要原因是什么？
   - 是证据不足、证据冲突还是解释多样性？

请基于以上分析，做出以下三种决策之一：
1. CONFIRM_ORIGINAL - 确认原始标签正确，尽管置信度不高
2. REVISE_TO_NEW - 修正为新标签（必须是LegalRequirement、BusinessDriven或UnspecifiedNeedsHuman之一）
3. NEEDS_HUMAN_REVIEW - 无法确定，需要人工审核

输出格式：
仅返回有效的JSON：
```
{{
  "decision": "CONFIRM_ORIGINAL" | "REVISE_TO_NEW" | "NEEDS_HUMAN_REVIEW",
  "revised_assessment": "LegalRequirement" | "BusinessDriven" | "UnspecifiedNeedsHuman" | null,
  "revised_confidence": 0.10-0.99,
  "reflection": {{
    "evidence_analysis": "对原始证据的分析 ≤150字",
    "logic_analysis": "对原始逻辑的分析 ≤150字",
    "alternative_explanations": "可能的替代解释 ≤150字",
    "uncertainty_source": "不确定性的主要来源 ≤100字"
  }},
  "reasoning": "最终决策的理由 ≤200字"
}}
```

约束条件：
- 如果decision="CONFIRM_ORIGINAL"，则revised_assessment必须等于原始标签
- 如果decision="REVISE_TO_NEW"，则revised_assessment必须是三个有效标签之一且不同于原始标签
- 如果decision="NEEDS_HUMAN_REVIEW"，则revised_assessment必须为null
- revised_confidence必须是0.10到0.99之间的浮点数
- 所有文本字段必须简洁明了，不超过指定字符数
"""
        
        return PromptTemplate(
            template=template,
            input_variables=[
                "original_result", "feature_description", "original_assessment", "original_confidence"
            ]
        )
    
    def evaluate_confidence(self, classification_result: Dict[str, Any], feature_description: str) -> Dict[str, Any]:
        """
        评估分类结果的置信度，并在必要时进行深度反思
        
        Args:
            classification_result: OptimizedLegalClassifier的分类结果
            feature_description: 原始功能描述
            
        Returns:
            评估结果字典，包含原始分类和可能的修正
        """
        # 提取原始分类信息
        original_assessment = classification_result.get("assessment", "UnspecifiedNeedsHuman")
        original_confidence = classification_result.get("confidence", 0.0)
        
        print(f"\n{'='*60}")
        print(f"开始评估功能置信度: {feature_description[:80]}...")
        print(f"{'='*60}")
        print(f"原始分类: {original_assessment}")
        print(f"原始置信度: {original_confidence}")
        
        # 检查置信度是否低于阈值
        if original_confidence >= self.confidence_threshold:
            print(f"✓ 置信度高于阈值 {self.confidence_threshold}，无需深度反思")
            return {
                "original_result": classification_result,
                "needs_reflection": False,
                "reflection_result": None,
                "final_assessment": original_assessment,
                "final_confidence": original_confidence,
                "needs_human_review": False
            }
        
        # 置信度低，进行深度反思
        print(f"! 置信度低于阈值 {self.confidence_threshold}，开始深度反思...")
        
        try:
            # 准备反思输入
            inputs = {
                "original_result": json.dumps(classification_result, indent=2),
                "feature_description": feature_description,
                "original_assessment": original_assessment,
                "original_confidence": original_confidence
            }
            
            # 执行反思分析
            formatted_prompt = self.reflection_prompt.format(**inputs)
            response = self.llm.invoke(formatted_prompt)
            reflection_result = self.parser.parse(response.content)
            
            # 提取反思结果
            decision = reflection_result.get("decision", "NEEDS_HUMAN_REVIEW")
            revised_assessment = reflection_result.get("revised_assessment")
            revised_confidence = reflection_result.get("revised_confidence", 0.5)
            
            # 确定最终评估结果
            if decision == "CONFIRM_ORIGINAL":
                final_assessment = original_assessment
                final_confidence = revised_confidence  # 使用反思后的置信度
                needs_human_review = False
                print(f"✓ 反思结果: 确认原始标签 {original_assessment}")
                print(f"✓ 修正后置信度: {revised_confidence}")
                
            elif decision == "REVISE_TO_NEW":
                final_assessment = revised_assessment
                final_confidence = revised_confidence
                needs_human_review = False
                print(f"✓ 反思结果: 修正标签为 {revised_assessment}")
                print(f"✓ 修正后置信度: {revised_confidence}")
                
            else:  # NEEDS_HUMAN_REVIEW
                final_assessment = "UnspecifiedNeedsHuman"  # 默认为不确定
                final_confidence = revised_confidence
                needs_human_review = True
                print(f"! 反思结果: 需要人工审核")
                print(f"! 置信度: {revised_confidence}")
            
            # 返回完整结果
            return {
                "original_result": classification_result,
                "needs_reflection": True,
                "reflection_result": reflection_result,
                "final_assessment": final_assessment,
                "final_confidence": final_confidence,
                "needs_human_review": needs_human_review
            }
            
        except Exception as e:
            print(f"✗ 反思分析失败: {e}")
            return {
                "original_result": classification_result,
                "needs_reflection": True,
                "reflection_result": None,
                "final_assessment": "UnspecifiedNeedsHuman",  # 出错时默认需要人工审核
                "final_confidence": 0.0,
                "needs_human_review": True,
                "error": str(e)
            }
    
    def process_feature(self, feature_description: str) -> Dict[str, Any]:
        """
        处理功能描述的完整流程：分类 + 置信度评估
        
        Args:
            feature_description: 功能描述
            
        Returns:
            完整的处理结果
        """
        print(f"\n{'='*80}")
        print(f"开始处理功能: {feature_description[:100]}...")
        print(f"{'='*80}")
        
        # 第一步：使用法律分类器进行初步分类
        print("第一步: 使用法律分类器进行初步分类...")
        classification_result = self.legal_classifier.classify_feature(feature_description)
        
        # 第二步：评估置信度并在必要时进行反思
        print("\n第二步: 评估置信度并在必要时进行反思...")
        evaluation_result = self.evaluate_confidence(classification_result, feature_description)
        
        # 第三步：整合结果
        print("\n第三步: 整合最终结果...")
        final_result = {
            "feature_description": feature_description,
            "original_classification": {
                "assessment": classification_result.get("assessment"),
                "confidence": classification_result.get("confidence"),
                "reasoning": classification_result.get("reasoning")
            },
            "confidence_evaluation": {
                "threshold": self.confidence_threshold,
                "needs_reflection": evaluation_result.get("needs_reflection", False),
                "reflection_details": evaluation_result.get("reflection_result")
            },
            "final_result": {
                "assessment": evaluation_result.get("final_assessment"),
                "confidence": evaluation_result.get("final_confidence"),
                "needs_human_review": evaluation_result.get("needs_human_review", False)
            }
        }
        
        # 输出结果摘要
        print(f"\n结果摘要:")
        print(f"  原始分类: {final_result['original_classification']['assessment']}")
        print(f"  原始置信度: {final_result['original_classification']['confidence']}")
        print(f"  是否需要反思: {final_result['confidence_evaluation']['needs_reflection']}")
        print(f"  最终分类: {final_result['final_result']['assessment']}")
        print(f"  最终置信度: {final_result['final_result']['confidence']}")
        print(f"  是否需要人工审核: {final_result['final_result']['needs_human_review']}")
        
        return final_result


def main():
    """
    主函数演示置信度评估Agent的工作流程
    """
    print("=== 置信度评估Agent - 工作流程演示 ===\n")
    
    # 初始化法律分类器和置信度评估Agent
    try:
        legal_classifier = OptimizedLegalClassifier(
            graph_db_path="/Users/yanjin/vscode/ReguLLM/legal_compliance_db1"  # 请根据实际路径修改
        )
        confidence_agent = ConfidenceAgent(
            confidence_threshold=0.7,
            legal_classifier=legal_classifier
        )
        print("✓ Agent初始化成功\n")
    except Exception as e:
        print(f"✗ Agent初始化失败: {e}")
        return
    
    # 测试数据 - 包含不同置信度的案例
    test_cases = [
        # 高置信度案例
        "Curfew login blocker with ASL and GH for Utah minors: To comply with the Utah Social Media Regulation Act, we are implementing a curfew-based login restriction for users under 18. The system uses ASL to detect minor accounts and routes enforcement through GH to apply only within Utah boundaries.",
        
        # 中等置信度案例
        "Content visibility lock with NSP for EU DSA: To meet the transparency expectations of the EU Digital Services Act, we are introducing a visibility lock for flagged user-generated content labeled under NSP. When such content is detected, a soft Softblock is applied.",
        
        # 低置信度案例
        "Universal PF deactivation on guest mode: By default, PF will be turned off for all users browsing in guest mode."
    ]
    
    # 处理测试案例
    results = []
    for i, feature in enumerate(test_cases):
        print(f"\n测试案例 {i+1}/{len(test_cases)}")
        
        # 执行完整处理流程
        result = confidence_agent.process_feature(feature)
        
        # 保存结果
        results.append(result)
        print(f"{'='*80}")
    
    # 保存结果到JSON文件
    try:
        with open('confidence_evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False, default=str)
        print(f"\n✓ 结果已保存到 confidence_evaluation_results.json")
        print(f"✓ 共处理 {len(results)} 个测试案例")
    except Exception as e:
        print(f"\n✗ 保存结果失败: {e}")


if __name__ == "__main__":
    main()
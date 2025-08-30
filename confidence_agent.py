import json
import re
import os
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from main import OptimizedLegalClassifier


class FeedbackKnowledgeBase:
    """
    人工反馈知识库 - 存储和管理人工审核的结果，用于提高Agent的判别能力
    
    工作流程：
    1. 存储人工审核的结果（功能描述、原始分类、人工修正的分类）
    2. 提供检索功能，根据功能描述相似度查找相关案例
    3. 支持导出和导入知识库，便于持久化存储
    4. 为置信度评估Agent提供参考案例，提高判别能力
    """
    
    def __init__(self, knowledge_base_path: str = "feedback_knowledge_base.pkl"):
        """
        初始化人工反馈知识库
        
        Args:
            knowledge_base_path: 知识库文件路径
        """
        self.knowledge_base_path = knowledge_base_path
        self.feedback_cases = []
        self.load_knowledge_base()
    
    def add_feedback(self, feature_description: str, original_assessment: str, 
                    human_assessment: str, metadata: Dict = None) -> Dict:
        """
        添加人工反馈案例到知识库
        
        Args:
            feature_description: 功能描述
            original_assessment: 原始分类标签
            human_assessment: 人工修正的分类标签
            metadata: 额外元数据
            
        Returns:
            添加的反馈案例
        """
        if metadata is None:
            metadata = {}
            
        # 创建反馈案例
        feedback_case = {
            "id": len(self.feedback_cases) + 1,
            "feature_description": feature_description,
            "original_assessment": original_assessment,
            "human_assessment": human_assessment,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata
        }
        
        # 添加到知识库
        self.feedback_cases.append(feedback_case)
        
        # 保存知识库
        self.save_knowledge_base()
        
        return feedback_case
    
    def get_similar_cases(self, feature_description: str, top_k: int = 3) -> List[Dict]:
        """
        根据功能描述相似度查找相关案例
        
        Args:
            feature_description: 功能描述
            top_k: 返回的最相似案例数量
            
        Returns:
            相似案例列表
        """
        # 简单实现：基于关键词匹配的相似度计算
        # 在实际应用中，可以使用更复杂的相似度计算方法，如向量嵌入
        
        if not self.feedback_cases:
            return []
        
        # 将查询转换为小写并分词
        query_words = set(feature_description.lower().split())
        
        # 计算每个案例的相似度
        scored_cases = []
        for case in self.feedback_cases:
            case_words = set(case["feature_description"].lower().split())
            # 计算Jaccard相似度
            intersection = len(query_words.intersection(case_words))
            union = len(query_words.union(case_words))
            similarity = intersection / union if union > 0 else 0
            
            scored_cases.append((case, similarity))
        
        # 按相似度排序并返回top_k个案例
        scored_cases.sort(key=lambda x: x[1], reverse=True)
        return [case for case, _ in scored_cases[:top_k]]
    
    def save_knowledge_base(self) -> None:
        """
        保存知识库到文件
        """
        try:
            with open(self.knowledge_base_path, 'wb') as f:
                pickle.dump(self.feedback_cases, f)
            print(f"✓ 知识库已保存到 {self.knowledge_base_path}")
        except Exception as e:
            print(f"✗ 保存知识库失败: {e}")
    
    def load_knowledge_base(self) -> None:
        """
        从文件加载知识库
        """
        if os.path.exists(self.knowledge_base_path):
            try:
                with open(self.knowledge_base_path, 'rb') as f:
                    self.feedback_cases = pickle.load(f)
                print(f"✓ 已加载 {len(self.feedback_cases)} 个反馈案例")
            except Exception as e:
                print(f"✗ 加载知识库失败: {e}")
                self.feedback_cases = []
        else:
            print(f"! 知识库文件不存在，创建新知识库")
            self.feedback_cases = []
    
    def export_to_json(self, json_path: str = "feedback_knowledge_base.json") -> None:
        """
        导出知识库到JSON文件
        
        Args:
            json_path: JSON文件路径
        """
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_cases, f, indent=2, ensure_ascii=False)
            print(f"✓ 知识库已导出到 {json_path}")
        except Exception as e:
            print(f"✗ 导出知识库失败: {e}")
    
    def import_from_json(self, json_path: str = "feedback_knowledge_base.json") -> None:
        """
        从JSON文件导入知识库
        
        Args:
            json_path: JSON文件路径
        """
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    imported_cases = json.load(f)
                
                # 合并导入的案例
                existing_ids = {case["id"] for case in self.feedback_cases}
                for case in imported_cases:
                    if case["id"] not in existing_ids:
                        self.feedback_cases.append(case)
                        existing_ids.add(case["id"])
                
                # 保存合并后的知识库
                self.save_knowledge_base()
                
                print(f"✓ 已从 {json_path} 导入 {len(imported_cases)} 个反馈案例")
            except Exception as e:
                print(f"✗ 导入知识库失败: {e}")
        else:
            print(f"✗ JSON文件不存在: {json_path}")
    
    def get_statistics(self) -> Dict:
        """
        获取知识库统计信息
        
        Returns:
            统计信息字典
        """
        if not self.feedback_cases:
            return {
                "total_cases": 0,
                "assessment_distribution": {},
                "correction_rate": 0.0
            }
        
        # 计算分类分布
        original_distribution = {}
        human_distribution = {}
        corrections = 0
        
        for case in self.feedback_cases:
            # 原始分类分布
            orig = case["original_assessment"]
            original_distribution[orig] = original_distribution.get(orig, 0) + 1
            
            # 人工分类分布
            human = case["human_assessment"]
            human_distribution[human] = human_distribution.get(human, 0) + 1
            
            # 计算修正率
            if orig != human:
                corrections += 1
        
        return {
            "total_cases": len(self.feedback_cases),
            "original_distribution": original_distribution,
            "human_distribution": human_distribution,
            "correction_rate": corrections / len(self.feedback_cases)
        }

class ConfidenceAgent:
    """
    置信度评估Agent - 对置信度低的功能标签进行反思和判断
    
    工作流程：
    1. 接收OptimizedLegalClassifier的分类结果
    2. 评估置信度是否低于阈值
    3. 如果置信度低，进行深度反思分析
    4. 输出最终判断：确认原标签、修正标签或标记为需要人工干预
    5. 支持从人工审核结果中学习，不断提高判别能力
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 model_name: str = "qwen-max",
                 legal_classifier: Optional[OptimizedLegalClassifier] = None,
                 knowledge_base_path: str = "feedback_knowledge_base.pkl",
                 use_feedback_learning: bool = True):
        """
        初始化置信度评估Agent
        
        Args:
            confidence_threshold: 置信度阈值，低于此值将触发深度反思
            model_name: 使用的大语言模型名称
            legal_classifier: 可选的法律分类器实例，如果为None则创建新实例
            knowledge_base_path: 人工反馈知识库路径
            use_feedback_learning: 是否启用反馈学习
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
        
        # 初始化人工反馈知识库
        print("5. 初始化人工反馈知识库...")
        self.use_feedback_learning = use_feedback_learning
        if self.use_feedback_learning:
            self.feedback_kb = FeedbackKnowledgeBase(knowledge_base_path)
            kb_stats = self.feedback_kb.get_statistics()
            print(f"   - 已加载 {kb_stats['total_cases']} 个反馈案例")
            print(f"   - 修正率: {kb_stats['correction_rate']:.2%}")
        else:
            self.feedback_kb = None
            print("   - 反馈学习功能已禁用")
        
        # 创建反思提示模板
        print("6. 创建反思提示模板...")
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

{similar_cases_prompt}

反思分析框架：
1. 证据评估：
   - 原始分类的证据是否充分？
   - 是否存在证据冲突或不一致？
   - 是否缺少关键信息？
   {similar_cases_available}- 相似案例的证据是否支持原始分类？

2. 逻辑评估：
   - 原始分类的推理是否合理？
   - 是否存在逻辑漏洞？
   - 是否考虑了所有相关因素？
   {similar_cases_available}- 相似案例的分类逻辑是否适用于当前案例？

3. 替代解释：
   - 是否存在其他合理的分类可能性？
   - 不同分类标签的支持证据如何？
   {similar_cases_available}- 相似案例是否提供了其他可能的解释？

4. 不确定性来源：
   - 置信度低的主要原因是什么？
   - 是证据不足、证据冲突还是解释多样性？
   {similar_cases_available}- 相似案例是否有助于减少不确定性？

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
    {similar_cases_available},"similar_cases_analysis": "相似案例的启示 ≤150字"
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
{similar_cases_available}- 在分析中必须考虑相似案例的人工审核结果
"""
        
        return PromptTemplate(
            template=template,
            input_variables=[
                "original_result", "feature_description", "original_assessment", "original_confidence",
                "similar_cases_prompt", "similar_cases_available"
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
        
        # 准备反思输入
        inputs = {
            "original_result": json.dumps(classification_result, indent=2),
            "feature_description": feature_description,
            "original_assessment": original_assessment,
            "original_confidence": original_confidence,
            "similar_cases_prompt": "",
            "similar_cases_available": ""
        }
        
        # 如果启用了反馈学习，查找相似案例
        similar_cases = []
        if self.use_feedback_learning and self.feedback_kb:
            similar_cases = self.feedback_kb.get_similar_cases(feature_description, top_k=3)
            if similar_cases:
                print(f"✓ 找到 {len(similar_cases)} 个相似反馈案例")
                
                # 构建相似案例提示
                similar_cases_text = "\n相似案例（来自人工反馈知识库）：\n"
                for i, case in enumerate(similar_cases):
                    similar_cases_text += f"\n案例 {i+1}:\n"
                    similar_cases_text += f"功能描述: {case['feature_description'][:200]}...\n"
                    similar_cases_text += f"原始分类: {case['original_assessment']}\n"
                    similar_cases_text += f"人工修正: {case['human_assessment']}\n"
                
                inputs["similar_cases_prompt"] = similar_cases_text
                inputs["similar_cases_available"] = ""  # 启用相似案例分析相关提示
            else:
                print("! 未找到相似反馈案例")
        
        try:
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
                "similar_cases": similar_cases,
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
                "similar_cases": similar_cases,
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
                "reflection_details": evaluation_result.get("reflection_result"),
                "similar_cases": evaluation_result.get("similar_cases", [])
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
        
        # 如果使用了相似案例
        similar_cases = evaluation_result.get("similar_cases", [])
        if similar_cases:
            print(f"  参考了 {len(similar_cases)} 个相似反馈案例")
        
        return final_result
        
    def add_human_feedback(self, feature_description: str, original_assessment: str, 
                          human_assessment: str, metadata: Dict = None) -> Dict[str, Any]:
        """
        添加人工反馈到知识库，实现伪RLHF功能
        
        Args:
            feature_description: 功能描述
            original_assessment: 原始分类标签
            human_assessment: 人工修正的分类标签
            metadata: 额外元数据
            
        Returns:
            添加的反馈案例
        """
        if not self.use_feedback_learning or not self.feedback_kb:
            print("! 反馈学习功能未启用，无法添加人工反馈")
            return None
        
        print(f"\n{'='*60}")
        print(f"添加人工反馈到知识库...")
        print(f"{'='*60}")
        print(f"功能描述: {feature_description[:80]}...")
        print(f"原始分类: {original_assessment}")
        print(f"人工修正: {human_assessment}")
        
        # 添加到知识库
        feedback_case = self.feedback_kb.add_feedback(
            feature_description=feature_description,
            original_assessment=original_assessment,
            human_assessment=human_assessment,
            metadata=metadata
        )
        
        print(f"✓ 成功添加反馈案例 #{feedback_case['id']}")
        
        # 获取知识库统计信息
        stats = self.feedback_kb.get_statistics()
        print(f"\n知识库统计:")
        print(f"  总案例数: {stats['total_cases']}")
        print(f"  修正率: {stats['correction_rate']:.2%}")
        
        return feedback_case


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
import json
import re
import os
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.embeddings import HuggingFaceEmbeddings
from main import OptimizedLegalClassifier


class FeedbackKnowledgeBase:
    """
    Human Feedback Knowledge Base - Stores and manages human review results to improve Agent's discrimination ability
    
    Workflow:
    1. Store human review results (feature description, original classification, human-corrected classification)
    2. Provide retrieval functionality, finding relevant cases based on feature description similarity
    3. Support export and import of knowledge base for persistent storage
    4. Provide reference cases for confidence evaluation Agent to improve discrimination ability
    """
    
    def __init__(self, knowledge_base_path: str = "feedback_knowledge_base.pkl", embedding_model_name: str = "BAAI/bge-base-en-v1.5"):
        """
        Initialize human feedback knowledge base
        
        Args:
            knowledge_base_path: Knowledge base file path
            embedding_model_name: Embedding model name
        """
        self.knowledge_base_path = knowledge_base_path
        self.feedback_cases = []
        
        # Initialize embedding model
        print("Initializing embedding model...")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        print(f"✓ Loaded embedding model: {embedding_model_name}")
        
        self.load_knowledge_base()
    
    def add_feedback(self, feature_description: str, original_assessment: str, 
                    human_assessment: str, metadata: Dict = None, reasoning: str = None) -> Dict:
        """
        Add human feedback case to knowledge base
        
        Args:
            feature_description: Feature description
            original_assessment: Original classification label
            human_assessment: Human-corrected classification label
            metadata: Additional metadata
            reasoning: Reasoning information
            
        Returns:
            Added feedback case
        """
        if metadata is None:
            metadata = {}
            
        # Generate embedding
        try:
            embedding = self.embeddings.embed_query(feature_description)
            print(f"✓ Generated embedding vector (dimension: {len(embedding)})")
        except Exception as e:
            print(f"✗ Failed to generate embedding: {e}")
            embedding = None
            
        # Create feedback case
        feedback_case = {
            "id": len(self.feedback_cases) + 1,
            "feature_description": feature_description,
            "original_assessment": original_assessment,
            "human_assessment": human_assessment,
            "timestamp": datetime.now().isoformat(),
            "embedding": embedding,  # Store embedding
            "metadata": metadata,
            "reasoning": reasoning  # Store reasoning information
        }
        
        # Add to knowledge base
        self.feedback_cases.append(feedback_case)
        
        # Save knowledge base
        self.save_knowledge_base()
        
        return feedback_case
    
    def get_similar_cases(self, feature_description: str, top_k: int = 3) -> List[Dict]:
        """
        Find relevant cases based on feature description similarity
        
        Args:
            feature_description: Feature description
            top_k: Number of most similar cases to return
            
        Returns:
            List of similar cases
        """
        if not self.feedback_cases:
            return []
        
        print(f"Searching for cases similar to '{feature_description[:50]}...'")
        
        # Try using vector embeddings to calculate similarity
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(feature_description)
            print(f"✓ Generated query embedding vector")
            
            # Calculate cosine similarity
            scored_cases = []
            embedding_count = 0
            
            for case in self.feedback_cases:
                if "embedding" in case and case["embedding"] is not None:
                    # Use vector similarity
                    similarity = self._cosine_similarity(query_embedding, case["embedding"])
                    scored_cases.append((case, similarity))
                    embedding_count += 1
                else:
                    # Fall back to keyword matching
                    query_words = set(feature_description.lower().split())
                    case_words = set(case["feature_description"].lower().split())
                    # Calculate Jaccard similarity
                    intersection = len(query_words.intersection(case_words))
                    union = len(query_words.union(case_words))
                    similarity = intersection / union if union > 0 else 0
                    scored_cases.append((case, similarity))
            
            print(f"✓ Compared {embedding_count}/{len(self.feedback_cases)} cases using vector similarity")
            if embedding_count < len(self.feedback_cases):
                print(f"! {len(self.feedback_cases) - embedding_count} cases without embeddings, used keyword matching")
                
        except Exception as e:
            print(f"✗ Vector similarity calculation failed: {e}, falling back to keyword matching")
            
            # Fall back to keyword matching
            query_words = set(feature_description.lower().split())
            scored_cases = []
            
            for case in self.feedback_cases:
                case_words = set(case["feature_description"].lower().split())
                # Calculate Jaccard similarity
                intersection = len(query_words.intersection(case_words))
                union = len(query_words.union(case_words))
                similarity = intersection / union if union > 0 else 0
                
                scored_cases.append((case, similarity))
        
        # Sort by similarity and return top_k cases
        scored_cases.sort(key=lambda x: x[1], reverse=True)
        return [case for case, _ in scored_cases[:top_k]]
    
    def _cosine_similarity(self, vec1, vec2):
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity value
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def save_knowledge_base(self) -> None:
        """
        Save knowledge base to file
        """
        try:
            with open(self.knowledge_base_path, 'wb') as f:
                pickle.dump(self.feedback_cases, f)
            print(f"✓ Knowledge base saved to {self.knowledge_base_path}")
        except Exception as e:
            print(f"✗ Failed to save knowledge base: {e}")
    
    def load_knowledge_base(self) -> None:
        """
        Load knowledge base from file
        """
        if os.path.exists(self.knowledge_base_path):
            try:
                with open(self.knowledge_base_path, 'rb') as f:
                    self.feedback_cases = pickle.load(f)
                print(f"✓ Loaded {len(self.feedback_cases)} feedback cases")
            except Exception as e:
                print(f"✗ Failed to load knowledge base: {e}")
                self.feedback_cases = []
        else:
            print(f"! Knowledge base file does not exist, creating new knowledge base")
            self.feedback_cases = []
    
    def export_to_json(self, json_path: str = "feedback_knowledge_base.json") -> None:
        """
        Export knowledge base to JSON file
        
        Args:
            json_path: JSON file path
        """
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_cases, f, indent=2, ensure_ascii=False)
            print(f"✓ Knowledge base exported to {json_path}")
        except Exception as e:
            print(f"✗ Failed to export knowledge base: {e}")
    
    def import_from_json(self, json_path: str = "feedback_knowledge_base.json") -> None:
        """
        Import knowledge base from JSON file
        
        Args:
            json_path: JSON file path
        """
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    imported_cases = json.load(f)
                
                # Merge imported cases
                existing_ids = {case["id"] for case in self.feedback_cases}
                for case in imported_cases:
                    if case["id"] not in existing_ids:
                        self.feedback_cases.append(case)
                        existing_ids.add(case["id"])
                
                # Save merged knowledge base
                self.save_knowledge_base()
                
                print(f"✓ Imported {len(imported_cases)} feedback cases from {json_path}")
            except Exception as e:
                print(f"✗ Failed to import knowledge base: {e}")
        else:
            print(f"✗ JSON file does not exist: {json_path}")
    
    def get_statistics(self) -> Dict:
        """
        Get knowledge base statistics
        
        Returns:
            Statistics dictionary
        """
        if not self.feedback_cases:
            return {
                "total_cases": 0,
                "assessment_distribution": {},
                "correction_rate": 0.0
            }
        
        # Calculate classification distribution
        original_distribution = {}
        human_distribution = {}
        corrections = 0
        
        for case in self.feedback_cases:
            # Original classification distribution
            orig = case["original_assessment"]
            original_distribution[orig] = original_distribution.get(orig, 0) + 1
            
            # Human classification distribution
            human = case["human_assessment"]
            human_distribution[human] = human_distribution.get(human, 0) + 1
            
            # Calculate correction rate
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
    Confidence Evaluation Agent - Reflects on and judges feature labels with low confidence
    
    Workflow:
    1. Receive classification results from OptimizedLegalClassifier
    2. Evaluate if confidence is below threshold
    3. If confidence is low, perform deep reflection analysis
    4. Output final judgment: confirm original label, revise label, or mark for human intervention
    5. Support learning from human review results to continuously improve discrimination ability
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 model_name: str = "qwen-max-latest",
                 legal_classifier: Optional[OptimizedLegalClassifier] = None,
                 knowledge_base_path: str = "feedback_knowledge_base.pkl",
                 use_feedback_learning: bool = True,
                 embedding_model_name: str = "BAAI/bge-base-en-v1.5"):
        """
        Initialize confidence evaluation Agent
        
        Args:
            confidence_threshold: Confidence threshold, below which triggers deep reflection
            model_name: Large language model name to use
            legal_classifier: Optional legal classifier instance, creates new if None
            knowledge_base_path: Human feedback knowledge base path
            use_feedback_learning: Whether to enable feedback learning
            embedding_model_name: Model name for vector embeddings
        """
        print("=== Initializing Confidence Evaluation Agent ===")
        
        # Set confidence threshold
        self.confidence_threshold = confidence_threshold
        print(f"1. Set confidence threshold: {confidence_threshold}")
        
        # Initialize large language model
        print("2. Initializing large language model...")
        self.llm = ChatTongyi(model=model_name, temperature=0.2)
        
        # Initialize or reuse legal classifier
        print("3. Initializing legal classifier...")
        self.legal_classifier = legal_classifier if legal_classifier else OptimizedLegalClassifier()
        
        # Initialize output parser
        print("4. Initializing output parser...")
        self.parser = JsonOutputParser()
        
        # Initialize human feedback knowledge base
        print("5. Initializing human feedback knowledge base...")
        self.use_feedback_learning = use_feedback_learning
        if self.use_feedback_learning:
            self.feedback_kb = FeedbackKnowledgeBase(
                knowledge_base_path=knowledge_base_path,
                embedding_model_name=embedding_model_name
            )
            kb_stats = self.feedback_kb.get_statistics()
            print(f"   - Loaded {kb_stats['total_cases']} feedback cases")
            print(f"   - Correction rate: {kb_stats['correction_rate']:.2%}")
        else:
            self.feedback_kb = None
            print("   - Feedback learning feature disabled")
        
        # Create reflection prompt template
        print("6. Creating reflection prompt template...")
        self.reflection_prompt = self._create_reflection_prompt()
        
        print("Initialization complete!\n")
    
    def _create_reflection_prompt(self) -> PromptTemplate:
        """
        Create reflection analysis prompt template
        """
        template = """
You are a senior legal compliance analysis expert with critical thinking and reflection capabilities.

Task: Perform deep reflection and re-evaluation on feature classification results with low confidence, deciding whether to revise labels or require human intervention.

Original classification result:
```
{original_result}
```

Original feature description:
{feature_description}

Original classification label: {original_assessment}
Original confidence: {original_confidence}

{similar_cases_prompt}

Reflection analysis framework:
1. Evidence evaluation:
   - Is the evidence for the original classification sufficient?
   - Are there evidence conflicts or inconsistencies?
   - Is key information missing?
   {similar_cases_available}- Does evidence from similar cases support the original classification?

2. Logic evaluation:
   - Is the reasoning for the original classification sound?
   - Are there logical gaps?
   - Were all relevant factors considered?
   {similar_cases_available}- Is the classification logic from similar cases applicable to the current case?

3. Alternative explanations:
   - Are there other reasonable classification possibilities?
   - What supporting evidence exists for different classification labels?
   {similar_cases_available}- Do similar cases provide other possible explanations?

4. Sources of uncertainty:
   - What are the main reasons for low confidence?
   - Is it insufficient evidence, conflicting evidence, or interpretation diversity?
   {similar_cases_available}- Do similar cases help reduce uncertainty?

Based on the above analysis, make one of the following three decisions:
1. CONFIRM_ORIGINAL - Confirm original label is correct despite low confidence
2. REVISE_TO_NEW - Revise to new label (must be one of LegalRequirement, BusinessDriven, or UnspecifiedNeedsHuman)
3. NEEDS_HUMAN_REVIEW - Cannot determine, requires human review

Output format:
Return only valid JSON:
```
{{
  "decision": "CONFIRM_ORIGINAL" | "REVISE_TO_NEW" | "NEEDS_HUMAN_REVIEW",
  "revised_assessment": "LegalRequirement" | "BusinessDriven" | "UnspecifiedNeedsHuman" | null,
  "revised_confidence": 0.10-0.99,
  "reflection": {{
    "evidence_analysis": "Analysis of original evidence ≤150 chars",
    "logic_analysis": "Analysis of original logic ≤150 chars",
    "alternative_explanations": "Possible alternative explanations ≤150 chars",
    "uncertainty_source": "Main source of uncertainty ≤100 chars"
    {similar_cases_available},"similar_cases_analysis": "Insights from similar cases ≤150 chars"
  }},
  "reasoning": "Rationale for final decision ≤200 chars"
}}
```

Constraints:
- If decision="CONFIRM_ORIGINAL", revised_assessment must equal original label
- If decision="REVISE_TO_NEW", revised_assessment must be one of three valid labels and different from original
- If decision="NEEDS_HUMAN_REVIEW", revised_assessment must be null
- revised_confidence must be float between 0.10 and 0.99
- All text fields must be concise and not exceed specified character limits
{similar_cases_available}- Analysis must consider human review results from similar cases
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
        Evaluate confidence of classification results and perform deep reflection if necessary
        
        Args:
            classification_result: Classification result from OptimizedLegalClassifier
            feature_description: Original feature description
            
        Returns:
            Evaluation result dictionary containing original classification and possible revisions
        """
        # Extract original classification information
        original_assessment = classification_result.get("assessment", "UnspecifiedNeedsHuman")
        original_confidence = classification_result.get("confidence", 0.0)
        
        print(f"\n{'='*60}")
        print(f"Starting confidence evaluation for feature: {feature_description[:80]}...")
        print(f"{'='*60}")
        print(f"Original classification: {original_assessment}")
        print(f"Original confidence: {original_confidence}")
        
        # Check if confidence is below threshold
        if original_confidence >= self.confidence_threshold:
            print(f"✓ Confidence above threshold {self.confidence_threshold}, no deep reflection needed")
            return {
                "original_result": classification_result,
                "needs_reflection": False,
                "reflection_result": None,
                "final_assessment": original_assessment,
                "final_confidence": original_confidence,
                "needs_human_review": False
            }
        
        # Low confidence, perform deep reflection
        print(f"! Confidence below threshold {self.confidence_threshold}, starting deep reflection...")
        
        # Prepare reflection input
        inputs = {
            "original_result": json.dumps(classification_result, indent=2),
            "feature_description": feature_description,
            "original_assessment": original_assessment,
            "original_confidence": original_confidence,
            "similar_cases_prompt": "",
            "similar_cases_available": ""
        }
        
        # If feedback learning is enabled, find similar cases
        similar_cases = []
        if self.use_feedback_learning and self.feedback_kb:
            similar_cases = self.feedback_kb.get_similar_cases(feature_description, top_k=3)
            if similar_cases:
                print(f"✓ Found {len(similar_cases)} similar feedback cases")
                
                # Build similar cases prompt
                similar_cases_text = "\nSimilar cases (from human feedback knowledge base):\n"
                for i, case in enumerate(similar_cases):
                    similar_cases_text += f"\nCase {i+1}:\n"
                    similar_cases_text += f"Feature description: {case['feature_description'][:200]}...\n"
                    similar_cases_text += f"Original classification: {case['original_assessment']}\n"
                    similar_cases_text += f"Human correction: {case['human_assessment']}\n"
                
                inputs["similar_cases_prompt"] = similar_cases_text
                inputs["similar_cases_available"] = ""  # Enable similar cases analysis related prompts
            else:
                print("! No similar feedback cases found")
        
        try:
            # Execute reflection analysis
            formatted_prompt = self.reflection_prompt.format(**inputs)
            response = self.llm.invoke(formatted_prompt)
            reflection_result = self.parser.parse(response.content)
            
            # Extract reflection results
            decision = reflection_result.get("decision", "NEEDS_HUMAN_REVIEW")
            revised_assessment = reflection_result.get("revised_assessment")
            revised_confidence = reflection_result.get("revised_confidence", 0.5)
            
            # Determine final evaluation result
            if decision == "CONFIRM_ORIGINAL":
                final_assessment = original_assessment
                final_confidence = revised_confidence  # Use post-reflection confidence
                needs_human_review = False
                print(f"✓ Reflection result: Confirmed original label {original_assessment}")
                print(f"✓ Revised confidence: {revised_confidence}")
                
            elif decision == "REVISE_TO_NEW":
                final_assessment = revised_assessment
                final_confidence = revised_confidence
                needs_human_review = False
                print(f"✓ Reflection result: Revised label to {revised_assessment}")
                print(f"✓ Revised confidence: {revised_confidence}")
                
            else:  # NEEDS_HUMAN_REVIEW
                final_assessment = "UnspecifiedNeedsHuman"  # Default to uncertain
                final_confidence = revised_confidence
                needs_human_review = True
                print(f"! Reflection result: Requires human review")
                print(f"! Confidence: {revised_confidence}")
            
            # Return complete result
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
            print(f"✗ Reflection analysis failed: {e}")
            return {
                "original_result": classification_result,
                "needs_reflection": True,
                "reflection_result": None,
                "similar_cases": similar_cases,
                "final_assessment": "UnspecifiedNeedsHuman",  # Default to human review on error
                "final_confidence": 0.0,
                "needs_human_review": True,
                "error": str(e)
            }
    
    def process_feature(self, feature_description: str) -> Dict[str, Any]:
        """
        Complete workflow for processing feature description: classification + confidence evaluation
        
        Args:
            feature_description: Feature description
            
        Returns:
            Complete processing result
        """
        print(f"\n{'='*80}")
        print(f"Starting to process feature: {feature_description[:100]}...")
        print(f"{'='*80}")
        
        # Step 1: Use legal classifier for initial classification
        print("Step 1: Using legal classifier for initial classification...")
        classification_result = self.legal_classifier.classify_feature(feature_description)
        
        # Step 2: Evaluate confidence and perform reflection if necessary
        print("\nStep 2: Evaluating confidence and performing reflection if necessary...")
        evaluation_result = self.evaluate_confidence(classification_result, feature_description)
        
        # Step 3: Integrate results
        print("\nStep 3: Integrating final results...")
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
        
        # Output result summary
        print(f"\nResult Summary:")
        print(f"  Original classification: {final_result['original_classification']['assessment']}")
        print(f"  Original confidence: {final_result['original_classification']['confidence']}")
        print(f"  Needs reflection: {final_result['confidence_evaluation']['needs_reflection']}")
        print(f"  Final classification: {final_result['final_result']['assessment']}")
        print(f"  Final confidence: {final_result['final_result']['confidence']}")
        print(f"  Needs human review: {final_result['final_result']['needs_human_review'] or final_result['final_result']['assessment'] == 'UnspecifiedNeedsHuman'}")
        
        # If similar cases were used
        similar_cases = evaluation_result.get("similar_cases", [])
        if similar_cases:
            print(f"  Referenced {len(similar_cases)} similar feedback cases")
        
        return final_result
        
    def add_human_feedback(self, feature_description: str, original_assessment: str, 
                          human_assessment: str, metadata: Dict = None, reasoning: str = None) -> Dict[str, Any]:
        """
        Add human feedback to knowledge base, implementing pseudo-RLHF functionality
        
        Args:
            feature_description: Feature description
            original_assessment: Original classification label
            human_assessment: Human-corrected classification label
            metadata: Additional metadata
            reasoning: Reasoning information
            
        Returns:
            Added feedback case
        """
        if not self.use_feedback_learning or not self.feedback_kb:
            print("! Feedback learning feature not enabled, cannot add human feedback")
            return None
        
        print(f"\n{'='*60}")
        print(f"Adding human feedback to knowledge base...")
        print(f"{'='*60}")
        print(f"Feature description: {feature_description[:80]}...")
        print(f"Original classification: {original_assessment}")
        print(f"Human correction: {human_assessment}")
        
        # Add to knowledge base
        feedback_case = self.feedback_kb.add_feedback(
            feature_description=feature_description,
            original_assessment=original_assessment,
            human_assessment=human_assessment,
            metadata=metadata,
            reasoning=reasoning
        )
        
        print(f"✓ Successfully added feedback case #{feedback_case['id']}")
        
        # Get knowledge base statistics
        stats = self.feedback_kb.get_statistics()
        print(f"\nKnowledge Base Statistics:")
        print(f"  Total cases: {stats['total_cases']}")
        print(f"  Correction rate: {stats['correction_rate']:.2%}")
        
        return feedback_case


# def main():
#     """
#     Main function demonstrating the workflow of confidence evaluation Agent
#     """
#     print("=== Confidence Evaluation Agent - Workflow Demonstration ===\n")
    
#     # Initialize legal classifier and confidence evaluation Agent
#     try:
#         legal_classifier = OptimizedLegalClassifier(
#             graph_db_path="/Users/yanjin/vscode/ReguLLM/legal_compliance_db1"  # Please modify according to actual path
#         )
#         confidence_agent = ConfidenceAgent(
#             confidence_threshold=0.7,
#             legal_classifier=legal_classifier
#         )
#         print("✓ Agent initialization successful\n")
#     except Exception as e:
#         print(f"✗ Agent initialization failed: {e}")
#         return
    
#     # Test data - including cases with different confidence levels
#     test_cases = [
#         # High confidence case
#         "Curfew login blocker with ASL and GH for Utah minors: To comply with the Utah Social Media Regulation Act, we are implementing a curfew-based login restriction for users under 18. The system uses ASL to detect minor accounts and routes enforcement through GH to apply only within Utah boundaries.",
        
#         # Medium confidence case
#         "Content visibility lock with NSP for EU DSA: To meet the transparency expectations of the EU Digital Services Act, we are introducing a visibility lock for flagged user-generated content labeled under NSP. When such content is detected, a soft Softblock is applied.",
        
#         # Low confidence case
#         "Universal PF deactivation on guest mode: By default, PF will be turned off for all users browsing in guest mode."
#     ]
    
#     # Process test cases
#     results = []
#     for i, feature in enumerate(test_cases):
#         print(f"\nTest Case {i+1}/{len(test_cases)}")
        
#         # Execute complete processing workflow
#         result = confidence_agent.process_feature(feature)
        
#         # Save result
#         results.append(result)
#         print(f"{'='*80}")
    
#     # Save results to JSON file
#     try:
#         with open('confidence_evaluation_results.json', 'w', encoding='utf-8') as f:
#             json.dump(results, f, indent=4, ensure_ascii=False, default=str)
#         print(f"\n✓ Results saved to confidence_evaluation_results.json")
#         print(f"✓ Processed {len(results)} test cases")
#     except Exception as e:
#         print(f"\n✗ Failed to save results: {e}")


# if __name__ == "__main__":
#     main()
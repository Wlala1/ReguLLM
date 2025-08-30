#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Legal Knowledge Base Test Script
For testing vector database functionality and automated human feedback workflow
"""

from trans_embedding import LegalDocumentVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import json
from confidence_agent import ConfidenceAgent
from main import OptimizedLegalClassifier

def automated_feedback_workflow():
    """
    Automated Human Feedback Workflow
    
    Implements automated human feedback process:
    1. Initialize confidence assessment Agent
    2. Process test cases
    3. Interactive review for cases requiring human review
    4. Add human feedback to knowledge base
    5. Process new cases with updated knowledge base
    """
    print("\n=== Automated Human Feedback Workflow ===")
    
    # Set paths
    vector_db_path = os.path.join(os.getcwd(), "dynamic_legal_graph_db")
    knowledge_base_path = os.path.join(os.getcwd(), "feedback_knowledge_base.pkl")
    
    print(f"Vector database path: {vector_db_path}")
    print(f"Feedback knowledge base path: {knowledge_base_path}")
    
    # Initialize components
    print("\nInitializing legal classifier and confidence assessment Agent...")
    legal_classifier = OptimizedLegalClassifier(graph_db_path=vector_db_path)
    confidence_agent = ConfidenceAgent(
        legal_classifier=legal_classifier,
        confidence_threshold=0.7,
        knowledge_base_path=knowledge_base_path,
        use_feedback_learning=True,  # Enable feedback learning functionality
        embedding_model_name="BAAI/bge-base-en-v1.5"  # Use vector embedding model
    )
    print("✓ Initialization successful")
    
    # Test cases
    test_cases = [
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
    
    # Process test cases
    results = []
    for i, case in enumerate(test_cases):
        print(f"\nTest case {i+1}/{len(test_cases)}: {case['name']}")
        print(f"Description: {case['description'][:100]}...")
        
        # Execute complete processing workflow
        feature_description = case['description']
        result = confidence_agent.process_feature(feature_description)
        
        # Add case name
        result['case_name'] = case['name']
        
        # Save result
        results.append(result)
        
        # Check if human review is needed
        # Human review is triggered when system flags for human review or final classification is UnspecifiedNeedsHuman
        if result['final_result']['needs_human_review'] or (result['final_result']['assessment'] == "UnspecifiedNeedsHuman"):
            print("\nHuman review needed, starting interactive review process...")
            
            # Display case information
            print(f"\n{'='*80}")
            print(f"Human Review Interface - Case #{i+1}")
            print(f"{'='*80}")
            print(f"Feature description: {feature_description}")
            print(f"Original classification: {result['final_result']['assessment']}")
            print(f"Confidence: {result['final_result']['confidence']}")
            
            # Get human input
            print("\nPlease select the correct classification:")
            print("1. LegalRequirement")
            print("2. BusinessDriven")
            
            while True:
                choice = input("Please enter your choice (1/2): ").strip()
                if choice in ["1", "2"]:
                    break
                print("Invalid choice, please try again")
            
            # Map choice to classification label
            human_assessment_map = {
                "1": "LegalRequirement",
                "2": "BusinessDriven",
            }
            human_assessment = human_assessment_map[choice]
            
            # Get reviewer notes
            notes = input("Please enter review notes (optional): ").strip()
            
            # Add human feedback to knowledge base
            metadata = {
                "reviewer": "Interactive Reviewer",
                "confidence": "high",
                "notes": notes if notes else "No notes"
            }
            
            # Use previously extracted feature_description
            feedback_case = confidence_agent.add_human_feedback(
                feature_description=feature_description,
                original_assessment=result['final_result']['assessment'],
                human_assessment=human_assessment,
                metadata=metadata,
                reasoning=notes  # Pass notes as reasoning information
            )
            
            # Update final classification in result to human review result
            result['original_assessment'] = result['final_result']['assessment']  # Save original classification
            result['final_result']['assessment'] = human_assessment
            result['final_result']['confidence'] = 1.0  # Set human review confidence to maximum
            result['human_reviewed'] = True  # Mark as human reviewed
            
            # Update reasoning information
            if notes:  # If human provided notes, use notes as reasoning
                result['reasoning'] = notes
                result['final_result']['reasoning'] = notes  # Update reasoning in final result
            else:  # Otherwise use default text
                result['reasoning'] = "Derived from human feedback, but no reason provided"
                result['final_result']['reasoning'] = "Derived from human feedback, but no reason provided"  # Update reasoning in final result
            
            print(f"\n✓ Feedback case added to knowledge base: #{feedback_case['id']}")
    
    # Results summary
    print("\n=== Test Results Summary ===")
    for i, result in enumerate(results):
        print(f"\nCase #{i+1}:")
        print(f"  Feature description: {test_cases[i]['description'][:50]}...")
        print(f"  Final classification: {result['final_result']['assessment']}")
        print(f"  Confidence: {result['final_result']['confidence']}")
        
        # Show if human reviewed
        if result.get('human_reviewed', False):
            print(f"  Human review: Yes (Original classification: {result['original_assessment']})")
            # Show human review reasoning
            if 'reasoning' in result:
                print(f"  Reasoning: {result['reasoning'][:100]}{'...' if len(result['reasoning']) > 100 else ''}")
        else:
            print(f"  Human review: No")
            # Show original reasoning or reflection results
            if result['confidence_evaluation'].get('reflection_details'):
                reflection = result['confidence_evaluation']['reflection_details']
                evidence_analysis = reflection.get('reflection', {}).get('evidence_analysis', 'N/A')
                print(f"  Reasoning: {evidence_analysis[:100]}{'...' if len(evidence_analysis) > 100 else ''}")
            elif 'reasoning' in result['original_classification']:
                reasoning = result['original_classification']['reasoning']
                print(f"  Reasoning: {reasoning[:100]}{'...' if len(reasoning) > 100 else ''}")
            elif 'reasoning' in result['final_result']:
                reasoning = result['final_result']['reasoning']
                print(f"  Reasoning: {reasoning[:100]}{'...' if len(reasoning) > 100 else ''}")
        
        # Show similar case information
        similar_cases = result['confidence_evaluation'].get('similar_cases', [])
        if similar_cases:
            print(f"  Used {len(similar_cases)} similar feedback cases")
    
    # Export knowledge base statistics
    if confidence_agent.feedback_kb:
        stats = confidence_agent.feedback_kb.get_statistics()
        print(f"\nKnowledge base statistics:")
        print(f"  Total cases: {stats['total_cases']}")
        print(f"  Correction rate: {stats['correction_rate']:.2%}")

if __name__ == "__main__":
        automated_feedback_workflow()
import json
import os
from confidence_agent import ConfidenceAgent
from main import OptimizedLegalClassifier

def test_confidence_agent(use_feedback_learning=False):
    """Test the functionality of the confidence assessment agent
    """
    print("=== Confidence Assessment Agent - Test Script ===\n")
    
    # Set vector database path
    vector_db_path = os.path.join(os.getcwd(), "dynamic_legal_graph_db")
    knowledge_base_path = os.path.join(os.getcwd(), "feedback_knowledge_base.pkl")
    print(f"Vector database path: {vector_db_path}")
    
    # Initialize legal classifier and confidence assessment agent
    try:
        print("Initializing legal classifier...")
        legal_classifier = OptimizedLegalClassifier(
            graph_db_path=vector_db_path
        )
        
        print("\nInitializing confidence assessment agent...")
        confidence_agent = ConfidenceAgent(
            confidence_threshold=0.7,  # Set confidence threshold
            legal_classifier=legal_classifier,
            knowledge_base_path=knowledge_base_path,
            use_feedback_learning=use_feedback_learning
        )
        print("✓ Initialization successful\n")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return
    
    # Test data - cases with different confidence levels
    test_cases = [
        # Case 1: Potentially high-confidence case - clear legal requirement
        {
            "name": "Curfew login blocker with ASL and GH for Utah minors",
            "description": "To comply with the Utah Social Media Regulation Act, we are implementing a curfew-based login restriction for users under 18. The system uses ASL to detect minor accounts and routes enforcement through GH to apply only within Utah boundaries."
        },
        
        # Case 2: Potentially medium-confidence case - partial legal basis
        {
            "name": "PF default toggle with NR enforcement for California teens",
            "description": "As part of compliance with California's SB976, the app will disable PF by default for users under 18 located in California. This default setting is considered NR to override, unless explicit parental opt-in is provided. Geo-detection is handled via GH, and rollout is monitored with FR logs. The design ensures minimal disruption while meeting the strict personalization requirements imposed by the law."
        },
        
        # Case 3: Potentially low-confidence case - lacking clear legal basis
        {
            "name": "Child abuse content scanner using T5 and CDS triggers",
            "description": "In line with the US federal law requiring providers to report child sexual abuse content to NCMEC, this feature scans uploads and flags suspected materials tagged as T5. Once flagged, the CDS auto-generates reports and routes them via secure channel APIs. The logic runs in real-time, supports human validation, and logs detection metadata for internal audits. Regional thresholds are governed by LCP parameters in the backend."
        },
        
        # Case 4: Potentially mixed signal case - legal and business factors combined
        {
            "name": "Feature reads user location to enforce France's copyright rules",
            "description": ""
        },
        
        # Case 5: Potentially requiring human intervention - insufficient information
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
        print(f"\nTest Case {i+1}/{len(test_cases)}: {case['name']}")
        print(f"Description: {case['description'][:100]}...")
        
        # Execute complete processing workflow
        try:
            result = confidence_agent.process_feature(case['description'])
            
            # Add case name
            result['case_name'] = case['name']
            
            # Save result
            results.append(result)
            
            # Display key results
            print(f"\nResult Summary:")
            print(f"  Original classification: {result['original_classification']['assessment']}")
            print(f"  Original confidence: {result['original_classification']['confidence']}")
            print(f"  Needs reflection: {result['confidence_evaluation']['needs_reflection']}")
            print(f"  Final classification: {result['final_result']['assessment']}")
            print(f"  Final confidence: {result['final_result']['confidence']}")
            print(f"  Needs human review: {result['final_result']['needs_human_review']}")
            
            if result['confidence_evaluation']['needs_reflection'] and result['confidence_evaluation']['reflection_details']:
                reflection = result['confidence_evaluation']['reflection_details']
                print(f"\nReflection Analysis:")
                print(f"  Decision: {reflection.get('decision', 'N/A')}")
                print(f"  Evidence analysis: {reflection.get('reflection', {}).get('evidence_analysis', 'N/A')[:100]}...")
                print(f"  Logic analysis: {reflection.get('reflection', {}).get('logic_analysis', 'N/A')[:100]}...")
                print(f"  Uncertainty source: {reflection.get('reflection', {}).get('uncertainty_source', 'N/A')[:100]}...")
                
                # Display similar case information
                similar_cases = result['confidence_evaluation'].get('similar_cases', [])
                if similar_cases:
                    print(f"\nReferenced similar cases:")
                    for i, case in enumerate(similar_cases):
                        print(f"  Case #{i+1}: {case['id']}")
                        print(f"    Original assessment: {case['original_assessment']}")
                        print(f"    Human correction: {case['human_assessment']}")
                        print(f"    Similarity: {case['similarity']:.2f}")
            
        except Exception as e:
            print(f"✗ Processing failed: {e}")
            results.append({
                "case_name": case['name'],
                "error": str(e),
                "status": "failed"
            })
        
        print(f"{('='*80)}")
    
    # Save results to JSON file
    try:
        output_file = 'test_confidence_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False, default=str)
        print(f"\n✓ Results saved to {output_file}")
        print(f"✓ Processed {len(results)} test cases")
        
        # Statistics
        needs_human_count = sum(1 for r in results if r.get('final_result', {}).get('needs_human_review', False))
        reflection_count = sum(1 for r in results if r.get('confidence_evaluation', {}).get('needs_reflection', False))
        
        print(f"\nStatistics:")
        print(f"  Cases requiring reflection: {reflection_count}/{len(results)}")
        print(f"  Cases requiring human review: {needs_human_count}/{len(results)}")
        
    except Exception as e:
        print(f"\n✗ Failed to save results: {e}")


def test_human_feedback():
    """Test human feedback functionality"""
    print(f"\n{'='*100}")
    print(f"Testing Human Feedback Functionality")
    print(f"{'='*100}")
    
    try:
        # Initialize agent with feedback learning enabled
        vector_db_path = os.path.join(os.getcwd(), "dynamic_legal_graph_db")
        knowledge_base_path = os.path.join(os.getcwd(), "feedback_knowledge_base.pkl")
        
        print(f"Vector database path: {vector_db_path}")
        print(f"Feedback knowledge base path: {knowledge_base_path}")
        
        # Initialize legal classifier
        print("\nInitializing legal classifier...")
        legal_classifier = OptimizedLegalClassifier(
            graph_db_path=vector_db_path
        )
        print("✓ Legal classifier initialization successful")
        
        # Initialize confidence assessment agent (with feedback learning enabled)
        print("\nInitializing confidence assessment agent (with feedback learning enabled)...")
        confidence_agent = ConfidenceAgent(
            confidence_threshold=0.7,
            legal_classifier=legal_classifier,
            knowledge_base_path=knowledge_base_path,
            use_feedback_learning=True
        )
        print("✓ Confidence assessment agent initialization successful\n")
        
        # Test case
        feature_description = "We need to add a feature that allows users to download all their data, including personal information and usage records. This is to comply with data portability rights requirements."
        
        # Step 1: Process feature and get initial results
        print("\nStep 1: Processing feature and getting initial results...")
        result = confidence_agent.process_feature(feature_description)
        
        original_assessment = result['final_result']['assessment']
        needs_human_review = result['final_result']['needs_human_review']
        
        # Step 2: Simulate human review and add feedback
        print("\nStep 2: Simulating human review and adding feedback...")
        if needs_human_review:
            print("Human review required, proceeding with human annotation...")
            # Simulate human review results
            human_assessment = "Legal requirement"  # Assume human review determines this is a legal requirement
            
            # Add human feedback to knowledge base
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
            
            print(f"\nFeedback case added: {feedback_case['id']}")
        else:
            print("Human review not required, skipping feedback addition")
        
        # Step 3: Test impact of similar cases
        print("\nStep 3: Testing impact of similar cases...")
        similar_feature = "We need to implement a feature that allows users to export all their data, as required by GDPR data portability rights."
        
        print("\nProcessing similar feature description...")
        result_with_feedback = confidence_agent.process_feature(similar_feature)
        
        # Check if similar cases were used
        similar_cases = result_with_feedback['confidence_evaluation'].get('similar_cases', [])
        if similar_cases:
            print(f"\nSuccess! The system used {len(similar_cases)} similar feedback cases for decision-making")
        else:
            print("\nNote: The system did not use any similar feedback cases")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_confidence_agent(use_feedback_learning=True)
    
    # Test human feedback functionality
    test_human_feedback()
from flask import Flask, request, jsonify
from flask_cors import CORS
from main import OptimizedLegalClassifier
from confidence_agent import ConfidenceAgent
import logging

app = Flask(__name__)
CORS(app)

# Initialize legal classifier with graph database path
legal_classifier = OptimizedLegalClassifier(
    graph_db_path="./dynamic_legal_graph_db"
)

# Initialize confidence agent with threshold and feedback learning
confidence_agent = ConfidenceAgent(
    confidence_threshold=0.7,
    legal_classifier=legal_classifier,
    use_feedback_learning=True
)

@app.route('/api/analyze', methods=['POST'])
def analyze_feature():
    try:
        data = request.json
        feature_description = data.get('feature_description')
        
        if not feature_description:
            return jsonify({'error': 'Please provide a feature description'}), 400
        
        # Process the feature through confidence agent
        result = confidence_agent.process_feature(feature_description)
        
        return jsonify(result)
    
    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.json
        
        # Submit human feedback
        feedback_result = confidence_agent.add_human_feedback(
            feature_description=data.get('feature_description'),
            original_assessment=data.get('original_assessment'),
            human_assessment=data.get('human_assessment'),
            metadata={
                'notes': data.get('notes', ''),
                'reviewer': data.get('reviewer', 'Web User'),
                'timestamp': data.get('timestamp')
            }
        )
        
        return jsonify({
            'status': 'success',
            'message': 'Human feedback successfully submitted',
            'feedback_id': feedback_result.get('id') if feedback_result else None
        })
    
    except Exception as e:
        logging.error(f"Error submitting feedback: {str(e)}")
        return jsonify({'error': f'Feedback submission failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
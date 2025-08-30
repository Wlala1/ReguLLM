from flask import Flask, request, jsonify
from flask_cors import CORS
from main import OptimizedLegalClassifier
from confidence_agent import ConfidenceAgent
import logging

app = Flask(__name__)
CORS(app)  # 允许前端跨域访问

# 初始化您的系统组件
legal_classifier = OptimizedLegalClassifier(
    graph_db_path="./dynamic_legal_graph_db"  # 修改为您的实际路径
)
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
            return jsonify({'error': '请提供功能描述'}), 400
        
        # 调用您的分析系统
        result = confidence_agent.process_feature(feature_description)
        
        return jsonify(result)
    
    except Exception as e:
        logging.error(f"分析过程出错: {str(e)}")
        return jsonify({'error': f'分析失败: {str(e)}'}), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.json
        
        # 提交人工反馈
        feedback_result = confidence_agent.add_human_feedback(
            feature_description=data.get('feature_description'),
            original_assessment=data.get('original_assessment'),
            human_assessment=data.get('human_assessment'),
            metadata={
                'notes': data.get('notes', ''),
                'reviewer': data.get('reviewer', 'Web用户'),
                'timestamp': data.get('timestamp')
            }
        )
        
        return jsonify({
            'status': 'success',
            'message': '人工反馈已成功提交',
            'feedback_id': feedback_result.get('id') if feedback_result else None
        })
    
    except Exception as e:
        logging.error(f"反馈提交出错: {str(e)}")
        return jsonify({'error': f'反馈提交失败: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
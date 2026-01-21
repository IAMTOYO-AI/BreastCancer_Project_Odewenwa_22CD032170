import os
import pandas as pd
from flask import Flask, render_template, request, json, Response
from model import BreastCancerModel

# Instance initialization
breast_app = Flask(__name__)
predictor_engine = BreastCancerModel()

def setup_classifier():
    """Ensures model is primed before requests start"""
    is_ready = predictor_engine.load_model()
    if not is_ready:
        print("Model file missing. Initiating training sequence...")
        from model import train_and_save_model
        train_and_save_model()
        predictor_engine.load_model()

setup_classifier()

@breast_app.route('/')
def index_view():
    """Displays the diagnostic interface"""
    return render_template('index.html', feature_names=predictor_engine.feature_names)

@breast_app.route('/predict', methods=['POST'])
def process_prediction():
    """Analyzes input metrics and returns classification results"""
    try:
        payload = request.get_json()
        input_metrics = {}

        # Integrity Check: verify all necessary bio-markers are present
        for metric in predictor_engine.feature_names:
            if metric not in payload:
                return json.jsonify({'success': False, 'error': f'Field required: {metric}'}), 400
            
            try:
                numeric_val = float(payload[metric])
                if numeric_val < 0:
                    raise ValueError("Negative value")
                input_metrics[metric] = numeric_val
            except (ValueError, TypeError):
                return json.jsonify({'success': False, 'error': f'Invalid entry for: {metric}'}), 400

        # Run inference
        class_label, prob_score = predictor_engine.predict(input_metrics)

        # Interpret labels (0: Malignant, 1: Benign)
        is_negative = bool(class_label == 1)
        certainty = prob_score if is_negative else (1.0 - prob_score)
        
        status_text = '✓ Tumor appears to be BENIGN (non-cancerous)' if is_negative \
                      else '⚠️ Tumor appears to be MALIGNANT (cancerous)'

        return json.jsonify({
            'status': 'success',
            'result': {
                'diagnosis_type': 'Benign' if is_negative else 'Malignant',
                'is_benign': is_negative,
                'confidence_pct': round(certainty * 100, 2),
                'summary': status_text,
                'notice': 'Educational tool only. Consult a specialist for clinical diagnosis.'
            }
        })

    except Exception as err:
        return json.jsonify({'status': 'error', 'msg': str(err)}), 500

@breast_app.route('/fetch-samples')
def get_test_examples():
    """Retrieves comparative data points for UI testing"""
    try:
        csv_loc = os.path.join(os.path.dirname(__file__), 'data', 'breast_cancer.csv')
        raw_df = pd.read_csv(csv_loc)

        # Select representative cases using filtering
        case_a = raw_df[raw_df['diagnosis'] == 1].sample(1).drop(columns=['diagnosis']).to_dict('records')[0]
        case_b = raw_df[raw_df['diagnosis'] == 0].sample(1).drop(columns=['diagnosis']).to_dict('records')[0]

        return json.jsonify({
            'success': True,
            'data': {'benign': case_a, 'malignant': case_b}
        })
    except Exception as exc:
        return json.jsonify({'success': False, 'error': str(exc)})

if __name__ == '__main__':
    # Launch server
    breast_app.run(host='0.0.0.0', port=5001, debug=False)
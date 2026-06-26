
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Disable caching to see changes immediately
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Load model and scaler
try:
    model = joblib.load('fraud_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_cols = joblib.load('feature_columns.pkl')
    metadata = joblib.load('model_metadata.pkl')
    print("Model loaded successfully!")
    print(f"Model: {metadata['model_name']}")
    print(f"Features: {len(feature_cols)}")
except Exception as e:
    print(f"Error loading model: {e}")
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    scaler = None
    feature_cols = ['cc_num', 'amt', 'zip', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']
    metadata = {'n_features': 9}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if 'features' not in data:
            return jsonify({'error': 'Missing "features" in request data'}), 400

        features = data['features']

        if len(features) != len(feature_cols):
            return jsonify({'error': f'Expected {len(feature_cols)} features, got {len(features)}'}), 400

        features_array = np.array(features).reshape(1, -1)

        if scaler:
            features_scaled = scaler.transform(features_array)
        else:
            features_scaled = features_array

        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)

        result = {
            'prediction': int(prediction[0]),
            'probability_fraud': float(probability[0][1]),
            'probability_normal': float(probability[0][0]),
            'is_fraud': bool(prediction[0])
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

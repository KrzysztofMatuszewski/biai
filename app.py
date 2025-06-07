from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import hashlib

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Global variables for model components
model = None
components = None
metadata = None

def load_model_components():
    """Load the trained model and preprocessing components"""
    global model, components, metadata
    
    try:
        model = load_model('model/best_model_simplified.h5')
        components = joblib.load('model/preprocessing_components_simplified.joblib')
        metadata = joblib.load('model/model_metadata_simplified.joblib')
        print("Successfully loaded model components")
        return True
    except Exception as e:
        print(f"Error loading model components: {e}")
        return False

def string_to_hash_int(text, max_value=1000):
    """Convert string to integer using hash function"""
    if pd.isna(text) or text == '' or text == 'nan':
        return 0
    
    hash_object = hashlib.md5(str(text).encode())
    hash_int = int(hash_object.hexdigest(), 16) % max_value
    return hash_int

def apply_hash_encoding_with_mappings(df, hash_mappings, categorical_columns):
    """Apply hash encoding using saved mappings from training"""
    df_encoded = df.copy()
    
    for col in categorical_columns:
        if col in df_encoded.columns and col in hash_mappings:
            def safe_map(value):
                if value in hash_mappings[col]:
                    return hash_mappings[col][value]
                else:
                    return string_to_hash_int(str(value), 1000)
            
            df_encoded[col] = df_encoded[col].apply(safe_map)
    
    return df_encoded

def preprocess_input_for_prediction(input_data):
    """Preprocess input data for prediction using saved components"""
    global components
    
    # Convert to DataFrame if needed
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()
    
    # Fill missing values
    numeric_cols_raw = ['Age', 'Fee', 'PhotoAmt']
    for col in numeric_cols_raw:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median() if not df[col].empty else 0)
    
    categorical_cols_raw = ['Type', 'Breed1', 'Gender', 'Color1', 'Color2', 
                           'MaturitySize', 'FurLength', 'Vaccinated', 
                           'Sterilized', 'Health']
    for col in categorical_cols_raw:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    
    # Apply hash encoding using saved mappings
    X_new = apply_hash_encoding_with_mappings(
        df, 
        components['hash_mappings'], 
        components['categorical_columns']
    )
    
    # Apply preprocessing
    X_imputed = components['imputer'].transform(X_new)
    X_scaled = components['scaler'].transform(X_imputed)
    
    return X_scaled

def predict_adoption_speed(input_data):
    """Predict adoption speed for new pet data"""
    global model, components, metadata
    
    if model is None or components is None:
        raise Exception("Model not loaded. Please check if model files exist.")
    
    # Preprocess input
    X_processed = preprocess_input_for_prediction(input_data)
    
    # Make prediction
    pred_proba = model.predict(X_processed, verbose=0)
    pred_class = np.argmax(pred_proba, axis=1)
    
    # Create result dictionary
    adoption_speed_map = {
        0: "Same day",
        1: "1-7 days",
        2: "8-30 days",
        3: "31-90 days",
        4: "No adoption after 100 days"
    }
    
    result = {
        'predicted_class': int(pred_class[0]),
        'predicted_label': adoption_speed_map[pred_class[0]],
        'probabilities': {f'Class {j}': float(pred_proba[0][j]) for j in range(5)},
        'confidence': float(np.max(pred_proba[0]))
    }
    
    return result

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    
    try:
        # Extract form data
        input_data = {
            'Type': request.form['type'],
            'Age': int(request.form['age']),
            'Breed1': request.form['breed'],
            'Gender': request.form['gender'],
            'Color1': request.form['color1'],
            'Color2': request.form.get('color2', ''),
            'MaturitySize': request.form['size'],
            'FurLength': request.form['fur'],
            'Vaccinated': request.form['vaccinated'],
            'Sterilized': request.form['sterilized'],
            'Health': request.form['health'],
            'Fee': float(request.form['fee']),
            'PhotoAmt': int(request.form['photos'])
        }
        
        # Make prediction
        result = predict_adoption_speed(input_data)
        
        return render_template('result.html', result=result, input_data=request.form)
        
    except Exception as e:
        flash(f'Error making prediction: {str(e)}', 'error')
        return redirect(url_for('predict'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        result = predict_adoption_speed(data)
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/health')
def health_check():
    """Health check endpoint"""
    model_loaded = model is not None and components is not None
    return jsonify({
        'status': 'healthy' if model_loaded else 'unhealthy',
        'model_loaded': model_loaded,
        'model_type': 'simplified'
    })

@app.route('/presentation')
def presentation():
    """Render the presentation page"""
    return render_template('presentation.html')

if __name__ == '__main__':
    if load_model_components():
        print("Model components loaded successfully!")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model components. Please ensure model files exist:")
        print("- best_model_simplified.h5")
        print("- preprocessing_components_simplified.joblib")
        print("- model_metadata_simplified.joblib")
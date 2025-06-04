from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import hashlib
from tensorflow.keras.models import load_model

# Download NLTK resources
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    print("Warning: Could not download NLTK resources")

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
        # Load hash-encoded model components
        model = load_model('best_hash_model.h5')
        components = joblib.load('hash_preprocessing_components.joblib')
        metadata = joblib.load('hash_model_metadata.joblib')
        print("Successfully loaded hash-encoded model components")
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

def extract_text_features(text):
    """Extract features from the pet description text"""
    try:
        sia = SentimentIntensityAnalyzer()
    except:
        return {
            'desc_length': len(text) if text else 0,
            'word_count': len(text.split()) if text else 0,
            'sentiment_compound': 0,
            'sentiment_pos': 0,
            'sentiment_neg': 0,
            'sentiment_neu': 1,
            'has_contact': 0,
            'has_health_mention': 0
        }
    
    if pd.isna(text) or text == '':
        return {
            'desc_length': 0,
            'word_count': 0,
            'sentiment_compound': 0,
            'sentiment_pos': 0,
            'sentiment_neg': 0,
            'sentiment_neu': 0,
            'has_contact': 0,
            'has_health_mention': 0
        }
    
    # Length features
    desc_length = len(str(text))
    word_count = len(str(text).split())
    
    # Sentiment analysis
    sentiment = sia.polarity_scores(str(text))
    
    # Check for specific content
    has_contact = 1 if re.search(r'\b(?:call|contact|phone|email)\b', str(text).lower()) else 0
    has_health_mention = 1 if re.search(r'\b(?:healthy|vaccinated|neutered|spayed|dewormed)\b', str(text).lower()) else 0
    
    return {
        'desc_length': desc_length,
        'word_count': word_count,
        'sentiment_compound': sentiment['compound'],
        'sentiment_pos': sentiment['pos'],
        'sentiment_neg': sentiment['neg'],
        'sentiment_neu': sentiment['neu'],
        'has_contact': has_contact,
        'has_health_mention': has_health_mention
    }

def create_interaction_features_hash(df):
    """Create interaction features using hash encoding"""
    df_new = df.copy()
    
    # Create age groups
    df_new['Age_Group'] = pd.cut(df_new['Age'], 
                                bins=[0, 3, 12, 36, 72, float('inf')], 
                                labels=['Baby', 'Young', 'Adult', 'Middle', 'Senior'])
    
    # Create health status groups
    df_new['Health_Status'] = df_new['Health']
    
    # Create interaction features
    interaction_combinations = [
        ('Vaccinated', 'Sterilized'),
        ('MaturitySize', 'FurLength'),
        ('Type', 'Breed1'),
        ('Color1', 'Color2'),
        ('Gender', 'MaturitySize'),
        ('Age_Group', 'Health_Status')
    ]
    
    for col1, col2 in interaction_combinations:
        if col1 in df_new.columns and col2 in df_new.columns:
            interaction_name = f'{col1}_{col2}_hash'
            combined = df_new[col1].astype(str) + '_' + df_new[col2].astype(str)
            df_new[interaction_name] = combined.apply(lambda x: string_to_hash_int(x, 1000))
    
    return df_new

def prepare_all_features(df):
    """Prepare all features including text features, interactions, and transformations"""
    # Extract text features from Description
    if 'Description' in df.columns:
        text_features = df['Description'].apply(extract_text_features).apply(pd.Series)
        df = pd.concat([df, text_features], axis=1)
    
    # Create basic mathematical features
    df['Age_Health_Product'] = df['Age'] * pd.to_numeric(df['Health'], errors='coerce').fillna(1)
    df['Price_Per_Photo'] = df['Fee'] / (df['PhotoAmt'] + 1)
    df['PhotoAmt_Log'] = np.log1p(df['PhotoAmt'])
    df['Fee_Log'] = np.log1p(df['Fee'])
    
    # Create categorical groupings before hashing
    df = create_interaction_features_hash(df)
    
    # Create premium indicator
    health_numeric = pd.to_numeric(df['Health'], errors='coerce').fillna(0)
    vaccinated_numeric = pd.to_numeric(df['Vaccinated'], errors='coerce').fillna(0)
    sterilized_numeric = pd.to_numeric(df['Sterilized'], errors='coerce').fillna(0)
    
    df['Is_Premium'] = ((health_numeric == 1) & 
                        (vaccinated_numeric == 1) & 
                        (sterilized_numeric == 1)).astype(int)
    
    return df

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
    
    if 'Description' in df.columns:
        df['Description'] = df['Description'].fillna('')
    
    # Apply same feature engineering as training
    df_processed = prepare_all_features(df)
    
    # Apply hash encoding using saved mappings
    df_encoded = apply_hash_encoding_with_mappings(
        df_processed, 
        components['hash_mappings'], 
        components['categorical_columns']
    )
    
    # Select same features as training
    try:
        X_new = df_encoded[components['feature_columns']]
    except KeyError as e:
        missing_cols = set(components['feature_columns']) - set(df_encoded.columns)
        print(f"Missing columns: {missing_cols}")
        for col in missing_cols:
            df_encoded[col] = 0
        X_new = df_encoded[components['feature_columns']]
    
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
            'PhotoAmt': int(request.form['photos']),
            'Description': request.form.get('description', '')
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
        'model_type': 'hash'
    })

@app.route('/presentation')
def presentation():
    """Render the presentation page"""
    return render_template('presentation.html')

if __name__ == '__main__':
    if load_model_components():
        print("Hash model components loaded successfully!")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model components. Please ensure model files exist:")
        print("- best_hash_model.h5")
        print("- hash_preprocessing_components.joblib")
        print("- hash_model_metadata.joblib")
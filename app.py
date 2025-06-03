from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import json
import os
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
preprocessor = None
metadata = None

def load_model_components():
    """Load the trained model and preprocessing components"""
    global model, preprocessor, metadata
    
    try:
        model = load_model('best_model.h5')
        preprocessor = joblib.load('pet_adoption_preprocessor.joblib')
        metadata = joblib.load('pet_adoption_metadata.joblib')
        return True
    except Exception as e:
        print(f"Error loading model components: {e}")
        return False

def extract_text_features(text):
    """Extract features from the pet description text"""
    try:
        sia = SentimentIntensityAnalyzer()
    except:
        # Fallback if NLTK resources are not available
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
    desc_length = len(text)
    word_count = len(text.split())
    
    # Sentiment analysis
    sentiment = sia.polarity_scores(text)
    
    # Check for specific content
    has_contact = 1 if re.search(r'\b(?:call|contact|phone|email)\b', text.lower()) else 0
    has_health_mention = 1 if re.search(r'\b(?:healthy|vaccinated|neutered|spayed|dewormed)\b', text.lower()) else 0
    
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

def create_derived_features(df):
    """Create derived features for the prediction"""
    # Interaction features
    df['Age_Health_Interaction'] = df['Age'] * df['Health']
    df['Vaccinated_Sterilized'] = df['Vaccinated'].astype(str) + '_' + df['Sterilized'].astype(str)
    df['Size_FurLength'] = df['MaturitySize'].astype(str) + '_' + df['FurLength'].astype(str)
    df['Price_Per_Photo'] = df['Fee'] / (df['PhotoAmt'] + 1)  # Avoid division by zero
    
    # Log transformations
    df['PhotoAmt_Log'] = np.log1p(df['PhotoAmt'])
    df['Fee_Log'] = np.log1p(df['Fee'])
    
    return df

def convert_categorical_values(data):
    """Convert string categorical values to numeric codes expected by the model"""
    # Mapping dictionaries for categorical variables
    type_map = {'Dog': 1, 'Cat': 2}
    gender_map = {'Male': 1, 'Female': 2, 'Mixed': 3}
    size_map = {'Small': 1, 'Medium': 2, 'Large': 3, 'Extra Large': 4, 'Not Specified': 0}
    fur_map = {'Short': 1, 'Medium': 2, 'Long': 3, 'Not Specified': 0}
    yesno_map = {'Yes': 1, 'No': 2, 'Not Sure': 3, 'Not Specified': 0}
    health_map = {'Healthy': 1, 'Minor Injury': 2, 'Serious Injury': 3, 'Not Specified': 0}
    
    # Apply mappings
    data_copy = data.copy()
    
    if 'Type' in data_copy and isinstance(data_copy['Type'], str):
        data_copy['Type'] = type_map.get(data_copy['Type'], data_copy['Type'])
    
    if 'Gender' in data_copy and isinstance(data_copy['Gender'], str):
        data_copy['Gender'] = gender_map.get(data_copy['Gender'], data_copy['Gender'])
    
    if 'MaturitySize' in data_copy and isinstance(data_copy['MaturitySize'], str):
        data_copy['MaturitySize'] = size_map.get(data_copy['MaturitySize'], data_copy['MaturitySize'])
    
    if 'FurLength' in data_copy and isinstance(data_copy['FurLength'], str):
        data_copy['FurLength'] = fur_map.get(data_copy['FurLength'], data_copy['FurLength'])
    
    if 'Vaccinated' in data_copy and isinstance(data_copy['Vaccinated'], str):
        data_copy['Vaccinated'] = yesno_map.get(data_copy['Vaccinated'], data_copy['Vaccinated'])
    
    if 'Sterilized' in data_copy and isinstance(data_copy['Sterilized'], str):
        data_copy['Sterilized'] = yesno_map.get(data_copy['Sterilized'], data_copy['Sterilized'])
    
    if 'Health' in data_copy and isinstance(data_copy['Health'], str):
        data_copy['Health'] = health_map.get(data_copy['Health'], data_copy['Health'])
    
    return data_copy

def preprocess_input(input_data, include_desc_features=True):
    """
    Preprocess the input data for prediction
    
    Parameters:
    input_data (dict or pd.DataFrame): Input data containing pet information
    include_desc_features (bool): Whether to extract features from Description
    
    Returns:
    pd.DataFrame: Preprocessed data ready for model prediction
    """
    # Convert dict to DataFrame if necessary
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()
    
    # Extract text features if Description is present and include_desc_features is True
    if 'Description' in df.columns and include_desc_features:
        text_features = df['Description'].apply(extract_text_features).apply(pd.Series)
        df = pd.concat([df, text_features], axis=1)
        df = df.drop('Description', axis=1)
    
    # Create derived features
    df = create_derived_features(df)
    
    return df

def predict_adoption_speed(input_data):
    """
    Predict adoption speed for new pet data
    
    Parameters:
    input_data (dict or pd.DataFrame): Input data containing pet information
    
    Returns:
    dict: Prediction results with class and probabilities
    """
    global model, preprocessor, metadata
    
    if model is None or preprocessor is None:
        raise Exception("Model not loaded. Please check if model files exist.")
    
    # Preprocess input
    preprocessed_data = preprocess_input(input_data)
    
    # Transform input with preprocessor
    X_processed = preprocessor.transform(preprocessed_data)
    
    # Make prediction
    pred_proba = model.predict(X_processed)
    pred_class = np.argmax(pred_proba, axis=1)
    
    # Create result dictionary
    adoption_speed_map = {
        0: "Same day",
        1: "1-7 days",
        2: "8-30 days",
        3: "31-90 days",
        4: "No adoption after 100 days"
    }
    
    results = []
    for i, pred in enumerate(pred_class):
        result = {
            'predicted_class': int(pred),
            'predicted_label': adoption_speed_map[pred],
            'probabilities': {f'Class {j}': float(pred_proba[i][j]) for j in range(5)},
            'confidence': float(np.max(pred_proba[i]))
        }
        results.append(result)
    
    return results[0] if len(results) == 1 else results

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
        
        # Convert categorical values
        input_data = convert_categorical_values(input_data)
        
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
        
        # Convert categorical values
        data = convert_categorical_values(data)
        
        # Make prediction
        result = predict_adoption_speed(data)
        
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/health')
def health_check():
    """Health check endpoint"""
    model_loaded = model is not None and preprocessor is not None
    return jsonify({
        'status': 'healthy' if model_loaded else 'unhealthy',
        'model_loaded': model_loaded
    })

@app.route('/presentation')
def presentation():
    """Render the presentation page"""
    return render_template('presentation.html')

if __name__ == '__main__':
    # Load model components on startup
    if load_model_components():
        print("Model components loaded successfully!")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model components. Please ensure model files exist:")
        print("- best_model.h5")
        print("- pet_adoption_preprocessor.joblib")
        print("- pet_adoption_metadata.joblib")
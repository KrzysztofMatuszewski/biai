import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import argparse
import json
import os
import csv
from tensorflow.keras.models import load_model

# Download NLTK resources (uncomment if needed)
nltk.download('vader_lexicon')
nltk.download('stopwords')

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

def extract_text_features(text):
    """Extract features from the pet description text"""
    sia = SentimentIntensityAnalyzer()
    
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

def predict_adoption_speed(input_data, model_path='best_model.h5', 
                          preprocessor_path='pet_adoption_preprocessor.joblib',
                          metadata_path='pet_adoption_metadata.joblib'):
    """
    Predict adoption speed for new pet data
    
    Parameters:
    input_data (dict or pd.DataFrame): Input data containing pet information
    model_path (str): Path to the saved model
    preprocessor_path (str): Path to the saved preprocessor
    metadata_path (str): Path to the saved model metadata
    
    Returns:
    dict: Prediction results with class and probabilities
    """
    # Load model, preprocessor, and metadata
    model = load_model(model_path)
    preprocessor = joblib.load(preprocessor_path)
    metadata = joblib.load(metadata_path)
    
    # Preprocess input
    preprocessed_data = preprocess_input(input_data)
    
    # Transform input with preprocessor
    X_processed = preprocessor.transform(preprocessed_data)
    
    # Make prediction
    pred_proba = model.predict(X_processed)
    pred_class = np.argmax(pred_proba, axis=1)
    
    # Create result dictionary
    adoption_speed_map = {
        0: "Same day (0)",
        1: "1-7 days (1)",
        2: "8-30 days (2)",
        3: "31-90 days (3)",
        4: "No adoption after 100 days (4)"
    }
    
    results = []
    for i, pred in enumerate(pred_class):
        result = {
            'predicted_class': int(pred),
            'predicted_label': adoption_speed_map[pred],
            'probabilities': {f'Class {j}': float(pred_proba[i][j]) for j in range(len(adoption_speed_map))}
        }
        results.append(result)
    
    return results[0] if len(results) == 1 else results

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

def process_batch_predictions(input_file, output_file, model_path='best_model.h5', 
                             preprocessor_path='pet_adoption_preprocessor.joblib',
                             metadata_path='pet_adoption_metadata.joblib'):
    """
    Process batch predictions from a CSV file
    
    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to output CSV file
    model_path (str): Path to the saved model
    preprocessor_path (str): Path to the saved preprocessor
    metadata_path (str): Path to the saved model metadata
    """
    # Read input CSV
    input_df = pd.read_csv(input_file)
    
    # Convert categorical string values to numeric codes
    for i in range(len(input_df)):
        row_dict = input_df.iloc[i].to_dict()
        input_df.iloc[i] = pd.Series(convert_categorical_values(row_dict))
    
    # Predict
    predictions = predict_adoption_speed(input_df, model_path, preprocessor_path, metadata_path)
    
    # If only one row, convert to list
    if not isinstance(predictions, list):
        predictions = [predictions]
    
    # Add predictions to the dataframe
    input_df['predicted_class'] = [p['predicted_class'] for p in predictions]
    input_df['predicted_label'] = [p['predicted_label'] for p in predictions]
    
    # Add probabilities as separate columns
    for i in range(5):  # 5 classes (0-4)
        input_df[f'probability_class_{i}'] = [p['probabilities'][f'Class {i}'] for p in predictions]
    
    # Save to output file
    input_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Predict pet adoption speed')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Single prediction from command line
    single_parser = subparsers.add_parser('single', help='Make a single prediction')
    single_parser.add_argument('--type', choices=['Dog', 'Cat'], required=True, help='Type of pet')
    single_parser.add_argument('--age', type=int, required=True, help='Age in months')
    single_parser.add_argument('--breed', type=str, required=True, help='Primary breed')
    single_parser.add_argument('--gender', choices=['Male', 'Female', 'Mixed'], required=True, help='Gender')
    single_parser.add_argument('--color1', type=str, required=True, help='Primary color')
    single_parser.add_argument('--color2', type=str, help='Secondary color')
    single_parser.add_argument('--size', choices=['Small', 'Medium', 'Large', 'Extra Large', 'Not Specified'], 
                             required=True, help='Maturity size')
    single_parser.add_argument('--fur', choices=['Short', 'Medium', 'Long', 'Not Specified'], 
                             required=True, help='Fur length')
    single_parser.add_argument('--vaccinated', choices=['Yes', 'No', 'Not Sure'], required=True, help='Vaccination status')
    single_parser.add_argument('--sterilized', choices=['Yes', 'No', 'Not Sure'], required=True, help='Sterilization status')
    single_parser.add_argument('--health', choices=['Healthy', 'Minor Injury', 'Serious Injury', 'Not Specified'], 
                             required=True, help='Health condition')
    single_parser.add_argument('--fee', type=float, required=True, help='Adoption fee')
    single_parser.add_argument('--photos', type=int, required=True, help='Number of photos')
    single_parser.add_argument('--description', type=str, help='Pet description')
    single_parser.add_argument('--model', type=str, default='best_model.h5', help='Path to model file')
    single_parser.add_argument('--preprocessor', type=str, default='pet_adoption_preprocessor.joblib', 
                             help='Path to preprocessor file')
    single_parser.add_argument('--metadata', type=str, default='pet_adoption_metadata.joblib', 
                             help='Path to metadata file')
    
    # Batch prediction from CSV
    batch_parser = subparsers.add_parser('batch', help='Make batch predictions from CSV')
    batch_parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    batch_parser.add_argument('--output', type=str, required=True, help='Output CSV file')
    batch_parser.add_argument('--model', type=str, default='best_model.h5', help='Path to model file')
    batch_parser.add_argument('--preprocessor', type=str, default='pet_adoption_preprocessor.joblib', 
                            help='Path to preprocessor file')
    batch_parser.add_argument('--metadata', type=str, default='pet_adoption_metadata.joblib', 
                            help='Path to metadata file')
    
    args = parser.parse_args()
    
    if args.command == 'single':
        # Create input data dictionary
        input_data = {
            'Type': args.type,
            'Age': args.age,
            'Breed1': args.breed,
            'Gender': args.gender,
            'Color1': args.color1,
            'Color2': args.color2 if args.color2 else '',
            'MaturitySize': args.size,
            'FurLength': args.fur,
            'Vaccinated': args.vaccinated,
            'Sterilized': args.sterilized,
            'Health': args.health,
            'Fee': args.fee,
            'PhotoAmt': args.photos
        }
        
        if args.description:
            input_data['Description'] = args.description
        
        # Convert string categorical values to numeric codes
        input_data = convert_categorical_values(input_data)
        
        # Make prediction
        result = predict_adoption_speed(
            input_data, 
            model_path=args.model, 
            preprocessor_path=args.preprocessor,
            metadata_path=args.metadata
        )
        
        # Print prediction
        print(json.dumps(result, indent=2))
        
    elif args.command == 'batch':
        process_batch_predictions(
            args.input,
            args.output,
            model_path=args.model,
            preprocessor_path=args.preprocessor,
            metadata_path=args.metadata
        )
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
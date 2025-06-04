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
import hashlib
from tensorflow.keras.models import load_model

# Download NLTK resources
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

def string_to_hash_int(text, max_value=1000):
    """Convert string to integer using hash function"""
    if pd.isna(text) or text == '' or text == 'nan':
        return 0
    
    hash_object = hashlib.md5(str(text).encode())
    hash_int = int(hash_object.hexdigest(), 16) % max_value
    return hash_int

def extract_text_features(text):
    """Extract features from the pet description text"""
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
    
    sia = SentimentIntensityAnalyzer()
    
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

def preprocess_input_for_prediction(input_data, components):
    """Preprocess input data for prediction using saved components"""
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

def predict_adoption_speed_hash(input_data, 
                               model_path='best_hash_model.h5',
                               components_path='hash_preprocessing_components.joblib',
                               metadata_path='hash_model_metadata.joblib'):
    """Predict adoption speed using hash-encoded model"""
    # Load all components
    model = load_model(model_path)
    components = joblib.load(components_path)
    metadata = joblib.load(metadata_path)
    
    # Preprocess input data
    X_processed = preprocess_input_for_prediction(input_data, components)
    
    # Make prediction
    pred_proba = model.predict(X_processed, verbose=0)
    pred_class = np.argmax(pred_proba, axis=1)
    
    # Format results
    adoption_speed_map = {
        0: "Same day (0)",
        1: "1-7 days (1)", 
        2: "8-30 days (2)",
        3: "31-90 days (3)",
        4: "No adoption after 100 days (4)"
    }
    
    result = {
        'predicted_class': int(pred_class[0]),
        'predicted_label': adoption_speed_map[pred_class[0]],
        'probabilities': {f'Class {i}': float(pred_proba[0][i]) 
                         for i in range(metadata['num_classes'])},
        'confidence': float(np.max(pred_proba[0]))
    }
    
    return result

def batch_predict_hash(input_file, output_file,
                      model_path='best_hash_model.h5',
                      components_path='hash_preprocessing_components.joblib',
                      metadata_path='hash_model_metadata.joblib'):
    """Make batch predictions from CSV file using hash model"""
    # Read input data
    input_df = pd.read_csv(input_file)
    print(f"Loaded {len(input_df)} rows for prediction")
    
    # Make predictions
    predictions = []
    for i, row in input_df.iterrows():
        try:
            pred = predict_adoption_speed_hash(
                row.to_dict(), model_path, components_path, metadata_path
            )
            predictions.append(pred)
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} rows...")
        except Exception as e:
            print(f"Error processing row {i}: {e}")
            predictions.append({
                'predicted_class': -1,
                'predicted_label': 'Error',
                'probabilities': {f'Class {j}': 0.0 for j in range(5)},
                'confidence': 0.0
            })
    
    # Add predictions to dataframe
    input_df['predicted_class'] = [p['predicted_class'] for p in predictions]
    input_df['predicted_label'] = [p['predicted_label'] for p in predictions]
    input_df['confidence'] = [p['confidence'] for p in predictions]
    
    # Add probability columns
    for i in range(5):
        input_df[f'probability_class_{i}'] = [
            p['probabilities'].get(f'Class {i}', 0.0) for p in predictions
        ]
    
    # Save results
    input_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Predict pet adoption speed using hash model')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Single prediction
    single_parser = subparsers.add_parser('single', help='Make a single prediction')
    single_parser.add_argument('--type', required=True, help='Type of pet (Dog/Cat)')
    single_parser.add_argument('--age', type=int, required=True, help='Age in months')
    single_parser.add_argument('--breed', required=True, help='Primary breed')
    single_parser.add_argument('--gender', required=True, help='Gender (Male/Female/Mixed)')
    single_parser.add_argument('--color1', required=True, help='Primary color')
    single_parser.add_argument('--color2', default='', help='Secondary color')
    single_parser.add_argument('--size', required=True, help='Maturity size')
    single_parser.add_argument('--fur', required=True, help='Fur length')
    single_parser.add_argument('--vaccinated', required=True, help='Vaccination status')
    single_parser.add_argument('--sterilized', required=True, help='Sterilization status')
    single_parser.add_argument('--health', required=True, help='Health condition')
    single_parser.add_argument('--fee', type=float, required=True, help='Adoption fee')
    single_parser.add_argument('--photos', type=int, required=True, help='Number of photos')
    single_parser.add_argument('--description', default='', help='Pet description')
    
    # Batch prediction
    batch_parser = subparsers.add_parser('batch', help='Make batch predictions from CSV')
    batch_parser.add_argument('--input', required=True, help='Input CSV file')
    batch_parser.add_argument('--output', required=True, help='Output CSV file')
    
    # Model paths (optional)
    for p in [single_parser, batch_parser]:
        p.add_argument('--model', default='best_hash_model.h5', help='Path to model file')
        p.add_argument('--components', default='hash_preprocessing_components.joblib', 
                      help='Path to preprocessing components')
        p.add_argument('--metadata', default='hash_model_metadata.joblib', 
                      help='Path to metadata file')
    
    args = parser.parse_args()
    
    if args.command == 'single':
        # Create input data
        input_data = {
            'Type': args.type,
            'Age': args.age,
            'Breed1': args.breed,
            'Gender': args.gender,
            'Color1': args.color1,
            'Color2': args.color2,
            'MaturitySize': args.size,
            'FurLength': args.fur,
            'Vaccinated': args.vaccinated,
            'Sterilized': args.sterilized,
            'Health': args.health,
            'Fee': args.fee,
            'PhotoAmt': args.photos,
            'Description': args.description
        }
        
        # Make prediction
        result = predict_adoption_speed_hash(
            input_data, args.model, args.components, args.metadata
        )
        
        # Print result
        print(json.dumps(result, indent=2))
        
    elif args.command == 'batch':
        batch_predict_hash(
            args.input, args.output, 
            args.model, args.components, args.metadata
        )
    else:
        parser.print_help()

def example_prediction():
    """Example of how to use the hash prediction function"""
    example_pet = {
        'Type': 'Dog',
        'Age': 12,
        'Breed1': 'Labrador Retriever',
        'Gender': 'Male',
        'Color1': 'Yellow',
        'Color2': 'White',
        'MaturitySize': 'Large',
        'FurLength': 'Short',
        'Vaccinated': 'Yes',
        'Sterilized': 'Yes',
        'Health': 'Healthy',
        'Fee': 250.0,
        'PhotoAmt': 5,
        'Description': 'Friendly and healthy dog, loves to play with children. Fully vaccinated and neutered.'
    }
    
    try:
        result = predict_adoption_speed_hash(example_pet)
        print("Example prediction:")
        print(json.dumps(result, indent=2))
        return result
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

if __name__ == '__main__':
    main()
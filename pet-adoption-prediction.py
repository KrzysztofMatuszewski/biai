import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import argparse
import json
import hashlib

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
    
    # Apply hash encoding using saved mappings
    df_encoded = apply_hash_encoding_with_mappings(
        df, 
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

def predict_adoption_speed(input_data, 
                          model_path='best_model_simplified.h5',
                          components_path='preprocessing_components_simplified.joblib',
                          metadata_path='model_metadata_simplified.joblib'):
    """Predict adoption speed using simplified model"""
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

def batch_predict(input_file, output_file,
                 model_path='best_model_simplified.h5',
                 components_path='preprocessing_components_simplified.joblib',
                 metadata_path='model_metadata_simplified.joblib'):
    """Make batch predictions from CSV file"""
    # Read input data
    input_df = pd.read_csv(input_file)
    print(f"Loaded {len(input_df)} rows for prediction")
    
    # Make predictions
    predictions = []
    for i, row in input_df.iterrows():
        try:
            pred = predict_adoption_speed(
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
    parser = argparse.ArgumentParser(description='Predict pet adoption speed')
    
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
    
    # Batch prediction
    batch_parser = subparsers.add_parser('batch', help='Make batch predictions from CSV')
    batch_parser.add_argument('--input', required=True, help='Input CSV file')
    batch_parser.add_argument('--output', required=True, help='Output CSV file')
    
    # Model paths (optional)
    for p in [single_parser, batch_parser]:
        p.add_argument('--model', default='best_model_simplified.h5', help='Path to model file')
        p.add_argument('--components', default='preprocessing_components_simplified.joblib', 
                      help='Path to preprocessing components')
        p.add_argument('--metadata', default='model_metadata_simplified.joblib', 
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
            'PhotoAmt': args.photos
        }
        
        # Make prediction
        result = predict_adoption_speed(
            input_data, args.model, args.components, args.metadata
        )
        
        # Print result
        print(json.dumps(result, indent=2))
        
    elif args.command == 'batch':
        batch_predict(
            args.input, args.output, 
            args.model, args.components, args.metadata
        )
    else:
        parser.print_help()

def example_prediction():
    """Example of how to use the prediction function"""
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
        'PhotoAmt': 5
    }
    
    try:
        result = predict_adoption_speed(example_pet)
        print("Example prediction:")
        print(json.dumps(result, indent=2))
        return result
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

if __name__ == '__main__':
    main()
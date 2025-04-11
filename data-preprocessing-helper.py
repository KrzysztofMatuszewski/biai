import pandas as pd
import numpy as np
import re
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download NLTK resources
nltk.download('vader_lexicon')
nltk.download('stopwords')

def load_and_clean_data(file_path):
    """
    Load and clean the pet adoption dataset
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    # Load the data
    data = pd.read_csv(file_path)
    
    # Display initial info
    print("Original data shape:", data.shape)
    print("\nMissing values:")
    print(data.isnull().sum())
    
    # Convert columns to correct types
    numeric_columns = ['Type', 'Age', 'Gender', 'MaturitySize', 'FurLength', 
                      'Vaccinated', 'Sterilized', 'Health', 'Fee', 'PhotoAmt', 'AdoptionSpeed']
    
    for col in numeric_columns:
        if col in data.columns and data[col].dtype == 'object':
            try:
                data[col] = pd.to_numeric(data[col])
                print(f"Converted {col} to numeric")
            except:
                print(f"Could not convert {col} to numeric, keeping as is")
    
    # For categorical columns, ensure they are strings
    categorical_cols = ['Breed1', 'Color1', 'Color2']
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype(str)
    
    # Handle missing values
    # For numeric columns, fill with median
    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())
    
    # For categorical columns, fill with most frequent value
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].mode()[0])
    
    # Description column - fill missing with empty string
    if 'Description' in data.columns:
        data['Description'] = data['Description'].fillna('')
    
    return data

def extract_text_features(text):
    """
    Extract features from the pet description text
    
    Parameters:
    text (str): Description text
    
    Returns:
    dict: Extracted features
    """
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
    
    # Ensure text is a string
    text = str(text)
    
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

def add_engineered_features(data):
    """
    Add engineered features to the dataframe
    
    Parameters:
    data (pd.DataFrame): Original dataframe
    
    Returns:
    pd.DataFrame: Dataframe with added features
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # 1. Extract text features if Description column exists
    if 'Description' in df.columns:
        print("Extracting features from descriptions...")
        text_features = df['Description'].apply(extract_text_features).apply(pd.Series)
        df = pd.concat([df, text_features], axis=1)
    
    # 2. Create interaction features
    print("Creating interaction features...")
    df['Age_Health_Interaction'] = df['Age'] * df['Health']
    df['Vaccinated_Sterilized'] = df['Vaccinated'].astype(str) + '_' + df['Sterilized'].astype(str)
    df['Size_FurLength'] = df['MaturitySize'].astype(str) + '_' + df['FurLength'].astype(str)
    
    # Avoid division by zero
    df['Price_Per_Photo'] = df['Fee'] / (df['PhotoAmt'] + 1)
    
    # 3. Normalize skewed features
    df['PhotoAmt_Log'] = np.log1p(df['PhotoAmt'])
    df['Fee_Log'] = np.log1p(df['Fee'])
    
    # 4. Create pet age categories
    age_bins = [0, 3, 12, 36, 72, float('inf')]
    age_labels = ['Baby', 'Young', 'Adult', 'Middle-aged', 'Senior']
    df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
    
    # 5. Premium pet indicator (combination of features that might make a pet more desirable)
    df['IsPremium'] = ((df['Vaccinated'] == 1) & 
                       (df['Sterilized'] == 1) & 
                       (df['Health'] == 1)).astype(int)
    
    return df

def prepare_for_modeling(data):
    """
    Prepare the data for modeling by defining column types
    
    Parameters:
    data (pd.DataFrame): Dataframe with all features
    
    Returns:
    tuple: (X, y, categorical_cols, numerical_cols)
    """
    # Define column types for preprocessing
    categorical_cols = ['Type', 'Breed1', 'Gender', 'Color1', 'Color2', 
                        'MaturitySize', 'FurLength', 'Vaccinated', 
                        'Sterilized', 'Health', 'Vaccinated_Sterilized', 
                        'Size_FurLength', 'AgeGroup']
                        
    numerical_cols = ['Age', 'Fee', 'PhotoAmt', 'desc_length', 'word_count', 
                      'sentiment_compound', 'sentiment_pos', 'sentiment_neg', 
                      'sentiment_neu', 'has_contact', 'has_health_mention',
                      'Age_Health_Interaction', 'Price_Per_Photo',
                      'PhotoAmt_Log', 'Fee_Log', 'IsPremium']
    
    # Remove columns that don't exist in the dataframe
    categorical_cols = [col for col in categorical_cols if col in data.columns]
    numerical_cols = [col for col in numerical_cols if col in data.columns]
    
    # Split the data into features and target
    X = data.drop(['AdoptionSpeed', 'Description'] if 'Description' in data.columns else ['AdoptionSpeed'], axis=1)
    y = data['AdoptionSpeed']
    
    return X, y, categorical_cols, numerical_cols

def process_data_pipeline(file_path):
    """
    Full pipeline for processing pet adoption data
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    tuple: (X, y, categorical_cols, numerical_cols, processed_data)
    """
    # Load and clean data
    data = load_and_clean_data(file_path)
    
    # Add engineered features
    processed_data = add_engineered_features(data)
    
    # Prepare for modeling
    X, y, categorical_cols, numerical_cols = prepare_for_modeling(processed_data)
    
    print("\nData preprocessing complete!")
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print(f"Number of categorical columns: {len(categorical_cols)}")
    print(f"Number of numerical columns: {len(numerical_cols)}")
    
    return X, y, categorical_cols, numerical_cols, processed_data

# Usage example
if __name__ == "__main__":
    # Replace with your file path
    file_path = "pet_adoption.csv"
    X, y, categorical_cols, numerical_cols, processed_data = process_data_pipeline(file_path)
    
    # Display sample of processed data
    print("\nSample of processed data:")
    print(processed_data.head())

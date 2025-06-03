import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import hashlib
import joblib

# Download NLTK resources
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def string_to_hash_int(text, max_value=10000):
    """
    Convert string to integer using hash function
    
    Parameters:
    text (str): Input string
    max_value (int): Maximum value for hash (controls range)
    
    Returns:
    int: Hash integer value
    """
    if pd.isna(text) or text == '' or text == 'nan':
        return 0
    
    # Convert to string and create hash
    hash_object = hashlib.md5(str(text).encode())
    hash_int = int(hash_object.hexdigest(), 16) % max_value
    return hash_int

def hash_encode_categorical(df, categorical_columns, hash_size=1000):
    """
    Hash encode categorical columns to integers
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    categorical_columns (list): List of categorical column names
    hash_size (int): Size of hash space
    
    Returns:
    pd.DataFrame: Dataframe with hash encoded categorical columns
    """
    df_encoded = df.copy()
    hash_mappings = {}
    
    for col in categorical_columns:
        if col in df_encoded.columns:
            print(f"Hash encoding column: {col}")
            # Create hash mapping for unique values
            unique_values = df_encoded[col].unique()
            hash_mappings[col] = {}
            
            for val in unique_values:
                hash_val = string_to_hash_int(str(val), hash_size)
                hash_mappings[col][val] = hash_val
            
            # Apply hash encoding
            df_encoded[col] = df_encoded[col].map(hash_mappings[col])
            
            print(f"  - Unique values: {len(unique_values)}")
            print(f"  - Hash range: 0-{hash_size}")
    
    return df_encoded, hash_mappings

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
    
    # Initialize sentiment analyzer
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
    """
    Create interaction features using hash encoding
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    
    Returns:
    pd.DataFrame: Dataframe with interaction features
    """
    df_new = df.copy()
    
    # Create interaction features by combining columns and hashing
    interaction_combinations = [
        ('Vaccinated', 'Sterilized'),
        ('MaturitySize', 'FurLength'),
        ('Type', 'Breed1'),
        ('Color1', 'Color2'),
        ('Gender', 'MaturitySize'),
        ('Age_Group', 'Health_Status')
    ]
    
    # Create age groups
    df_new['Age_Group'] = pd.cut(df_new['Age'], 
                                bins=[0, 3, 12, 36, 72, float('inf')], 
                                labels=['Baby', 'Young', 'Adult', 'Middle', 'Senior'])
    
    # Create health status groups
    df_new['Health_Status'] = df_new['Health']
    
    # Create interaction features
    for col1, col2 in interaction_combinations:
        if col1 in df_new.columns and col2 in df_new.columns:
            interaction_name = f'{col1}_{col2}_hash'
            # Combine values and hash
            combined = df_new[col1].astype(str) + '_' + df_new[col2].astype(str)
            df_new[interaction_name] = combined.apply(lambda x: string_to_hash_int(x, 1000))
            print(f"Created interaction feature: {interaction_name}")
    
    return df_new

def prepare_all_features(df):
    """
    Prepare all features including text features, interactions, and transformations
    
    Parameters:
    df (pd.DataFrame): Raw dataframe
    
    Returns:
    pd.DataFrame: Processed dataframe with all features
    """
    print("Starting feature preparation...")
    
    # 1. Extract text features from Description
    if 'Description' in df.columns:
        print("Extracting text features...")
        text_features = df['Description'].apply(extract_text_features).apply(pd.Series)
        df = pd.concat([df, text_features], axis=1)
    
    # 2. Create basic mathematical features
    print("Creating mathematical features...")
    df['Age_Health_Product'] = df['Age'] * pd.to_numeric(df['Health'], errors='coerce').fillna(1)
    df['Price_Per_Photo'] = df['Fee'] / (df['PhotoAmt'] + 1)  # Avoid division by zero
    df['PhotoAmt_Log'] = np.log1p(df['PhotoAmt'])
    df['Fee_Log'] = np.log1p(df['Fee'])
    
    # 3. Create categorical groupings before hashing
    df = create_interaction_features_hash(df)
    
    # 4. Create premium indicator
    # Convert categorical health to numeric for comparison
    health_numeric = pd.to_numeric(df['Health'], errors='coerce').fillna(0)
    vaccinated_numeric = pd.to_numeric(df['Vaccinated'], errors='coerce').fillna(0)
    sterilized_numeric = pd.to_numeric(df['Sterilized'], errors='coerce').fillna(0)
    
    df['Is_Premium'] = ((health_numeric == 1) & 
                        (vaccinated_numeric == 1) & 
                        (sterilized_numeric == 1)).astype(int)
    
    print("Feature preparation complete!")
    return df

# Load the data
print("Loading data...")
data = pd.read_csv('petfinder.csv')

# Display basic information
print("Data shape:", data.shape)
print("\nBasic statistics:")
print(data.describe())

# Check data types
print("\nData types:")
print(data.dtypes)

# Fill missing values before processing
print("\nHandling missing values...")
# Fill numeric columns with median
numeric_cols_raw = ['Age', 'Fee', 'PhotoAmt']
for col in numeric_cols_raw:
    if col in data.columns:
        data[col] = data[col].fillna(data[col].median())

# Fill categorical columns with mode or 'Unknown'
categorical_cols_raw = ['Type', 'Breed1', 'Gender', 'Color1', 'Color2', 
                       'MaturitySize', 'FurLength', 'Vaccinated', 
                       'Sterilized', 'Health']
for col in categorical_cols_raw:
    if col in data.columns:
        data[col] = data[col].fillna('Unknown')

# Fill Description with empty string
if 'Description' in data.columns:
    data['Description'] = data['Description'].fillna('')

# Prepare all features
data_with_features = prepare_all_features(data)

# Define categorical columns for hash encoding
categorical_columns = ['Type', 'Breed1', 'Gender', 'Color1', 'Color2', 
                      'MaturitySize', 'FurLength', 'Vaccinated', 
                      'Sterilized', 'Health', 'Age_Group', 'Health_Status']

# Hash encode categorical columns
print("\nApplying hash encoding to categorical columns...")
data_encoded, hash_mappings = hash_encode_categorical(data_with_features, categorical_columns, hash_size=1000)

# Define final feature columns (all should be numeric now)
feature_columns = [col for col in data_encoded.columns if col not in ['AdoptionSpeed', 'Description']]

# Verify all columns are numeric
print("\nVerifying data types after encoding:")
non_numeric_cols = []
for col in feature_columns:
    if not pd.api.types.is_numeric_dtype(data_encoded[col]):
        non_numeric_cols.append(col)
        print(f"Non-numeric column found: {col} - {data_encoded[col].dtype}")

if non_numeric_cols:
    print(f"Converting remaining non-numeric columns: {non_numeric_cols}")
    for col in non_numeric_cols:
        data_encoded[col] = pd.to_numeric(data_encoded[col], errors='coerce').fillna(0)

# Final verification
print("\nFinal data types check:")
for col in feature_columns[:10]:  # Show first 10 columns
    print(f"{col}: {data_encoded[col].dtype}")

# Split features and target
X = data_encoded[feature_columns]
y = data_encoded['AdoptionSpeed']

print(f"\nFinal feature matrix shape: {X.shape}")
print(f"All features are numeric: {all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns)}")

# Plot class distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='AdoptionSpeed', data=data_encoded)
plt.title('Distribution of Adoption Speed')
plt.xlabel('Adoption Speed (0: Fastest, 4: Slowest)')
plt.ylabel('Count')
plt.savefig('class_distribution.png')
plt.close()

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Simple preprocessing pipeline for all-numeric data
print("\nApplying preprocessing...")
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

# Fit on training data
X_train_imputed = imputer.fit_transform(X_train)
X_train_scaled = scaler.fit_transform(X_train_imputed)

# Transform validation and test data
X_val_imputed = imputer.transform(X_val)
X_val_scaled = scaler.transform(X_val_imputed)

X_test_imputed = imputer.transform(X_test)
X_test_scaled = scaler.transform(X_test_imputed)

print(f"Preprocessed training data shape: {X_train_scaled.shape}")

# Convert target to one-hot encoding
num_classes = len(np.unique(y))
y_train_encoded = keras.utils.to_categorical(y_train, num_classes)
y_val_encoded = keras.utils.to_categorical(y_val, num_classes)
y_test_encoded = keras.utils.to_categorical(y_test, num_classes)

# Class weights for imbalanced data
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(class_weight='balanced',
                                    classes=np.unique(y_train),
                                    y=y_train)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
print("Class weights:", class_weight_dict)

# Define model architecture
def create_hash_model(input_shape, num_classes):
    """Create neural network model optimized for hash-encoded features"""
    inputs = keras.Input(shape=(input_shape,))
    
    # First layer - larger since we have hash-encoded features
    x = layers.Dense(512, activation='relu', 
                     kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    # Second layer
    x = layers.Dense(256, activation='relu', 
                     kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    # Third layer
    x = layers.Dense(128, activation='relu', 
                     kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Fourth layer
    x = layers.Dense(64, activation='relu', 
                     kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create the model
input_shape = X_train_scaled.shape[1]
model = create_hash_model(input_shape, num_classes)

print("\nModel architecture:")
model.summary()

# Define callbacks
checkpoint = keras.callbacks.ModelCheckpoint(
    'best_hash_model.h5', 
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True, 
    mode='min'
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

callbacks = [checkpoint, early_stopping, reduce_lr]

# Train the model
print("\nStarting training...")
history = model.fit(
    X_train_scaled, y_train_encoded,
    validation_data=(X_val_scaled, y_val_encoded),
    epochs=100,
    batch_size=64,  # Increased batch size for hash-encoded data
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# Load the best model
try:
    model = keras.models.load_model('best_hash_model.h5')
    print("Loaded best model from disk")
except:
    print("Could not load best model, using current model")

# Evaluate the model
print("\nEvaluating model...")
y_pred_proba = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_proba, axis=1)
y_test_decoded = np.argmax(y_test_encoded, axis=1)

# Calculate metrics
accuracy = accuracy_score(y_test_decoded, y_pred)
f1 = f1_score(y_test_decoded, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test_decoded, y_pred)

print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test F1 Score (weighted): {f1:.4f}")

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(num_classes),
            yticklabels=range(num_classes))
plt.title('Confusion Matrix - Hash Encoded Model')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix_hash.png')
plt.close()

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test_decoded, y_pred))

# Plot training history
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy - Hash Encoded')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss - Hash Encoded')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])

plt.tight_layout()
plt.savefig('training_history_hash.png')
plt.close()

# Save all components
print("\nSaving model components...")

# Save the preprocessing components
preprocessing_components = {
    'imputer': imputer,
    'scaler': scaler,
    'hash_mappings': hash_mappings,
    'feature_columns': feature_columns,
    'categorical_columns': categorical_columns
}

joblib.dump(preprocessing_components, 'hash_preprocessing_components.joblib')

# Save model metadata
model_metadata = {
    'num_classes': num_classes,
    'input_shape': input_shape,
    'class_weights': class_weight_dict,
    'feature_count': len(feature_columns)
}
joblib.dump(model_metadata, 'hash_model_metadata.joblib')

print("Hash-encoded model training complete!")
print(f"Total features used: {len(feature_columns)}")
print("All components saved successfully.")

# Example prediction function for new data
def predict_with_hash_model(new_data, model_path='best_hash_model.h5'):
    """
    Predict adoption speed for new data using hash-encoded model
    
    Parameters:
    new_data (dict or pd.DataFrame): New pet data
    model_path (str): Path to saved model
    
    Returns:
    dict: Prediction results
    """
    # Load components
    model = keras.models.load_model(model_path)
    components = joblib.load('hash_preprocessing_components.joblib')
    metadata = joblib.load('hash_model_metadata.joblib')
    
    # Convert to DataFrame if needed
    if isinstance(new_data, dict):
        df = pd.DataFrame([new_data])
    else:
        df = new_data.copy()
    
    # Apply same feature engineering
    df_processed = prepare_all_features(df)
    
    # Apply hash encoding
    for col in components['categorical_columns']:
        if col in df_processed.columns:
            # Map using saved hash mappings
            if col in components['hash_mappings']:
                df_processed[col] = df_processed[col].map(
                    components['hash_mappings'][col]
                ).fillna(0)
    
    # Select features
    X_new = df_processed[components['feature_columns']]
    
    # Apply preprocessing
    X_imputed = components['imputer'].transform(X_new)
    X_scaled = components['scaler'].transform(X_imputed)
    
    # Predict
    pred_proba = model.predict(X_scaled)
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
                         for i in range(metadata['num_classes'])}
    }
    
    return result

print("\nHash-encoded model ready for predictions!")
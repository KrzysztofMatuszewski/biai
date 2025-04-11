import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re

# Download NLTK resources
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load the data
data = pd.read_csv('petfinder.csv')

# Display basic information
print("Data shape:", data.shape)
print("\nBasic statistics:")
print(data.describe())

# Check data types to ensure proper handling
print("\nData types:")
print(data.dtypes)

# Convert string columns to appropriate types if needed
# The error in the original script was caused by data type issues

# If any columns are detected as strings but should be numeric, convert them
numeric_columns = ['Type', 'Age', 'Gender', 'MaturitySize', 'FurLength', 
                   'Vaccinated', 'Sterilized', 'Health', 'Fee', 'PhotoAmt', 'AdoptionSpeed']

for col in numeric_columns:
    if col in data.columns and data[col].dtype == 'object':
        try:
            data[col] = pd.to_numeric(data[col])
            print(f"Converted {col} to numeric")
        except:
            print(f"Could not convert {col} to numeric, keeping as object")

# Plot class distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='AdoptionSpeed', data=data)
plt.title('Distribution of Adoption Speed')
plt.xlabel('Adoption Speed (0: Fastest, 4: Slowest)')
plt.ylabel('Count')
plt.savefig('class_distribution.png')
plt.close()

# Feature Engineering

# 1. Text Features from Description
# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

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

# Extract text features
text_features = data['Description'].apply(extract_text_features).apply(pd.Series)
data = pd.concat([data, text_features], axis=1)

# 2. Create interaction features
data['Age_Health_Interaction'] = data['Age'] * data['Health']

# Handle categorical combinations separately to avoid mixed types
data['Vaccinated_Sterilized'] = data['Vaccinated'].astype(str) + '_' + data['Sterilized'].astype(str)
data['Size_FurLength'] = data['MaturitySize'].astype(str) + '_' + data['FurLength'].astype(str)

# Avoid division by zero
data['Price_Per_Photo'] = data['Fee'] / (data['PhotoAmt'] + 1)  

# 3. Normalize skewed features
data['PhotoAmt_Log'] = np.log1p(data['PhotoAmt'])
data['Fee_Log'] = np.log1p(data['Fee'])

# Define column types for preprocessing
categorical_cols = ['Type', 'Breed1', 'Gender', 'Color1', 'Color2', 
                    'MaturitySize', 'FurLength', 'Vaccinated', 
                    'Sterilized', 'Health', 'Vaccinated_Sterilized', 'Size_FurLength']
                    
numerical_cols = ['Age', 'Fee', 'PhotoAmt', 'desc_length', 'word_count', 
                  'sentiment_compound', 'sentiment_pos', 'sentiment_neg', 
                  'sentiment_neu', 'has_contact', 'has_health_mention',
                  'Age_Health_Interaction', 'Price_Per_Photo',
                  'PhotoAmt_Log', 'Fee_Log']

# Verify column types before preprocessing
print("\nChecking data types for numeric columns:")
for col in numerical_cols:
    if col in data.columns:
        print(f"{col}: {data[col].dtype}")

# Split the data into features and target
X = data.drop(['AdoptionSpeed', 'Description'], axis=1)  # Remove Description as we've extracted features
y = data['AdoptionSpeed']

# Print some samples to verify data quality
print("\nSample data after feature engineering:")
print(X.iloc[:3])

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Create preprocessing pipeline
# The key fix: Ensure numeric columns are actually numeric and separate categorical preprocessing
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Check for any non-numeric data in numerical_cols
for col in numerical_cols:
    if col in X_train.columns and not pd.api.types.is_numeric_dtype(X_train[col]):
        print(f"Warning: {col} contains non-numeric data. Converting to numeric...")
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_val[col] = pd.to_numeric(X_val[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Preprocess the data
try:
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_val_preprocessed = preprocessor.transform(X_val)
    X_test_preprocessed = preprocessor.transform(X_test)
    print("Preprocessing successful!")
except Exception as e:
    print(f"Error during preprocessing: {e}")
    # Fallback to simpler preprocessing if needed
    simple_numerical_cols = ['Age', 'Fee', 'PhotoAmt']
    
    simple_preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), simple_numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])
    
    X_train_preprocessed = simple_preprocessor.fit_transform(X_train)
    X_val_preprocessed = simple_preprocessor.transform(X_val)
    X_test_preprocessed = simple_preprocessor.transform(X_test)
    print("Fallback preprocessing successful!")

print(f"Preprocessed training data shape: {X_train_preprocessed.shape}")

# Convert target to one-hot encoding for categorical cross-entropy
num_classes = len(np.unique(y))
y_train_encoded = keras.utils.to_categorical(y_train, num_classes)
y_val_encoded = keras.utils.to_categorical(y_val, num_classes)
y_test_encoded = keras.utils.to_categorical(y_test, num_classes)

# Class weights to handle imbalance
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(class_weight='balanced',
                                    classes=np.unique(y_train),
                                    y=y_train)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
print("Class weights:", class_weight_dict)

# Define model architecture
def create_model(input_shape, num_classes):
    # Input layer
    inputs = keras.Input(shape=(input_shape,))
    
    # First hidden layer
    x = layers.Dense(256, activation='relu', 
                     kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    # Second hidden layer
    x = layers.Dense(128, activation='relu', 
                     kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Third hidden layer
    x = layers.Dense(64, activation='relu', 
                     kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create the model
input_shape = X_train_preprocessed.shape[1]
model = create_model(input_shape, num_classes)

print("\nModel architecture:")
model.summary()

# Define callbacks
checkpoint = keras.callbacks.ModelCheckpoint(
    'best_model.h5', 
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
history = model.fit(
    X_train_preprocessed, y_train_encoded,
    validation_data=(X_val_preprocessed, y_val_encoded),
    epochs=100,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# Load the best model
try:
    model = keras.models.load_model('best_model.h5')
    print("Loaded best model from disk")
except:
    print("Could not load best model, using current model")

# Evaluate the model on test data
y_pred_proba = model.predict(X_test_preprocessed)
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
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')
plt.close()

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test_decoded, y_pred))

# Plot training history
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])

plt.tight_layout()
plt.savefig('training_history.png')
plt.close()

# Save the model and preprocessor
import joblib

# Save preprocessor
joblib.dump(preprocessor, 'pet_adoption_preprocessor.joblib')

# Save additional model metadata
model_metadata = {
    'numerical_cols': numerical_cols,
    'categorical_cols': categorical_cols,
    'num_classes': num_classes,
    'class_weights': class_weight_dict,
    'input_shape': input_shape,
}
joblib.dump(model_metadata, 'pet_adoption_metadata.joblib')

print("\nModel training and evaluation complete. Model and preprocessor saved.")

# Function to predict adoption speed for new pets
def predict_adoption_speed(new_data):
    """
    Predict adoption speed for new pet data
    
    Parameters:
    new_data (pd.DataFrame): DataFrame with required features
    
    Returns:
    dict: Dictionary with prediction information
    """
    # Preprocess the data
    processed_data = preprocessor.transform(new_data)
    
    # Make prediction
    predictions_proba = model.predict(processed_data)
    predictions = np.argmax(predictions_proba, axis=1)
    
    # Create result dictionary
    adoption_speed_map = {
        0: "Same day (0)",
        1: "1-7 days (1)",
        2: "8-30 days (2)",
        3: "31-90 days (3)",
        4: "No adoption after 100 days (4)"
    }
    
    results = []
    for i, pred in enumerate(predictions):
        result = {
            'predicted_class': int(pred),
            'predicted_label': adoption_speed_map[pred],
            'probabilities': {f'Class {j}': float(predictions_proba[i][j]) for j in range(num_classes)}
        }
        results.append(result)
    
    return results[0] if len(results) == 1 else results

print("\nModel ready for predictions!")
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def create_output_directories():
    """Create output directories for organized file storage"""
    directories = ['png', 'model']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")

def string_to_hash_int(text, max_value=1000):
    """Convert string to integer using hash function"""
    if pd.isna(text) or text == '' or text == 'nan':
        return 0
    
    hash_object = hashlib.md5(str(text).encode())
    hash_int = int(hash_object.hexdigest(), 16) % max_value
    return hash_int

def hash_encode_categorical(df, categorical_columns, hash_size=1000):
    """Hash encode categorical columns to integers"""
    df_encoded = df.copy()
    hash_mappings = {}
    
    for col in categorical_columns:
        if col in df_encoded.columns:
            print(f"Hash encoding column: {col}")
            unique_values = df_encoded[col].unique()
            hash_mappings[col] = {}
            
            for val in unique_values:
                hash_val = string_to_hash_int(str(val), hash_size)
                hash_mappings[col][val] = hash_val
            
            df_encoded[col] = df_encoded[col].map(hash_mappings[col])
            print(f"  - Unique values: {len(unique_values)}")
    
    return df_encoded, hash_mappings

def create_simplified_model(input_shape, num_classes):
    """Create neural network model with basic features only"""
    inputs = keras.Input(shape=(input_shape,))
    
    x = layers.Dense(128, activation='relu', 
                     kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(64, activation='relu', 
                     kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(32, activation='relu', 
                     kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_data_for_correlation(data):
    """
    Prepare data for correlation analysis - convert categorical to numerical
    """
    df_corr = data.copy()
    
    # List of categorical columns to encode
    categorical_columns = ['Type', 'Breed1', 'Gender', 'Color1', 'Color2', 
                          'MaturitySize', 'FurLength', 'Vaccinated', 
                          'Sterilized', 'Health']
    
    # Encoder for categorical variables
    label_encoders = {}
    
    for col in categorical_columns:
        if col in df_corr.columns:
            le = LabelEncoder()
            # Fill missing values before encoding
            df_corr[col] = df_corr[col].fillna('Unknown')
            df_corr[col] = le.fit_transform(df_corr[col].astype(str))
            label_encoders[col] = le
    
    # Fill missing numerical values
    numeric_cols = ['Age', 'Fee', 'PhotoAmt']
    for col in numeric_cols:
        if col in df_corr.columns:
            df_corr[col] = df_corr[col].fillna(df_corr[col].median())
    
    return df_corr, label_encoders

def create_correlation_plots(data):
    """
    Create correlation plots for ALL neural network input data
    """
    # Prepare all features like in neural network
    df_processed, _ = prepare_data_for_correlation(data)
    
    # All columns that go into the network (without AdoptionSpeed)
    network_input_columns = [
        # Basic features
        'Age', 'Fee', 'PhotoAmt', 'Type', 'Breed1', 'Gender', 
        'Color1', 'Color2', 'MaturitySize', 'FurLength', 
        'Vaccinated', 'Sterilized', 'Health'
    ]
    
    # Filter columns that actually exist in data
    available_columns = [col for col in network_input_columns if col in df_processed.columns]
    
    print(f"Network input features: {len(available_columns)}")
    print(f"Feature list: {available_columns}")
    
    # Add target variable if exists
    target_column = None
    if 'AdoptionSpeed' in df_processed.columns:
        available_columns.append('AdoptionSpeed')
        target_column = 'AdoptionSpeed'
    
    # Select data for correlation
    correlation_data = df_processed[available_columns]
    
    # Calculate correlation matrix
    correlation_matrix = correlation_data.corr()
    
    # Create plots - 2 plots for better readability
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    fig.suptitle(f'Correlation Analysis of ALL Neural Network Inputs - PetFinder\n({len(available_columns)} features)', 
                 fontsize=16, fontweight='bold')
    
    # 1. Full correlation matrix
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.2f',
                annot_kws={'size': 8},
                cbar_kws={'shrink': 0.8},
                ax=axes[0])
    axes[0].set_title(f'Full Correlation Matrix\n({len(correlation_matrix.columns)} x {len(correlation_matrix.columns)} features)', 
                      fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45, labelsize=9)
    axes[0].tick_params(axis='y', rotation=0, labelsize=9)
    
    # 2. Correlations with AdoptionSpeed (if exists)
    if target_column and target_column in correlation_matrix.columns:
        adoption_corr = correlation_matrix[target_column].drop(target_column).sort_values(key=abs, ascending=False)
        
        # Color code for better readability
        colors = []
        for x in adoption_corr.values:
            if x > 0.3:
                colors.append('darkblue')
            elif x > 0.1:
                colors.append('blue')
            elif x > -0.1:
                colors.append('gray')
            elif x > -0.3:
                colors.append('red')
            else:
                colors.append('darkred')
        
        bars = axes[1].barh(range(len(adoption_corr)), adoption_corr.values, color=colors, alpha=0.7)
        axes[1].set_yticks(range(len(adoption_corr)))
        axes[1].set_yticklabels(adoption_corr.index, fontsize=9)
        axes[1].set_xlabel('Correlation Coefficient', fontsize=12)
        axes[1].set_title(f'Correlation with Adoption Speed\n(sorted by correlation strength)', 
                          fontsize=14, fontweight='bold')
        axes[1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        axes[1].axvline(x=0.3, color='blue', linestyle='--', alpha=0.5, label='Moderate (+)')
        axes[1].axvline(x=-0.3, color='red', linestyle='--', alpha=0.5, label='Moderate (-)')
        axes[1].legend()
        
        # Add values on bars
        for i, (bar, value) in enumerate(zip(bars, adoption_corr.values)):
            axes[1].text(value + (0.01 if value >= 0 else -0.01), i, f'{value:.3f}', 
                          ha='left' if value >= 0 else 'right', va='center', fontsize=8)
        
        # Display statistics
        strong_positive = sum(1 for x in adoption_corr.values if x > 0.3)
        strong_negative = sum(1 for x in adoption_corr.values if x < -0.3)
        axes[1].text(0.02, 0.98, f'Strong positive: {strong_positive}\nStrong negative: {strong_negative}', 
                     transform=axes[1].transAxes, va='top', ha='left', 
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                     
    else:
        axes[1].text(0.5, 0.5, f'{target_column if target_column else "AdoptionSpeed"}\nnot found\nin data', 
                       ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
        axes[1].set_title('Correlation with Adoption Speed', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig, correlation_matrix

def analyze_top_correlations(correlation_matrix, threshold=0.3):
    """
    Analyze and display top correlations
    """
    print("\n=== TOP CORRELATIONS ANALYSIS ===\n")
    
    # Find pairs with highest correlation
    corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                corr_pairs.append({
                    'var1': correlation_matrix.columns[i],
                    'var2': correlation_matrix.columns[j],
                    'correlation': corr_value
                })
    
    # Sort by absolute value
    corr_pairs = sorted(corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)
    
    print(f"Found correlations stronger than {threshold}:")
    print("-" * 60)
    
    for pair in corr_pairs:
        direction = "positive" if pair['correlation'] > 0 else "negative"
        strength = "very strong" if abs(pair['correlation']) > 0.7 else "strong" if abs(pair['correlation']) > 0.5 else "moderate"
        
        print(f"{pair['var1']} ↔ {pair['var2']}")
        print(f"  Correlation: {pair['correlation']:.3f} ({direction}, {strength})")
        print()
    
    if not corr_pairs:
        print(f"No correlations stronger than {threshold} found")
    
    return corr_pairs

def analyze_class_distribution(y, class_names=None):
    """
    Analizuje rozkład klas i wyświetla szczegółowe statystyki
    """
    print("\n" + "="*60)
    print("ANALIZA ROZKŁADU KLAS")
    print("="*60)
    
    # Oblicz statystyki klas
    unique_classes, class_counts = np.unique(y, return_counts=True)
    total_samples = len(y)
    
    # Utwórz nazwy klas jeśli nie podano
    if class_names is None:
        class_names = [f"AdoptionSpeed_{i}" for i in unique_classes]
    
    print(f"Całkowita liczba próbek: {total_samples}")
    print(f"Liczba klas: {len(unique_classes)}")
    print("\nRozkład klas:")
    print("-" * 50)
    
    for i, (cls, count) in enumerate(zip(unique_classes, class_counts)):
        percentage = (count / total_samples) * 100
        print(f"Klasa {cls} ({class_names[i]}): {count:5d} próbek ({percentage:5.1f}%)")
    
    # Sprawdź czy dane są niezbalansowane
    min_count = np.min(class_counts)
    max_count = np.max(class_counts)
    imbalance_ratio = max_count / min_count
    
    print(f"\nStatystyki niezbalansowania:")
    print(f"Najmniejsza klasa: {min_count} próbek")
    print(f"Największa klasa: {max_count} próbek") 
    print(f"Współczynnik niezbalansowania: {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 2:
        print("⚠️  OSTRZEŻENIE: Dane są niezbalansowane!")
    else:
        print("✅ Dane są względnie zbalansowane")
        
    return unique_classes, class_counts, imbalance_ratio

def calculate_enhanced_class_weights(y, method='balanced', custom_weights=None):
    """
    Oblicza wagi klas różnymi metodami
    
    Parameters:
    - method: 'balanced', 'inverse_freq', 'sqrt_inverse', 'log_inverse', 'custom'
    - custom_weights: słownik z wagami dla metody 'custom'
    """
    unique_classes, class_counts = np.unique(y, return_counts=True)
    total_samples = len(y)
    
    print(f"\n🔧 Obliczanie wag klas metodą: {method}")
    
    if method == 'balanced':
        # Standardowa metoda sklearn
        weights = compute_class_weight(class_weight='balanced',
                                     classes=unique_classes,
                                     y=y)
        weight_dict = {i: weight for i, weight in enumerate(weights)}
        
    elif method == 'inverse_freq':
        # Odwrotność częstotliwości
        weights = total_samples / (len(unique_classes) * class_counts)
        weight_dict = {cls: weight for cls, weight in zip(unique_classes, weights)}
        
    elif method == 'sqrt_inverse':
        # Pierwiastek z odwrotności częstotliwości (łagodniejsze)
        frequencies = class_counts / total_samples
        weights = 1 / np.sqrt(frequencies)
        # Normalizuj tak aby średnia waga = 1
        weights = weights / np.mean(weights)
        weight_dict = {cls: weight for cls, weight in zip(unique_classes, weights)}
        
    elif method == 'log_inverse':
        # Logarytmiczna odwrotność (bardzo łagodne)
        frequencies = class_counts / total_samples
        weights = 1 / np.log(frequencies + 1)
        weights = weights / np.mean(weights)
        weight_dict = {cls: weight for cls, weight in zip(unique_classes, weights)}
        
    elif method == 'custom' and custom_weights is not None:
        # Niestandardowe wagi
        weight_dict = custom_weights.copy()
        
    else:
        raise ValueError(f"Nieznana metoda: {method}")
    
    # Wyświetl obliczone wagi
    print("Obliczone wagi klas:")
    print("-" * 30)
    for cls in unique_classes:
        count = class_counts[list(unique_classes).index(cls)]
        weight = weight_dict[cls]
        print(f"Klasa {cls}: waga = {weight:.4f} (próbek: {count})")
    
    return weight_dict

def compare_weight_methods(y):
    """
    Porównuje różne metody obliczania wag
    """
    print("\n" + "="*60)
    print("PORÓWNANIE METOD OBLICZANIA WAG")
    print("="*60)
    
    methods = ['balanced', 'inverse_freq', 'sqrt_inverse', 'log_inverse']
    all_weights = {}
    
    for method in methods:
        try:
            weights = calculate_enhanced_class_weights(y, method=method)
            all_weights[method] = weights
        except Exception as e:
            print(f"Błąd dla metody {method}: {e}")
    
    # Utwórz tabelę porównawczą
    unique_classes = np.unique(y)
    print(f"\n📊 Tabela porównawcza wag:")
    print("-" * 80)
    header = f"{'Klasa':<8}"
    for method in methods:
        header += f"{method:<15}"
    print(header)
    print("-" * 80)
    
    for cls in unique_classes:
        row = f"{cls:<8}"
        for method in methods:
            if method in all_weights:
                weight = all_weights[method].get(cls, 0)
                row += f"{weight:<15.4f}"
            else:
                row += f"{'N/A':<15}"
        print(row)
    
    return all_weights

def create_class_distribution_plot(y, class_weights=None):
    """
    Tworzy wykres rozkładu klas z wagami
    """
    unique_classes, class_counts = np.unique(y, return_counts=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Wykres 1: Rozkład klas
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_classes)))
    bars1 = axes[0].bar(unique_classes, class_counts, color=colors, alpha=0.7)
    axes[0].set_xlabel('Klasa AdoptionSpeed')
    axes[0].set_ylabel('Liczba próbek')
    axes[0].set_title('Rozkład klas w zbiorze danych')
    axes[0].grid(True, alpha=0.3)
    
    # Dodaj wartości na słupkach
    for bar, count in zip(bars1, class_counts):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count}', ha='center', va='bottom')
    
    # Wykres 2: Wagi klas (jeśli podano)
    if class_weights is not None:
        weights = [class_weights.get(cls, 0) for cls in unique_classes]
        bars2 = axes[1].bar(unique_classes, weights, color=colors, alpha=0.7)
        axes[1].set_xlabel('Klasa AdoptionSpeed')
        axes[1].set_ylabel('Waga klasy')
        axes[1].set_title('Wagi klas dla zbalansowania')
        axes[1].grid(True, alpha=0.3)
        
        # Dodaj wartości na słupkach
        for bar, weight in zip(bars2, weights):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{weight:.3f}', ha='center', va='bottom')
    else:
        axes[1].text(0.5, 0.5, 'Brak wag klas', ha='center', va='center',
                    transform=axes[1].transAxes, fontsize=14)
        axes[1].set_title('Wagi klas')
    
    plt.tight_layout()
    return fig

def main():
    # Create output directories first
    create_output_directories()
    
    # Load the data
    print("Loading data...")
    data = pd.read_csv('petfinder.csv')
    print("Data shape:", data.shape)
    print(f"Columns: {list(data.columns)}")
    
    # Drop Description column if it exists
    if 'Description' in data.columns:
        data = data.drop('Description', axis=1)
        print("Dropped 'Description' column")
        print(f"New data shape: {data.shape}")
        print(f"Updated columns: {list(data.columns)}")
    else:
        print("'Description' column not found in dataset")

    # Create correlation analysis before preprocessing
    print("\n" + "="*60)
    print("CREATING CORRELATION ANALYSIS")
    print("="*60)
    
    try:
        # Create correlation plots for all network input features
        fig, correlation_matrix = create_correlation_plots(data)
        
        # Save the plot to png folder
        plt.savefig('png/correlation_analysis_simplified.png', dpi=300, bbox_inches='tight')
        print("\nCorrelation plot saved as 'png/correlation_analysis_simplified.png'")
        
        # Analyze top correlations
        top_correlations = analyze_top_correlations(correlation_matrix, threshold=0.2)
        
        # Display feature information
        print(f"\n=== NEURAL NETWORK INPUT FEATURES INFO ===")
        available_features = [col for col in correlation_matrix.columns if col != 'AdoptionSpeed']
        print(f"Basic features: {len(available_features)} - {available_features}")
        
        total_features = len(available_features)
        print(f"\nTotal input features: {total_features}")
        
        # Display correlation statistics
        all_corr_values = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                all_corr_values.append(abs(correlation_matrix.iloc[i, j]))
        
        print(f"\n=== CORRELATION STATISTICS ===")
        print(f"Mean absolute correlation: {np.mean(all_corr_values):.3f}")
        print(f"Median absolute correlation: {np.median(all_corr_values):.3f}")
        print(f"Maximum correlation: {np.max(all_corr_values):.3f}")
        print(f"Correlations > 0.5: {sum(1 for x in all_corr_values if x > 0.5)}")
        print(f"Correlations > 0.3: {sum(1 for x in all_corr_values if x > 0.3)}")
        
        plt.close()  # Close the correlation plot to free memory
        
    except Exception as e:
        print(f"Error creating correlation analysis: {e}")
        print("Continuing with model training...")
    
    print("\n" + "="*60)
    print("STARTING DATA PREPROCESSING")
    print("="*60)
    
    # Fill missing values
    print("Handling missing values...")
    numeric_cols_raw = ['Age', 'Fee', 'PhotoAmt']
    for col in numeric_cols_raw:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())
    
    categorical_cols_raw = ['Type', 'Breed1', 'Gender', 'Color1', 'Color2', 
                           'MaturitySize', 'FurLength', 'Vaccinated', 
                           'Sterilized', 'Health']
    for col in categorical_cols_raw:
        if col in data.columns:
            data[col] = data[col].fillna('Unknown')
    
    # Define categorical columns for hash encoding
    categorical_columns = ['Type', 'Breed1', 'Gender', 'Color1', 'Color2', 
                          'MaturitySize', 'FurLength', 'Vaccinated', 
                          'Sterilized', 'Health']
    
    # Hash encode categorical columns
    print("Applying hash encoding to categorical columns...")
    data_encoded, hash_mappings = hash_encode_categorical(data, categorical_columns, hash_size=1000)
    
    # Define final feature columns
    feature_columns = [col for col in data_encoded.columns if col not in ['AdoptionSpeed']]
    
    print(f"Features being used: {len(feature_columns)}")
    
    # Verify all columns are numeric
    for col in feature_columns:
        if not pd.api.types.is_numeric_dtype(data_encoded[col]):
            data_encoded[col] = pd.to_numeric(data_encoded[col], errors='coerce').fillna(0)
    
    # Split features and target
    X = data_encoded[feature_columns]
    y = data_encoded['AdoptionSpeed']
    
    print(f"Final feature matrix shape: {X.shape}")

    # ========================================
    # NOWA SEKCJA: ANALIZA I WAGI KLAS
    # ========================================
    print("\n" + "="*60)
    print("ANALIZA I WAGI KLAS")
    print("="*60)
    
    # 1. Analizuj rozkład klas
    class_names = ["Bardzo szybka (0)", "Szybka (1)", "Średnia (2)", "Wolna (3)", "Bardzo wolna (4)"]
    unique_classes, class_counts, imbalance_ratio = analyze_class_distribution(y, class_names)
    
    # 2. Porównaj różne metody wag
    all_weight_methods = compare_weight_methods(y)
    
    # 3. Wybierz najlepszą metodę na podstawie stopnia niezbalansowania
    if imbalance_ratio > 5:
        chosen_method = 'balanced'
        print(f"\n🎯 Wybrano metodę 'balanced' (silne niezbalansowanie: {imbalance_ratio:.2f})")
    elif imbalance_ratio > 3:
        chosen_method = 'sqrt_inverse'
        print(f"\n🎯 Wybrano metodę 'sqrt_inverse' (umiarkowane niezbalansowanie: {imbalance_ratio:.2f})")
    else:
        chosen_method = 'inverse_freq'
        print(f"\n🎯 Wybrano metodę 'inverse_freq' (łagodne niezbalansowanie: {imbalance_ratio:.2f})")
    
    # 4. Oblicz finalne wagi
    final_class_weights = calculate_enhanced_class_weights(y, method=chosen_method)
    
    # 5. Utwórz wykres rozkładu klas z wagami
    fig = create_class_distribution_plot(y, final_class_weights)
    plt.savefig('png/class_distribution_with_weights.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Wykres rozkładu klas zapisany jako 'png/class_distribution_with_weights.png'")
    
    # ========================================
    # KONTYNUACJA ORYGINALNEGO KODU
    # ========================================
    
    # Plot class distribution (zostaw oryginalny kod)
    plt.figure(figsize=(10, 6))
    sns.countplot(x='AdoptionSpeed', data=data_encoded)
    plt.title('Distribution of Adoption Speed')
    plt.xlabel('Adoption Speed (0: Fastest, 4: Slowest)')
    plt.ylabel('Count')
    plt.savefig('png/class_distribution_simplified.png')
    plt.close()
    
    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Przeanalizuj rozkład klas w każdym zbiorze
    print(f"\n📊 Rozkład klas w podzielonych zbiorach:")
    for name, y_subset in [("Treningowy", y_train), ("Walidacyjny", y_val), ("Testowy", y_test)]:
        subset_unique, subset_counts = np.unique(y_subset, return_counts=True)
        print(f"{name}: {dict(zip(subset_unique, subset_counts))}")
    
    # Preprocessing
    print("Applying preprocessing...")
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_train_imputed = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    
    X_val_imputed = imputer.transform(X_val)
    X_val_scaled = scaler.transform(X_val_imputed)
    
    X_test_imputed = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Convert target to one-hot encoding
    num_classes = len(np.unique(y))
    y_train_encoded = keras.utils.to_categorical(y_train, num_classes)
    y_val_encoded = keras.utils.to_categorical(y_val, num_classes)
    y_test_encoded = keras.utils.to_categorical(y_test, num_classes)
    
    # UŻYWAMY NOWYCH WAG ZAMIAST STARYCH
    print(f"\n🎯 Używam wag klas obliczonych metodą '{chosen_method}':")
    for cls, weight in final_class_weights.items():
        print(f"  Klasa {cls}: {weight:.4f}")
    
    # Create the model
    input_shape = X_train_scaled.shape[1]
    model = create_simplified_model(input_shape, num_classes)
    
    print("Model architecture:")
    model.summary()
    
    # Define callbacks with model folder path
    checkpoint = keras.callbacks.ModelCheckpoint(
        'model/best_model_simplified.h5', 
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
    
    # Train the model - UŻYWAMY NOWYCH WAG!
    print("Starting training...")
    history = model.fit(
        X_train_scaled, y_train_encoded,
        validation_data=(X_val_scaled, y_val_encoded),
        epochs=200,
        batch_size=32,
        class_weight=final_class_weights,  # <-- ZMIANA: używamy nowych wag
        callbacks=callbacks,
        verbose=1
    )
    
    # Load the best model from model folder
    try:
        model = keras.models.load_model('model/best_model_simplified.h5')
        print("Loaded best model from disk")
    except:
        print("Could not load best model, using current model")
    
    # Evaluate the model
    print("Evaluating model...")
    y_pred_proba = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_test_decoded = np.argmax(y_test_encoded, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_decoded, y_pred)
    f1 = f1_score(y_test_decoded, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test_decoded, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score (weighted): {f1:.4f}")
    
    # Dodatkowa analiza wyników z wagami klas
    print(f"\n📈 SZCZEGÓŁOWA ANALIZA WYNIKÓW Z WAGAMI KLAS")
    print("="*60)
    
    # Oblicz metryki dla każdej klasy
    precision, recall, f1_per_class, support = precision_recall_fscore_support(y_test_decoded, y_pred)
    
    print(f"Wyniki dla każdej klasy:")
    print("-" * 70)
    print(f"{'Klasa':<10} {'Precyzja':<10} {'Czułość':<10} {'F1-Score':<10} {'Próbki':<10} {'Waga':<10}")
    print("-" * 70)
    
    for i in range(len(precision)):
        weight = final_class_weights.get(i, 'N/A')
        print(f"{i:<10} {precision[i]:<10.3f} {recall[i]:<10.3f} {f1_per_class[i]:<10.3f} {support[i]:<10} {weight:<10.3f}")
    
    # Plot confusion matrix and save to png folder
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(num_classes),
                yticklabels=range(num_classes))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('png/confusion_matrix_simplified.png')
    plt.close()
    
    # Display classification report
    print("Classification Report:")
    print(classification_report(y_test_decoded, y_pred))
    
    # Plot training history and save to png folder
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
    plt.savefig('png/training_history_simplified.png')
    plt.close()
    
    # Save all components to model folder (dodajemy informację o wagach)
    print("Saving model components...")
    
    preprocessing_components = {
        'imputer': imputer,
        'scaler': scaler,
        'hash_mappings': hash_mappings,
        'feature_columns': feature_columns,
        'categorical_columns': categorical_columns
    }
    
    joblib.dump(preprocessing_components, 'model/preprocessing_components_simplified.joblib')
    
    model_metadata = {
        'num_classes': num_classes,
        'input_shape': input_shape,
        'class_weights': final_class_weights,  # <-- DODANE
        'class_weight_method': chosen_method,  # <-- DODANE
        'class_imbalance_ratio': imbalance_ratio,  # <-- DODANE
        'feature_count': len(feature_columns),
        'simplified': True
    }
    joblib.dump(model_metadata, 'model/model_metadata_simplified.joblib')
    
    print("Model training complete!")
    print(f"Total features used: {len(feature_columns)}")
    print(f"Class weights method: {chosen_method}")
    print(f"Class imbalance ratio: {imbalance_ratio:.2f}")
    print("All components saved successfully.")
    print("\nFile organization:")
    print("- PNG files saved to: png/ folder")
    print("- Model files saved to: model/ folder")
    
    # Create final correlation analysis with encoded features
    print("\n" + "="*60)
    print("FINAL CORRELATION ANALYSIS (ENCODED FEATURES)")
    print("="*60)

    
    return final_class_weights, chosen_method, imbalance_ratio

if __name__ == "__main__":
    main()
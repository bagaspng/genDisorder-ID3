# Specific Data Processor for Genetic Disorders Dataset
# This script handles the specific structure of your CSV file

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from id3_algorithm import id3, print_tree

def load_genetic_disorders_data(file_path):
    """Load the specific genetic disorders CSV file"""
    print("Loading genetic disorders dataset...")
    
    try:
        # Load the data
        df = pd.read_csv("../dataset/train_genetic_disorders.csv")
        print(f"Dataset loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {len(df.columns)}")
        
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def clean_and_preprocess_genetic_data(df):
    """Clean and preprocess the genetic disorders dataset"""
    print("\n=== CLEANING AND PREPROCESSING DATA ===")
    
    # Create a copy
    df_clean = df.copy()
    
    # Print initial info
    print(f"Initial shape: {df_clean.shape}")
    print(f"Missing values per column:")
    print(df_clean.isnull().sum().sort_values(ascending=False))
    
    # Remove unnecessary columns for prediction
    columns_to_drop = [
        'Patient Id', 'Patient First Name', 'Family Name', 
        'Father\'s name', 'Institute Name', 'Location of Institute',
        'Place of birth'  # These are identifiers, not predictive features
    ]
    
    # Only drop columns that exist
    existing_columns_to_drop = [col for col in columns_to_drop if col in df_clean.columns]
    df_clean = df_clean.drop(columns=existing_columns_to_drop)
    print(f"Dropped {len(existing_columns_to_drop)} identifier columns")
    
    # Handle specific data cleaning based on your dataset
    
    # 1. Clean Yes/No columns
    yes_no_columns = [
        'Genes in mother\'s side', 'Inherited from father', 
        'Maternal gene', 'Paternal gene', 'Parental consent',
        'Birth asphyxia', 'Folic acid details (peri-conceptional)',
        'H/O serious maternal illness', 'H/O radiation exposure (x-ray)',
        'H/O substance abuse', 'Assisted conception IVF/ART',
        'History of anomalies in previous pregnancies'
    ]
    
    for col in yes_no_columns:
        if col in df_clean.columns:
            # Standardize Yes/No values
            df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
            df_clean[col] = df_clean[col].map({
                'yes': 1, 'no': 0, 'true': 1, 'false': 0,
                '1': 1, '0': 0, 'nan': 0, 'none': 0
            })
            # Fill remaining NaN with 0
            df_clean[col] = df_clean[col].fillna(0).astype(int)
    
    # 2. Clean numerical columns
    numerical_cols = [
        'Patient Age', 'Mother\'s age', 'Father\'s age',
        'Blood cell count (mcL)', 'White Blood cell count (thousand per microliter)',
        'No. of previous abortion', 'Respiratory Rate (breaths/min)'
    ]
    
    for col in numerical_cols:
        if col in df_clean.columns:
            # Convert to numeric, errors='coerce' will turn invalid values to NaN
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            # Fill NaN with median
            if df_clean[col].notna().sum() > 0:
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
    
    # 3. Clean categorical columns
    categorical_cols = ['Status', 'Gender', 'Follow-up', 'Birth defects', 
                       'Blood test result', 'Autopsy shows birth defect (if applicable)']
    
    for col in categorical_cols:
        if col in df_clean.columns:
            # Fill NaN with 'Unknown'
            df_clean[col] = df_clean[col].fillna('Unknown')
            # Clean string values
            df_clean[col] = df_clean[col].astype(str).str.strip()
    
    # 4. Handle test results columns (Test 1, Test 2, etc.)
    test_cols = ['Test 1', 'Test 2', 'Test 3', 'Test 4', 'Test 5']
    for col in test_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
    
    # 5. Handle symptom columns
    symptom_cols = ['Symptom 1', 'Symptom 2', 'Symptom 3', 'Symptom 4', 'Symptom 5']
    for col in symptom_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
    
    # 6. Clean respiratory and heart rate columns
    if 'Respiratory Rate (breaths/min)' in df_clean.columns:
        # Extract numeric values from text like "Normal (30-60)"
        df_clean['Respiratory_Rate_Category'] = df_clean['Respiratory Rate (breaths/min)'].astype(str)
        df_clean['Respiratory_Rate_Numeric'] = 0
        df_clean.loc[df_clean['Respiratory_Rate_Category'].str.contains('Normal', na=False), 'Respiratory_Rate_Numeric'] = 1
        df_clean.loc[df_clean['Respiratory_Rate_Category'].str.contains('Tachypnea', na=False), 'Respiratory_Rate_Numeric'] = 2
    
    if 'Heart Rate (rates/min' in df_clean.columns:
        df_clean['Heart_Rate_Category'] = df_clean['Heart Rate (rates/min'].astype(str)
        df_clean['Heart_Rate_Numeric'] = 0
        df_clean.loc[df_clean['Heart_Rate_Category'].str.contains('Normal', na=False), 'Heart_Rate_Numeric'] = 1
        df_clean.loc[df_clean['Heart_Rate_Category'].str.contains('Tachycardia', na=False), 'Heart_Rate_Numeric'] = 2
    
    print(f"Final shape after cleaning: {df_clean.shape}")
    print(f"Missing values after cleaning:")
    print(df_clean.isnull().sum().sum())
    
    return df_clean

def encode_features(df):
    """Encode categorical features for machine learning"""
    print("\n=== ENCODING FEATURES ===")
    
    df_encoded = df.copy()
    label_encoders = {}
    
    # Get categorical columns (excluding target variables)
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
    
    # Don't encode target variables
    target_cols = ['Genetic Disorder', 'Disorder Subclass']
    categorical_cols = [col for col in categorical_cols if col not in target_cols]
    
    print(f"Encoding {len(categorical_cols)} categorical columns...")
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le
            print(f"  - {col}: {len(le.classes_)} unique values")
    
    return df_encoded, label_encoders

def prepare_data_for_modeling(df):
    """Prepare data for Decision Tree modeling"""
    print("\n=== PREPARING DATA FOR MODELING ===")
    
    # Check if target column exists
    target_col = 'Genetic Disorder'
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found!")
        print(f"Available columns: {list(df.columns)}")
        return None, None, None
    
    # Separate features and target
    X = df.drop(columns=[target_col, 'Disorder Subclass'], errors='ignore')
    y = df[target_col]
    
    # Encode target variable
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y.astype(str))
    
    print(f"Features shape: {X.shape}")
    print(f"Target classes: {len(le_target.classes_)}")
    print(f"Target distribution:")
    unique, counts = np.unique(y_encoded, return_counts=True)
    for i, (cls, count) in enumerate(zip(le_target.classes_, counts)):
        print(f"  {cls}: {count} samples")
    
    return X, y_encoded, le_target

def train_decision_tree_model(X, y, test_size=0.2, random_state=42):
    """Train a Decision Tree model"""
    print("\n=== TRAINING DECISION TREE MODEL ===")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=y, shuffle=True
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create Decision Tree with optimal parameters
    dt_model = DecisionTreeClassifier(
        criterion='gini',           # Split criterion
        max_depth=15,              # Maximum depth to prevent overfitting
        min_samples_split=5,       # Minimum samples required to split
        min_samples_leaf=3,        # Minimum samples in leaf node
        max_features='sqrt',       # Number of features for best split
        random_state=random_state,
        class_weight='balanced'    # Handle class imbalance
    )
    
    # Train the model
    print("Training model...")
    dt_model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = dt_model.predict(X_train)
    y_test_pred = dt_model.predict(X_test)
    
    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    return dt_model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred

def evaluate_and_visualize(model, X_train, X_test, y_train, y_test, 
                          y_train_pred, y_test_pred, le_target, feature_names):
    """Evaluate model and create visualizations"""
    print("\n=== MODEL EVALUATION ===")
    
    # Classification Report
    print("Classification Report (Test Set):")
    report = classification_report(y_test, y_test_pred, 
                                 target_names=le_target.classes_,
                                 zero_division=0)
    print(report)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\nTop 15 Most Important Features:")
    print(feature_importance.head(15))
    
    # Visualizations
    plt.figure(figsize=(15, 12))
    
    # 1. Confusion Matrix Heatmap
    plt.subplot(2, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le_target.classes_,
                yticklabels=le_target.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 2. Feature Importance
    plt.subplot(2, 3, 2)
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['Importance'])
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importance')
    plt.gca().invert_yaxis()
    
    # 3. Target Distribution
    plt.subplot(2, 3, 3)
    unique, counts = np.unique(y_test, return_counts=True)
    class_names = [le_target.classes_[i] for i in unique]
    plt.pie(counts, labels=class_names, autopct='%1.1f%%')
    plt.title('Target Distribution (Test Set)')
    
    # 4. Model Performance Comparison
    plt.subplot(2, 3, 4)
    accuracies = [accuracy_score(y_train, y_train_pred),
                 accuracy_score(y_test, y_test_pred)]
    plt.bar(['Training', 'Test'], accuracies, color=['blue', 'orange'])
    plt.ylabel('Accuracy')
    plt.title('Model Performance')
    plt.ylim(0, 1)
    
    # Add accuracy values on bars
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center')
    
    plt.tight_layout()
    plt.show()
    
    return feature_importance

def predict_new_samples(model, le_target, X_sample):
    """Make predictions on new samples"""
    print("\n=== MAKING PREDICTIONS ===")
    
    predictions = model.predict(X_sample)
    probabilities = model.predict_proba(X_sample)
    
    # Decode predictions
    decoded_predictions = le_target.inverse_transform(predictions)
    
    print("Predictions for sample data:")
    for i, (pred, prob) in enumerate(zip(decoded_predictions, probabilities)):
        max_prob = np.max(prob)
        print(f"Sample {i+1}: {pred} (Confidence: {max_prob:.4f})")
    
    return decoded_predictions, probabilities

def load_data(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_test

def predict_all(tree, X):
    # Prediksi semua baris X menggunakan pohon ID3
    preds = []
    for _, row in X.iterrows():
        node = tree
        while 'label' not in node:
            feat = node['feature']
            val = row[feat]
            if val in node['nodes']:
                node = node['nodes'][val]
            else:
                # Jika nilai tidak ada di node, ambil label mayoritas
                node = list(node['nodes'].values())[0]
        preds.append(node['label'])
    return preds

def main_pipeline(file_path):
    """Complete pipeline for genetic disorders prediction"""
    print("🧬 GENETIC DISORDERS PREDICTION PIPELINE 🧬")
    print("=" * 60)
    
    try:
        # 1. Load data
        df = load_genetic_disorders_data(file_path)
        if df is None:
            return None
        
        # 2. Clean and preprocess
        df_clean = clean_and_preprocess_genetic_data(df)
        
        # 3. Encode features
        df_encoded, encoders = encode_features(df_clean)
        
        # 4. Prepare for modeling
        X, y, le_target = prepare_data_for_modeling(df_encoded)
        if X is None:
            return None
        
        # 5. Train model
        model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred = train_decision_tree_model(X, y)
        
        # 6. Evaluate and visualize
        feature_importance = evaluate_and_visualize(
            model, X_train, X_test, y_train, y_test, 
            y_train_pred, y_test_pred, le_target, X_train.columns.tolist()
        )
        
        # 7. Predict new samples (if any)
        # Sample new data for prediction (you can replace this with real new data)
        X_sample = X_test.sample(n=5, random_state=42)
        print("\nSample data for prediction:")
        print(X_sample)
        
        predictions, probabilities = predict_new_samples(model, le_target, X_sample)
        
    except Exception as e:
        print(f"Error in pipeline: {e}")
        return None

def main():
    train_path = "../dataset/train_genetic_disorders.csv"
    test_path = "../dataset/test_genetic_disorders.csv"

    df_train, df_test = load_data(train_path, test_path)

    # Pastikan kolom target ada di data latih
    if "Genetic Disorder" not in df_train.columns:
        print("Kolom 'Genetic Disorder' tidak ditemukan di data latih.")
        return

    # Siapkan fitur dan target
    X_train = df_train.drop(columns=["Genetic Disorder"])
    y_train = df_train["Genetic Disorder"]

    # Pastikan tidak ada NaN di target
    print(y_train.isnull().sum())
    print(y_train.unique())
    # Drop baris yang targetnya NaN
    mask = y_train.notnull()
    X_train = X_train[mask]
    y_train = y_train[mask]

    # Gabungkan X_train dan y_train untuk drop baris yang targetnya NaN
    df_train_clean = pd.concat([X_train, y_train], axis=1)
    df_train_clean = df_train_clean.dropna(subset=["Genetic Disorder"])

    # Pisahkan kembali fitur dan target
    X_train = df_train_clean.drop(columns=["Genetic Disorder"])
    y_train = df_train_clean["Genetic Disorder"]

    # Isi NaN pada fitur dengan string 'missing', lalu ubah semua ke string
    X_train = X_train.fillna('missing').astype(str)

    # Latih pohon ID3
    features = list(X_train.columns)
    tree = id3(X_train, y_train, features)

    print("Pohon keputusan:")
    print_tree(tree)

    # Prediksi data uji
    predictions = predict_all(tree, X_test)
    print("\nPrediksi untuk data uji:")
    for i, pred in enumerate(predictions):
        print(f"Sample {i+1}: {pred}")

if __name__ == "__main__":
    main()
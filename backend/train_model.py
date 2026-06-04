# backend/train_model.py
import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Ensure backend directory is in path for imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from feature_extractor import compute_signal_features

def main():
    print("==================================================")
    print("  EDGE AI ECG MODEL TRAINING PIPELINE")
    print("==================================================")
    
    dataset_path = "data/mitbih/mitbih.csv"
    model_dir = "models"
    model_path = os.path.join(model_dir, "ecg_model.pkl")
    
    if not os.path.exists(dataset_path):
        print(f"Error: MIT-BIH dataset not found at '{dataset_path}'.")
        print("\nPlease download the MIT-BIH Arrhythmia Dataset CSV format and place it there.")
        print("Note: You can run 'backend/generate_mitbih_data.py' to generate a mock dataset for testing.")
        return
        
    print(f"Loading dataset from {dataset_path}...")
    try:
        # Load the CSV
        df = pd.read_csv(dataset_path)
        
        # Robustly handle headerless Kaggle files
        if "label" not in df.columns:
            print("No 'label' column found. Re-loading as headerless Kaggle format...")
            df = pd.read_csv(dataset_path, header=None)
            df.columns = [f"f_{i}" for i in range(df.shape[1] - 1)] + ["label"]
            
        print(f"Dataset successfully loaded. Shape: {df.shape}")
        print("Class counts in dataset:")
        print(df["label"].value_counts().sort_index())
        
        # Separate signal and label
        signals = df.drop("label", axis=1)
        y = df["label"].astype(int)
        
        # Extract features row-by-row
        print("\nExtracting 6 statistical features from each heartbeat...")
        print("Features: Mean, Std Dev, Max, Min, Skewness, Kurtosis...")
        
        feature_list = []
        total_rows = len(signals)
        
        for idx, (_, row) in enumerate(signals.iterrows()):
            feats = compute_signal_features(row)
            feature_list.append(feats)
            
            # Print progress every 10%
            if (idx + 1) % (total_rows // 10 if total_rows >= 10 else 1) == 0:
                print(f"Processed {idx + 1}/{total_rows} heartbeats...")
                
        X = pd.DataFrame(feature_list)
        print("Feature extraction complete.")
        print(X.head())
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining Random Forest Classifier on {len(X_train)} samples...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight="balanced"
        )
        model.fit(X_train, y_train)
        
        # Evaluate model
        print("\nEvaluating model performance...")
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Validation Accuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        # Label map: 0 = Normal, 1 = PVC, 2 = Supraventricular, 3 = Fusion, 4 = Unknown
        # We handle whatever labels are present in the dataset dynamically
        present_labels = sorted(list(y_test.unique()))
        label_names = []
        label_map = {
            0: "Normal (N)", 
            1: "PVC (V)", 
            2: "Supraventricular (S)", 
            3: "Fusion (F)", 
            4: "Unclassifiable (Q)"
        }
        for lbl in present_labels:
            label_names.append(label_map.get(lbl, f"Class {lbl}"))
            
        print(classification_report(y_test, y_pred, target_names=label_names))
        
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Save model
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, model_path)
        print(f"\nTrained model successfully saved to: {model_path}")
        
        # Make predictions on the entire dataset for clinical display
        print("\nGenerating predictions for the entire clinical dataset...")
        import time
        all_preds = model.predict(X)
        
        # Label descriptions
        label_map_desc = {
            0: "Normal (N)", 
            1: "PVC (V)", 
            2: "Supraventricular (S)", 
            3: "Fusion (F)", 
            4: "Unclassifiable (Q)"
        }
        
        predictions_df = X.copy()
        predictions_df["beat_index"] = range(len(X))
        predictions_df["true_label"] = y
        predictions_df["prediction"] = all_preds
        predictions_df["prediction_label"] = [label_map_desc.get(p, f"Class {p}") for p in all_preds]
        predictions_df["status"] = ["NORMAL" if p == 0 else "ABNORMAL" for p in all_preds]
        # Distribute timestamps sequentially in the past
        start_ts = time.time() - len(X)
        predictions_df["timestamp"] = [start_ts + i for i in range(len(X))]
        
        mitbih_predictions_path = "data/processed/mitbih_predictions.csv"
        predictions_df.to_csv(mitbih_predictions_path, index=False)
        print(f"Clinical dataset predictions saved to: {mitbih_predictions_path}")
        print("==================================================")
        
    except Exception as e:
        print(f"Error during model training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
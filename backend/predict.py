# backend/predict.py
import os
import sys
import time
import pandas as pd
import joblib

# Ensure backend directory is in path for imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from session_manager import get_active_session, get_session_paths

# Label mapping for clinical descriptions
LABEL_MAP = {
    0: "Normal (N)",
    1: "PVC (V)",
    2: "Supraventricular premature (S)",
    3: "Fusion (F)",
    4: "Unclassifiable (Q)"
}

def make_prediction(model_path="models/ecg_model.pkl", 
                    features_path=None, 
                    predictions_log_path=None):
    """
    Loads the trained model and the latest feature vector, performs prediction,
    logs the result to a history file, and returns the prediction and diagnostic label.
    
    If features_path or predictions_log_path are not provided, they default to the
    active session's paths (data/sessions/default/).
    """
    # Resolve session-based default paths
    if features_path is None or predictions_log_path is None:
        session_name = get_active_session()
        paths = get_session_paths(session_name)
        if features_path is None:
            features_path = os.path.join(paths["dir"], "features.csv")
        if predictions_log_path is None:
            predictions_log_path = paths["predictions"]
    
    if not os.path.exists(model_path):
        return None, "Model file not found. Please train the model first."
    if not os.path.exists(features_path):
        return None, "Features file not found. Please run feature extraction."
        
    try:
        # Load model and features
        model = joblib.load(model_path)
        features_df = pd.read_csv(features_path)
        
        if len(features_df) == 0:
            return None, "Feature vector is empty."
            
        # Select the latest features row (should be only one row)
        latest_features = features_df.iloc[[-1]]
        
        # Ensure correct column ordering
        expected_cols = ["mean", "std", "max", "min", "skew", "kurt"]
        latest_features = latest_features[expected_cols]
        
        # Predict
        pred_array = model.predict(latest_features)
        prediction = int(pred_array[0])
        
        # Map label
        pred_label = LABEL_MAP.get(prediction, f"Unknown Class {prediction}")
        status = "NORMAL" if prediction == 0 else "ABNORMAL"
        
        # Log prediction history
        new_log = pd.DataFrame([{
            "timestamp": time.time(),
            "prediction": prediction,
            "prediction_label": pred_label,
            "status": status
        }])
        
        if os.path.exists(predictions_log_path):
            try:
                history_df = pd.read_csv(predictions_log_path)
                history_df = pd.concat([history_df, new_log], ignore_index=True)
                # Keep history size bounded to last 1000 entries
                if len(history_df) > 1000:
                    history_df = history_df.tail(1000)
            except Exception:
                history_df = new_log
        else:
            history_df = new_log
            
        os.makedirs(os.path.dirname(predictions_log_path), exist_ok=True)
        history_df.to_csv(predictions_log_path, index=False)
        
        return prediction, pred_label
        
    except Exception as e:
        return None, f"Error in prediction: {e}"

if __name__ == "__main__":
    pred, label = make_prediction()
    if pred is not None:
        print(f"Prediction: {pred} ({label})")
    else:
        print(f"Error: {label}")
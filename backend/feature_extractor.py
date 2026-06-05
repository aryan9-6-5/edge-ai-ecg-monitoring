# backend/feature_extractor.py
import os
import pandas as pd
import numpy as np

def compute_signal_features(signal, normalize=True):
    """
    Computes 6 statistical features of an ECG signal array or Series:
    mean, std, max, min, skewness, and kurtosis.
    Handles NaN values and empty signals gracefully.
    If normalize=True, Min-Max normalizes the signal to [0, 1] range first
    to ensure scale consistency with the model training dataset.
    """
    # Convert to pandas Series for clean built-in skew/kurtosis calculation
    if not isinstance(signal, pd.Series):
        s = pd.Series(signal)
    else:
        s = signal
        
    # Drop any NaNs
    s = s.dropna()
    
    if len(s) == 0:
        return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0, "skew": 0.0, "kurt": 0.0}
        
    if normalize:
        s_min = s.min()
        s_max = s.max()
        if s_max - s_min > 1e-8:
            s = (s - s_min) / (s_max - s_min)
        else:
            s = s - s_min
            
    mean_val = float(s.mean())
    std_val = float(s.std() if len(s) > 1 else 0.0)
    max_val = float(s.max())
    min_val = float(s.min())
    skew_val = float(s.skew() if len(s) > 2 else 0.0)
    kurt_val = float(s.kurt() if len(s) > 3 else 0.0)
    
    # Handle NaN results from skewness/kurtosis on uniform signals
    if np.isnan(skew_val):
        skew_val = 0.0
    if np.isnan(kurt_val):
        kurt_val = 0.0
        
    return {
        "mean": mean_val,
        "std": std_val,
        "max": max_val,
        "min": min_val,
        "skew": skew_val,
        "kurt": kurt_val
    }

def main():
    import sys
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    from session_manager import get_active_session, get_session_paths
    
    print("Running feature extraction on live processed data...")
    session_name = get_active_session()
    paths = get_session_paths(session_name)
    input_file = paths["filtered"]
    output_file = os.path.join(paths["dir"], "features.csv")
    
    if not os.path.exists(input_file):
        print(f"Error: Preprocessed signal file '{input_file}' not found.")
        return
        
    try:
        df = pd.read_csv(input_file)
        
        # Verify columns
        if "filtered_ecg" not in df.columns:
            print(f"Error: 'filtered_ecg' column not found in {input_file}.")
            return
            
        # Compute features
        features = compute_signal_features(df["filtered_ecg"])
        
        # Write to DataFrame and save
        feature_df = pd.DataFrame([features])
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        feature_df.to_csv(output_file, index=False)
        
        print("Feature extraction complete. Features:")
        print(feature_df)
        
    except Exception as e:
        print(f"Error extracting features: {e}")

if __name__ == "__main__":
    main()

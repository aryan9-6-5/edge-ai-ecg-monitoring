# backend/feature_extractor.py
import os
import pandas as pd
import numpy as np

def compute_signal_features(signal):
    """
    Computes 6 statistical features of an ECG signal array or Series:
    mean, std, max, min, skewness, and kurtosis.
    Handles NaN values and empty signals gracefully.
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
    print("Running feature extraction on live processed data...")
    input_file = "data/processed/ecg_filtered.csv"
    output_file = "data/processed/features.csv"
    
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

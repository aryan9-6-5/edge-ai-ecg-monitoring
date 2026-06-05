# backend/preprocess.py
import os
import json
import pandas as pd
import numpy as np

def preprocess_data(raw_path="data/raw/ecglive.csv", processed_path="data/processed/ecg_filtered.csv"):
    """
    Cleans raw ECG data, detects and interpolates sensor lead-off/noise spikes,
    filters noise using a rolling moving average, and saves the preprocessed signal.
    """
    if not os.path.exists(raw_path):
        return False, f"Raw data file '{raw_path}' not found."
        
    try:
        df = pd.read_csv(raw_path)
        if len(df) == 0:
            return False, "Raw data file is empty."
            
        if "ecg" not in df.columns:
            return False, "Required column 'ecg' not found in raw data."
            
        # Ensure timestamp column exists
        if "timestamp" not in df.columns:
            df["timestamp"] = range(len(df))
            
        df_cleaned = df.copy()
        
        # Load configuration to check if sensor adaptation is enabled
        adaptation_enabled = True
        config_path = "data/processed/session_config.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    cfg = json.load(f)
                    adaptation_enabled = cfg.get("sensor_adaptation_enabled", True)
            except Exception:
                pass
                
        if adaptation_enabled:
            # 1. Detect extreme outliers (rails >= 4080 or <= 15 representing lead-off, or huge jumps)
            rolling_median = df_cleaned["ecg"].rolling(window=15, center=True, min_periods=1).median()
            # Use dynamic spike threshold: 4x the rolling IQR, with a minimum floor of 500
            rolling_q1 = df_cleaned["ecg"].rolling(window=31, center=True, min_periods=1).quantile(0.25)
            rolling_q3 = df_cleaned["ecg"].rolling(window=31, center=True, min_periods=1).quantile(0.75)
            rolling_iqr = (rolling_q3 - rolling_q1).clip(lower=125)
            spike_threshold = (rolling_iqr * 4).clip(lower=500)
            is_outlier = (df_cleaned["ecg"] >= 4080) | (df_cleaned["ecg"] <= 15) | ((df_cleaned["ecg"] - rolling_median).abs() > spike_threshold)
            
            # 2. Mask outliers with NaN so they are interpolated linearly
            df_cleaned["ecg_adapted"] = df_cleaned["ecg"].mask(is_outlier)
            
            # 3. Linearly interpolate the masked values (preserves monotonic trends)
            df_cleaned["ecg_adapted"] = df_cleaned["ecg_adapted"].interpolate(method="linear", limit_direction="both")
            
            # 4. Fill remaining NaNs if entire series was outlier (e.g. startup)
            df_cleaned["ecg_adapted"] = df_cleaned["ecg_adapted"].fillna(2048.0)
            
            # Apply 5-Point SMA smoothing filter on the adapted ECG signal
            df_cleaned["filtered_ecg"] = (
                df_cleaned["ecg_adapted"]
                .rolling(window=5, min_periods=1)
                .mean()
            )
        else:
            # Simple SMA filter without adaptive interpolation
            df_cleaned["filtered_ecg"] = (
                df_cleaned["ecg"]
                .rolling(window=5, min_periods=1)
                .mean()
            )
            
        # Save processed data
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df_cleaned.to_csv(processed_path, index=False)
        return True, "Preprocessing completed successfully."
        
    except Exception as e:
        return False, f"Error in preprocessing: {e}"

if __name__ == "__main__":
    success, msg = preprocess_data()
    print(msg)


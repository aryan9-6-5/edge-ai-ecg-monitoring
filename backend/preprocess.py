# backend/preprocess.py
import os
import pandas as pd

def preprocess_data(raw_path="data/raw/ecglive.csv", processed_path="data/processed/ecg_filtered.csv"):
    """
    Cleans raw ECG data, filters noise using a rolling moving average,
    and saves the preprocessed signal to a CSV file.
    """
    if not os.path.exists(raw_path):
        return False, f"Raw data file '{raw_path}' not found."
        
    try:
        df = pd.read_csv(raw_path)
        if len(df) == 0:
            return False, "Raw data file is empty."
            
        if "ecg" not in df.columns:
            return False, "Required column 'ecg' not found in raw data."
            
        # Keep all raw samples to prevent graph freeze on lead-off (4095)
        df_cleaned = df.copy()
            
        # Apply simple moving average filter to smooth high-frequency serial noise
        df_cleaned["filtered_ecg"] = (
            df_cleaned["ecg"]
            .rolling(window=5, min_periods=1)
            .mean()
        )
        
        # Ensure timestamp column exists
        if "timestamp" not in df_cleaned.columns:
            df_cleaned["timestamp"] = range(len(df_cleaned))
            
        # Save processed data
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df_cleaned.to_csv(processed_path, index=False)
        return True, "Preprocessing completed successfully."
        
    except Exception as e:
        return False, f"Error in preprocessing: {e}"

if __name__ == "__main__":
    success, msg = preprocess_data()
    print(msg)

# backend/alert_engine.py
import os
import pandas as pd

def check_latest_alert(predictions_log_path="data/processed/predictions.csv"):
    """
    Checks the latest status in the prediction log.
    Returns "NORMAL" or "ABNORMAL".
    """
    if not os.path.exists(predictions_log_path):
        return "NORMAL"
        
    try:
        df = pd.read_csv(predictions_log_path)
        if len(df) == 0:
            return "NORMAL"
            
        latest = df.iloc[-1]
        return latest["status"]
    except Exception:
        return "NORMAL"

if __name__ == "__main__":
    status = check_latest_alert()
    print(status)
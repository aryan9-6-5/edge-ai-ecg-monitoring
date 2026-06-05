# backend/alert_engine.py
import os
import sys
import pandas as pd

# Ensure backend directory is in path for imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from session_manager import get_active_session, get_session_paths

def check_latest_alert(predictions_log_path=None):
    """
    Checks the latest status in the prediction log.
    Returns "NORMAL" or "ABNORMAL".
    
    If predictions_log_path is not provided, defaults to the active session's
    predictions file (data/sessions/default/predictions.csv).
    """
    if predictions_log_path is None:
        session_name = get_active_session()
        paths = get_session_paths(session_name)
        predictions_log_path = paths["predictions"]
    
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
# backend/session_manager.py
import os
import json

SESSIONS_DIR = "data/sessions"
ACTIVE_SESSION_FILE = "data/active_session.json"
STATUS_FILE_PATH = "data/processed/acquisition_status.json"

def init_sessions_dir():
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    os.makedirs(os.path.join(SESSIONS_DIR, "default"), exist_ok=True)
    os.makedirs(os.path.join(SESSIONS_DIR, "live"), exist_ok=True)

def get_all_sessions():
    """Returns only the two allowed sessions: 'default' and 'live'."""
    init_sessions_dir()
    return ["default", "live"]

def get_latest_session():
    """Determines the default session to load based on sensor presence."""
    if os.path.exists(STATUS_FILE_PATH):
        try:
            with open(STATUS_FILE_PATH, "r") as f:
                data = json.load(f)
            # If serial port is connected, load live, otherwise default
            if data.get("serial_connected", False):
                return "live"
        except Exception:
            pass
    return "default"

def get_active_session():
    """Reads the active session from active_session.json, falling back to latest."""
    init_sessions_dir()
    if os.path.exists(ACTIVE_SESSION_FILE):
        try:
            with open(ACTIVE_SESSION_FILE, "r") as f:
                data = json.load(f)
                session = data.get("active_session", "default")
                if session in ["default", "live"]:
                    return session
        except Exception:
            pass
    
    # Fallback to sensor detection
    fallback = get_latest_session()
    set_active_session(fallback)
    return fallback

def set_active_session(session_name):
    """Saves the active session name (must be 'default' or 'live') to active_session.json."""
    if session_name not in ["default", "live"]:
        session_name = "default"
        
    os.makedirs("data", exist_ok=True)
    try:
        with open(ACTIVE_SESSION_FILE, "w") as f:
            json.dump({"active_session": session_name}, f, indent=4)
        return True
    except Exception:
        return False

def get_session_paths(session_name):
    """Returns file paths for raw, filtered, predictions, and report files in a session."""
    if session_name not in ["default", "live"]:
        session_name = "default"
        
    session_dir = os.path.join(SESSIONS_DIR, session_name)
    os.makedirs(session_dir, exist_ok=True)
    
    return {
        "dir": session_dir,
        "raw": os.path.join(session_dir, "ecglive.csv"),
        "filtered": os.path.join(session_dir, "ecg_filtered.csv"),
        "predictions": os.path.join(session_dir, "predictions.csv"),
        "report": os.path.join(session_dir, "report.json")
    }

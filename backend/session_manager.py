# backend/session_manager.py
import os
import glob
import json

SESSIONS_DIR = "data/sessions"
ACTIVE_SESSION_FILE = "data/active_session.json"

def init_sessions_dir():
    os.makedirs(SESSIONS_DIR, exist_ok=True)

def get_all_sessions():
    """Returns a sorted list of all session names found in the sessions directory."""
    init_sessions_dir()
    session_paths = glob.glob(os.path.join(SESSIONS_DIR, "session_*"))
    sessions = []
    for path in session_paths:
        if os.path.isdir(path):
            name = os.path.basename(path)
            sessions.append(name)
    return sorted(sessions)

def get_latest_session():
    """Returns the name of the most recent session, or None if none exist."""
    sessions = get_all_sessions()
    if sessions:
        return sessions[-1]
    return None

def create_new_session():
    """Increments session count, creates a new session directory, and returns (path, name)."""
    init_sessions_dir()
    sessions = get_all_sessions()
    
    if not sessions:
        next_num = 1
    else:
        # Extract the highest number from existing session names (e.g. session_003 -> 3)
        try:
            numbers = [int(s.split("_")[1]) for s in sessions if "_" in s]
            next_num = max(numbers) + 1
        except Exception:
            next_num = len(sessions) + 1
            
    session_name = f"session_{next_num:03d}"
    session_path = os.path.join(SESSIONS_DIR, session_name)
    os.makedirs(session_path, exist_ok=True)
    
    # Set this as the active session
    set_active_session(session_name)
    
    return session_path, session_name

def get_active_session():
    """Returns the name of the currently active/latest recording session."""
    if os.path.exists(ACTIVE_SESSION_FILE):
        try:
            with open(ACTIVE_SESSION_FILE, "r") as f:
                data = json.load(f)
            name = data.get("active_session")
            # Verify the session directory exists
            if name and os.path.exists(os.path.join(SESSIONS_DIR, name)):
                return name
        except Exception:
            pass
            
    # Fallback to latest session, or create session_001 if none exist
    latest = get_latest_session()
    if latest:
        set_active_session(latest)
        return latest
    
    # Create the first session if nothing exists
    _, name = create_new_session()
    return name

def set_active_session(session_name):
    """Saves the active session name to a JSON configuration file."""
    os.makedirs("data", exist_ok=True)
    try:
        with open(ACTIVE_SESSION_FILE, "w") as f:
            json.dump({"active_session": session_name}, f)
        return True
    except Exception:
        return False

def get_session_paths(session_name):
    """Returns absolute/relative file paths for raw, filtered, predictions, and report files in a session."""
    session_dir = os.path.join(SESSIONS_DIR, session_name)
    return {
        "dir": session_dir,
        "raw": os.path.join(session_dir, "ecglive.csv"),
        "filtered": os.path.join(session_dir, "ecg_filtered.csv"),
        "predictions": os.path.join(session_dir, "predictions.csv"),
        "report": os.path.join(session_dir, "report.json")
    }

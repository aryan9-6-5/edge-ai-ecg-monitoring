# backend/patient_manager.py
import os
import json

PATIENTS_FILE = "data/patients.json"

DEFAULT_PATIENTS = {
    "P-101": {
        "id": "P-101",
        "name": "John Doe",
        "age": 68,
        "gender": "Male",
        "baseline_hr": 72.0,
        "arrhythmia_history": True,
        "notes": "History of hypertension and coronary artery disease."
    },
    "P-102": {
        "id": "P-102",
        "name": "Jane Smith",
        "age": 32,
        "gender": "Female",
        "baseline_hr": 68.0,
        "arrhythmia_history": False,
        "notes": "Routine screening. Physically active. No cardiac history."
    },
    "P-103": {
        "id": "P-103",
        "name": "Robert Chen",
        "age": 45,
        "gender": "Male",
        "baseline_hr": 75.0,
        "arrhythmia_history": True,
        "notes": "History of supraventricular premature beats and anxiety."
    },
    "P-104": {
        "id": "P-104",
        "name": "Emma Watson",
        "age": 71,
        "gender": "Female",
        "baseline_hr": 65.0,
        "arrhythmia_history": True,
        "notes": "Prior myocardial infarction. Normal pacemaker baseline."
    }
}

def load_patients():
    """Loads patient profiles from JSON, creating defaults if not present."""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(PATIENTS_FILE):
        save_patients(DEFAULT_PATIENTS)
        return DEFAULT_PATIENTS
    try:
        with open(PATIENTS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return DEFAULT_PATIENTS

def save_patients(patients):
    """Saves patient profiles to JSON."""
    os.makedirs("data", exist_ok=True)
    try:
        with open(PATIENTS_FILE, "w") as f:
            json.dump(patients, f, indent=4)
        return True
    except Exception:
        return False

def get_patient(patient_id):
    """Returns details for a specific patient ID."""
    patients = load_patients()
    return patients.get(patient_id, DEFAULT_PATIENTS["P-101"])

def get_active_patient_id():
    """Gets the currently selected patient ID from config."""
    config_path = "data/processed/session_config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
                return cfg.get("active_patient_id", "P-101")
        except Exception:
            pass
    return "P-101"

def set_active_patient_id(patient_id):
    """Saves the selected patient ID to the active session config."""
    config_path = "data/processed/session_config.json"
    cfg = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
        except Exception:
            pass
    cfg["active_patient_id"] = patient_id
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=4)
        return True
    except Exception:
        return False

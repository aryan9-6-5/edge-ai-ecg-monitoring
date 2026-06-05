# run.py
import os
import sys
import time
import json
import subprocess
import threading
import webbrowser
import numpy as np
import pandas as pd
import joblib

# Default Settings
BAUD_RATE = 115200
SAMPLE_RATE = 125  # Hz
WINDOW_SIZE = 250  # Samples to evaluate (2 seconds)

# Add backend to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))
from preprocess import preprocess_data
from feature_extractor import compute_signal_features
from session_manager import get_active_session, get_session_paths, set_active_session, get_all_sessions, get_latest_session
from patient_manager import get_active_patient_id, get_patient
from analytics_engine import analyze_session_signal

# Control paths
SIM_CONTROL_PATH = "data/processed/simulation_control.json"
STATUS_FILE_PATH = "data/processed/acquisition_status.json"
SESSION_CONFIG_PATH = "data/processed/session_config.json"
TELEMETRY_COMMAND_PATH = "data/processed/telemetry_command.json"

# State variables
stop_event = threading.Event()
serial_connected = False

LABEL_MAP = {
    0: "Normal (N)",
    1: "PVC (V)",
    2: "Supraventricular premature (S)",
    3: "Fusion (F)",
    4: "Unclassifiable (Q)"
}

def read_session_config():
    """Reads session parameters from JSON config or returns default settings."""
    default_config = {
        "limit_type": "samples",
        "max_duration": 60,
        "max_samples": 5000,
        "port": "Auto-Detect",
        "sensor_adaptation_enabled": True,
        "active_patient_id": "P-101"
    }
    if os.path.exists(SESSION_CONFIG_PATH):
        try:
            with open(SESSION_CONFIG_PATH, "r") as f:
                config = json.load(f)
                for k, v in default_config.items():
                    if k not in config:
                        config[k] = v
                return config
        except Exception:
            pass
    return default_config

def init_files(session_name):
    """Initializes directories and empty files for a given session."""
    paths = get_session_paths(session_name)
    os.makedirs(paths["dir"], exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    with open(paths["raw"], "w", newline="") as f:
        f.write("timestamp,ecg\n")
        
    with open(paths["filtered"], "w", newline="") as f:
        f.write("timestamp,ecg,filtered_ecg\n")
        
    with open(paths["predictions"], "w", newline="") as f:
        f.write("timestamp,prediction,prediction_label,status\n")
        
    if os.path.exists(paths["report"]):
        try:
            os.remove(paths["report"])
        except Exception:
            pass
            
    if not os.path.exists(SIM_CONTROL_PATH):
        with open(SIM_CONTROL_PATH, "w") as f:
            json.dump({"simulate_pvc": False, "heart_rate": 75, "noise_level": 0.015, "force_simulation": False}, f)
            
    if not os.path.exists(SESSION_CONFIG_PATH):
        with open(SESSION_CONFIG_PATH, "w") as f:
            json.dump({
                "limit_type": "samples",
                "max_duration": 60,
                "max_samples": 5000,
                "port": "Auto-Detect",
                "sensor_adaptation_enabled": True,
                "active_patient_id": "P-101"
            }, f, indent=4)

def read_simulation_config():
    """Reads simulation control settings from JSON file."""
    if os.path.exists(SIM_CONTROL_PATH):
        try:
            with open(SIM_CONTROL_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"simulate_pvc": False, "heart_rate": 75, "noise_level": 0.015, "force_simulation": False}

def get_simulated_ecg_point(phase, is_pvc):
    """Generates a single ECG value based on cardiac phase (0 to 1)."""
    if not is_pvc:
        p = 0.10 * np.exp(-((phase - 0.20) / 0.035)**2)
        q = -0.12 * np.exp(-((phase - 0.32) / 0.012)**2)
        r = 1.00 * np.exp(-((phase - 0.35) / 0.015)**2)
        s = -0.18 * np.exp(-((phase - 0.38) / 0.015)**2)
        t = 0.25 * np.exp(-((phase - 0.55) / 0.055)**2)
        val = p + q + r + s + t
    else:
        r = 1.20 * np.exp(-((phase - 0.38) / 0.055)**2)
        s = -0.40 * np.exp(-((phase - 0.48) / 0.040)**2)
        t = -0.30 * np.exp(-((phase - 0.65) / 0.070)**2)
        val = r + s + t
        
    adc_val = 1500 + val * 1200
    return adc_val

def check_serial_ports(configured_port="Auto-Detect"):
    """Scans and attempts to verify physical hardware connection status."""
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None, False
        
    port_name = None
    if configured_port in ["Auto-Detect", "Auto"]:
        port_name = sorted(ports, key=lambda x: x.device)[0].device
    else:
        # Check if configured port is physically present
        for p in ports:
            if p.device == configured_port:
                port_name = configured_port
                break
                
    if not port_name:
        return None, False
        
    # Test open
    try:
        import serial
        ser = serial.Serial(port_name, BAUD_RATE, timeout=0.1)
        ser.close()
        return port_name, True
    except Exception:
        return port_name, False

def run_ai_diagnostics(session_name):
    """Executes scale-normalized batch inference and saves patient-adapted reports."""
    paths = get_session_paths(session_name)
    if not os.path.exists(paths["filtered"]) or os.path.getsize(paths["filtered"]) < 100:
        print(f"[Diagnostics] Session {session_name} has insufficient filtered data.")
        return
        
    print(f"[Diagnostics] Running scale-normalized inference for: {session_name}...")
    sys.stdout.flush()
    
    try:
        df_filt = pd.read_csv(paths["filtered"])
        filtered_ecg = df_filt["filtered_ecg"].values
        timestamps = df_filt["timestamp"].values
        
        # Load RF model
        model = joblib.load("models/ecg_model.pkl")
        
        predictions = []
        normal_count = 0
        abnormal_count = 0
        step = 125  # 1 second step
        
        for i in range(0, len(filtered_ecg) - WINDOW_SIZE + 1, step):
            window_signal = filtered_ecg[i : i + WINDOW_SIZE]
            window_timestamp = timestamps[i + WINDOW_SIZE - 1]
            
            # Extract features (normalization = True scales window to [0,1])
            feats = compute_signal_features(window_signal, normalize=True)
            feats_df = pd.DataFrame([feats])
            
            expected_cols = ["mean", "std", "max", "min", "skew", "kurt"]
            feats_df = feats_df[expected_cols]
            
            pred = int(model.predict(feats_df)[0])
            pred_label = LABEL_MAP.get(pred, f"Class {pred}")
            status = "NORMAL" if pred == 0 else "ABNORMAL"
            
            if pred == 0:
                normal_count += 1
            else:
                abnormal_count += 1
                
            predictions.append({
                "timestamp": window_timestamp,
                "prediction": pred,
                "prediction_label": pred_label,
                "status": status
            })
            
        if predictions:
            pred_df = pd.DataFrame(predictions)
            pred_df.to_csv(paths["predictions"], index=False)
            
        # Calculate HRV and R-Peaks
        hrv = analyze_session_signal(paths["filtered"])
        bpm = hrv.get("bpm", 0.0) if hrv else 0.0
        sdnn = hrv.get("sdnn", 0.0) if hrv else 0.0
        rmssd = hrv.get("rmssd", 0.0) if hrv else 0.0
        sqi = hrv.get("sqi", 100.0) if hrv else 100.0
        
        # Patient-specific threshold adaptation
        patient_id = get_active_patient_id()
        patient = get_patient(patient_id)
        has_arrhythmia_history = patient.get("arrhythmia_history", False)
        age = patient.get("age", 50)
        
        # Senior or history patients have a lower threshold for High Risk warning
        if has_arrhythmia_history or age >= 65:
            if abnormal_count >= 2:
                risk_level = "HIGH"
            elif abnormal_count > 0:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
        else:
            if abnormal_count >= 4:
                risk_level = "HIGH"
            elif abnormal_count > 0:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
                
        total_windows = len(predictions)
        pct_normal = (normal_count / total_windows * 100) if total_windows > 0 else 0.0
        pct_abnormal = (abnormal_count / total_windows * 100) if total_windows > 0 else 0.0
        
        report = {
            "duration": round(len(df_filt) / 125.0, 1),
            "total_samples": len(df_filt),
            "windows_analysed": total_windows,
            "normal_windows": normal_count,
            "abnormal_windows": abnormal_count,
            "percentage_normal": round(pct_normal, 1),
            "percentage_abnormal": round(pct_abnormal, 1),
            "overall_risk": risk_level,
            "bpm": bpm,
            "sdnn": sdnn,
            "rmssd": rmssd,
            "sqi": sqi,
            "patient_id": patient_id,
            "patient_name": patient.get("name", "Unknown"),
            "patient_age": age,
            "patient_gender": patient.get("gender", "Unknown"),
            "timestamp": time.time()
        }
        
        with open(paths["report"], "w") as f:
            json.dump(report, f, indent=4)
            
        print("\n==================================================")
        print(" 🛑 SESSION COMPLETE: CLINICAL AI REPORT")
        print("==================================================")
        print(f" Patient:             {report['patient_name']} ({report['patient_age']}y, {report['patient_gender']})")
        print(f" Session ID:          {session_name}")
        print(f" Overall Risk:        {report['overall_risk']}")
        print(f" Average Heart Rate:  {report['bpm']} BPM")
        print(f" HRV (SDNN / RMSSD):  {report['sdnn']}ms / {report['rmssd']}ms")
        print(f" Signal Quality index: {report['sqi']}%")
        print("==================================================")
        sys.stdout.flush()
        
    except Exception as e:
        print(f"[Diagnostics] Error running diagnostics: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()

def data_acquisition_pipeline():
    """Persistent Telemetry Acquisition Daemon Loop."""
    global serial_connected
    
    print("[Launcher] Telemetry Daemon started. Waiting for commands...")
    import serial
    
    # Pre-load dataset fallback
    mitbih_beats = {0: [], 1: [], 2: [], 3: []}
    mitbih_loaded = False
    if os.path.exists("data/mitbih/mitbih.csv"):
        try:
            mit_df = pd.read_csv("data/mitbih/mitbih.csv")
            if "label" not in mit_df.columns:
                mit_df = pd.read_csv("data/mitbih/mitbih.csv", header=None)
                mit_df.columns = [f"f_{i}" for i in range(mit_df.shape[1] - 1)] + ["label"]
            for lbl in [0, 1, 2, 3]:
                df_lbl = mit_df[mit_df["label"] == lbl]
                if len(df_lbl) > 0:
                    mitbih_beats[lbl] = df_lbl.drop("label", axis=1).values.tolist()
            mitbih_loaded = True
            print("[Launcher] Preloaded synthetic MIT-BIH beats for simulator.")
        except Exception as e:
            print(f"[Launcher] Error loading MIT-BIH for simulator: {e}")
            
    # Command tracking
    last_command_ts = 0.0
    
    # State machine
    is_recording = False
    session_name = "default"
    sample_count = 0
    start_time = time.time()
    last_process_time = time.time()
    last_data_time = time.time()
    
    sample_buffer = []
    consecutive_leadoff = 0
    leads_attached = False
    beat_buffer = []
    current_beat_label = 0
    
    # Active serial handler
    ser = None
    active_port = None
    
    # Continuous loop
    while not stop_event.is_set():
        current_time = time.time()
        
        # 1. Parse JSON Telemetry commands
        if os.path.exists(TELEMETRY_COMMAND_PATH):
            try:
                with open(TELEMETRY_COMMAND_PATH, "r") as f:
                    cmd_data = json.load(f)
                cmd_ts = cmd_data.get("timestamp", 0.0)
                if cmd_ts > last_command_ts:
                    last_command_ts = cmd_ts
                    command = cmd_data.get("command", "")
                    
                    if command == "start":
                        session_name = cmd_data.get("session_name", "default")
                        set_active_session(session_name)
                        init_files(session_name)
                        
                        # Reset telemetry state
                        is_recording = True
                        sample_count = 0
                        start_time = time.time()
                        last_process_time = time.time()
                        last_data_time = time.time()
                        sample_buffer.clear()
                        beat_buffer.clear()
                        consecutive_leadoff = 0
                        leads_attached = False
                        
                        print(f"\n[Launcher] COMMAND RECEIVED: Start recording session '{session_name}'")
                        sys.stdout.flush()
                        
                    elif command == "stop":
                        if is_recording:
                            print("\n[Launcher] COMMAND RECEIVED: Stop recording session")
                            is_recording = False
                            # Flush remaining
                            paths = get_session_paths(session_name)
                            if sample_buffer:
                                try:
                                    df_new = pd.DataFrame(sample_buffer, columns=["timestamp", "ecg"])
                                    df_new.to_csv(paths["raw"], mode='a', header=False, index=False)
                                    sample_buffer.clear()
                                except Exception:
                                    pass
                            preprocess_data(paths["raw"], paths["filtered"])
                            run_ai_diagnostics(session_name)
                            sys.stdout.flush()
            except Exception as ce:
                print(f"[Launcher] Command parsing error: {ce}")
                
        # Load configuration details
        session_config = read_session_config()
        limit_type = session_config.get("limit_type", "samples")
        max_duration = float(session_config.get("max_duration", 60))
        max_samples = int(session_config.get("max_samples", 5000))
        configured_port = session_config.get("port", "Auto-Detect")
        
        # 2. Check and handle hardware connection status
        port_detected, is_connected = check_serial_ports(configured_port)
        serial_connected = is_connected
        
        # Handle serial port changes
        if serial_connected and ser is None:
            try:
                ser = serial.Serial(port_detected, BAUD_RATE, timeout=0.1)
                active_port = port_detected
                print(f"[Launcher] Serial port {active_port} opened successfully.")
            except Exception:
                serial_connected = False
        elif not serial_connected and ser is not None:
            try:
                ser.close()
            except Exception:
                pass
            ser = None
            active_port = None
            print("[Launcher] Serial port connection lost.")
            
        # Determine simulation mode override
        sim_cfg = read_simulation_config()
        force_simulation = sim_cfg.get("force_simulation", False)
        use_serial = serial_connected and not force_simulation
        
        # 3. Handle Acquisition Step
        if is_recording:
            elapsed_time = current_time - start_time
            
            # Limit check
            if limit_type == "samples":
                session_complete = (sample_count >= max_samples)
            else:
                session_complete = (elapsed_time >= max_duration)
                
            if session_complete:
                is_recording = False
                paths = get_session_paths(session_name)
                # Flush
                if sample_buffer:
                    try:
                        df_new = pd.DataFrame(sample_buffer, columns=["timestamp", "ecg"])
                        df_new.to_csv(paths["raw"], mode='a', header=False, index=False)
                        sample_buffer.clear()
                    except Exception:
                        pass
                preprocess_data(paths["raw"], paths["filtered"])
                run_ai_diagnostics(session_name)
                
                # Write final status
                try:
                    sensor_status = "Active" if (leads_attached and use_serial) else ("Leads Off (Flatline)" if use_serial else "Active (Simulation)")
                    status_data = {
                        "active": False,
                        "samples_collected": sample_count,
                        "max_samples": max_samples,
                        "samples_remaining": 0,
                        "elapsed_time": round(elapsed_time, 1),
                        "max_duration": max_duration,
                        "time_remaining": 0.0,
                        "serial_connected": use_serial,
                        "port_name": active_port if use_serial else "None",
                        "leads_attached": leads_attached if use_serial else True,
                        "sensor_status": sensor_status,
                        "active_session": session_name,
                        "last_update": time.time()
                    }
                    with open(STATUS_FILE_PATH, "w") as f:
                        json.dump(status_data, f)
                except Exception:
                    pass
                continue
                
            # Perform single point acquisition
            ecg_val = None
            
            if use_serial and ser is not None:
                # 3-sec data timeout
                if (current_time - last_data_time) > 3.0:
                    print("[Launcher] Timeout: Serial connected but no data. Switching to Simulation...")
                    use_serial = False
                    leads_attached = False
                else:
                    try:
                        line = ser.readline().decode().strip()
                        if line:
                            try:
                                ecg_val = int(line)
                                last_data_time = current_time
                            except ValueError:
                                pass
                    except Exception:
                        use_serial = False
                        leads_attached = False
                        
            if not use_serial:
                # Simulation fallback
                simulate_pvc = sim_cfg.get("simulate_pvc", False)
                noise_level = sim_cfg.get("noise_level", 0.015)
                
                if not beat_buffer:
                    if simulate_pvc:
                        current_beat_label = 1
                        try:
                            sim_cfg["simulate_pvc"] = False
                            with open(SIM_CONTROL_PATH, "w") as f:
                                json.dump(sim_cfg, f)
                        except Exception:
                            pass
                    else:
                        current_beat_label = np.random.choice([0, 1, 2, 3], p=[0.90, 0.06, 0.03, 0.01])
                        
                    if mitbih_loaded and mitbih_beats[current_beat_label]:
                        idx = np.random.randint(0, len(mitbih_beats[current_beat_label]))
                        beat_buffer = list(mitbih_beats[current_beat_label][idx])
                    else:
                        t_beat = np.linspace(0, 1, 187)
                        beat_buffer = [get_simulated_ecg_point(p, current_beat_label == 1) for p in t_beat]
                        
                raw_val = beat_buffer.pop(0)
                base_val = 1200.0 + raw_val * 1600.0 if mitbih_loaded else raw_val
                noise = np.random.normal(0, noise_level * 1000)
                ecg_val = int(base_val + noise)
                
                # Regulate rate
                time.sleep(1.0 / SAMPLE_RATE)
                
            if ecg_val is not None:
                # AD8232 leads check
                if ecg_val >= 4085 or ecg_val <= 10:
                    consecutive_leadoff += 1
                else:
                    consecutive_leadoff = 0
                    
                leads_attached = (consecutive_leadoff <= 10)
                sample_buffer.append((current_time, ecg_val))
                sample_count += 1
                
            # Periodic flush (every 1s)
            if (current_time - last_process_time) >= 1.0:
                last_process_time = current_time
                paths = get_session_paths(session_name)
                
                if sample_buffer:
                    try:
                        df_new = pd.DataFrame(sample_buffer, columns=["timestamp", "ecg"])
                        if os.path.exists(paths["raw"]) and os.path.getsize(paths["raw"]) > 10:
                            df_new.to_csv(paths["raw"], mode='a', header=False, index=False)
                        else:
                            df_new.to_csv(paths["raw"], index=False)
                        sample_buffer.clear()
                    except Exception:
                        pass
                
                preprocess_data(paths["raw"], paths["filtered"])
                
                # Write current status
                try:
                    sensor_status = "Active" if (leads_attached and use_serial) else ("Leads Off (Flatline)" if use_serial else "Active (Simulation)")
                    status_data = {
                        "active": True,
                        "samples_collected": sample_count,
                        "max_samples": max_samples,
                        "samples_remaining": max(0, max_samples - sample_count),
                        "elapsed_time": round(elapsed_time, 1),
                        "max_duration": max_duration,
                        "time_remaining": round(max(0.0, max_duration - elapsed_time), 1),
                        "serial_connected": use_serial,
                        "port_name": active_port if use_serial else "None",
                        "leads_attached": leads_attached if use_serial else True,
                        "sensor_status": sensor_status,
                        "active_session": session_name,
                        "last_update": time.time()
                    }
                    with open(STATUS_FILE_PATH, "w") as f:
                        json.dump(status_data, f)
                except Exception:
                    pass
        else:
            # Idle Mode - update status periodically (every 0.5s)
            try:
                status_data = {
                    "active": False,
                    "samples_collected": 0,
                    "max_samples": max_samples,
                    "samples_remaining": max_samples,
                    "elapsed_time": 0.0,
                    "max_duration": max_duration,
                    "time_remaining": max_duration,
                    "serial_connected": use_serial,
                    "port_name": active_port if use_serial else "None",
                    "leads_attached": leads_attached if use_serial else True,
                    "sensor_status": "Idle",
                    "active_session": session_name,
                    "last_update": time.time()
                }
                with open(STATUS_FILE_PATH, "w") as f:
                    json.dump(status_data, f)
            except Exception:
                pass
            time.sleep(0.5)
            
    if ser is not None:
        try:
            ser.close()
        except Exception:
            pass
    print("[Launcher] Telemetry Daemon stopped.")

def check_model_exists():
    """Verifies that the trained classification model and clinical dataset are ready."""
    model_path = "models/ecg_model.pkl"
    dataset_path = "data/mitbih/mitbih.csv"
    
    if not os.path.exists(dataset_path):
        print("[Launcher] MIT-BIH clinical dataset not found. Generating synthesized calibration dataset...")
        try:
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))
            import generate_mitbih_data
            generate_mitbih_data.main()
        except Exception as e:
            print(f"[Launcher] Error generating calibration dataset: {e}")
            return False
            
    if not os.path.exists(model_path):
        print("[Launcher] Classifier model not found. Training model on calibration dataset...")
        try:
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))
            import train_model
            train_model.main()
        except Exception as e:
            print(f"[Launcher] Error training Random Forest model: {e}")
            return False
            
    return True

def main():
    print("==================================================")
    print("    STARTING EDGE AI PATIENT MONITOR DEMO")
    print("==================================================")
    
    if not check_model_exists():
        sys.exit(1)
        
    # Determine initial session based on port detection
    active_session_name = get_latest_session()
    set_active_session(active_session_name)
    init_files(active_session_name)
    
    # Make sure command file is cleared
    if os.path.exists(TELEMETRY_COMMAND_PATH):
        try:
            os.remove(TELEMETRY_COMMAND_PATH)
        except Exception:
            pass
            
    # Start telemetry daemon
    acq_thread = threading.Thread(target=data_acquisition_pipeline, name="AcquisitionDaemonThread")
    acq_thread.daemon = True
    acq_thread.start()
    
    # Start Streamlit Dashboard
    print("[Launcher] Launching Streamlit dashboard...")
    try:
        dashboard_proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "dashboard/app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        time.sleep(2.5)
        
        print("[Launcher] Opening dashboard in browser: http://localhost:8501")
        webbrowser.open("http://localhost:8501")
        
        print("\n==================================================")
        print("  Demo is running! Press CTRL+C to terminate.")
        print("==================================================")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n[Launcher] Shutting down demo...")
    finally:
        stop_event.set()
        acq_thread.join(timeout=3)
        try:
            dashboard_proc.terminate()
            print("[Launcher] Dashboard processes terminated successfully.")
        except Exception:
            pass
        print("[Launcher] Shutdown complete.")

if __name__ == "__main__":
    main()

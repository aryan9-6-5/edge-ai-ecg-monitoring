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
PORT = "COM11"
BAUD_RATE = 115200
SAMPLE_RATE = 125  # Hz (matches standard clinical rates and spacing)
WINDOW_SIZE = 250  # Samples to evaluate (2 seconds of data)
MAX_SAMPLES = 7500  # Exactly 60 seconds at 125Hz
MAX_DURATION = 60  # seconds

# Add backend to path for direct imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))
from preprocess import preprocess_data
from feature_extractor import compute_signal_features
from session_manager import get_active_session, get_session_paths, create_new_session

# Control files
SIM_CONTROL_PATH = "data/processed/simulation_control.json"
STATUS_FILE_PATH = "data/processed/acquisition_status.json"

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

def init_files(session_name):
    """Initializes workspace directories and empty data files for the active session."""
    paths = get_session_paths(session_name)
    os.makedirs(paths["dir"], exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Reset raw and filtered data files on launcher start to prevent file bloat
    with open(paths["raw"], "w", newline="") as f:
        f.write("timestamp,ecg\n")
        
    with open(paths["filtered"], "w", newline="") as f:
        f.write("timestamp,ecg,filtered_ecg\n")
        
    with open(paths["predictions"], "w", newline="") as f:
        f.write("timestamp,prediction,prediction_label,status\n")
        
    # Write default simulation config
    with open(SIM_CONTROL_PATH, "w") as f:
        json.dump({"simulate_pvc": False, "heart_rate": 75, "noise_level": 0.015, "force_simulation": False}, f)
        
    # Write initial acquisition status
    initial_status = {
        "active": True,
        "samples_collected": 0,
        "max_samples": MAX_SAMPLES,
        "samples_remaining": MAX_SAMPLES,
        "elapsed_time": 0.0,
        "max_duration": MAX_DURATION,
        "time_remaining": float(MAX_DURATION),
        "serial_connected": False,
        "sensor_status": "Sensor Not Connected",
        "active_session": session_name,
        "last_update": time.time()
    }
    with open(STATUS_FILE_PATH, "w") as f:
        json.dump(initial_status, f)

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

def data_acquisition_pipeline():
    """Background thread running acquisition (Serial or Simulation) and processing."""
    global serial_connected
    
    # Get active session paths
    active_session_name = get_active_session()
    paths = get_session_paths(active_session_name)
    
    print(f"[Launcher] Initializing data acquisition loop for {active_session_name}...")
    import serial
    sample_buffer = []
    
    # Load MIT-BIH dataset for live clinical simulation fallback
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
            print("[Launcher] Loaded real MIT-BIH dataset for live clinical playback simulation fallback.")
        except Exception as e:
            print(f"[Launcher] Could not load MIT-BIH dataset for playback: {e}")
            
    beat_buffer = []
    current_beat_label = 0
    
    # Try opening serial port
    ser = None
    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=0.1)
        serial_connected = True
        print(f"[Launcher] Serial port connected successfully on {PORT}.")
    except Exception as e:
        print(f"[Launcher] WARNING: Could not connect to serial port {PORT} ({e}).")
        print("[Launcher] Launching in SIMULATION MODE. A realistic live ECG stream will be generated.")
        
    # Initialize simulation config with connection status
    try:
        config = read_simulation_config()
        config["serial_connected"] = serial_connected
        with open(SIM_CONTROL_PATH, "w") as f:
            json.dump(config, f)
    except Exception:
        pass
        
    sample_count = 0
    limit_reached_printed = False
    
    # Track start time, processing intervals, and serial data receipt
    start_time = time.time()
    last_process_time = time.time()
    last_data_time = time.time()
    
    while not stop_event.is_set():
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # Read simulation config and force simulation flag
        config = read_simulation_config()
        force_simulation = config.get("force_simulation", False)
        use_serial = serial_connected and not force_simulation
        
        # Enforce limits (7500 samples or 60 seconds)
        session_complete = (sample_count >= MAX_SAMPLES) or (elapsed_time >= MAX_DURATION)
        
        # Write acquisition status JSON every 10 samples or when complete
        if sample_count % 10 == 0 or session_complete:
            try:
                sensor_status = "Sensor Connected on COM11" if use_serial else "Sensor Not Connected"
                status_data = {
                    "active": not session_complete,
                    "samples_collected": sample_count,
                    "max_samples": MAX_SAMPLES,
                    "samples_remaining": max(0, MAX_SAMPLES - sample_count),
                    "elapsed_time": round(elapsed_time, 1),
                    "max_duration": MAX_DURATION,
                    "time_remaining": round(max(0.0, MAX_DURATION - elapsed_time), 1),
                    "serial_connected": use_serial,
                    "sensor_status": sensor_status,
                    "active_session": active_session_name,
                    "last_update": time.time()
                }
                with open(STATUS_FILE_PATH, "w") as f:
                    json.dump(status_data, f)
            except Exception:
                pass
                
        if session_complete:
            break
            
        ecg_val = None
        
        if use_serial and ser is not None:
            # Check for silent/dummy port timeout (no valid data for 3 seconds)
            if (current_time - last_data_time) > 3.0:
                print("\n[Launcher] WARNING: COM port connected, but no valid data received for 3 seconds.")
                print("[Launcher] Automatically switching to SIMULATION MODE...")
                sys.stdout.flush()
                serial_connected = False
                try:
                    ser.close()
                except Exception:
                    pass
                ser = None
                # Update config
                try:
                    config = read_simulation_config()
                    config["serial_connected"] = False
                    with open(SIM_CONTROL_PATH, "w") as f:
                        json.dump(config, f)
                except Exception:
                    pass
            else:
                # Read from serial port
                try:
                    line = ser.readline().decode().strip()
                    if line:
                        try:
                            ecg_val = int(line)
                            # Reset timeout since we received a valid data point
                            last_data_time = current_time
                        except ValueError:
                            pass
                except Exception:
                    # If serial drops, switch to simulation
                    print("[Launcher] Serial connection lost. Switching to Simulation Mode...")
                    serial_connected = False
                    if ser:
                        ser.close()
                        ser = None
                    try:
                        config = read_simulation_config()
                        config["serial_connected"] = False
                        with open(SIM_CONTROL_PATH, "w") as f:
                            json.dump(config, f)
                    except Exception:
                        pass
        
        # Simulation Mode
        if not use_serial:
            simulate_pvc = config.get("simulate_pvc", False)
            noise_level = config.get("noise_level", 0.015)
            
            # If current beat buffer is empty, select and load a new beat
            if not beat_buffer:
                if simulate_pvc:
                    current_beat_label = 1
                    try:
                        config["simulate_pvc"] = False
                        with open(SIM_CONTROL_PATH, "w") as f:
                            json.dump(config, f)
                    except Exception:
                        pass
                else:
                    if np.random.rand() < 0.08:
                        current_beat_label = np.random.choice([1, 2, 3])
                    else:
                        current_beat_label = 0
                        
                if mitbih_loaded and mitbih_beats[current_beat_label]:
                    idx = np.random.randint(0, len(mitbih_beats[current_beat_label]))
                    beat_buffer = list(mitbih_beats[current_beat_label][idx])
                else:
                    t_beat = np.linspace(0, 1, 187)
                    beat_buffer = [get_simulated_ecg_point(p, current_beat_label == 1) for p in t_beat]
            
            raw_val = beat_buffer.pop(0)
            
            if mitbih_loaded:
                base_val = 1200.0 + raw_val * 1600.0
            else:
                base_val = raw_val
                
            noise = np.random.normal(0, noise_level * 1000)
            ecg_val = int(base_val + noise)
            
            # Sleep to match sample rate
            time.sleep(1.0 / SAMPLE_RATE)
            
        if ecg_val is not None:
            # Buffer sample in memory to reduce Windows file I/O locks
            sample_buffer.append((current_time, ecg_val))
            sample_count += 1
                
        # Flush to CSV and preprocess every 1.0 second
        if (current_time - last_process_time) >= 1.0:
            last_process_time = current_time
            
            # Flush buffered samples to raw CSV in active session directory
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
            
            # Perform Preprocessing (SMA filtering) on session raw data
            preprocess_data(paths["raw"], paths["filtered"])
            
            # Print console telemetry progress
            remaining_samples = max(0, MAX_SAMPLES - sample_count)
            remaining_time = max(0.0, MAX_DURATION - elapsed_time)
            mode_str = "SERIAL" if use_serial else "SIMULATION"
            
            print(f"[Telemetry] Session: {active_session_name} | Mode: {mode_str} | Samples: {sample_count}/{MAX_SAMPLES} ({remaining_samples} left) | Time: {elapsed_time:.1f}s/{MAX_DURATION}s ({remaining_time:.1f}s left)")
            sys.stdout.flush()

    # Flush any remaining samples in buffer
    if sample_buffer:
        try:
            df_new = pd.DataFrame(sample_buffer, columns=["timestamp", "ecg"])
            if os.path.exists(paths["raw"]) and os.path.getsize(paths["raw"]) > 10:
                df_new.to_csv(paths["raw"], mode='a', header=False, index=False)
            else:
                df_new.to_csv(paths["raw"], index=False)
        except Exception:
            pass
            
    # Final preprocess to ensure filtered dataset has all raw records
    preprocess_data(paths["raw"], paths["filtered"])
    
    # ------------------------------------------------------------------
    # RUN PATIENT INFRASTRUCTURE AI DIAGNOSTICS (BATCH WINDOWED INFERENCE)
    # ------------------------------------------------------------------
    print(f"\n[Launcher] Data collection complete! Running AI Diagnostics on {active_session_name}...")
    sys.stdout.flush()
    
    if os.path.exists(paths["filtered"]) and os.path.getsize(paths["filtered"]) > 50:
        try:
            df_filt = pd.read_csv(paths["filtered"])
            filtered_ecg = df_filt["filtered_ecg"].values
            timestamps = df_filt["timestamp"].values
            
            # Load RandomForest classifier
            model = joblib.load("models/ecg_model.pkl")
            
            predictions = []
            normal_count = 0
            abnormal_count = 0
            step = 125 # 1 second step
            
            for i in range(0, len(filtered_ecg) - WINDOW_SIZE + 1, step):
                window_signal = filtered_ecg[i : i + WINDOW_SIZE]
                window_timestamp = timestamps[i + WINDOW_SIZE - 1]
                
                # Extract features
                feats = compute_signal_features(window_signal)
                feats_df = pd.DataFrame([feats])
                
                # Re-order features correctly
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
                
            # Save predictions to session predictions.csv
            if predictions:
                pred_df = pd.DataFrame(predictions)
                pred_df.to_csv(paths["predictions"], index=False)
                
            # Create session report summary
            total_windows = len(predictions)
            pct_normal = (normal_count / total_windows * 100) if total_windows > 0 else 0.0
            pct_abnormal = (abnormal_count / total_windows * 100) if total_windows > 0 else 0.0
            
            if abnormal_count >= 3:
                risk_level = "HIGH"
            elif abnormal_count > 0:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
                
            report = {
                "duration": 60,
                "total_samples": len(df_filt),
                "windows_analysed": total_windows,
                "normal_windows": normal_count,
                "abnormal_windows": abnormal_count,
                "percentage_normal": round(pct_normal, 1),
                "percentage_abnormal": round(pct_abnormal, 1),
                "overall_risk": risk_level,
                "timestamp": time.time()
            }
            
            with open(paths["report"], "w") as f:
                json.dump(report, f, indent=4)
                
            # Print beautiful summary table in terminal
            print("\n==================================================")
            print(" 🛑 SESSION COMPLETE: AI DIAGNOSIS GENERATED")
            print("==================================================")
            print(f" Session Name:        {active_session_name}")
            print(f" Recording Duration:  {report['duration']}s")
            print(f" Total Samples:       {report['total_samples']}")
            print(f" Windows Analysed:    {report['windows_analysed']}")
            print(f" Normal Windows:      {report['normal_windows']}")
            print(f" Abnormal Windows:    {report['abnormal_windows']}")
            print(f" Percentage Normal:   {report['percentage_normal']}%")
            print(f" Percentage Abnormal: {report['percentage_abnormal']}%")
            print(f" Overall Risk Level:  {report['overall_risk']}")
            print("==================================================")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"[Launcher] Error running AI diagnostics: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            
    # Write final completed status to acquisition_status.json
    try:
        sensor_status = "Sensor Connected on COM11" if use_serial else "Sensor Not Connected"
        status_data = {
            "active": False,
            "samples_collected": sample_count,
            "max_samples": MAX_SAMPLES,
            "samples_remaining": 0,
            "elapsed_time": 60.0,
            "max_duration": MAX_DURATION,
            "time_remaining": 0.0,
            "serial_connected": use_serial,
            "sensor_status": sensor_status,
            "active_session": active_session_name,
            "last_update": time.time()
        }
        with open(STATUS_FILE_PATH, "w") as f:
            json.dump(status_data, f)
    except Exception:
        pass
        
    if ser is not None:
        ser.close()
    print("[Launcher] Data acquisition loop stopped.")
    sys.stdout.flush()

def check_model_exists():
    """Verifies that the trained classification model and clinical dataset are ready."""
    model_path = "models/ecg_model.pkl"
    dataset_path = "data/mitbih/mitbih.csv"
    
    # 1. Verify / generate dataset
    if not os.path.exists(dataset_path):
        print("[Launcher] MIT-BIH clinical dataset not found. Generating synthesized calibration dataset...")
        try:
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))
            import generate_mitbih_data
            generate_mitbih_data.main()
        except Exception as e:
            print(f"[Launcher] Error generating calibration dataset: {e}")
            return False
            
    # 2. Verify / train model
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
    
    # 1. Verify model and dataset
    if not check_model_exists():
        sys.exit(1)
        
    # 2. Create and initialize a new recording session folder
    session_path, session_name = create_new_session()
    init_files(session_name)
    
    # 3. Start data acquisition thread
    acq_thread = threading.Thread(target=data_acquisition_pipeline, name="AcquisitionThread")
    acq_thread.daemon = True
    acq_thread.start()
    
    # 4. Start Streamlit Dashboard
    print("[Launcher] Launching Streamlit dashboard...")
    try:
        # Use python -m streamlit to guarantee running inside the current venv environment
        dashboard_proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "dashboard/app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a moment for Streamlit to initialize
        time.sleep(2)
        
        # Open browser to default port
        print("[Launcher] Opening dashboard in browser: http://localhost:8501")
        webbrowser.open("http://localhost:8501")
        
        print("\n==================================================")
        print("  Demo is running! Press CTRL+C to terminate.")
        print("==================================================")
        
        # Wait for user exit (CTRL+C)
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

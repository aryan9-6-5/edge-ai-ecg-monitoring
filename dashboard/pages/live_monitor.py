# dashboard/pages/live_monitor.py
import os
import sys
import time
import json
import struct
import math
import base64
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import serial.tools.list_ports

# Setup sys.path to allow importing from run.py and backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from session_manager import get_session_paths, get_active_session, set_active_session
from patient_manager import get_active_patient_id, get_patient

# Custom styling for medical aesthetic
st.markdown("""
    <style>
    .medical-card {
        background-color: #121620;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #1e2638;
        margin-bottom: 15px;
    }
    .status-badge-ok {
        background-color: #064e3b;
        color: #34d399;
        border: 1px solid #047857;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .status-badge-warn {
        background-color: #78350f;
        color: #fde047;
        border: 1px solid #d97706;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .status-badge-info {
        background-color: #1e3a8a;
        color: #93c5fd;
        border: 1px solid #2563eb;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .alert-banner-high {
        background-color: #7f1d1d;
        color: #fca5a5;
        padding: 18px;
        border-radius: 8px;
        border-left: 5px solid #ef4444;
        margin-bottom: 20px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .alert-banner-med {
        background-color: #78350f;
        color: #fef3c7;
        padding: 18px;
        border-radius: 8px;
        border-left: 5px solid #f59e0b;
        margin-bottom: 20px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .alert-banner-low {
        background-color: #064e3b;
        color: #d1fae5;
        padding: 18px;
        border-radius: 8px;
        border-left: 5px solid #10b981;
        margin-bottom: 20px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Data paths
SIM_CONTROL_PATH = "data/processed/simulation_control.json"
STATUS_FILE_PATH = "data/processed/acquisition_status.json"
SESSION_CONFIG_PATH = "data/processed/session_config.json"
TELEMETRY_COMMAND_PATH = "data/processed/telemetry_command.json"

LABEL_MAP = {
    0: "Normal (N)",
    1: "PVC (V)",
    2: "Supraventricular premature (S)",
    3: "Fusion (F)",
    4: "Unclassifiable (Q)"
}

def read_json_file(file_path, default_val):
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return default_val

def generate_beep_wav_base64(frequency=750, duration_seconds=0.3):
    """Generates a pure sine wave WAV file in memory and returns it as a base64 string."""
    sample_rate = 8000
    num_samples = int(sample_rate * duration_seconds)
    
    # WAV header values
    header = struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + num_samples * 2,
        b'WAVE',
        b'fmt ',
        16,
        1,  # PCM
        1,  # Mono
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b'data',
        num_samples * 2
    )
    
    samples = []
    for i in range(num_samples):
        t = float(i) / sample_rate
        val = int(30000.0 * math.sin(2.0 * math.pi * frequency * t))
        samples.append(struct.pack('<h', val))
        
    wav_data = header + b''.join(samples)
    return base64.b64encode(wav_data).decode('utf-8')

def send_telemetry_command(command, session_name="default"):
    """Writes a JSON IPC command for the background run.py daemon."""
    os.makedirs(os.path.dirname(TELEMETRY_COMMAND_PATH), exist_ok=True)
    try:
        with open(TELEMETRY_COMMAND_PATH, "w") as f:
            json.dump({
                "command": command,
                "session_name": session_name,
                "timestamp": time.time()
            }, f, indent=4)
        return True
    except Exception:
        return False

def run_diagnostics_on_the_fly(filtered_df):
    """Computes AI Predictions and report dynamically using scale-normalized features."""
    model_path = "models/ecg_model.pkl"
    if not os.path.exists(model_path):
        return None
    try:
        import joblib
        from feature_extractor import compute_signal_features
        model = joblib.load(model_path)
        
        filtered_ecg = filtered_df["filtered_ecg"].values
        timestamps = filtered_df["timestamp"].values
        
        predictions = []
        normal_count = 0
        abnormal_count = 0
        window_size = 250
        step = 125
        
        for i in range(0, len(filtered_ecg) - window_size + 1, step):
            window_signal = filtered_ecg[i : i + window_size]
            window_timestamp = timestamps[i + window_size - 1]
            
            # Extracts features with min-max [0, 1] scale-normalization enabled
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
            
        # HRV & R-Peaks
        from analytics_engine import detect_r_peaks, compute_hrv_metrics
        peaks = detect_r_peaks(filtered_ecg, 125)
        hrv = compute_hrv_metrics(peaks, 125)
        bpm = hrv.get("bpm", 0.0)
        sdnn = hrv.get("sdnn", 0.0)
        rmssd = hrv.get("rmssd", 0.0)
        
        # Load Patient Profile for adaptive risk thresholding
        patient_id = get_active_patient_id()
        patient = get_patient(patient_id)
        has_history = patient.get("arrhythmia_history", False)
        age = patient.get("age", 50)
        
        if has_history or age >= 65:
            risk_level = "HIGH" if abnormal_count >= 2 else ("MEDIUM" if abnormal_count > 0 else "LOW")
        else:
            risk_level = "HIGH" if abnormal_count >= 4 else ("MEDIUM" if abnormal_count > 0 else "LOW")
            
        total_windows = len(predictions)
        pct_normal = (normal_count / total_windows * 100) if total_windows > 0 else 0.0
        pct_abnormal = (abnormal_count / total_windows * 100) if total_windows > 0 else 0.0
        
        report = {
            "duration": round(len(filtered_df) / 125.0, 1),
            "total_samples": len(filtered_df),
            "windows_analysed": total_windows,
            "normal_windows": normal_count,
            "abnormal_windows": abnormal_count,
            "percentage_normal": round(pct_normal, 1),
            "percentage_abnormal": round(pct_abnormal, 1),
            "overall_risk": risk_level,
            "bpm": bpm,
            "sdnn": sdnn,
            "rmssd": rmssd,
            "sqi": 100.0,
            "patient_id": patient_id,
            "patient_name": patient.get("name", "Unknown"),
            "timestamp": time.time()
        }
        return report, predictions
    except Exception as e:
        st.error(f"Failed to generate dynamic AI report: {e}")
        return None

def apply_normalization(series, mode):
    """Normalizes an ECG series based on selected display mode."""
    if "Min-Max" in mode:
        s_min = series.min()
        s_max = series.max()
        if s_max - s_min > 0:
            return (series - s_min) / (s_max - s_min)
        return series.fillna(0.5)
    elif "Z-Score" in mode:
        s_mean = series.mean()
        s_std = series.std()
        if s_std > 1e-8:
            return (series - s_mean) / s_std
        return series - s_mean
    return series

# ------------------------------------------------------------------
# CONFIG & STATE LOAD
# ------------------------------------------------------------------
active_session_name = get_active_session()
paths = get_session_paths(active_session_name)
patient_id = get_active_patient_id()
patient = get_patient(patient_id)

# Check if background telemetry daemon is running
status_data = read_json_file(STATUS_FILE_PATH, {
    "active": False,
    "samples_collected": 0,
    "max_samples": 5000,
    "samples_remaining": 5000,
    "elapsed_time": 0.0,
    "max_duration": 60,
    "time_remaining": 60.0,
    "serial_connected": False,
    "leads_attached": False,
    "sensor_status": "Daemon Offline",
    "active_session": active_session_name,
    "last_update": 0.0
})

daemon_active = (time.time() - status_data.get("last_update", 0.0) < 4.0)
is_recording = daemon_active and status_data.get("active", False)

# ------------------------------------------------------------------
# SIDEBAR CONTROLS
# ------------------------------------------------------------------
st.sidebar.header("🕹️ Telemetry Control Panel")
st.sidebar.subheader(f"🏥 Session: {active_session_name}")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Acquisition Limits")

session_config = read_json_file(SESSION_CONFIG_PATH, {
    "limit_type": "samples",
    "max_duration": 60,
    "max_samples": 5000,
    "port": "Auto-Detect",
    "sensor_adaptation_enabled": True
})

limit_type = st.sidebar.radio(
    "Stop Telemetry By:",
    ["Time Duration (sec)", "Sample Count"],
    index=0 if session_config.get("limit_type", "samples") == "time" else 1
)

limit_type_config = "time" if limit_type == "Time Duration (sec)" else "samples"

max_duration = session_config.get("max_duration", 60)
max_samples = session_config.get("max_samples", 5000)

if limit_type_config == "time":
    max_duration = st.sidebar.slider("Duration limit (s)", 10, 120, int(max_duration), step=5)
    session_config["limit_type"] = "time"
    session_config["max_duration"] = max_duration
else:
    max_samples = st.sidebar.slider("Sample count limit", 500, 10000, int(max_samples), step=500)
    session_config["limit_type"] = "samples"
    session_config["max_samples"] = max_samples

# Port Selector
st.sidebar.markdown("### 🔌 Microcontroller Port")
ports_detected = [p.device for p in serial.tools.list_ports.comports()]
port_options = ["Auto-Detect"] + sorted(ports_detected)
current_port = session_config.get("port", "Auto-Detect")
port_idx = port_options.index(current_port) if current_port in port_options else 0
selected_port = st.sidebar.selectbox("Select COM Port:", port_options, index=port_idx)
session_config["port"] = selected_port

# Outlier Adaptation
sensor_adapt = st.sidebar.checkbox("Adapt to Sensor Lead-Off (Outliers)", value=session_config.get("sensor_adaptation_enabled", True))
session_config["sensor_adaptation_enabled"] = sensor_adapt

# Write configs
try:
    with open(SESSION_CONFIG_PATH, "w") as sf:
        json.dump(session_config, sf, indent=4)
except Exception:
    pass

st.sidebar.markdown("---")

# Telemetry Commands via File IPC
col_cmd1, col_cmd2 = st.sidebar.columns(2)
with col_cmd1:
    if st.button("▶️ Start", use_container_width=True, disabled=not daemon_active or is_recording):
        send_telemetry_command("start", active_session_name)
        st.toast("Starting telemetry acquisition...", icon="📈")
        time.sleep(0.3)
        st.rerun()

with col_cmd2:
    if st.button("⏹️ Stop", use_container_width=True, disabled=not is_recording):
        send_telemetry_command("stop", active_session_name)
        st.toast("Stopping telemetry acquisition...", icon="🛑")
        time.sleep(0.3)
        st.rerun()

if st.sidebar.button("🧹 Clear Telemetry Data", use_container_width=True):
    try:
        for file_key in ["raw", "filtered", "predictions"]:
            file_path = paths[file_key]
            if os.path.exists(file_path):
                os.remove(file_path)
        if os.path.exists(paths["report"]):
            os.remove(paths["report"])
        st.sidebar.success("Session telemetry cleared!")
        time.sleep(0.4)
        st.rerun()
    except Exception as ce:
        st.sidebar.error(f"Error resetting telemetry data: {ce}")

# Simulation controls
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧬 Simulated Anomalies")
sim_control_cfg = read_json_file(SIM_CONTROL_PATH, {"force_simulation": False, "simulate_pvc": False})
force_sim = st.sidebar.checkbox("Force simulated patient source", value=sim_control_cfg.get("force_simulation", False))
sim_control_cfg["force_simulation"] = force_sim

if st.sidebar.button("💥 Trigger PVC Beats anomaly", disabled=not is_recording or force_sim and not sim_control_cfg.get("force_simulation", False)):
    sim_control_cfg["simulate_pvc"] = True
    st.sidebar.success("PVC beat triggered in next cycle!")

try:
    with open(SIM_CONTROL_PATH, "w") as f:
        json.dump(sim_control_cfg, f)
except Exception:
    pass


# ------------------------------------------------------------------
# MAIN LAYOUT
# ------------------------------------------------------------------
st.title("🩺 Real-time Cardiac Telemetry")
st.markdown(f"Active Patient: **{patient['name']}** (Age: {patient['age']}, Gender: {patient['gender']}) | Session: **{active_session_name}**")
st.markdown("---")

# Visual Source Badges
is_simulation = force_sim or not status_data.get("serial_connected", False)

leads_attached = status_data.get("leads_attached", False)
serial_connected = status_data.get("serial_connected", False)

# Sensor lead warnings
if not is_simulation:
    if not serial_connected:
        st.markdown(
            '<div style="background-color: #7f1d1d; color: #fca5a5; padding: 18px; border-radius: 8px; border-left: 5px solid #ef4444; margin-bottom: 20px; font-weight: bold; text-align: center; font-size: 1.15rem;">'
            '🚨 SENSOR PLUGGED OUT. PLEASE CONNECT (Microcontroller USB Serial Connection Not Detected)'
            '</div>',
            unsafe_allow_html=True
        )
    elif not leads_attached:
        st.markdown(
            '<div style="background-color: #7f1d1d; color: #fca5a5; padding: 18px; border-radius: 8px; border-left: 5px solid #ef4444; margin-bottom: 20px; font-weight: bold; text-align: center; font-size: 1.15rem;">'
            '⚠️ ELECTRODE LEADS DETACHED. PLEASE ATTACH (AD8232 Electrode Leads Detached / Flatline Signal)'
            '</div>',
            unsafe_allow_html=True
        )

# Normalization Selector
plot_mode = st.radio(
    "📈 Waveform Visualization Mode:",
    ["SMA Filtered (Raw Scale)", "Min-Max Normalized (0 to 1)", "Z-Score Normalized (Mean=0, Std=1)"],
    horizontal=True
)

# Load telemetry datasets
raw_df = None
filtered_df = None
if os.path.exists(paths["raw"]) and os.path.getsize(paths["raw"]) > 50:
    try:
        raw_df = pd.read_csv(paths["raw"])
        filtered_df = pd.read_csv(paths["filtered"])
    except Exception:
        pass

# ------------------------------------------------------------------
# DURING RECORDING LAYOUT
# ------------------------------------------------------------------
if is_recording:
    st.subheader(f"🩺 Patient Telemetry Stream: Recording...")
    
    elapsed = status_data.get("elapsed_time", 0.0)
    samples = status_data.get("samples_collected", 0)
    
    # Progress Display
    if limit_type_config == "time":
        st.progress(min(1.0, elapsed / max(max_duration, 1.0)))
        c_timer, c_samples, c_base = st.columns(3)
        with c_timer:
            st.metric("Countdown Timer", f"{max(0.0, max_duration - elapsed):.1f} s", delta="-1.0s / sec")
        with c_samples:
            st.metric("Samples Logged", f"{samples} / {max_samples}", delta="+125 Hz")
        with c_base:
            st.metric("Patient Baseline HR", f"{patient['baseline_hr']} BPM")
    else:
        st.progress(min(1.0, samples / max(max_samples, 1)))
        c_timer, c_samples, c_base = st.columns(3)
        with c_timer:
            st.metric("Elapsed Time", f"{elapsed:.1f} s", delta="+1.0s / sec")
        with c_samples:
            st.metric("Samples Logged", f"{samples} / {max_samples}", delta=f"{max(0, max_samples - samples)} remaining")
        with c_base:
            st.metric("Patient Baseline HR", f"{patient['baseline_hr']} BPM")
            
    st.markdown("---")
    
    # Scrolling Plot
    if filtered_df is not None and len(filtered_df) > 10:
        plot_len = 250
        tail_raw = raw_df.tail(plot_len).reset_index(drop=True)
        tail_filt = filtered_df.tail(plot_len).reset_index(drop=True)
        
        # Apply Normalization
        raw_plotted = apply_normalization(tail_raw["ecg"], plot_mode)
        filt_plotted = apply_normalization(tail_filt["filtered_ecg"], plot_mode)
        
        graph_df = pd.DataFrame({
            "Raw ECG Telemetry": raw_plotted,
            "SMA Filtered Signal": filt_plotted
        })
        
        st.line_chart(graph_df, height=380)
    else:
        st.info("Initializing telemetry stream buffers... Please wait.")
        
    st.info("💡 Model diagnostics, reports, alerts, and analytics will process and display automatically once the limit is reached.")
    
    # Refresh loop during recording
    time.sleep(0.8)
    st.rerun()

# ------------------------------------------------------------------
# AFTER RECORDING DIAGNOSTICS LAYOUT
# ------------------------------------------------------------------
else:
    st.subheader(f"📊 Session Diagnostic Summary Report")
    
    # Verify report JSON, run on-the-fly if missing but data is present
    report = read_json_file(paths["report"], None)
    on_the_fly_preds = None
    
    if not report and filtered_df is not None and len(filtered_df) > 10:
        with st.spinner("⏳ Analyzing recorded signals on the fly..."):
            otf = run_diagnostics_on_the_fly(filtered_df)
            if otf:
                report, on_the_fly_preds = otf
                
    if not report:
        st.warning("⏳ No diagnostic report generated for this session yet.")
        if raw_df is not None and len(raw_df) > 0:
            st.info(f"Loaded {len(raw_df)} partial samples. Click 'Start' in control panel to record a telemetry run.")
            if filtered_df is not None and len(filtered_df) > 0:
                st.markdown("##### Partial Signal View")
                partial_filt = apply_normalization(filtered_df["filtered_ecg"], plot_mode)
                st.line_chart(partial_filt.tail(500))
        else:
            st.info("Choose a session/patient and click 'Start' in the Sidebar panel to run a patient ECG acquisition test.")
        st.stop()
        
    # Report exists! Load values
    risk_level = report.get("overall_risk", "LOW")
    normal_w = report.get("normal_windows", 0)
    abnormal_w = report.get("abnormal_windows", 0)
    total_w = report.get("windows_analysed", 0)
    pct_normal = report.get("percentage_normal", 0.0)
    pct_abnormal = report.get("percentage_abnormal", 0.0)
    total_samples = report.get("total_samples", 0)
    
    # 1. Flashing/Colored Diagnostic Risk Banner
    if risk_level == "HIGH":
        st.markdown(f"""
            <div class="alert-banner-high">
                🚨 CRITICAL RISK DETECTED: HIGH CARDIAC ARRHYTHMIC ACTIVITY FOUND! <br/>
                Overall Diagnosis: High ectopic ventricular anomaly index. Patient monitoring advised.
            </div>
        """, unsafe_allow_html=True)
        
        # Inject base64 browser WAV beep sound
        # Storing play state in session state to avoid playing non-stop on other clicks
        if 'last_played_high' not in st.session_state or st.session_state.last_played_high != paths["report"]:
            st.session_state.last_played_high = paths["report"]
            beep_b64 = generate_beep_wav_base64(880, 0.4)
            st.markdown(f'<audio src="data:audio/wav;base64,{beep_b64}" autoplay></audio>', unsafe_allow_html=True)
            
    elif risk_level == "MEDIUM":
        st.markdown(f"""
            <div class="alert-banner-med">
                ⚠️ ELEVATED RISK DETECTED: MODERATE ARRHYTHMIC BEATS FOUND! <br/>
                Overall Diagnosis: Elevated ectopic activity. Patient baseline deviated.
            </div>
        """, unsafe_allow_html=True)
        
        if 'last_played_med' not in st.session_state or st.session_state.last_played_med != paths["report"]:
            st.session_state.last_played_med = paths["report"]
            beep_b64 = generate_beep_wav_base64(660, 0.3)
            st.markdown(f'<audio src="data:audio/wav;base64,{beep_b64}" autoplay></audio>', unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="alert-banner-low">
                🟢 PATIENT DIAGNOSIS: NORMAL CARDIAC RHYTHM (LOW RISK) <br/>
                Overall Diagnosis: Ectopic and arrhythmia metrics fall within safe clinical bounds.
            </div>
        """, unsafe_allow_html=True)
        
    # 2. Session Summary Columns
    col_dur, col_samp, col_wind, col_risk = st.columns(4)
    with col_dur:
        st.metric("Recording Duration", f"{report.get('duration', 60):.1f} s")
    with col_samp:
        st.metric("Samples Collected", f"{total_samples} points")
    with col_wind:
        st.metric("Windows Analysed", f"{total_w} windows")
    with col_risk:
        st.metric("Patient ECG Risk Level", risk_level)
        
    # Styled HTML Session Summary Details
    st.markdown(f"""
    <div style="background-color: #121620; padding: 20px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #1e2638;">
      <h4 style="color: #60a5fa; margin-top:0;">📊 Patient Session Statistics Summary</h4>
      <table style="width: 100%; font-size: 0.95rem; border-collapse: collapse; line-height: 2.2;">
        <tr style="border-bottom: 1px solid #1e2638;">
          <td><b>Normal Heartbeat Windows</b></td>
          <td style="text-align: right; color: #34d399; font-weight: bold;">{normal_w} windows</td>
          <td style="text-align: right; color: #34d399; font-weight: bold;">{pct_normal}%</td>
        </tr>
        <tr style="border-bottom: 1px solid #1e2638;">
          <td><b>Abnormal Heartbeat Windows</b></td>
          <td style="text-align: right; color: #f87171; font-weight: bold;">{abnormal_w} windows</td>
          <td style="text-align: right; color: #f87171; font-weight: bold;">{pct_abnormal}%</td>
        </tr>
        <tr style="border-bottom: 1px solid #1e2638;">
          <td><b>Clinical BPM (Computed average)</b></td>
          <td colspan="2" style="text-align: right; color: #60a5fa; font-weight: bold;">{report.get("bpm", 0.0)} BPM (Baseline: {patient["baseline_hr"]} BPM)</td>
        </tr>
        <tr>
          <td><b>Heart Rate Variability (SDNN / RMSSD)</b></td>
          <td colspan="2" style="text-align: right; color: #a78bfa; font-weight: bold;">{report.get("sdnn", 0.0)} ms / {report.get("rmssd", 0.0)} ms</td>
        </tr>
      </table>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 3. Full Session Interactive Plotly Viewer
    st.subheader("🔍 Full Waveform Viewer")
    st.caption("Scroll, zoom, and select sections of the patient's entire ECG signal recording.")
    
    if filtered_df is not None:
        try:
            # Sub-sample for plotly load performance (take every 2nd point = 62.5Hz)
            plot_df = filtered_df.iloc[::2].copy()
            plot_df["Seconds"] = (plot_df["timestamp"] - plot_df["timestamp"].iloc[0])
            
            # Apply chosen normalization / preprocessing display mode
            plot_df["plotted_ecg"] = apply_normalization(plot_df["filtered_ecg"], plot_mode)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=plot_df["Seconds"],
                y=plot_df["plotted_ecg"],
                mode='lines',
                line=dict(color='#ef4444', width=1.5),
                name='Filtered ECG'
            ))
            
            y_title = "Amplitude (Normalized)" if "Normalized" in plot_mode else "Amplitude (ADC Units)"
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(18,22,32,0.85)',
                font_color='#ffffff',
                xaxis=dict(
                    title="Time (Seconds)",
                    showgrid=True,
                    gridcolor='#1e2638',
                    rangeslider=dict(visible=True)
                ),
                yaxis=dict(
                    title=y_title,
                    showgrid=True,
                    gridcolor='#1e2638'
                ),
                margin=dict(t=10, b=10, l=10, r=10),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as ge:
            st.error(f"Plotly drawing error: {ge}")
            
    # 4. Predictions Timeline and Diagnostic Table
    col_timeline, col_table = st.columns([1, 1])
    
    p_df = None
    if on_the_fly_preds is not None:
        p_df = pd.DataFrame(on_the_fly_preds)
    else:
        pred_csv_path = paths["predictions"]
        if os.path.exists(pred_csv_path) and os.path.getsize(pred_csv_path) > 40:
            try:
                p_df = pd.read_csv(pred_csv_path)
            except Exception:
                pass
                
    with col_timeline:
        st.subheader("⏱️ Prediction Timeline")
        st.caption("AI classification category mapped to the recording timeline.")
        
        if p_df is not None and len(p_df) > 0:
            try:
                p_df_timeline = p_df.copy()
                p_df_timeline["Seconds"] = range(len(p_df_timeline))
                
                fig_timeline = px.scatter(
                    p_df_timeline,
                    x="Seconds",
                    y="prediction_label",
                    color="status",
                    color_discrete_map={"NORMAL": "#34d399", "ABNORMAL": "#f87171"},
                    labels={"prediction_label": "Class Diagnosis"},
                    height=300
                )
                fig_timeline.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(18,22,32,0.5)',
                    font_color='#ffffff',
                    xaxis_title="Window Index (Seconds)",
                    yaxis_title="Diagnosis Category",
                    margin=dict(t=10, b=10, l=10, r=10)
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
            except Exception as pe:
                st.write(f"Could not render timeline: {pe}")
        else:
            st.info("No predictions timeline data available.")
            
    with col_table:
        st.subheader("📋 Session Diagnostic Log")
        st.caption("Windowed diagnostic reports details.")
        if p_df is not None and len(p_df) > 0:
            try:
                p_df_table = p_df.copy()
                p_df_table["Window Offset"] = [f"Second {i+2}" for i in range(len(p_df_table))]
                st.dataframe(
                    p_df_table[["Window Offset", "prediction_label", "status"]].iloc[::-1],
                    use_container_width=True,
                    height=260
                )
            except Exception:
                st.write("Loading log details...")
        else:
            st.info("Log is empty.")
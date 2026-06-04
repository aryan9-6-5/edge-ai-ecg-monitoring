# dashboard/pages/live_monitor.py
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Setup sys.path to allow importing from run.py and backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Custom styling for medical aesthetic
st.markdown("""
    <style>
    .medical-card {
        background-color: #111827;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #1f2937;
        margin-bottom: 15px;
    }
    .status-badge-ok {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .status-badge-warn {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .status-badge-info {
        background-color: #1e3a8a;
        color: #93c5fd;
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
        background-color: #065f46;
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

from session_manager import get_all_sessions, get_session_paths, get_active_session, create_new_session, set_active_session

def read_json_file(file_path, default_val):
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return default_val

def is_pipeline_active():
    """Checks if the telemetry acquisition pipeline is running and recently active."""
    if os.path.exists(STATUS_FILE_PATH):
        try:
            with open(STATUS_FILE_PATH, "r") as f:
                data = json.load(f)
            last_update = data.get("last_update", 0.0)
            if time.time() - last_update < 4.0:
                return data.get("active", False)
        except Exception:
            pass
    return False

@st.cache_resource
def start_local_acquisition_thread():
    """Starts the background telemetry thread locally inside Streamlit if run.py is offline."""
    import threading
    try:
        from run import data_acquisition_pipeline, init_files, stop_event
        active_session = get_active_session()
        init_files(active_session)
        stop_event.clear()
        t = threading.Thread(target=data_acquisition_pipeline, name="LocalAcquisitionThread", daemon=True)
        t.start()
        return True
    except Exception as e:
        st.sidebar.error(f"Failed to start local thread: {e}")
    return False

# Initialize / load available sessions
sessions = get_all_sessions()
if not sessions:
    # Auto-initialize session_001 if none exist
    _, initial_name = create_new_session()
    sessions = [initial_name]

# Check if pipeline is running or needs startup
pipeline_is_active = is_pipeline_active()
status_data = read_json_file(STATUS_FILE_PATH, {
    "active": False,
    "samples_collected": 0,
    "max_samples": 7500,
    "samples_remaining": 7500,
    "elapsed_time": 0.0,
    "max_duration": 60,
    "time_remaining": 60.0,
    "serial_connected": False,
    "sensor_status": "Sensor Not Connected",
    "active_session": sessions[-1],
    "last_update": 0.0
})

active_session_name = status_data.get("active_session", sessions[-1])

# If active in status file but last_update is too old, pipeline has crashed/stopped
is_recording = pipeline_is_active and status_data.get("active", False)

# Sidebar Demo & Simulation Panel
st.sidebar.header("🕹️ Telemetry Session Control")

# Session Selection
selected_session_name = st.sidebar.selectbox(
    "Select Session Folder",
    sessions,
    index=sessions.index(active_session_name) if active_session_name in sessions else len(sessions)-1
)

set_active_session(selected_session_name)
paths = get_session_paths(selected_session_name)

# Display connection / sensor status from the active pipeline
sensor_msg = status_data.get("sensor_status", "Sensor Not Connected")
if "Connected" in sensor_msg:
    st.sidebar.markdown(f'<span class="status-badge-ok">🔌 {sensor_msg.upper()}</span>', unsafe_allow_html=True)
else:
    st.sidebar.markdown(f'<span class="status-badge-warn">⚠️ SENSOR NOT CONNECTED</span>', unsafe_allow_html=True)

# Telemetry Source Badge
pipeline_running_externally = (time.time() - status_data.get("last_update", 0.0) < 4.0)
if pipeline_running_externally:
    st.sidebar.markdown('<span class="status-badge-info">⚙️ BACKEND: TERMINAL ENGINE (RUN.PY)</span>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<span class="status-badge-info">⚙️ BACKEND: LOCAL STANDALONE</span>', unsafe_allow_html=True)

st.sidebar.markdown("---")

# Trigger a brand new session recording
if st.sidebar.button("▶️ Start New Recording Session", use_container_width=True):
    # Stop any running process
    try:
        from run import stop_event
        stop_event.set()
        time.sleep(0.5)
    except Exception:
        pass
        
    # Create new session directory
    _, new_session = create_new_session()
    
    # Initialize session files
    try:
        from run import init_files
        init_files(new_session)
    except Exception:
        pass
        
    # Start thread
    try:
        from run import stop_event, data_acquisition_pipeline
        stop_event.clear()
        import threading
        t = threading.Thread(target=data_acquisition_pipeline, name="LocalAcquisitionThread", daemon=True)
        t.start()
    except Exception as e:
        st.sidebar.error(f"Error starting local pipeline: {e}")
        
    st.rerun()

# Serial / Simulation Controls
config = read_json_file(SIM_CONTROL_PATH, {"simulate_pvc": False, "heart_rate": 75, "noise_level": 0.015, "serial_connected": False})
serial_connected = config.get("serial_connected", False)

if serial_connected:
    source = st.sidebar.radio(
        "Active Data Source",
        ["Physical Sensor", "Simulated Patient"],
        index=1 if config.get("force_simulation", False) else 0
    )
    force_sim = (source == "Simulated Patient")
    if force_sim != config.get("force_simulation", False):
        config["force_simulation"] = force_sim
        config["simulate_pvc"] = False
        try:
            with open(SIM_CONTROL_PATH, "w") as f:
                json.dump(config, f)
        except Exception:
            pass
        st.rerun()

# Simulator sliders
if not serial_connected or config.get("force_simulation", False):
    st.sidebar.markdown("##### Patient Simulator Tuning")
    hr_val = st.sidebar.slider("Heart Rate (BPM)", min_value=50, max_value=140, value=config.get("heart_rate", 75))
    noise_val = st.sidebar.slider("Signal Noise level", min_value=0.005, max_value=0.05, value=config.get("noise_level", 0.015), step=0.005)
    
    if hr_val != config.get("heart_rate") or noise_val != config.get("noise_level"):
        config["heart_rate"] = hr_val
        config["noise_level"] = noise_val
        try:
            with open(SIM_CONTROL_PATH, "w") as f:
                json.dump(config, f)
        except Exception:
            pass
            
    if st.sidebar.button("🚨 Inject PVC Arrhythmia Beat"):
        config["simulate_pvc"] = True
        try:
            with open(SIM_CONTROL_PATH, "w") as f:
                json.dump(config, f)
            st.sidebar.success("PVC beat scheduled!")
        except Exception:
            pass

# Clear session logs completely
st.sidebar.markdown("---")
if st.sidebar.button("🧹 Clear All Telemetry Sessions"):
    import shutil
    try:
        if os.path.exists("data/sessions"):
            shutil.rmtree("data/sessions")
        if os.path.exists(STATUS_FILE_PATH):
            os.remove(STATUS_FILE_PATH)
        st.sidebar.success("All session logs cleared!")
        st.rerun()
    except Exception as ce:
        st.sidebar.error(f"Error resetting sessions: {ce}")

# Load active raw telemetry dataset
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
if is_recording and selected_session_name == active_session_name:
    st.subheader(f"🩺 Telemetry Session Active: {selected_session_name}")
    st.caption("The system is streaming raw ECG telemetry from your sensor. Watch the signal live.")
    
    # Progress Display
    elapsed = status_data.get("elapsed_time", 0.0)
    samples = status_data.get("samples_collected", 0)
    st.progress(min(1.0, elapsed / 60.0))
    
    c_timer, c_samples = st.columns(2)
    with c_timer:
        st.metric("Recording Countdown Timer", f"{max(0.0, 60.0 - elapsed):.1f} s remaining", delta="-1.0s / sec")
    with c_samples:
        st.metric("Total Samples Collected", f"{samples} / 7500", delta="+125 Hz")
        
    st.markdown("---")
    
    # Scrolling Plot
    if filtered_df is not None and len(filtered_df) > 10:
        plot_len = 250
        tail_raw = raw_df.tail(plot_len).reset_index(drop=True)
        tail_filt = filtered_df.tail(plot_len).reset_index(drop=True)
        
        graph_df = pd.DataFrame({
            "Raw ECG Telemetry": tail_raw["ecg"],
            "SMA Filtered Signal": tail_filt["filtered_ecg"]
        })
        
        st.line_chart(graph_df, height=380)
    else:
        st.info("Initializing telemetry stream buffers... Please wait.")
        
    st.info("💡 Model diagnostics, reports, alerts, and analytics will process and display automatically once the countdown reaches 0.")
    
    # Refresh loop during recording
    time.sleep(0.8)
    st.rerun()

# ------------------------------------------------------------------
# AFTER RECORDING LAYOUT
# ------------------------------------------------------------------
else:
    st.subheader(f"📊 Recording Session Diagnostics: {selected_session_name}")
    
    # Verify session report JSON exists
    report = read_json_file(paths["report"], None)
    
    if not report:
        st.warning("⏳ No diagnostic report generated. Telemetry acquisition may have been interrupted, or the session is empty.")
        if raw_df is not None and len(raw_df) > 0:
            st.info(f"Loaded {len(raw_df)} partial samples. Click 'Start New Recording Session' to perform a complete 60-second logging run.")
            
            # Show what raw data exists
            st.markdown("##### Partial Signal View")
            st.line_chart(raw_df["ecg"].tail(500))
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
        st.markdown("""
            <div class="alert-banner-high">
                🚨 CRITICAL RISK DETECTED: HIGH CARDIAC ARRHYTHMIC ACTIVITY FOUND! <br/>
                Overall Diagnosis: High ventricular anomalies detected. Risk level elevated.
            </div>
        """, unsafe_allow_html=True)
        # winsound acoustic beep
        try:
            import winsound
            winsound.Beep(950, 200)
            winsound.Beep(950, 200)
        except Exception:
            pass
    elif risk_level == "MEDIUM":
        st.markdown("""
            <div class="alert-banner-med">
                ⚠️ ELEVATED RISK DETECTED: MODERATE ARRHYTHMIC BEATS FOUND! <br/>
                Overall Diagnosis: Elevated ectopic activity. Patient monitoring advised.
            </div>
        """, unsafe_allow_html=True)
        try:
            import winsound
            winsound.Beep(850, 150)
        except Exception:
            pass
    else:
        st.markdown("""
            <div class="alert-banner-low">
                🟢 PATIENT DIAGNOSIS: NORMAL CARDIAC RHYTHM (LOW RISK) <br/>
                Overall Diagnosis: Ectopic and arrhythmia metrics fall within safe clinical bounds.
            </div>
        """, unsafe_allow_html=True)
        
    # 2. Session Summary Columns
    col_dur, col_samp, col_wind, col_risk = st.columns(4)
    with col_dur:
        st.metric("Recording Duration", f"{report.get('duration', 60)} s")
    with col_samp:
        st.metric("Samples Collected", f"{total_samples} points")
    with col_wind:
        st.metric("Windows Analysed", f"{total_w} windows")
    with col_risk:
        st.metric("Patient EGC Risk Level", risk_level)
        
    # Styled HTML Session Summary Details
    st.markdown(f"""
    <div style="background-color: #1f2937; padding: 20px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #374151;">
      <h4 style="color: #60a5fa; margin-top:0;">📊 Patient Session Statistics Summary</h4>
      <table style="width: 100%; font-size: 0.95rem; border-collapse: collapse;">
        <tr style="border-bottom: 1px solid #374151; height: 35px;">
          <td><b>Normal Windows Count</b></td>
          <td style="text-align: right; color: #10b981; font-weight: bold;">{normal_w} windows</td>
          <td style="text-align: right; color: #10b981; font-weight: bold;">{pct_normal}%</td>
        </tr>
        <tr style="border-bottom: 1px solid #374151; height: 35px;">
          <td><b>Abnormal Windows Count</b></td>
          <td style="text-align: right; color: #ef4444; font-weight: bold;">{abnormal_w} windows</td>
          <td style="text-align: right; color: #ef4444; font-weight: bold;">{pct_abnormal}%</td>
        </tr>
      </table>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 3. Full Session Interactive Plotly Viewer
    st.subheader("🔍 Full 60-Second Filtered Waveform Viewer")
    st.caption("Scroll, zoom, and select sections of the patient's entire 60-second ECG signal recording.")
    
    if filtered_df is not None:
        try:
            # We sub-sample for plotly load performance (take every 2nd point = 62.5Hz)
            # This is still incredibly high resolution but speeds up rendering drastically
            plot_df = filtered_df.iloc[::2].copy()
            plot_df["Seconds"] = (plot_df["timestamp"] - plot_df["timestamp"].iloc[0])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=plot_df["Seconds"],
                y=plot_df["filtered_ecg"],
                mode='lines',
                line=dict(color='#ef4444', width=1.5),
                name='SMA Filtered ECG'
            ))
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(17,24,39,0.85)',
                font_color='#ffffff',
                xaxis=dict(
                    title="Time (Seconds)",
                    showgrid=True,
                    gridcolor='#374151',
                    rangeslider=dict(visible=True) # Adds range slider for easy navigation
                ),
                yaxis=dict(
                    title="Amplitude (ADC Units)",
                    showgrid=True,
                    gridcolor='#374151'
                ),
                margin=dict(t=10, b=10, l=10, r=10),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as ge:
            st.error(f"Plotly drawing error: {ge}")
            
    # 4. Predictions Timeline and Diagnostic Table
    col_timeline, col_table = st.columns([1, 1])
    
    with col_timeline:
        st.subheader("⏱️ Prediction Timeline")
        st.caption("AI classification category mapped to the recording timeline.")
        
        pred_csv_path = paths["predictions"]
        if os.path.exists(pred_csv_path) and os.path.getsize(pred_csv_path) > 40:
            try:
                p_df = pd.read_csv(pred_csv_path)
                p_df["Seconds"] = range(len(p_df))
                
                fig_timeline = px.scatter(
                    p_df,
                    x="Seconds",
                    y="prediction_label",
                    color="status",
                    color_discrete_map={"NORMAL": "#10b981", "ABNORMAL": "#ef4444"},
                    labels={"prediction_label": "Class Diagnosis"},
                    height=300
                )
                fig_timeline.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(17,24,39,0.5)',
                    font_color='#ffffff',
                    xaxis_title="Window Index (Seconds)",
                    yaxis_title="Diagnosis Category",
                    margin=dict(t=10, b=10, l=10, r=10)
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
            except Exception as pe:
                st.write(f"Could not render timeline: {pe}")
        else:
            st.info("No predictions timeline data.")
            
    with col_table:
        st.subheader("📋 Session Diagnostic Log")
        st.caption("Windowed diagnostic reports details.")
        pred_csv_path = paths["predictions"]
        if os.path.exists(pred_csv_path) and os.path.getsize(pred_csv_path) > 40:
            try:
                p_df = pd.read_csv(pred_csv_path)
                p_df["Window Offset"] = [f"Second {i+2}" for i in range(len(p_df))]
                st.dataframe(
                    p_df[["WindowOffset" if "WindowOffset" in p_df.columns else "Window Offset", "prediction_label", "status"]].iloc[::-1],
                    use_container_width=True,
                    height=260
                )
            except Exception:
                st.write("Loading log details...")
        else:
            st.info("Log is empty.")
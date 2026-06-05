# dashboard/app.py
import os
import sys
import json
import time
import pandas as pd
import streamlit as st
import serial.tools.list_ports

st.set_page_config(
    page_title="Edge AI Patient Monitor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup path for backend imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from session_manager import get_active_session, set_active_session, get_session_paths, get_all_sessions
from patient_manager import load_patients, get_patient, get_active_patient_id, set_active_patient_id

# ------------------------------------------------------------------
# PREMIUM STYLING
# ------------------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    .main {
        background-color: #0d0f14;
        color: #ffffff;
    }
    
    .card {
        background-color: #121620;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #1e2638;
        margin-bottom: 20px;
        box-shadow: 0 8px 16px -2px rgba(0,0,0,0.3);
    }
    
    .status-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-ok {
        background-color: #064e3b;
        color: #34d399;
        border: 1px solid #047857;
    }
    
    .status-error {
        background-color: #7f1d1d;
        color: #fca5a5;
        border: 1px solid #b91c1c;
    }
    
    .status-warn {
        background-color: #78350f;
        color: #fde047;
        border: 1px solid #d97706;
    }
    
    .status-info {
        background-color: #1e3a8a;
        color: #93c5fd;
        border: 1px solid #2563eb;
    }
    
    .medical-badge {
        font-size: 0.85rem;
        font-weight: 600;
        background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
        padding: 3px 8px;
        border-radius: 4px;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# SIDEBAR - PATIENT & SESSION MANAGER
# ------------------------------------------------------------------
st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <span style="font-size: 3rem;">🏥</span>
        <h2 style="margin: 0; color: #ef4444;">Patient Monitor</h2>
        <span style="font-size: 0.8rem; color: #6b7280;">Edge AI Diagnostics v1.0</span>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.subheader("👤 Patient Profile Selection")

# Load patients & selector
patients_db = load_patients()
patient_options = list(patients_db.keys())
active_p_id = get_active_patient_id()

selected_p_id = st.sidebar.selectbox(
    "Choose Active Patient:",
    patient_options,
    index=patient_options.index(active_p_id) if active_p_id in patient_options else 0
)

# Update patient if changed
if selected_p_id != active_p_id:
    set_active_patient_id(selected_p_id)
    st.rerun()

# Display selected Patient Card
patient = get_patient(selected_p_id)
history_badge = '<span class="status-pill status-error">Arrhythmia History</span>' if patient["arrhythmia_history"] else '<span class="status-pill status-ok">No Prior Arrhythmia</span>'

st.sidebar.markdown(f"""
    <div style="background-color: #161b26; border: 1px solid #283046; border-radius: 8px; padding: 16px; margin-bottom: 15px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <strong style="color: #60a5fa; font-size: 1.05rem;">{patient["name"]}</strong>
            <span style="color: #9ca3af; font-size: 0.8rem;">ID: {patient["id"]}</span>
        </div>
        <div style="font-size: 0.88rem; color: #d1d5db; line-height: 1.5;">
            <b>Age / Gender:</b> {patient["age"]} yrs / {patient["gender"]}<br/>
            <b>Baseline HR:</b> {patient["baseline_hr"]} BPM<br/>
            <div style="margin-top: 8px; margin-bottom: 8px;">{history_badge}</div>
            <p style="margin: 5px 0 0 0; font-size: 0.8rem; color: #9ca3af; font-style: italic;">"{patient["notes"]}"</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Session Selector
st.sidebar.subheader("📂 Patient Session Selection")
all_sessions = get_all_sessions()
active_session = get_active_session()
selected_session = st.sidebar.selectbox(
    "Select Session Folder:",
    all_sessions,
    index=all_sessions.index(active_session) if active_session in all_sessions else 0
)

if selected_session != active_session:
    set_active_session(selected_session)
    st.rerun()

# Check daemon / connection status for Sidebar indicator
status_path = "data/processed/acquisition_status.json"
serial_connected = False
daemon_active = False

if os.path.exists(status_path):
    try:
        with open(status_path, "r") as f:
            sd = json.load(f)
        last_up = sd.get("last_update", 0.0)
        daemon_active = (time.time() - last_up < 4.0)
        serial_connected = sd.get("serial_connected", False)
    except Exception:
        pass

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔌 Connection Status")
if serial_connected:
    st.sidebar.markdown('<span class="status-pill status-ok">🟢 Hardware Connected (live)</span>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<span class="status-pill status-warn">🟡 Simulation Active (default)</span>', unsafe_allow_html=True)

if daemon_active:
    st.sidebar.markdown('<span class="status-pill status-ok">🟢 Telemetry Daemon: Active</span>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<span class="status-pill status-error">🔴 Telemetry Daemon: Offline</span>', unsafe_allow_html=True)


# ------------------------------------------------------------------
# MAIN HOME DASHBOARD
# ------------------------------------------------------------------
st.title("🏥 Edge AI Patient Monitoring System")
st.markdown("##### Version 1 (Polished MVP Development Release)")
st.markdown("---")

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### 🩺 Clinical Overview")
    st.markdown("""
    The **Edge AI Patient Monitoring System** is a low-latency cardiology telemetry platform. It transitions from traditional raw-signal streaming to **Edge-embedded intelligence**, executing cleaning, features extraction, and anomaly classification locally on the hardware node.
    
    This design guarantees patient data privacy, eliminates cloud dependencies, and delivers immediate diagnostic alerts for cardiac anomalies like Premature Ventricular Contractions (PVCs).
    """)
    
    st.markdown("#### ⚙️ Edge Telemetry Pipeline Steps")
    
    st.markdown("""
    <div style="display: flex; flex-direction: column; gap: 12px; margin-top: 10px; margin-bottom: 20px;">
      <div style="background-color: #121620; padding: 15px; border-radius: 8px; border-left: 5px solid #3b82f6; border: 1px solid #1e2638; border-left-width: 5px;">
        <strong style="color: #60a5fa; font-size: 1.0rem;">📥 1. High-Fidelity Signal Acquisition</strong><br/>
        <span style="font-size: 0.88rem; color: #9ca3af;">Continuously streams ECG signals at 125 Hz from physical hardware (ESP32 microcontrollers via serial port) or simulated patient cardiac profiles.</span>
      </div>
      <div style="background-color: #121620; padding: 15px; border-radius: 8px; border-left: 5px solid #10b981; border: 1px solid #1e2638; border-left-width: 5px;">
        <strong style="color: #34d399; font-size: 1.0rem;">🧹 2. Signal Preprocessing & Lead-Off Adaptation</strong><br/>
        <span style="font-size: 0.88rem; color: #9ca3af;">Detects high-amplitude noise spikes or lead disconnection flatlines. Dynamically interpolates outliers and applies a 5-Point SMA smoothing filter.</span>
      </div>
      <div style="background-color: #121620; padding: 15px; border-radius: 8px; border-left: 5px solid #fbbf24; border: 1px solid #1e2638; border-left-width: 5px;">
        <strong style="color: #fbbf24; font-size: 1.0rem;">📊 3. Clinical Feature Extraction</strong><br/>
        <span style="font-size: 0.88rem; color: #9ca3af;">Segments data into sliding windows. Computes scale-normalized moments: Mean, Standard Deviation, Max, Min, Skewness, and Kurtosis.</span>
      </div>
      <div style="background-color: #121620; padding: 15px; border-radius: 8px; border-left: 5px solid #8b5cf6; border: 1px solid #1e2638; border-left-width: 5px;">
        <strong style="color: #a78bfa; font-size: 1.0rem;">🧠 4. Patient-Specific AI Inference</strong><br/>
        <span style="font-size: 0.88rem; color: #9ca3af;">Classifies rhythm patterns in real-time using a Random Forest classifier trained on MIT-BIH clinical data. Thresholds adapt based on the patient's risk profile.</span>
      </div>
      <div style="background-color: #121620; padding: 15px; border-radius: 8px; border-left: 5px solid #ef4444; border: 1px solid #1e2638; border-left-width: 5px;">
        <strong style="color: #f87171; font-size: 1.0rem;">🚨 5. Risk Alert & Clinical Reporting</strong><br/>
        <span style="font-size: 0.88rem; color: #9ca3af;">Triggers instantaneous audio-visual browser alerts and logs details. Provides printable patient reports with heart rate and HRV metrics.</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### 🔍 System Health & Diagnostics")
    
    # Check ML Model
    model_path = "models/ecg_model.pkl"
    model_ok = os.path.exists(model_path)
    model_status = '<span class="status-pill status-ok">🟢 Ready</span>' if model_ok else '<span class="status-pill status-error">🔴 Missing</span>'
    
    # Check calibration dataset
    dataset_path = "data/mitbih/mitbih.csv"
    dataset_ok = os.path.exists(dataset_path)
    dataset_status = '<span class="status-pill status-ok">🟢 Loaded</span>' if dataset_ok else '<span class="status-pill status-error">🔴 Missing</span>'
    
    # Scan serial ports
    com_ports = [p.device for p in serial.tools.list_ports.comports()]
    ports_display = ", ".join(com_ports) if com_ports else "None Detected"
    ports_status = '<span class="status-pill status-ok">🟢 Available</span>' if com_ports else '<span class="status-pill status-info">🔵 Simulated Fallback</span>'
    
    st.markdown(f"""
        <div class="card">
            <h4 style="margin-top:0; color: #ef4444; border-bottom: 1px solid #1e2638; padding-bottom: 10px;">📋 Calibration Verification</h4>
            <table style="width:100%; font-size: 0.9rem; line-height: 2.0; border-collapse: collapse;">
                <tr>
                    <td><b>AI Classifier (Random Forest)</b></td>
                    <td style="text-align:right;">{model_status}</td>
                </tr>
                <tr>
                    <td><b>MIT-BIH Calibration Dataset</b></td>
                    <td style="text-align:right;">{dataset_status}</td>
                </tr>
                <tr>
                    <td><b>Active Patient Session</b></td>
                    <td style="text-align:right;"><span class="status-pill status-info">{active_session}</span></td>
                </tr>
                <tr>
                    <td><b>Available COM Ports</b></td>
                    <td style="text-align:right;">{ports_status}</td>
                </tr>
                <tr>
                    <td colspan="2" style="font-size:0.75rem; color:#6b7280; padding-top: 5px;">
                        Detected Ports: {ports_display}
                    </td>
                </tr>
            </table>
        </div>
    """, unsafe_allow_html=True)
    
    st.info("👈 Use the Streamlit sidebar to navigate between pages. Explore Live Monitor, Analytics, Alerts, and Model Info.")
    
    # Show active patient parameters
    st.markdown(f"""
        <div class="card" style="margin-top:10px;">
            <h4 style="margin-top:0; color: #60a5fa; border-bottom: 1px solid #1e2638; padding-bottom: 10px;">🩺 Dynamic Alarm Thresholds</h4>
            <p style="font-size: 0.85rem; line-height: 1.4; color: #9ca3af;">
                Risk classification warning sensitivities adapt based on age and history:
            </p>
            <table style="width:100%; font-size: 0.88rem; line-height: 1.8;">
                <tr>
                    <td><b>Selected Profile Type:</b></td>
                    <td style="text-align:right; color:#ffffff;"><b>{"High Risk Category" if (patient["arrhythmia_history"] or patient["age"]>=65) else "Normal Category"}</b></td>
                </tr>
                <tr>
                    <td><b>High Risk Trigger Limit:</b></td>
                    <td style="text-align:right; color:#ef4444;"><b>{"&ge; 2 abnormal windows" if (patient["arrhythmia_history"] or patient["age"]>=65) else "&ge; 4 abnormal windows"}</b></td>
                </tr>
                <tr>
                    <td><b>Medium Risk Trigger Limit:</b></td>
                    <td style="text-align:right; color:#fbbf24;"><b>&ge; 1 abnormal window</b></td>
                </tr>
            </table>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align: center; color: #6b7280; font-size: 0.8rem;'>Edge AI Patient Monitor | Academic Research Project</p>", unsafe_allow_html=True)

# dashboard/pages/alerts.py
import os
import time
import json
import pandas as pd
import streamlit as st
import sys

# Setup sys.path to allow importing from run.py and backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from session_manager import get_active_session, get_session_paths
from patient_manager import get_active_patient_id, get_patient

st.set_page_config(page_title="Cardiovascular Alerts Log", page_icon="🚨", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .critical-alert-card {
        background-color: #7f1d1d;
        color: #fca5a5;
        padding: 18px;
        border-radius: 8px;
        border-left: 5px solid #ef4444;
        margin-bottom: 12px;
        border: 1px solid #b91c1c;
    }
    .acknowledged-alert-card {
        background-color: #111827;
        color: #9ca3af;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #4b5563;
        margin-bottom: 12px;
        border: 1px solid #1e2638;
        opacity: 0.75;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚨 Critical Cardiac Alerts History")
st.markdown("---")

active_session_name = get_active_session()
paths = get_session_paths(active_session_name)
pred_csv_path = paths["predictions"]
alerts_status_path = os.path.join(paths["dir"], "alerts_status.json")

patient_id = get_active_patient_id()
patient = get_patient(patient_id)

st.info(f"📋 Displaying alerts log for Selected Patient Profile: **{patient['name']}** | Session: **{active_session_name}**")

def load_alerts_status():
    if os.path.exists(alerts_status_path):
        try:
            with open(alerts_status_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_alerts_status(status_dict):
    try:
        with open(alerts_status_path, "w") as f:
            json.dump(status_dict, f, indent=4)
        return True
    except Exception:
        return False

if not os.path.exists(pred_csv_path) or os.path.getsize(pred_csv_path) < 50:
    st.warning("⚠️ No predictions logged for this session yet. Please navigate to the Live Monitor tab and record a session first.")
else:
    try:
        df = pd.read_csv(pred_csv_path)
        
        if len(df) == 0:
            st.info("Waiting for incoming telemetry predictions...")
        else:
            # Load report for overall patient risk
            report_data = None
            if os.path.exists(paths["report"]):
                try:
                    with open(paths["report"], "r") as f:
                        report_data = json.load(f)
                except Exception:
                    pass
            
            risk_level = report_data.get("overall_risk", "LOW") if report_data else "LOW"
            duration = report_data.get("duration", 60.0) if report_data else 60.0
            
            # Filter warnings (abnormal beats)
            abnormal_df = df[df["status"] == "ABNORMAL"].copy()
            
            # Format offset for readability
            abnormal_df["Window Offset"] = [f"Second {int(row.name)+2}" for _, row in abnormal_df.iterrows()]
            
            # Load persistent alert acknowledgment status
            alerts_status = load_alerts_status()
            
            # Calculate alert rate
            minutes = duration / 60.0
            alert_rate = round(len(abnormal_df) / minutes, 2) if minutes > 0 else 0.0
            
            # Count unacknowledged
            unack_count = 0
            for _, row in abnormal_df.iterrows():
                key = row["Window Offset"]
                if key not in alerts_status or not alerts_status[key].get("acknowledged", False):
                    unack_count += 1
            
            # Top metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Windows Analysed", f"{len(df)}")
            with col2:
                st.metric("Abnormal Events", len(abnormal_df))
            with col3:
                st.metric("Alerts Frequency Rate", f"{alert_rate} / min")
            with col4:
                st.metric("Unacknowledged Alerts", unack_count, delta=f"-{len(abnormal_df)-unack_count} ack", delta_color="inverse")
                
            st.markdown("---")
            
            # Acknowledge all button
            if unack_count > 0:
                if st.button("✅ Acknowledge All Active Alerts", use_container_width=True):
                    for _, row in abnormal_df.iterrows():
                        key = row["Window Offset"]
                        alerts_status[key] = {"acknowledged": True, "timestamp": time.time()}
                    save_alerts_status(alerts_status)
                    st.rerun()
                    
            # Display Alerts list
            col_l, col_r = st.columns([2, 1])
            
            with col_l:
                st.subheader("📋 Chronological Warning Log")
                if len(abnormal_df) == 0:
                    st.success("🟢 No abnormal heartbeats detected in the patient's monitoring session.")
                else:
                    # Reverse chronological order
                    reversed_alerts = abnormal_df.iloc[::-1]
                    
                    for idx, row in reversed_alerts.iterrows():
                        key = row["Window Offset"]
                        is_ack = key in alerts_status and alerts_status[key].get("acknowledged", False)
                        
                        if is_ack:
                            card_html = f"""
                                <div class="acknowledged-alert-card">
                                    <strong>✅ {row['prediction_label']} (Acknowledged)</strong><br/>
                                    <span style="font-size: 0.85rem; opacity: 0.85;">Offset: {row['Window Offset']} | Classification Code: Class {row['prediction']}</span>
                                </div>
                            """
                            st.markdown(card_html, unsafe_allow_html=True)
                        else:
                            col_alert, col_btn = st.columns([4, 1])
                            with col_alert:
                                card_html = f"""
                                    <div class="critical-alert-card">
                                        <strong>⚠️ ALERT: {row['prediction_label']} Detected</strong><br/>
                                        <span style="font-size: 0.85rem; opacity: 0.85;">Offset: {row['Window Offset']} | Classification Code: Class {row['prediction']}</span>
                                    </div>
                                """
                                st.markdown(card_html, unsafe_allow_html=True)
                            with col_btn:
                                st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
                                if st.button("Ack Alert", key=f"ack_{key}", use_container_width=True):
                                    alerts_status[key] = {"acknowledged": True, "timestamp": time.time()}
                                    save_alerts_status(alerts_status)
                                    st.rerun()
            
            with col_r:
                st.subheader("💡 Emergency Protocol Guidelines")
                st.markdown(f"""
                **Active Patient Category:** {"HIGH RISK SENIOR" if (patient['arrhythmia_history'] or patient['age']>=65) else "STANDARD RISK"}
                **Patient Risk Level:** `{risk_level}`
                
                If the monitoring system generates multiple consecutive **ABNORMAL** PVC alerts:
                
                1. **Patient Check**: Confirm physical sensor placement (leads off AD8232 can cause spike artifacts).
                2. **Symptom Review**: Check if patient is experiencing chest discomfort, shortness of breath, or palpitations.
                3. **Medical Intervention**: Log clinical data and contact supervising physician.
                4. **Record Event**: Use the Analytics tab to examine signal morphology around the timestamp of the alert.
                """)
                
    except Exception as e:
        st.error(f"Error loading alerts database: {e}")
        import traceback
        traceback.print_exc()

# Only auto-refresh if a recording is actively running
_status_path = "data/processed/acquisition_status.json"
_is_active = False
if os.path.exists(_status_path):
    try:
        with open(_status_path, "r") as _f:
            _sd = json.load(_f)
        if time.time() - _sd.get("last_update", 0.0) < 4.0 and _sd.get("active", False):
            _is_active = True
    except Exception:
        pass

if _is_active:
    time.sleep(2.5)
    st.rerun()
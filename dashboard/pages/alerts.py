# dashboard/pages/alerts.py
import os
import time
import pandas as pd
import streamlit as st
import sys

# Setup sys.path to allow importing from run.py and backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from session_manager import get_active_session, get_session_paths

st.set_page_config(page_title="Cardiovascular Alerts Log", page_icon="🚨", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .critical-alert-card {
        background-color: #7f1d1d;
        color: #fca5a5;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ef4444;
        margin-bottom: 12px;
    }
    .normal-card {
        background-color: #065f46;
        color: #a7f3d0;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #10b981;
        margin-bottom: 12px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚨 Critical Cardiac Alerts History")
st.markdown("---")

# Load current session configurations
active_session_name = get_active_session()
paths = get_session_paths(active_session_name)
pred_csv_path = paths["predictions"]

st.info(f"📋 Displaying alerts log for Selected Patient Session: **{active_session_name}**")

if not os.path.exists(pred_csv_path) or os.path.getsize(pred_csv_path) < 50:
    st.warning("⚠️ No predictions logged for this session yet. Please navigate to the Live Monitor tab and record a 60-second session first.")
else:
    try:
        df = pd.read_csv(pred_csv_path)
        
        if len(df) == 0:
            st.info("Waiting for incoming telemetry predictions...")
        else:
            # Filter warnings (abnormal beats)
            abnormal_df = df[df["status"] == "ABNORMAL"].copy()
            
            # Format offset for readability
            abnormal_df["Window Offset"] = [f"Second {int(row.name)+2}" for _, row in abnormal_df.iterrows()]
            
            # Top metrics row
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Windows Analysed", f"{len(df)}")
            with col2:
                st.metric("Abnormal Events Logged", len(abnormal_df))
            with col3:
                last_event = abnormal_df.iloc[-1]["Window Offset"] if len(abnormal_df) > 0 else "N/A"
                st.metric("Last Critical Event Offset", last_event)
                
            st.markdown("---")
            
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
                        st.markdown(f"""
                            <div class="critical-alert-card">
                                <strong>⚠️ {row['prediction_label']} Detected</strong><br/>
                                <span style="font-size: 0.85rem; opacity: 0.85;">Session Timestamp Offset: {row['Window Offset']} | Classification Code: Class {row['prediction']}</span>
                            </div>
                        """, unsafe_allow_html=True)
            
            with col_r:
                st.subheader("💡 Emergency Protocol Guidelines")
                st.markdown("""
                If the monitoring system generates multiple consecutive **ABNORMAL** PVC alerts:
                
                1. **Patient Check**: Confirm physical sensor placement (leads off AD8232 can cause spike artifacts).
                2. **Symptom Review**: Check if patient is experiencing chest discomfort, shortness of breath, or palpitations.
                3. **Medical Intervention**: Log clinical data and contact supervising physician.
                4. **Record Event**: Use the Analytics tab to examine signal morphology around the timestamp of the alert.
                """)
                
    except Exception as e:
        st.error(f"Error loading alerts database: {e}")

# Autorefresh
time.sleep(2.5)
st.rerun()
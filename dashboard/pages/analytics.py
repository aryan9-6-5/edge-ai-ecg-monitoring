# dashboard/pages/analytics.py
import os
import pandas as pd
import streamlit as st
import plotly.express as px
import sys

# Setup sys.path to allow importing from run.py and backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from session_manager import get_active_session, get_session_paths

st.set_page_config(page_title="Cardiology Data Analytics", page_icon="📊", layout="wide")

st.title("📊 Patient ECG Signal & Model Analytics")
st.markdown("---")

active_session_name = get_active_session()
paths = get_session_paths(active_session_name)
pred_csv_path = paths["predictions"]
filtered_csv_path = paths["filtered"]

st.info(f"📊 Displaying analytics charts for Selected Patient Session: **{active_session_name}**")

if not os.path.exists(pred_csv_path) or os.path.getsize(pred_csv_path) < 50:
    st.warning("⚠️ No prediction logs found for this session. Please navigate to the Live Monitor tab and complete a 60-second telemetry session first.")
else:
    try:
        pred_df = pd.read_csv(pred_csv_path)
        filt_df = pd.read_csv(filtered_csv_path)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🍰 AI Diagnosis Distribution (This Session)")
            st.caption("Distribution of cardiac rhythm classifications during the 60-second session.")
            if len(pred_df) > 0:
                class_counts = pred_df["prediction_label"].value_counts().reset_index()
                class_counts.columns = ["Diagnosis", "Windows Count"]
                
                # Interactive Plotly Pie Chart
                fig_pie = px.pie(
                    class_counts, 
                    names="Diagnosis", 
                    values="Windows Count",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    hole=0.4
                )
                fig_pie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#ffffff',
                    margin=dict(t=10, b=10, l=10, r=10),
                    showlegend=True
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No predictions to display yet.")
                
        with col2:
            st.subheader("📉 Patient ECG Amplitude Distribution")
            st.caption("Frequency distribution of signal amplitudes (ADC values) in this session's recording.")
            if len(filt_df) > 0:
                # Sub-sample to speed up loading
                hist_data = filt_df["filtered_ecg"].tail(3000)
                fig_hist = px.histogram(
                    hist_data,
                    x=hist_data,
                    nbins=45,
                    labels={"x": "ADC Amplitude (Filtered)"},
                    color_discrete_sequence=['#ef4444']
                )
                fig_hist.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#ffffff',
                    xaxis_title="Patient ECG Amplitude",
                    yaxis_title="Samples Count",
                    margin=dict(t=10, b=10, l=10, r=10)
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("No ECG data logged yet.")
                
        st.markdown("---")
        
        # Session metrics analysis
        st.subheader("📊 Session Classification Statistics")
        st.markdown("Detailed breakdown of classification frequencies in the patient's recording:")
        
        c_counts = pred_df["prediction_label"].value_counts().reset_index()
        c_counts.columns = ["Cardiac Category", "Windows (Seconds) Count"]
        c_counts["Percentage"] = (c_counts["Windows (Seconds) Count"] / len(pred_df) * 100).round(1)
        c_counts["Percentage"] = c_counts["Percentage"].astype(str) + "%"
        
        st.dataframe(c_counts, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error loading analytics datasets: {e}")
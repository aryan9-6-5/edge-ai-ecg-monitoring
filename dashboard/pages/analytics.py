# dashboard/pages/analytics.py
import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import sys

# Setup sys.path to allow importing from run.py and backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from session_manager import get_active_session, get_session_paths
from patient_manager import get_active_patient_id, get_patient

st.set_page_config(page_title="Cardiology Data Analytics", page_icon="📊", layout="wide")

# Custom CSS for clinical layout
st.markdown("""
    <style>
    .analytics-card {
        background-color: #121620;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #1e2638;
        margin-bottom: 20px;
    }
    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        color: #ef4444;
    }
    .report-frame {
        background-color: #ffffff;
        color: #000000;
        padding: 40px;
        border-radius: 8px;
        border: 1px solid #d1d5db;
        font-family: 'Courier New', Courier, monospace;
        line-height: 1.5;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Clinical ECG Signal & Anomaly Analytics")
st.markdown("---")

active_session_name = get_active_session()
paths = get_session_paths(active_session_name)
pred_csv_path = paths["predictions"]
filtered_csv_path = paths["filtered"]

patient_id = get_active_patient_id()
patient = get_patient(patient_id)

st.info(f"📋 Selected Patient: **{patient['name']}** ({patient['age']}y / {patient['gender']}) | Active Session: **{active_session_name}**")

def read_json_file(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return None

report_data = read_json_file(paths["report"])

if not os.path.exists(pred_csv_path) or os.path.getsize(pred_csv_path) < 50:
    st.warning("⚠️ No prediction logs found for this session. Please navigate to the Live Monitor tab and complete a telemetry session first.")
else:
    try:
        pred_df = pd.read_csv(pred_csv_path)
        filt_df = pd.read_csv(filtered_csv_path)
        
        # Create Tabs
        tab1, tab2, tab3 = st.tabs(["📈 Waveform & Anomaly Distribution", "🧬 Clinical Metrics & Features", "📋 Printable Clinical Report"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🍰 AI Diagnosis Distribution")
                st.caption("Breakdown of cardiac rhythm classification windows.")
                if len(pred_df) > 0:
                    class_counts = pred_df["prediction_label"].value_counts().reset_index()
                    class_counts.columns = ["Diagnosis", "Windows Count"]
                    
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
                        margin=dict(t=10, b=10, l=10, r=10)
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("No predictions to display yet.")
                    
            with col2:
                st.subheader("📉 ECG Signal Amplitude Distribution")
                st.caption("Frequency distribution of filtered ADC values in this session's recording.")
                if len(filt_df) > 0:
                    hist_data = filt_df["filtered_ecg"].tail(3000)
                    fig_hist = px.histogram(
                        hist_data,
                        x=hist_data,
                        nbins=40,
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
            st.subheader("📊 Session Classification Statistics Table")
            c_counts = pred_df["prediction_label"].value_counts().reset_index()
            c_counts.columns = ["Cardiac Category", "Windows Count"]
            c_counts["Percentage"] = (c_counts["Windows Count"] / len(pred_df) * 100).round(1)
            c_counts["Percentage"] = c_counts["Percentage"].astype(str) + "%"
            st.dataframe(c_counts, use_container_width=True)
            
        with tab2:
            st.subheader("🧬 Statistical Feature Distrubition Moments")
            st.caption("Box plots of extracted feature moments across all sliding windows in this session.")
            
            # Extract features on the fly for box plot visualization
            from feature_extractor import compute_signal_features
            
            feats_list = []
            filtered_ecg = filt_df["filtered_ecg"].values
            window_size = 250
            step = 125
            
            for i in range(0, len(filtered_ecg) - window_size + 1, step):
                w = filtered_ecg[i : i + window_size]
                feats_list.append(compute_signal_features(w, normalize=True))
                
            if feats_list:
                feats_df = pd.DataFrame(feats_list)
                
                # Melt feature df for plotting
                melted_df = pd.melt(feats_df, value_vars=["mean", "std", "skew", "kurt"], var_name="Moment", value_name="Value")
                
                fig_box = px.box(
                    melted_df,
                    x="Moment",
                    y="Value",
                    color="Moment",
                    color_discrete_sequence=px.colors.qualitative.Safe,
                    points="outliers"
                )
                fig_box.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(18,22,32,0.5)',
                    font_color='#ffffff',
                    xaxis_title="Statistical Moment Feature",
                    yaxis_title="Scale-Normalized Value",
                    margin=dict(t=10, b=10, l=10, r=10)
                )
                st.plotly_chart(fig_box, use_container_width=True)
                
            # HRV Plots
            if report_data and "rr_intervals" in report_data and report_data["rr_intervals"]:
                st.markdown("---")
                col_hrv_text, col_hrv_plot = st.columns([1, 2])
                with col_hrv_text:
                    st.subheader("⏱️ HRV & R-R Interval Analysis")
                    st.markdown(f"""
                        - **Average Heart Rate**: `{report_data.get('bpm', 0.0)}` BPM
                        - **SDNN (Autonomic Tone)**: `{report_data.get('sdnn', 0.0)}` ms
                          *Clinical range: 30-100 ms. Low values indicate sympathetic dominance/stress.*
                        - **RMSSD (Vagal Activity)**: `{report_data.get('rmssd', 0.0)}` ms
                          *Reflects parasympathetic regulation of the heart.*
                        - **Signal Quality Index (SQI)**: `{report_data.get('sqi', 100.0)}` %
                    """)
                with col_hrv_plot:
                    rr_intervals = report_data["rr_intervals"]
                    fig_rr = px.line(
                        x=range(len(rr_intervals)),
                        y=rr_intervals,
                        labels={"x": "Beat Interval Index", "y": "R-R Interval (ms)"},
                        title="R-R Intervals Timeline (ms)",
                        color_discrete_sequence=['#a78bfa']
                    )
                    fig_rr.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(18,22,32,0.5)',
                        font_color='#ffffff',
                        margin=dict(t=30, b=10, l=10, r=10),
                        height=250
                    )
                    st.plotly_chart(fig_rr, use_container_width=True)
            else:
                st.info("HRV metrics not fully logged yet. Run a complete telemetry acquisition session to compute R-peaks.")
                
        with tab3:
            st.subheader("📋 Clinical Summary PDF/HTML Report Generator")
            st.caption("View and download a clinical summary report for the selected patient.")
            
            # Extract metadata
            bpm_val = report_data.get("bpm", 0.0) if report_data else "N/A"
            sdnn_val = report_data.get("sdnn", 0.0) if report_data else "N/A"
            rmssd_val = report_data.get("rmssd", 0.0) if report_data else "N/A"
            sqi_val = report_data.get("sqi", 100.0) if report_data else "N/A"
            risk_val = report_data.get("overall_risk", "LOW") if report_data else "N/A"
            duration_val = report_data.get("duration", 0.0) if report_data else "N/A"
            abnormal_count = report_data.get("abnormal_windows", 0) if report_data else 0
            
            # Format classification details for text report
            counts_summary = ""
            for idx, row in c_counts.iterrows():
                counts_summary += f"{row['Cardiac Category']:<35} : {row['Windows Count']} windows ({row['Percentage']})\n"
                
            report_text = f"""================================================================================
                    EDGE AI CARDIOLOGY PATIENT DIAGNOSTIC SUMMARY
================================================================================
PATIENT METADATA:
--------------------------------------------------------------------------------
Patient ID           : {patient['id']}
Patient Name         : {patient['name']}
Age / Gender         : {patient['age']} yrs / {patient['gender']}
Baseline HR          : {patient['baseline_hr']} BPM
History of Arrhythmia: {"YES" if patient['arrhythmia_history'] else "NO"}
Clinical Notes       : {patient['notes']}

TELEMETRY CONFIGURATION:
--------------------------------------------------------------------------------
Session Identifier   : {active_session_name}
Acquisition Duration : {duration_val} seconds
Sampling Rate        : 125 Hz
Total Samples Logged : {len(filt_df)} points
AI Classifier        : Random Forest (trained on MIT-BIH clinical data)

CLINICAL METRICS & ANOMALIES:
--------------------------------------------------------------------------------
Average Heart Rate   : {bpm_val} BPM
Heart Rate Variabilty: SDNN = {sdnn_val} ms | RMSSD = {rmssd_val} ms
Signal Quality Index : {sqi_val} %
Arrhythmic Anomalies : {abnormal_count} abnormal windows detected
Overall Risk Level   : {risk_val}

DIAGNOSIS CLASSIFICATION COUNTS:
--------------------------------------------------------------------------------
{counts_summary}
CLINICAL ASSESSMENT / SIGNATURE:
--------------------------------------------------------------------------------
Diagnosis: Patient presents with {risk_val} risk. {"Ventricular ectopic activity observed (PVCs)." if abnormal_count > 0 else "Rhythm falls within standard limits."}

Supervising Physician Signature: ____________________________________
Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
================================================================================
"""
            # Display report inside code frame
            st.text_area("Report Preview", report_text, height=450)
            
            # Create a HTML printable version for downloading
            html_report = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: 'Helvetica Neue', Arial, sans-serif; color: #333; margin: 40px; line-height: 1.6; }}
                    .header {{ border-bottom: 2px solid #ef4444; padding-bottom: 10px; margin-bottom: 30px; text-align: center; }}
                    .section-title {{ font-size: 1.2rem; color: #ef4444; border-bottom: 1px solid #e5e7eb; padding-bottom: 5px; margin-top: 25px; font-weight: bold; }}
                    .meta-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                    .meta-table td {{ padding: 8px 0; }}
                    .meta-table td:nth-child(odd) {{ font-weight: bold; color: #4b5563; width: 25%; }}
                    .footer {{ margin-top: 50px; border-top: 1px solid #e5e7eb; padding-top: 20px; font-size: 0.85rem; color: #6b7280; text-align: center; }}
                    .risk-badge {{ display: inline-block; padding: 4px 12px; border-radius: 4px; font-weight: bold; color: white; }}
                    .risk-HIGH {{ background-color: #ef4444; }}
                    .risk-MEDIUM {{ background-color: #f59e0b; }}
                    .risk-LOW {{ background-color: #10b981; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h2>🏥 EDGE AI PATIENT MONITORING CLINICAL REPORT</h2>
                    <p style="color:#6b7280; font-size:0.9rem;">Edge Diagnostics Platform | Session: {active_session_name}</p>
                </div>
                
                <div class="section-title">👤 Patient Profile</div>
                <table class="meta-table">
                    <tr>
                        <td>Patient ID:</td><td>{patient['id']}</td>
                        <td>Patient Name:</td><td>{patient['name']}</td>
                    </tr>
                    <tr>
                        <td>Age / Gender:</td><td>{patient['age']} yrs / {patient['gender']}</td>
                        <td>Baseline HR:</td><td>{patient['baseline_hr']} BPM</td>
                    </tr>
                    <tr>
                        <td>History:</td><td colspan="3">{"Yes" if patient['arrhythmia_history'] else "No"}</td>
                    </tr>
                    <tr>
                        <td>Clinical Notes:</td><td colspan="3">{patient['notes']}</td>
                    </tr>
                </table>
                
                <div class="section-title">⏱️ Acquisition Parameters</div>
                <table class="meta-table">
                    <tr>
                        <td>Duration:</td><td>{duration_val} seconds</td>
                        <td>Sampling Rate:</td><td>125 Hz</td>
                    </tr>
                    <tr>
                        <td>Total Samples:</td><td>{len(filt_df)} points</td>
                        <td>Model Type:</td><td>Random Forest Classifier</td>
                    </tr>
                </table>
                
                <div class="section-title">🩺 Clinical Signal & Anomaly Metrics</div>
                <table class="meta-table">
                    <tr>
                        <td>Average Heart Rate:</td><td>{bpm_val} BPM</td>
                        <td>Overall Risk:</td><td><span class="risk-badge risk-{risk_val}">{risk_val}</span></td>
                    </tr>
                    <tr>
                        <td>HRV (SDNN):</td><td>{sdnn_val} ms</td>
                        <td>HRV (RMSSD):</td><td>{rmssd_val} ms</td>
                    </tr>
                    <tr>
                        <td>Signal Quality:</td><td>{sqi_val} %</td>
                        <td>Ectopic Beats Count:</td><td>{abnormal_count} abnormal windows</td>
                    </tr>
                </table>
                
                <div class="section-title">📊 Cardiac rhythm distributions</div>
                <pre style="font-family: inherit; font-size: 0.95rem; margin-top: 15px; padding-left: 10px;">
{counts_summary}
                </pre>
                
                <div class="section-title">✍️ Diagnostic Assessment</div>
                <p>
                    Patient cardiac rhythm evaluated locally on the edge. Classification risk index is <b>{risk_val}</b>. 
                    {"Ventricular ectopic activity (PVCs) was observed during telemetry recording. Follow-up diagnostic monitoring recommended." if abnormal_count > 0 else "ECG waveform exhibits stable sinus rhythm. Normal autonomic tone."}
                </p>
                
                <br/><br/>
                <table style="width:100%; border:0; margin-top: 30px;">
                    <tr>
                        <td style="width:50%; border:0;">Supervising Physician Signature: _______________________</td>
                        <td style="width:50%; border:0; text-align:right;">Date: {time.strftime('%Y-%m-%d')}</td>
                    </tr>
                </table>
                
                <div class="footer">
                    Edge AI Patient Monitor | Academic Research Project Summary | Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </body>
            </html>
            """
            st.download_button(
                label="📥 Download Printable HTML Clinical Report",
                data=html_report,
                file_name=f"clinical_report_{patient['id']}_{active_session_name}.html",
                mime="text/html",
                use_container_width=True
            )
            
    except Exception as e:
        st.error(f"Error loading analytics datasets: {e}")
        import traceback
        traceback.print_exc()
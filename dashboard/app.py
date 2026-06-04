# dashboard/app.py
import streamlit as st

st.set_page_config(
    page_title="Edge AI Patient Monitor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stMetric {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #374151;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .header-text {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        color: #ef4444;
    }
    .card {
        background-color: #111827;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #1f2937;
        margin-bottom: 20px;
    }
    .highlight {
        color: #10b981;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Main Title & Subtitle
st.title("🏥 Edge AI Patient Monitoring System")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🩺 Project Overview")
    st.markdown("""
    Welcome to the **Edge AI Patient Monitoring System**. This platform represents the transition from a traditional IoT architecture to an **Edge AI pipeline** for real-time cardiac health analysis. 
    
    By executing signal preprocessing, statistical feature extraction, and machine learning classification locally on the edge node, the system minimizes telemetry latency, eliminates cloud dependencies, and provides instant diagnostic warnings.
    """)
    
    st.markdown("#### ⚙️ Edge AI System Architecture & Flow")
    
    # Styled HTML cards
    st.markdown("""
    <div style="display: flex; flex-direction: column; gap: 12px; margin-top: 10px; margin-bottom: 20px;">
      <div style="background-color: #1f2937; padding: 15px; border-radius: 8px; border-left: 5px solid #3b82f6;">
        <strong style="color: #60a5fa; font-size: 1.0rem;">📥 1. Signal Acquisition</strong><br/>
        <span style="font-size: 0.88rem; color: #d1d5db;">Continuous high-fidelity ECG streaming from physical hardware (ESP32 via COM serial) or real-time simulation buffers.</span>
      </div>
      <div style="background-color: #1f2937; padding: 15px; border-radius: 8px; border-left: 5px solid #10b981;">
        <strong style="color: #34d399; font-size: 1.0rem;">🧹 2. Signal Preprocessing</strong><br/>
        <span style="font-size: 0.88rem; color: #d1d5db;">Artifact filtering (removing lead-off float values) and 5-Point Simple Moving Average (SMA) smoothing filter on live data.</span>
      </div>
      <div style="background-color: #1f2937; padding: 15px; border-radius: 8px; border-left: 5px solid #f59e0b;">
        <strong style="color: #fbbf24; font-size: 1.0rem;">📊 3. Feature Extraction</strong><br/>
        <span style="font-size: 0.88rem; color: #d1d5db;">Real-time segmentation and estimation of 6 statistical moments: Mean, Standard Deviation, Max, Min, Skewness, and Kurtosis.</span>
      </div>
      <div style="background-color: #1f2937; padding: 15px; border-radius: 8px; border-left: 5px solid #8b5cf6;">
        <strong style="color: #a78bfa; font-size: 1.0rem;">🧠 4. Edge AI Classifier</strong><br/>
        <span style="font-size: 0.88rem; color: #d1d5db;">Local RandomForest Classifier (trained on clinical datasets) mapping the extracted feature vector to cardiac rhythm states.</span>
      </div>
      <div style="background-color: #1f2937; padding: 15px; border-radius: 8px; border-left: 5px solid #ef4444;">
        <strong style="color: #f87171; font-size: 1.0rem;">🚨 5. Risk Alert Engine</strong><br/>
        <span style="font-size: 0.88rem; color: #d1d5db;">Continuous diagnostic classification logging, terminal output streaming, and visual/acoustic UI alerts.</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 🚀 Features Included
    1. **Live ECG Monitoring**: Continuous visual plotting of both raw and filtered signals.
    2. **Real-time Feature Tracking**: Extraction of key statistical features (Mean, Std Dev, Max, Min, Skewness, Kurtosis) every second.
    3. **Edge AI Predictions**: Local inference using a Random Forest model. The system supports standard **MIT-BIH formatted ECG datasets** for training and evaluation.
    4. **Smart Alerts**: Visual warning banners and instant acoustic alerts during PVC or arrhythmia detection.
    5. **Deep Analytics**: Signal distribution histograms, classification counts, and model feature importances.
    """)

with col2:
    st.markdown("""
    <div class="card">
        <h4 style="color: #ef4444; margin-top:0;">📊 System Configuration</h4>
        <p><b>Status:</b> <span class="highlight">Active & Ready</span></p>
        <p><b>Sampling Rate:</b> 125 Hz</p>
        <p><b>Model Type:</b> Random Forest Classifier</p>
        <p><b>Features Used:</b> 6 statistical moments</p>
        <p><b>Classification Categories:</b></p>
        <ul>
            <li>Normal (N)</li>
            <li>Premature Ventricular Contraction (PVC/V)</li>
            <li>Supraventricular Premature (S)</li>
            <li>Fusion Beat (F)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("👈 Use the Sidebar navigation to access the Live ECG Monitor, past Alerts logs, and Analytics tabs.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #6b7280; font-size: 0.8rem;'>Edge AI Patient Monitor | Academic Research Project</p>", unsafe_allow_html=True)

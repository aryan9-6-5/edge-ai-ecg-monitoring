# dashboard/pages/model_info.py
import os
import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.metrics import confusion_matrix
import sys

# Setup sys.path to allow importing from backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from feature_extractor import compute_signal_features

st.set_page_config(page_title="Classifier Specifications", page_icon="🧠", layout="wide")

# Custom Styles
st.markdown("""
    <style>
    .section-card {
        background-color: #121620;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #1e2638;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Edge AI Classifier Specifications & Sandbox")
st.markdown("---")

MODEL_PATH = "models/ecg_model.pkl"
MITBIH_DATA_PATH = "data/mitbih/mitbih.csv"
MITBIH_PREDS_PATH = "data/processed/mitbih_predictions.csv"

LABEL_MAP_DESC = {
    0: "Normal (N)", 
    1: "PVC (V)", 
    2: "Supraventricular (S)", 
    3: "Fusion (F)", 
    4: "Unclassifiable (Q)"
}

if not os.path.exists(MODEL_PATH):
    st.warning("Classifier model file not found in models/ecg_model.pkl. Please run the background daemon to calibrate.")
else:
    try:
        # Load the Random Forest model
        model = joblib.load(MODEL_PATH)
        
        # Get creation timestamp
        m_time = os.path.getmtime(MODEL_PATH)
        m_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(m_time))
        
        # Load calibration data info
        ds_shape = "Unknown (mitbih.csv missing)"
        ds_count_str = "N/A"
        if os.path.exists(MITBIH_DATA_PATH):
            try:
                mit_df = pd.read_csv(MITBIH_DATA_PATH)
                if "label" not in mit_df.columns:
                    mit_df.columns = [f"f_{i}" for i in range(mit_df.shape[1] - 1)] + ["label"]
                ds_shape = f"{mit_df.shape[0]} beats × {mit_df.shape[1] - 1} points"
                counts_dict = mit_df["label"].value_counts().to_dict()
                ds_count_str = ", ".join([f"{LABEL_MAP_DESC.get(int(k), f'Class {k}')}: {v}" for k, v in counts_dict.items()])
            except Exception:
                pass
        
        # 1. Top Specifications Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Classifier Algorithm", "Random Forest Classifier")
        with col2:
            st.metric("Decision Trees Count", model.n_estimators)
        with col3:
            st.metric("Moment Features Checked", model.n_features_in_)
            
        st.markdown("---")
        
        col_l, col_r = st.columns([1, 1])
        
        # 2. Left Column - Feature Importance Bar Chart
        with col_l:
            st.subheader("📊 Trained Feature Importances")
            st.caption("Contribution of each of the 6 statistical moments in classification decisions.")
            
            feats = ["mean", "std", "max", "min", "skew", "kurt"]
            importances = model.feature_importances_
            
            imp_df = pd.DataFrame({
                "Feature": [f.upper() for f in feats],
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)
            
            fig_bar = px.bar(
                imp_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="reds"
            )
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#ffffff',
                yaxis={'categoryorder':'total ascending'},
                margin=dict(t=10, b=10, l=10, r=10),
                height=280
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        # 3. Right Column - Confusion Matrix Heatmap
        with col_r:
            st.subheader("🔥 Calibration Confusion Matrix")
            st.caption("Verification accuracy matrix on the synthetic calibration dataset.")
            
            if os.path.exists(MITBIH_PREDS_PATH):
                try:
                    preds_df = pd.read_csv(MITBIH_PREDS_PATH)
                    y_true = preds_df["true_label"].values.astype(int)
                    y_pred = preds_df["prediction"].values.astype(int)
                    
                    # Compute CM
                    labels = sorted(list(set(y_true)))
                    cm = confusion_matrix(y_true, y_pred, labels=labels)
                    
                    label_names = [LABEL_MAP_DESC.get(l, f"Class {l}") for l in labels]
                    
                    fig_cm = px.imshow(
                        cm,
                        labels=dict(x="Predicted Beat", y="True Beat", color="Count"),
                        x=label_names,
                        y=label_names,
                        text_auto=True,
                        color_continuous_scale="reds"
                    )
                    fig_cm.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#ffffff',
                        margin=dict(t=10, b=10, l=10, r=10),
                        height=280
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
                except Exception as cme:
                    st.write(f"Could not compute matrix heatmap: {cme}")
            else:
                st.info("Predictions file missing. Run telemetry background daemon to compile performance statistics.")
                
        # 4. Model Training Metadata
        st.markdown("---")
        st.subheader("⚙️ Calibration Training Metadata")
        st.markdown(f"""
        - **Trained Model File**: `models/ecg_model.pkl`
        - **Training Completed At**: `{m_time_str}`
        - **Clinical Dataset Calibration Size**: `{ds_shape}`
        - **Class Distribution in Dataset**: `{ds_count_str}`
        """)
        
        st.info("💡 The Random Forest classifier extracts statistical properties of the waveform (moments) rather than absolute temporal points, making it highly robust against minor frequency or lead placement variations.")
        
        # ------------------------------------------------------------------
        # 5. INTERACTIVE SANDBOX PLAYGROUND
        # ------------------------------------------------------------------
        st.markdown("---")
        st.subheader("🧬 Interactive Model Sandbox Testing")
        st.caption("Select a beat class category, load a random calibration heartbeat, and test the model predictions in real-time.")
        
        sandbox_col1, sandbox_col2 = st.columns([1, 1])
        
        with sandbox_col1:
            beat_choice = st.selectbox(
                "Select Heartbeat Class to Sample:",
                ["Normal Beat (Class 0)", "PVC Anomaly Beat (Class 1)", "Supraventricular Beat (Class 2)", "Fusion Beat (Class 3)"]
            )
            target_lbl = int(beat_choice.split("Class ")[1].replace(")", ""))
            
            if st.button("🎲 Load and Test Random Beat", use_container_width=True):
                if os.path.exists(MITBIH_DATA_PATH):
                    try:
                        df_mit = pd.read_csv(MITBIH_DATA_PATH)
                        if "label" not in df_mit.columns:
                            df_mit.columns = [f"f_{i}" for i in range(df_mit.shape[1] - 1)] + ["label"]
                            
                        # Filter to class
                        df_class = df_mit[df_mit["label"] == target_lbl]
                        if len(df_class) > 0:
                            # Choose random row
                            random_row = df_class.sample(n=1, random_state=int(time.time()) % 1000).iloc[0]
                            signal = random_row.drop("label").values.astype(float)
                            
                            st.session_state.sandbox_signal = signal
                            st.session_state.sandbox_label = target_lbl
                            st.session_state.sandbox_loaded = True
                        else:
                            st.error(f"No samples of Class {target_lbl} found in dataset.")
                    except Exception as le:
                        st.error(f"Error loading beat: {le}")
                else:
                    st.error("MIT-BIH Calibration dataset file missing.")
                    
            if 'sandbox_loaded' in st.session_state and st.session_state.sandbox_loaded:
                # Plot the single beat waveform
                signal = st.session_state.sandbox_signal
                fig_beat = px.line(
                    y=signal,
                    labels={"x": "Sample Point", "y": "Normalized Amplitude"},
                    title=f"Sample Waveform - {LABEL_MAP_DESC.get(st.session_state.sandbox_label)}"
                )
                fig_beat.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(18,22,32,0.5)',
                    font_color='#ffffff',
                    height=240,
                    margin=dict(t=30, b=10, l=10, r=10)
                )
                st.plotly_chart(fig_beat, use_container_width=True)
                
        with sandbox_col2:
            if 'sandbox_loaded' in st.session_state and st.session_state.sandbox_loaded:
                signal = st.session_state.sandbox_signal
                
                # Extract features (Since the beat is already in [0,1], compute_signal_features works perfectly)
                feats = compute_signal_features(signal, normalize=False)
                feats_df = pd.DataFrame([feats])
                expected_cols = ["mean", "std", "max", "min", "skew", "kurt"]
                feats_df = feats_df[expected_cols]
                
                # Predict probability
                probs = model.predict_proba(feats_df)[0]
                pred_class = int(model.predict(feats_df)[0])
                
                # Display metrics
                st.markdown("#### 📊 Real-time Feature Extraction Results:")
                col_f1, col_f2, col_f3 = st.columns(3)
                col_f1.metric("MEAN", f"{feats['mean']:.3f}")
                col_f2.metric("STD DEV", f"{feats['std']:.3f}")
                col_f3.metric("MAX / MIN", f"{feats['max']:.2f} / {feats['min']:.2f}")
                
                col_f4, col_f5 = st.columns(2)
                col_f4.metric("SKEWNESS", f"{feats['skew']:.3f}")
                col_f5.metric("KURTOSIS", f"{feats['kurt']:.3f}")
                
                # Display predicted class
                prediction_text = LABEL_MAP_DESC.get(pred_class, f"Class {pred_class}")
                is_correct = (pred_class == st.session_state.sandbox_label)
                result_pill = '<span class="status-pill status-ok">🟢 Match (Correct)</span>' if is_correct else '<span class="status-pill status-error">🔴 Mismatch</span>'
                
                st.markdown(f"""
                    <div style="background-color:#161b26; border: 1px solid #283046; border-radius:8px; padding:15px; margin-top:10px;">
                        <b>AI Model Prediction:</b> <span style="font-size:1.1rem; color:#ef4444; font-weight:bold;">{prediction_text}</span> &nbsp;&nbsp; {result_pill}
                    </div>
                """, unsafe_allow_html=True)
                
                # Draw probability bar chart
                prob_data = pd.DataFrame({
                    "Cardiac Rhythm": [LABEL_MAP_DESC.get(c) for c in model.classes_],
                    "Probability": probs
                })
                
                fig_probs = px.bar(
                    prob_data,
                    x="Probability",
                    y="Cardiac Rhythm",
                    orientation="h",
                    color="Probability",
                    color_continuous_scale="greens",
                    range_x=[0, 1]
                )
                fig_probs.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#ffffff',
                    margin=dict(t=10, b=10, l=10, r=10),
                    height=200,
                    showlegend=False
                )
                st.plotly_chart(fig_probs, use_container_width=True)
            else:
                st.info("Click 'Load and Test Random Beat' to activate the Sandbox playground.")
                
    except Exception as e:
        st.error(f"Error loading model details: {e}")
        import traceback
        traceback.print_exc()

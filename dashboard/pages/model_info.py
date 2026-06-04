# dashboard/pages/model_info.py
import os
import time
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib

st.set_page_config(page_title="Classifier Specifications", page_icon="🧠", layout="wide")

st.title("🧠 Edge AI Classifier Specifications")
st.markdown("---")

MODEL_PATH = "models/ecg_model.pkl"
MITBIH_DATA_PATH = "data/mitbih/mitbih.csv"

if not os.path.exists(MODEL_PATH):
    st.warning("Classifier model file not found in models/ecg_model.pkl. Please run train_model.py first.")
else:
    try:
        # Load the Random Forest model
        model = joblib.load(MODEL_PATH)
        
        # Get creation timestamp
        m_time = os.path.getmtime(MODEL_PATH)
        m_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(m_time))
        
        # Get dataset shape if available
        ds_shape = "Unknown (mitbih.csv missing)"
        ds_count_str = "N/A"
        if os.path.exists(MITBIH_DATA_PATH):
            try:
                mit_df = pd.read_csv(MITBIH_DATA_PATH)
                ds_shape = f"{mit_df.shape[0]} rows × {mit_df.shape[1]} columns"
                ds_count_str = str(mit_df["label"].value_counts().to_dict())
            except Exception:
                pass
        
        # Top Specifications Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Classifier Algorithm", "Random Forest")
        with col2:
            st.metric("Number of Decision Trees", model.n_estimators)
        with col3:
            st.metric("Number of Input Features", model.n_features_in_)
            
        st.markdown("---")
        
        col_l, col_r = st.columns([1, 1])
        
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
                margin=dict(t=10, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col_r:
            st.subheader("⚙️ Training Metadata")
            st.markdown(f"""
            - **Trained Model File**: `models/ecg_model.pkl`
            - **Training Completed At**: `{m_time_str}`
            - **Clinical Training Dataset**: `MIT-BIH Arrhythmia Dataset`
            - **Training Shape**: `{ds_shape}`
            - **Class Counts in Training Dataset**:
              - Normal (Class 0): Normal rhythm beats.
              - PVC (Class 1): Premature Ventricular Contractions.
              - Supraventricular (Class 2): Supraventricular ectopic beats.
              - Fusion (Class 3): Fusion beats.
            - **Training Distribution**: `{ds_count_str}`
            """)
            
            st.info("💡 The Random Forest classifier extracts statistical properties of the waveform (moments) rather than absolute temporal points, making it highly robust against minor frequency or lead placement variations.")
            
    except Exception as e:
        st.error(f"Error loading model details: {e}")

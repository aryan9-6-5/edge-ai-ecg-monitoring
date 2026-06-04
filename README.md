# Edge AI Patient Monitoring System

An academic-grade, end-to-end Edge AI Patient Monitoring System that interfaces with an AD8232 ECG sensor and an ESP32 microcontroller, performs local preprocessing and statistical feature extraction, and runs classification using a lightweight Random Forest model to perform live patient-specific arrhythmia diagnosis and statistics in real time.

---

## 📐 System Architecture

### 1. Training Phase (Offline / One-Time)
The classifier is trained on a high-quality clinical dataset (MIT-BIH Arrhythmia Database) to learn cardiac rhythm features.
```text
  +------------------+      +--------------------+      +--------------------+      +--------------------+
  |  MIT-BIH Dataset  | ---> | Feature Extraction | ---> |   Random Forest    | ---> | Saved Model File   |
  |  (187 amplitudes) |      | (6 stats metrics)  |      |   Classification   |      |  (ecg_model.pkl)   |
  +------------------+      +--------------------+      +--------------------+      +--------------------+
```

### 2. Runtime / Demonstration Phase (Live Session AI Pipeline)
During live operation, the system records the patient's ECG for exactly 60 seconds (7500 samples), filters it in real time, and runs batch windowed feature extraction and RandomForest inference on the patient's signal at the end of the session.
```text
  AD8232 ECG Pin
       ↓
     ESP32
       ↓
  60-Second Recording Session (7500 samples @ 125Hz)
       ↓
  Saved to data/sessions/session_XXX/ecglive.csv
       ↓
  Moving Average Preprocessing -> ecg_filtered.csv
       ↓
  Window Generation (size=250 samples, step=125 samples)
       ↓
  Feature Extraction (6 statistical features per window)
       ↓
  Random Forest Inference (models/ecg_model.pkl)
       ↓
  Live Patient Diagnostics, Alerts, Analytics & Session Report
```

---

## 🛠️ Project Structure

```text
edge/
│
├── backend/
│   ├── alert_engine.py         # Reusable risk classifier evaluator
│   ├── feature_extractor.py    # Computes 6 statistical features (mean, std, max, min, skew, kurt)
│   ├── generate_mitbih_data.py # Synthesizes calibration MIT-BIH dataset for training
│   ├── predict.py              # Performs RandomForest inference on features
│   ├── preprocess.py           # Smooths signal using a rolling moving average filter
│   ├── session_manager.py      # Manages session folders under data/sessions/
│   └── train_model.py          # Extracts features and trains the Random Forest model
│
├── dashboard/
│   ├── app.py                  # Main landing page for Streamlit
│   └── pages/
│       ├── alerts.py           # Session abnormal alerts history
│       ├── analytics.py        # Patient-specific beat pie charts and amplitude distribution
│       ├── live_monitor.py     # Live ECG graphs, countdowns, and session summary reports
│       └── model_info.py       # Serialized model specs and feature importances bar chart
│
├── data/
│   ├── mitbih/                 # Folder containing training dataset (mitbih.csv)
│   └── sessions/               # Stores session-specific data folders (git-ignored)
│
├── models/
│   └── ecg_model.pkl           # Trained Random Forest classifier binary (git-ignored)
│
├── README.md                   # Comprehensive system documentation
└── run.py                      # Single-command pipeline and dashboard launcher
```

---

## 🚀 Setup & Execution Guide

### 📦 Prerequisites
Install the required packages using the virtual environment:
```bash
pip install -r ../requirements.txt
```

### 🏋️ Phase 1: Training (One-Time Setup)

#### 1. Place the MIT-BIH Dataset
Place the standard **MIT-BIH Arrhythmia Dataset** (in CSV format) inside the data directory:
*   File Location: `data/mitbih/mitbih.csv`
*   *Note: If you want to test before downloading, running the launcher automatically generates a synthesized clinical dataset for training.*

#### 2. Execute Training Pipeline
Train the classifier on the statistical features extracted from the dataset:
```bash
python backend/train_model.py
```
This prints the validation accuracy, generates a classification report, and outputs the serialized classifier to `models/ecg_model.pkl`.

---

### 💻 Phase 2: Demonstration (Primary Interface)

Launch the entire pipeline with a **single command**:
```bash
python run.py
```

This single launcher will automatically:
1.  Verify the trained classification model exists (trains it automatically if missing).
2.  Attempt to connect to the ESP32 via serial port (default: `COM11`, `115200` baud).
    *   **Auto-Simulation Fallback**: If the ESP32 is not connected, the script automatically launches in **Simulation Mode**, streaming highly realistic synthetic ECG waveforms at `125Hz` (including random arrhythmias and noise).
3.  Establish a new recording session folder (e.g. `data/sessions/session_001/`).
4.  Log raw telemetry, run real-time noise preprocessing, extract rolling window statistical features at 60 seconds, run AI predictions, and generate reports.
5.  Launch the Streamlit web dashboard and open it in your browser automatically at `http://localhost:8501`.

---

## 🩺 Physiological Features Extracted
Rather than evaluating single amplitude values, the Edge AI classifier extracts 6 statistical moments representing the ECG waveform morphology:
*   **Mean**: The average electrical baseline.
*   **Standard Deviation**: Signal dispersion, capturing the high-frequency polarization amplitudes.
*   **Maximum / Minimum**: Peaks representing ventricular depolarization (QRS complex bounds).
*   **Skewness**: Asymmetry of the wave. Abnormal ventricular beats (PVCs) have deep, asymmetrical wide complexes, drastically shifting skewness values.
*   **Kurtosis**: Shape peakiness, helping separate sharp R-peaks from smooth, wide ectopic PVC configurations.

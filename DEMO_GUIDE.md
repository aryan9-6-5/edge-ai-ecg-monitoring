# Edge AI Patient Monitor — Demonstration Guide

This guide explains how to configure and run the Edge AI Patient Monitor (Version 1) in two scenarios: **With a Physical Hardware Sensor** and **Without a Sensor (Simulation Mode)**.

---

## 🚀 Quick Start (Launching the Platform)

To start the system:
1. Double-click the launcher script: **[run.bat](file:///d:/iot/run.bat)**.
2. The launcher will automatically:
   - Verify that the machine learning models and calibration data are ready.
   - Boot up the background telemetry acquisition loop.
   - Launch the Streamlit dashboard server.
   - Open your web browser automatically to: `http://localhost:8501`.

---

## 🔌 Scenario 1: Running WITH a Physical Sensor

Use this mode when you have the physical ESP32 microcontroller and AD8232 ECG sensor connected.

### 1. Hardware Connection
- Connect the **ESP32 microcontroller** to your computer via a USB cable.
- Connect the **AD8232 sensor board** to the ESP32 as follows (pre-configured in the firmware):
  - `GND` ➡️ `GND`
  - `3.3V` ➡️ `3.3V`
  - `Output (OUT)` ➡️ `ADC pin (GPIO 34)`
  - `LO+` ➡️ `GPIO 25`
  - `LO-` ➡️ `GPIO 26`
- Attach the **3 ECG electrodes** to your body (or a simulator):
  - **Red Electrode (RA)**: Right Arm (or near right collarbone).
  - **Yellow Electrode (LA)**: Left Arm (or near left collarbone).
  - **Green Electrode (RL)**: Right Leg (or near right lower rib).

### 2. Dashboard Configuration
1. Open the **Live ECG Monitor** tab.
2. In the sidebar, select the COM Port where your ESP32 is connected under **Microcontroller Port** (e.g. `COM3` or keep it on `Auto-Detect`).
3. Under **Telemetry Control Panel**, click **▶️ Start**.
4. The system will detect the incoming signal and display:
   - `SOURCE: PHYSICAL SENSOR` in green.
   - If the leads are disconnected, you will see a red warning: `⚠️ ELECTRODE LEADS DETACHED`. Placing the electrodes on your skin will resolve the warning and start showing your raw heartbeat waveform.

---

## 🧬 Scenario 2: Running WITHOUT a Sensor (Simulation Mode)

Use this mode to demonstrate the full platform capabilities when no physical hardware is plugged in.

### 1. Automatic Fallback
- If the telemetry daemon starts and finds no active serial USB COM port connected, it **automatically falls back to Simulation Mode**.
- In the sidebar, the source status will show: `SOURCE: SIMULATED PATIENT` in blue.

### 2. Manual Simulation Override
- If you have an ESP32 plugged in but want to force simulation, check the box **Force simulated patient source** in the sidebar.

### 3. Demonstrating Real-Time Anomaly Detection
1. Select a Patient Profile in the sidebar (e.g. **John Doe** - high risk, or **Jane Smith** - low risk).
2. Go to the **Live ECG Monitor** tab and click **▶️ Start**.
3. While the telemetry progress bar is filling up:
   - Click the **Trigger PVC Beats anomaly** button in the sidebar.
   - The simulator will inject an abnormal Premature Ventricular Contraction (PVC) heartbeat into the live waveform stream on the next cycle.
   - The waveform plot will show the abnormal wide wave shape.
4. **Hear the Alarm**: Once the acquisition countdown completes, the scale-normalized AI classifier will diagnose the session. If PVCs were triggered, the browser will play a **clinical alarm beep sound** and display a red **CRITICAL RISK DETECTED** banner.

---

## 📊 Interactive Features to Demonstrate

### 1. Visualizations
- Switch the **Waveform Visualization Mode** radio buttons on the fly between:
  - `SMA Filtered (Raw Scale)`: shows original ADC amplitude.
  - `Min-Max Normalized (0 to 1)`: normalizes the height.
  - `Z-Score Normalized`: centers the wave around mean 0.

### 2. Deep Clinical Analytics
- Navigate to the **Analytics** page to show:
  - **ECG Amplitude Distribution**: A histogram of the raw values.
  - **Clinical Metrics**: HRV analysis showing **BPM**, **SDNN (ms)**, and **RMSSD (ms)** based on detected R-peaks.
  - **Moment Box Plots**: Interactive statistical variance (Mean, Std, Skewness, Kurtosis).
  - **Clinical Report**: Scroll down to preview and click **Download Printable HTML Clinical Report** to save a PDF-ready clinical summary on your machine.

### 3. Alerts Management
- Navigate to the **Alerts** page to show:
  - The warning log history.
  - Individual **Ack Alert** buttons and **Acknowledge All** buttons to demonstrate alert resolution.

### 4. Interactive Model Sandbox
- Navigate to the **Model Info** page.
- Scroll to the bottom to find the **Interactive Model Sandbox**:
  - Select a beat type (Normal, PVC, Supraventricular, Fusion).
  - Click **Load and Test Random Beat**.
  - The sandbox loads a clinical beat, graphs it, extracts moments in real-time, runs the classifier, and plots a horizontal bar chart of the model's class probabilities!

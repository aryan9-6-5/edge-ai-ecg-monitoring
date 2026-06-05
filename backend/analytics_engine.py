# backend/analytics_engine.py
import numpy as np
import pandas as pd

def detect_r_peaks(signal_values, sample_rate=125):
    """
    Detects R-peaks in an ECG signal using a rolling-threshold and local maxima search.
    Does not depend on external signal processing libraries (like scipy).
    """
    if len(signal_values) < 100:
        return []
        
    s = np.array(signal_values)
    
    # 1. Take first derivative to emphasize steep slopes of the QRS complex
    diff = np.diff(s)
    # Square the derivative to make all values positive and amplify high slopes
    diff_sq = diff ** 2
    
    # 2. Smooth using convolution (5-point moving average)
    kernel = np.ones(5) / 5.0
    smoothed_diff = np.convolve(diff_sq, kernel, mode='same')
    
    # 3. Find peaks exceeding a threshold (e.g. 15% of the maximum derivative intensity)
    max_val = np.max(smoothed_diff)
    threshold = 0.15 * max_val if max_val > 0 else 1.0
    
    peaks = []
    # Minimum refractory period between heartbeats is ~0.35 seconds (43 samples at 125 Hz)
    min_dist = int(0.35 * sample_rate)
    
    i = min_dist
    while i < len(smoothed_diff) - min_dist:
        val = smoothed_diff[i]
        if val > threshold:
            # Check if local maximum within window
            window = smoothed_diff[i - min_dist : i + min_dist + 1]
            if val == np.max(window):
                # Search original signal around this area for the actual highest value (R-peak)
                search_start = max(0, i - 15)
                search_end = min(len(s), i + 15)
                orig_peak_idx = search_start + np.argmax(s[search_start:search_end])
                
                if not peaks or (orig_peak_idx - peaks[-1] >= min_dist):
                    peaks.append(orig_peak_idx)
                i += min_dist
                continue
        i += 1
        
    return peaks

def compute_hrv_metrics(r_peaks, sample_rate=125):
    """
    Computes average BPM, SDNN (ms), and RMSSD (ms) from detected R-peaks indices.
    """
    if len(r_peaks) < 2:
        return {
            "bpm": 0.0,
            "sdnn": 0.0,
            "rmssd": 0.0,
            "rr_intervals": []
        }
        
    # Convert peak indices to timestamps in seconds
    r_peaks_sec = np.array(r_peaks) / sample_rate
    # Calculate R-R intervals
    rr_intervals_sec = np.diff(r_peaks_sec)
    rr_intervals_ms = rr_intervals_sec * 1000.0
    
    # Compute Average Heart Rate
    mean_rr_sec = np.mean(rr_intervals_sec)
    bpm = 60.0 / mean_rr_sec if mean_rr_sec > 0 else 0.0
    
    # Compute SDNN (standard deviation of RR intervals)
    sdnn = np.std(rr_intervals_ms) if len(rr_intervals_ms) >= 2 else 0.0
    
    # Compute RMSSD (root mean square of successive differences)
    if len(rr_intervals_ms) >= 2:
        successive_diffs = np.diff(rr_intervals_ms)
        rmssd = np.sqrt(np.mean(successive_diffs ** 2))
    else:
        rmssd = 0.0
        
    return {
        "bpm": round(float(bpm), 1),
        "sdnn": round(float(sdnn), 1),
        "rmssd": round(float(rmssd), 1),
        "rr_intervals": [round(float(rr), 2) for rr in rr_intervals_ms]
    }

def analyze_session_signal(filtered_csv_path, sample_rate=125):
    """
    Reads the session's filtered_ecg data and returns HR & HRV metrics.
    """
    try:
        df = pd.read_csv(filtered_csv_path)
        if "filtered_ecg" not in df.columns:
            return None
        
        filtered_ecg = df["filtered_ecg"].values
        r_peaks = detect_r_peaks(filtered_ecg, sample_rate)
        metrics = compute_hrv_metrics(r_peaks, sample_rate)
        
        # Calculate Signal Quality Index (SQI)
        # Check percentage of values that are not flatlined (e.g. not 0 or 4095)
        # Note: If sensor adaptation was active, outliers were masked & interpolated.
        # So we can calculate SQI based on the number of actual spikes in raw data.
        # But here we just return a high SQI based on filtered content stability
        raw_df_path = filtered_csv_path.replace("ecg_filtered.csv", "ecglive.csv")
        sqi = 100.0
        if os.path.exists(raw_df_path):
            raw_df = pd.read_csv(raw_df_path)
            if "ecg" in raw_df.columns and len(raw_df) > 0:
                outliers = (raw_df["ecg"] >= 4080) | (raw_df["ecg"] <= 15)
                sqi = round(float((1.0 - (outliers.sum() / len(raw_df))) * 100.0), 1)
                
        metrics["sqi"] = sqi
        return metrics
    except Exception as e:
        print(f"Error analyzing session signal: {e}")
        return None

# backend/generate_mitbih_data.py
import os
import numpy as np
import pandas as pd

def generate_ecg_beat(label, length=187, noise_std=0.015):
    """
    Generates a synthetic ECG heartbeat based on Gaussian components.
    Length: 187 points.
    label 0: Normal Beat
    label 1: PVC (Premature Ventricular Contraction - wide, tall/inverted, no P wave)
    label 2: Supraventricular premature (premature, slightly narrow, normal shape)
    label 3: Fusion / Abnormal (irregular shape)
    """
    t = np.linspace(0, 1, length)
    signal = np.zeros(length)
    
    if label == 0:  # Normal
        # P wave
        signal += 0.10 * np.exp(-((t - 0.20) / 0.035)**2)
        # Q wave
        signal -= 0.12 * np.exp(-((t - 0.32) / 0.012)**2)
        # R peak (sharp, tall)
        signal += 1.00 * np.exp(-((t - 0.35) / 0.015)**2)
        # S wave
        signal -= 0.18 * np.exp(-((t - 0.38) / 0.015)**2)
        # T wave (broad)
        signal += 0.25 * np.exp(-((t - 0.55) / 0.055)**2)
    elif label == 1:  # PVC
        # No P wave
        # Tall, very wide QRS complex
        signal += 1.20 * np.exp(-((t - 0.38) / 0.055)**2)
        # Deep, wide S-like recovery
        signal -= 0.40 * np.exp(-((t - 0.48) / 0.040)**2)
        # Inverted T wave
        signal -= 0.30 * np.exp(-((t - 0.65) / 0.070)**2)
    elif label == 2:  # Supraventricular premature (premature R-peak)
        # P wave
        signal += 0.08 * np.exp(-((t - 0.15) / 0.030)**2)
        # Q wave
        signal -= 0.10 * np.exp(-((t - 0.24) / 0.010)**2)
        # R peak (premature center)
        signal += 0.95 * np.exp(-((t - 0.26) / 0.012)**2)
        # S wave
        signal -= 0.15 * np.exp(-((t - 0.29) / 0.012)**2)
        # T wave
        signal += 0.22 * np.exp(-((t - 0.48) / 0.050)**2)
    else:  # Fusion / Other Abnormal
        # Irregular shape
        signal += 0.60 * np.exp(-((t - 0.30) / 0.040)**2)
        signal -= 0.30 * np.exp(-((t - 0.42) / 0.030)**2)
        signal += 0.15 * np.exp(-((t - 0.60) / 0.080)**2)
        
    # Normalize to [0, 1] range to match MIT-BIH dataset format
    s_min = np.min(signal)
    s_max = np.max(signal)
    if s_max - s_min > 0:
        signal = (signal - s_min) / (s_max - s_min)
        
    # Add random noise
    if noise_std > 0:
        signal += np.random.normal(0, noise_std, length)
        # Re-clip to [0, 1] after noise
        signal = np.clip(signal, 0.0, 1.0)
        
    return signal

def main():
    print("Generating compatible MIT-BIH formatted calibration dataset for evaluation...")
    
    # Let's generate 1200 beats total
    num_normal = 800
    num_pvc = 250
    num_sp = 100
    num_fusion = 50
    
    data = []
    
    # Generate Normal (label 0)
    for _ in range(num_normal):
        beat = generate_ecg_beat(label=0)
        row = list(beat) + [0.0]
        data.append(row)
        
    # Generate PVC (label 1)
    for _ in range(num_pvc):
        beat = generate_ecg_beat(label=1)
        row = list(beat) + [1.0]
        data.append(row)
        
    # Generate Supraventricular (label 2)
    for _ in range(num_sp):
        beat = generate_ecg_beat(label=2)
        row = list(beat) + [2.0]
        data.append(row)
        
    # Generate Fusion (label 3)
    for _ in range(num_fusion):
        beat = generate_ecg_beat(label=3)
        row = list(beat) + [3.0]
        data.append(row)
        
    # Create DataFrame
    columns = [f"f_{i}" for i in range(187)] + ["label"]
    df = pd.DataFrame(data, columns=columns)
    
    # Shuffle the dataset
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Create directory if not exists
    os.makedirs("data/mitbih", exist_ok=True)
    
    output_path = "data/mitbih/mitbih.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset generated with shape {df.shape} and saved to: {output_path}")
    print("Class distribution:")
    print(df["label"].value_counts())

if __name__ == "__main__":
    main()

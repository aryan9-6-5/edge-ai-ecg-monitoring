import serial
import csv
import time
import os

# Configuration
PORT = "COM11"
BAUD_RATE = 115200
MAX_SAMPLES = 5000
MAX_DURATION = 60  # seconds

# Create folder if it doesn't exist
os.makedirs("data/raw", exist_ok=True)

ser = serial.Serial(PORT, BAUD_RATE)

samples = 0
start_time = time.time()

with open("data/raw/ecglive.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "ecg"])

    print(f"Collecting up to {MAX_SAMPLES} ECG samples...")

    while samples < MAX_SAMPLES and (time.time() - start_time) < MAX_DURATION:
        try:
            value = int(ser.readline().decode().strip())
            writer.writerow([time.time(), value])
            print(value)

            samples += 1

            # After every 100 samples, print how many are left
            if samples % 100 == 0:
                remaining = MAX_SAMPLES - samples
                print(f"{remaining} samples remaining...")

        except Exception:
            continue

print("\nCollection Complete.")
print(f"Collected {samples} samples in {round(time.time() - start_time, 2)} seconds.")
print("Saved data to: data/raw/ecglive.csv")

ser.close()

import os
import numpy as np
import matplotlib.pyplot as plt

#Select the sample for visualizing
#sample_path = "/home/main/Desktop/timit_ppg_raw/train/FAEM0_SI762.npy" 
sample_path = "/home/main/Desktop/librispeech_ppg_raw/dev_clean/lbi-84-121123-0000.npy"

# Load .npy
ppg = np.load(sample_path)  # shape = (T, D), e.g., (300, 1920)

# Print Information
print(f"Shape: {ppg.shape}")  # (Time, PDF-Dim)

# visualization
plt.figure(figsize=(12, 6))
plt.imshow(ppg.T, aspect='auto', origin='lower', interpolation='nearest', cmap='viridis')
plt.colorbar(label="Posterior probability")
plt.xlabel("Time frame")
plt.ylabel("PDF index")
plt.title("Raw PPG Heatmap")
plt.tight_layout()
plt.show()

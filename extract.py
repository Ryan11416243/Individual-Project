import numpy as np
import pandas as pd
import os
from scipy.signal import find_peaks

def parse_ltspice_txt(filepath):
    runs = []
    current_freq = []
    current_mag = []
    
    with open(filepath, 'r') as f:
        for line in f:
            # Skip the header line
            if line.startswith('Freq.'):
                continue
            # New step, save the current run and reset
            elif line.startswith('Step Information:'):
                if current_freq:
                    runs.append({'freq': np.array(current_freq), 'mag': np.array(current_mag)})
                    current_freq = []
                    current_mag = []
            # Parse the actual data lines (added line.strip() to prevent crashes on empty lines)
            elif line.strip() and line[0].isdigit():
                parts = line.split('\t')
                freq = float(parts[0])
                
                # extracting just the dB value from "(-20dB,45°)"
                mag_str = parts[1].split('dB')[0].replace('(', '')
                mag = float(mag_str)
                
                current_freq.append(freq)
                current_mag.append(mag)
                
        # Append the very last run after the loop finishes
        if current_freq:
            runs.append({'freq': np.array(current_freq), 'mag': np.array(current_mag)})
            
    return runs

def calculate_statistical_indices(ref_mag, fault_mag):
    """Calculates CC, ASLE, and RMSE between a reference and a fault array."""
    # Correlation Coefficient (CC)
    cc = np.corrcoef(ref_mag, fault_mag)[0, 1]
    
    # Absolute Sum of Logarithmic Error (ASLE)
    # Note: LTSpice data is ALREADY in dB (logarithmic), use absolute difference
    n = len(ref_mag)
    asle = np.sum(np.abs(fault_mag - ref_mag)) / n
    
    # Root Mean Square Error (RMSE) / Spectral Deviation
    rmse = np.sqrt(np.sum((fault_mag - ref_mag)**2) / n)
    
    return cc, asle, rmse

def extract_features(parsed_runs, ref_run):
    """Loops through parsed runs, splits into sub-bands, and extracts ML features."""
    dataset = []
    
    ref_freq = ref_run['freq']
    ref_mag_all = ref_run['mag']
    
    # Create masks for sub-banding (LF: <10kHz, MF: 10k-100k, HF: >100k)
    mask_lf = ref_freq < 10000
    mask_mf = (ref_freq >= 10000) & (ref_freq <= 100000)
    mask_hf = ref_freq > 100000
    
    for i, run in enumerate(parsed_runs):
        freq = run['freq']
        mag = run['mag']
        
        # 1. Calculate Statistical Indices for each sub-band
        cc_lf, asle_lf, rmse_lf = calculate_statistical_indices(ref_mag_all[mask_lf], mag[mask_lf])
        cc_mf, asle_mf, rmse_mf = calculate_statistical_indices(ref_mag_all[mask_mf], mag[mask_mf])
        cc_hf, asle_hf, rmse_hf = calculate_statistical_indices(ref_mag_all[mask_hf], mag[mask_hf])
        
        # 2. Extract Physical Features (Resonant Peaks in the HF band)
        # Using a prominence of 3dB filters out tiny numerical noise ripples
        peaks, _ = find_peaks(mag[mask_hf], prominence=3) 
        
        # Get the actual frequencies of the first 3 peaks (pad with NaN if fewer than 3 are found)
        hf_freqs = freq[mask_hf]
        peak_freqs = hf_freqs[peaks]
        
        p1 = peak_freqs[0] if len(peak_freqs) > 0 else np.nan
        p2 = peak_freqs[1] if len(peak_freqs) > 1 else np.nan
        p3 = peak_freqs[2] if len(peak_freqs) > 2 else np.nan
        
        # Compile the feature row for this specific run
        feature_row = {
            'Run_ID': i + 1,
            'CC_LF': cc_lf, 'ASLE_LF': asle_lf, 'RMSE_LF': rmse_lf,
            'CC_MF': cc_mf, 'ASLE_MF': asle_mf, 'RMSE_MF': rmse_mf,
            'CC_HF': cc_hf, 'ASLE_HF': asle_hf, 'RMSE_HF': rmse_hf,
            'Peak_1_Hz': p1, 'Peak_2_Hz': p2, 'Peak_3_Hz': p3
        }
        dataset.append(feature_row)
        
    return pd.DataFrame(dataset)

# --- EXECUTION ---
# 1. Load the Golden Baseline (NO NOISE)
baseline_runs = parse_ltspice_txt('Baseline_NoNoise.txt')
golden_reference = baseline_runs[0] # Single Run

# 2. Process all datasets and add Labels for the Machine Learning algorithm 
all_datasets = []

files_to_process = [
    {'filename': 'Data_Healthy_Noise.txt', 'label': 0, 'severity': '0%'},
    {'filename': 'Data_5Percent.txt',      'label': 1, 'severity': '5%'},
    {'filename': 'Data_10Percent.txt',     'label': 1, 'severity': '10%'},
    {'filename': 'Data_20Percent.txt',     'label': 1, 'severity': '20%'}
]

for file_info in files_to_process:
    filepath = file_info['filename']
    label = file_info['label']
    severity = file_info['severity']
    
    if os.path.exists(filepath):
        print(f"Processing {filepath}...")
        parsed_runs = parse_ltspice_txt(filepath)
        
        # compares the parsed runs against No-Noise Baseline
        df = extract_features(parsed_runs, golden_reference) 
        
        df['True_Label'] = label
        df['Severity'] = severity
        
        all_datasets.append(df)
    else:
        print(f"File not found: {filepath}. Skipping.")

# --- THE MISSING COMBINE AND EXPORT BLOCK ---
# 3. Combine everything into one giant Master Dataset
if all_datasets:
    master_dataset = pd.concat(all_datasets, ignore_index=True)
    
    # Reorder columns so Label and Severity are at the front
    cols = ['Run_ID', 'True_Label', 'Severity'] + [c for c in master_dataset.columns if c not in ['Run_ID', 'True_Label', 'Severity']]
    master_dataset = master_dataset[cols]
    
    print("\nExtraction Complete. Preview:")
    print(master_dataset.head())
    
    # Export to CSV
    master_dataset.to_csv('Master_ML_Dataset.csv', index=False)
    print("\nSaved successfully as Master_ML_Dataset.csv")
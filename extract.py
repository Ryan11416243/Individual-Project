import numpy as np
import pandas as pd
import os
from scipy.signal import find_peaks

# ==========================================
# CONFIGURATION SECTION
# ==========================================
FILE_CONFIG = [
    ('Data_Healthy_Noise.txt', 0, '0%'),
    ('Data_5Percent.txt',      1, '5%'),
    ('Data_10Percent.txt',     1, '10%'),
    ('Data_20Percent.txt',     1, '20%'),
    ('./Local_Axial/High_5.txt', 2, '5%')
    # Add your remaining 5 files below:
    # ('Your_Next_File.txt',   1, '30%'),
]

BASELINE_FILE = 'Baseline_NoNoise.txt'
OUTPUT_FILE = 'Master_ML_Dataset.csv'

# ==========================================
# FUNCTIONS
# ==========================================

def parse_ltspice_txt(filepath):
    runs = []
    current_freq = []
    current_mag = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('Freq.'):
                continue
            elif line.startswith('Step Information:'):
                if current_freq:
                    runs.append({'freq': np.array(current_freq), 'mag': np.array(current_mag)})
                    current_freq = []
                    current_mag = []
            elif line.strip() and line[0].isdigit():
                parts = line.split('\t')
                freq = float(parts[0])
                
                mag_str = parts[1].split('dB')[0].replace('(', '')
                mag = float(mag_str)
                
                current_freq.append(freq)
                current_mag.append(mag)
                
        if current_freq:
            runs.append({'freq': np.array(current_freq), 'mag': np.array(current_mag)})
            
    return runs

def calculate_statistical_indices(ref_mag, fault_mag):
    """Calculates CC, ASLE, and RMSE between a reference and a fault array."""
    cc = np.corrcoef(ref_mag, fault_mag)[0, 1]
    
    n = len(ref_mag)
    asle = np.sum(np.abs(fault_mag - ref_mag)) / n
    rmse = np.sqrt(np.sum((fault_mag - ref_mag)**2) / n)
    
    return cc, asle, rmse

def extract_features(parsed_runs, ref_run):
    """Loops through parsed runs, splits into sub-bands, and extracts ML features."""
    dataset = []
    
    ref_freq = ref_run['freq']
    ref_mag_all = ref_run['mag']
    
    mask_lf = ref_freq < 10000
    mask_mf = (ref_freq >= 10000) & (ref_freq <= 100000)
    mask_hf = ref_freq > 100000
    
    for i, run in enumerate(parsed_runs):
        freq = run['freq']
        mag = run['mag']
        
        cc_lf, asle_lf, rmse_lf = calculate_statistical_indices(ref_mag_all[mask_lf], mag[mask_lf])
        cc_mf, asle_mf, rmse_mf = calculate_statistical_indices(ref_mag_all[mask_mf], mag[mask_mf])
        cc_hf, asle_hf, rmse_hf = calculate_statistical_indices(ref_mag_all[mask_hf], mag[mask_hf])
        
        peaks, _ = find_peaks(mag[mask_hf], prominence=3) 
        
        hf_freqs = freq[mask_hf]
        peak_freqs = hf_freqs[peaks]
        
        # --- UPDATED: Now extracting up to 5 peaks ---
        p1 = peak_freqs[0] if len(peak_freqs) > 0 else np.nan
        p2 = peak_freqs[1] if len(peak_freqs) > 1 else np.nan
        p3 = peak_freqs[2] if len(peak_freqs) > 2 else np.nan
        p4 = peak_freqs[3] if len(peak_freqs) > 3 else np.nan
        p5 = peak_freqs[4] if len(peak_freqs) > 4 else np.nan
        
        feature_row = {
            'Run_ID': i + 1,
            'CC_LF': cc_lf, 'ASLE_LF': asle_lf, 'RMSE_LF': rmse_lf,
            'CC_MF': cc_mf, 'ASLE_MF': asle_mf, 'RMSE_MF': rmse_mf,
            'CC_HF': cc_hf, 'ASLE_HF': asle_hf, 'RMSE_HF': rmse_hf,
            # --- UPDATED: Added Peak 4 and Peak 5 to the dataset ---
            'Peak_1_Hz': p1, 'Peak_2_Hz': p2, 'Peak_3_Hz': p3,
            'Peak_4_Hz': p4, 'Peak_5_Hz': p5
        }
        dataset.append(feature_row)
        
    return pd.DataFrame(dataset)

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    if not os.path.exists(BASELINE_FILE):
        print(f"Error: Baseline file '{BASELINE_FILE}' not found. Cannot proceed.")
        exit()
        
    print(f"Loading golden reference from {BASELINE_FILE}...")
    baseline_runs = parse_ltspice_txt(BASELINE_FILE)
    golden_reference = baseline_runs[0] 

    all_datasets = []

    for filename, label, severity in FILE_CONFIG:
        if os.path.exists(filename):
            print(f"Processing {filename}...")
            parsed_runs = parse_ltspice_txt(filename)
            
            df = extract_features(parsed_runs, golden_reference) 
            
            df['Source_File'] = filename 
            df['True_Label'] = label
            df['Severity'] = severity
            
            all_datasets.append(df)
        else:
            print(f"Warning: File not found: {filename}. Skipping.")

    if all_datasets:
        master_dataset = pd.concat(all_datasets, ignore_index=True)
        
        front_cols = ['Source_File', 'Run_ID', 'True_Label', 'Severity']
        remaining_cols = [c for c in master_dataset.columns if c not in front_cols]
        master_dataset = master_dataset[front_cols + remaining_cols]
        
        print("\nExtraction Complete. Preview:")
        print(master_dataset.head())
        
        master_dataset.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSaved successfully as {OUTPUT_FILE}")
    else:
        print("\nNo valid datasets were processed. Check your file paths.")

import numpy as np
import pandas as pd
import os
from scipy.signal import find_peaks

# ==========================================
# CONFIGURATION SECTION
# ==========================================
# Format: ('Filename.txt', True_Label, 'Severity', 'Location_of_fault')
FILE_CONFIG = [
    ('Healthy\Data_Healthy_Noise.txt', 0, '0%',  'None'), # Example: No fault location
    ('Healthy\More_Healthy.txt', 0, '0%', 'None'),
    ('Healthy\healthy_extra' , 0, '0%', 'None'),

    ('Data_5Percent.txt',      1, '5%',  'LV_Mid'),   # Example: Fault at Inductor 1
    ('Data_10Percent.txt',     1, '10%', 'LV_Mid'),   # Example: Fault at Capacitor 2
    ('Data_20Percent.txt',     1, '20%', 'LV_Mid'),

    ('./Local_Axial/Low_5.txt', 2, '5%', 'LV_Low'),
    ('./Local_Axial/Low_10.txt', 2, '10%', 'LV_Low'),
    ('./Local_Axial/Low_20.txt', 2, '20%', 'LV_Low'),
    ('./Local_Axial/Mid_5.txt', 2, '5%', 'LV_Mid'),
    ('./Local_Axial/Mid_10.txt', 2, '10%', 'LV_Mid'),
    ('./Local_Axial/Mid_20.txt', 2, '20%', 'LV_Mid'),
    ('./Local_Axial/High_5.txt', 2, '5%', 'LV_High'),
    ('./Local_Axial/High_10.txt', 2, '10%', 'LV_High'),
    ('./Local_Axial/High_20.txt', 2, '20%', 'LV_High'),

    ('./LCP/LCP_5percent.txt', 3, '5%', 'None'),
    ('./LCP/LCP_10percent.txt', 3, '10%', 'None'),
    ('./LCP/LCP_20percent.txt', 3, '20%', 'None'),


    ('SC\SC_left_5percent', 4, '5%', 'LV_Low'),
    ('SC\SC_left_10percent', 4, '10%', 'LV_Low'),
    ('SC\SC_left_10percent', 4, '20%', 'LV_Low'),
    ('SC\SC_mid_5percent', 4, '5%', 'LV_Mid'),
    ('SC\SC_mid_10percent', 4, '10%', 'LV_Mid'),
    ('SC\SC_mid_20percent', 4, '20%', 'LV_Mid'),
    ('SC\SC_Right_5percent', 4, '5%', 'LV_High'),
    ('SC\SC_Right_10percent', 4, '10%', 'LV_High'),
    ('SC\SC_Right_20percent', 4, '20%', 'LV_High')

    # ('.txt File',   Class, 'Severity', 'Location'),
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
    """
    Calculates CCF, LCC, SDA, SE, and CSD between a reference and a fault array.
    Assumes inputs are NumPy arrays in dB.
    """
    n = len(ref_mag)
    
    # 1. CCF (Cross-Correlation Factor)
    ccf = np.corrcoef(ref_mag, fault_mag)[0, 1]
    
    # Pre-calculate means for LCC and CSD
    mean_ref = np.mean(ref_mag)
    mean_fault = np.mean(fault_mag)
    
    # 2. LCC (Lin's Concordance Coefficient)
    var_ref = np.var(ref_mag) 
    var_fault = np.var(fault_mag) 
    covar = np.cov(ref_mag, fault_mag, bias=True)[0, 1] 
    
    lcc = (2 * covar) / ((mean_fault - mean_ref)**2 + var_fault + var_ref)
    
    # 3. SDA (Standard Difference Area)
    sda = np.sum(np.abs(fault_mag - ref_mag)) / np.sum(np.abs(ref_mag))
    
    # 4. SE (Sum of Errors)
    se = np.sum(fault_mag - ref_mag) / n
    
    # 5. CSD (Comparative Standard Deviation)
    # Calculates the variance of the mean-centered differences
    mean_centered_diff = (fault_mag - mean_fault) - (ref_mag - mean_ref)
    csd = np.sqrt(np.sum(mean_centered_diff**2) / (n - 1))
    
    return ccf, lcc, sda, se, csd

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
        
        # Unpack the 5 variables per frequency band
        ccf_lf, lcc_lf, sda_lf, se_lf, csd_lf = calculate_statistical_indices(ref_mag_all[mask_lf], mag[mask_lf])
        ccf_mf, lcc_mf, sda_mf, se_mf, csd_mf = calculate_statistical_indices(ref_mag_all[mask_mf], mag[mask_mf])
        ccf_hf, lcc_hf, sda_hf, se_hf, csd_hf = calculate_statistical_indices(ref_mag_all[mask_hf], mag[mask_hf])
        
        # 2. Extract Physical Features (Resonant Peaks in the HF band)
        peaks, _ = find_peaks(mag[mask_hf], prominence=3) 
        
        hf_freqs = freq[mask_hf]
        hf_mags = mag[mask_hf] # <-- NEW: Isolate the HF magnitudes
        
        peak_freqs = hf_freqs[peaks]
        peak_mags = hf_mags[peaks]   # <-- NEW: Get the dB values at those peaks
        
        # Extract Frequencies (Hz)
        p1 = peak_freqs[0] if len(peak_freqs) > 0 else np.nan
        p2 = peak_freqs[1] if len(peak_freqs) > 1 else np.nan
        p3 = peak_freqs[2] if len(peak_freqs) > 2 else np.nan
        p4 = peak_freqs[3] if len(peak_freqs) > 3 else np.nan
        p5 = peak_freqs[4] if len(peak_freqs) > 4 else np.nan
        
        # Extract Magnitudes (dB)
        m1 = peak_mags[0] if len(peak_mags) > 0 else np.nan
        m2 = peak_mags[1] if len(peak_mags) > 1 else np.nan
        m3 = peak_mags[2] if len(peak_mags) > 2 else np.nan
        m4 = peak_mags[3] if len(peak_mags) > 3 else np.nan
        m5 = peak_mags[4] if len(peak_mags) > 4 else np.nan
        
        # Compile the feature row
        feature_row = {
            'Run_ID': i + 1,
            'CCF_LF': ccf_lf, 'LCC_LF': lcc_lf, 'SDA_LF': sda_lf, 'SE_LF': se_lf, 'CSD_LF': csd_lf,
            'CCF_MF': ccf_mf, 'LCC_MF': lcc_mf, 'SDA_MF': sda_mf, 'SE_MF': se_mf, 'CSD_MF': csd_mf,
            'CCF_HF': ccf_hf, 'LCC_HF': lcc_hf, 'SDA_HF': sda_hf, 'SE_HF': se_hf, 'CSD_HF': csd_hf,
            'Peak_1_Hz': p1, 'Peak_1_dB': m1,
            'Peak_2_Hz': p2, 'Peak_2_dB': m2,
            'Peak_3_Hz': p3, 'Peak_3_dB': m3,
            'Peak_4_Hz': p4, 'Peak_4_dB': m4,
            'Peak_5_Hz': p5, 'Peak_5_dB': m5}
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

    # --- UPDATED: Unpacking 4 variables now instead of 3 ---
    for filename, label, severity, fault_location in FILE_CONFIG:
        if os.path.exists(filename):
            print(f"Processing {filename}...")
            parsed_runs = parse_ltspice_txt(filename)
            
            df = extract_features(parsed_runs, golden_reference) 
            
            df['Source_File'] = filename 
            df['True_Label'] = label
            df['Severity'] = severity
            # --- UPDATED: Added the new column here ---
            df['Location_of_fault'] = fault_location 
            
            all_datasets.append(df)
        else:
            print(f"Warning: File not found: {filename}. Skipping.")

    if all_datasets:
        master_dataset = pd.concat(all_datasets, ignore_index=True)
        
        # --- UPDATED: Added Location_of_fault to the front columns list ---
        front_cols = ['Source_File', 'Run_ID', 'True_Label', 'Severity', 'Location_of_fault']
        remaining_cols = [c for c in master_dataset.columns if c not in front_cols]
        master_dataset = master_dataset[front_cols + remaining_cols]
        
        print("\nExtraction Complete. Preview:")
        print(master_dataset.head())
        
        master_dataset.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSaved successfully as {OUTPUT_FILE}")
    else:
        print("\nNo valid datasets were processed. Check your file paths.")
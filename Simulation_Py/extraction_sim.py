import numpy as np
import pandas as pd
import os
import glob
from scipy.signal import find_peaks

# ==========================================
# 1. CONFIGURATION SECTION
# ==========================================
# Instead of hardcoding every file, we map the directories we generated.
# Format: 'Directory_Name': (True_Label, 'Severity', 'Location_of_fault')
DIR_CONFIG = {
    'Simulation_Healthy': (0, '0%', 'None'),
    'Simulation_Radial':  (1, '10%', 'LV_Mid')  # Update severity/location based on your batch runs
}

# The nominal baseline we generated (Run 0 from the Healthy batch)
BASELINE_FILE = os.path.join('Simulation_Py\Simulation_Healthy', 'Trace_0.txt')
OUTPUT_FILE = 'Master_ML_Dataset.csv'

# Frequency boundaries for sub-bands
FREQ_BANDS = {
    'LF': (0, 10000),
    'MF': (10000, 100000),
    'HF': (100000, float('inf'))
}

# ==========================================
# 2. CORE FUNCTIONS
# ==========================================

def read_sim_data(filepath):
    """
    Reads the clean tab-separated generated text files.
    Returns frequencies and magnitudes as numpy arrays.
    """
    df = pd.read_csv(filepath, sep='\t')
    # Column names match the export script: "Frequency(Hz)" and "Magnitude(dB)"
    return df['Frequency(Hz)'].values, df['Magnitude(dB)'].values

def calculate_statistical_indices(ref_mag, fault_mag):
    """
    Calculates CCF, LCC, SDA, SE, and CSD between a reference and a fault array.
    """
    n = len(ref_mag)
    
    # 1. CCF (Cross-Correlation Factor)
    ccf = np.corrcoef(ref_mag, fault_mag)[0, 1]
    
    # Pre-calculate means 
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
    mean_centered_diff = (fault_mag - mean_fault) - (ref_mag - mean_ref)
    csd = np.sqrt(np.sum(mean_centered_diff**2) / (n - 1))
    
    return ccf, lcc, sda, se, csd

def extract_features_single(freqs, mags, ref_mags, file_id):
    """
    Splits a single run into sub-bands, extracts statistical features, 
    and identifies HF resonance peaks.
    """
    # Create logical masks for frequency bands
    mask_lf = (freqs >= FREQ_BANDS['LF'][0]) & (freqs < FREQ_BANDS['LF'][1])
    mask_mf = (freqs >= FREQ_BANDS['MF'][0]) & (freqs <= FREQ_BANDS['MF'][1])
    mask_hf = (freqs > FREQ_BANDS['HF'][0])
    
    # Calculate Statistical Indices per band
    ccf_lf, lcc_lf, sda_lf, se_lf, csd_lf = calculate_statistical_indices(ref_mags[mask_lf], mags[mask_lf])
    ccf_mf, lcc_mf, sda_mf, se_mf, csd_mf = calculate_statistical_indices(ref_mags[mask_mf], mags[mask_mf])
    ccf_hf, lcc_hf, sda_hf, se_hf, csd_hf = calculate_statistical_indices(ref_mags[mask_hf], mags[mask_hf])
    
    # Extract Physical Features (Resonant Peaks in the HF band)
    hf_freqs = freqs[mask_hf]
    hf_mags = mags[mask_hf]
    peaks, _ = find_peaks(hf_mags, prominence=3) 
    
    peak_freqs = hf_freqs[peaks]
    peak_mags = hf_mags[peaks]
    
    # Safely extract up to 5 peaks (pad with NaN if fewer exist)
    def safe_get(arr, idx):
        return arr[idx] if len(arr) > idx else np.nan

    # Compile the feature row dictionary
    return {
        'Run_ID': file_id,
        'CCF_LF': ccf_lf, 'LCC_LF': lcc_lf, 'SDA_LF': sda_lf, 'SE_LF': se_lf, 'CSD_LF': csd_lf,
        'CCF_MF': ccf_mf, 'LCC_MF': lcc_mf, 'SDA_MF': sda_mf, 'SE_MF': se_mf, 'CSD_MF': csd_mf,
        'CCF_HF': ccf_hf, 'LCC_HF': lcc_hf, 'SDA_HF': sda_hf, 'SE_HF': se_hf, 'CSD_HF': csd_hf,
        'Peak_1_Hz': safe_get(peak_freqs, 0), 'Peak_1_dB': safe_get(peak_mags, 0),
        'Peak_2_Hz': safe_get(peak_freqs, 1), 'Peak_2_dB': safe_get(peak_mags, 1),
        'Peak_3_Hz': safe_get(peak_freqs, 2), 'Peak_3_dB': safe_get(peak_mags, 2),
        'Peak_4_Hz': safe_get(peak_freqs, 3), 'Peak_4_dB': safe_get(peak_mags, 3),
        'Peak_5_Hz': safe_get(peak_freqs, 4), 'Peak_5_dB': safe_get(peak_mags, 4)
    }

# ==========================================
# 3. MAIN BATCH EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # 1. Get absolute path of the script's directory
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Build the absolute path to the Golden Reference
    baseline_filepath = os.path.join(SCRIPT_DIR, 'Simulation_Healthy', 'Trace_0.txt')
    
    if not os.path.exists(baseline_filepath):
        print(f"Error: Golden baseline '{baseline_filepath}' not found.")
        print("Please run your generation script first.")
        exit()
        
    print(f"Loading golden reference from {baseline_filepath}...")
    ref_freqs, ref_mags = read_sim_data(baseline_filepath)
    
    all_rows = []

    # 3. Iterate through configured directories using Absolute Paths
    for folder_name, (label, severity, fault_location) in DIR_CONFIG.items():
        
        # Resolve exactly where this folder should be
        target_folder_path = os.path.join(SCRIPT_DIR, folder_name)
        
        if not os.path.exists(target_folder_path):
            print(f"Warning: Directory '{target_folder_path}' not found. Skipping.")
            continue
            
        print(f"Processing directory: {folder_name}...")
        
        # Grab all .txt files in the directory
        txt_files = glob.glob(os.path.join(target_folder_path, '*.txt'))
        
        if len(txt_files) == 0:
            print(f"  -> Folder found, but no .txt files inside. Skipping.")
            continue
            
        for filepath in txt_files:
            filename = os.path.basename(filepath)
            
            # Read the target trace
            freqs, mags = read_sim_data(filepath)
            
            # Extract features against the baseline
            feature_row = extract_features_single(freqs, mags, ref_mags, filename)
            
            # Append Metadata
            feature_row['Source_Directory'] = folder_name
            feature_row['True_Label'] = label
            feature_row['Severity'] = severity
            feature_row['Location_of_fault'] = fault_location
            
            all_rows.append(feature_row)

    # 4. Build and format the final DataFrame
    if all_rows:
        master_dataset = pd.DataFrame(all_rows)
        
        # Reorder columns to put metadata at the front
        front_cols = ['Source_Directory', 'Run_ID', 'True_Label', 'Severity', 'Location_of_fault']
        remaining_cols = [c for c in master_dataset.columns if c not in front_cols]
        master_dataset = master_dataset[front_cols + remaining_cols]
        
        # Export to CSV (also using absolute path for safety)
        output_filepath = os.path.join(SCRIPT_DIR, OUTPUT_FILE)
        master_dataset.to_csv(output_filepath, index=False)
        
        print("\nExtraction Complete. Preview:")
        print(master_dataset.head())
        print(f"\nSaved successfully as {output_filepath}")
        
    else:
        print("\nNo data processed. Check your DIR_CONFIG and ensure the generated folders contain .txt files.")
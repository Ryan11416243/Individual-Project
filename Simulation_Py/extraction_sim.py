import numpy as np
import pandas as pd
import os
import glob
from scipy.signal import find_peaks
import warnings

# ==========================================
# 1. CONFIGURATION SECTION
# ==========================================
DIR_CONFIG = {
    'Simulation_Healthy':   (0, '0%', 'None'),
    'Simulation_Radial_10': (1, '10%', 'LV_Mid')
}

OUTPUT_FILE = 'Master_ML_Dataset.csv'

FREQ_BANDS = {
    'LF': (0, 10000),
    'MF': (10000, 100000),
    'HF': (100000, float('inf'))
}

# ==========================================
# 2. CORE FUNCTIONS (With Defensive Programming)
# ==========================================

def read_sim_data(filepath):
    """
    Reads tab-separated text files.
    CS Feature: Explicit try-catch blocks and data validation.
    """
    try:
        df = pd.read_csv(filepath, sep='\t')
        
        # Enforce strict column naming
        if 'Frequency(Hz)' not in df.columns or 'Magnitude(dB)' not in df.columns:
            raise KeyError(f"Missing required columns in {filepath}.")
            
        freqs = df['Frequency(Hz)'].values
        mags = df['Magnitude(dB)'].values
        
        # Enforce data integrity (No NaNs or Infs allowed from the simulator)
        if np.isnan(mags).any() or np.isinf(mags).any():
            raise ValueError(f"Corrupted data (NaN/Inf) found in {filepath}.")
            
        return freqs, mags
        
    except Exception as e:
        print(f"  [ERROR] Failed to read {filepath}: {e}")
        return None, None

def calculate_statistical_indices(ref_mag, fault_mag):
    """
    Calculates statistical features with mathematical safeguards.
    CS Feature: Zero-division protection and length assertions.
    """
    assert len(ref_mag) == len(fault_mag), "Length mismatch between Reference and Fault arrays."
    n = len(ref_mag)
    epsilon = 1e-12 # Prevent divide-by-zero errors in mathematically flat signals
    
    # 1. CCF (Cross-Correlation Factor)
    # Catch Constant Input Warnings which return NaN in correlation matrices
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ccf_matrix = np.corrcoef(ref_mag, fault_mag)
        ccf = ccf_matrix[0, 1] if not np.isnan(ccf_matrix[0, 1]) else 0.0
    
    # Pre-calculate components
    mean_ref = np.mean(ref_mag)
    mean_fault = np.mean(fault_mag)
    var_ref = np.var(ref_mag) 
    var_fault = np.var(fault_mag) 
    
    # 2. LCC (Lin's Concordance Coefficient)
    covar_matrix = np.cov(ref_mag, fault_mag, bias=True)
    covar = covar_matrix[0, 1] if covar_matrix.ndim > 1 else 0.0
    lcc_denominator = ((mean_fault - mean_ref)**2 + var_fault + var_ref)
    lcc = (2 * covar) / (lcc_denominator + epsilon)
    
    # 3. SDA (Standard Difference Area)
    sda_denominator = np.sum(np.abs(ref_mag))
    sda = np.sum(np.abs(fault_mag - ref_mag)) / (sda_denominator + epsilon)
    
    # 4. SE (Sum of Errors)
    se = np.sum(fault_mag - ref_mag) / n
    
    # 5. CSD (Comparative Standard Deviation)
    mean_centered_diff = (fault_mag - mean_fault) - (ref_mag - mean_ref)
    csd = np.sqrt(np.sum(mean_centered_diff**2) / (n - 1))
    
    return ccf, lcc, sda, se, csd

def extract_features_single(freqs, mags, ref_mags, file_id, unit_id):
    """
    Splits a run into sub-bands and extracts features/peaks.
    """
    mask_lf = (freqs >= FREQ_BANDS['LF'][0]) & (freqs < FREQ_BANDS['LF'][1])
    mask_mf = (freqs >= FREQ_BANDS['MF'][0]) & (freqs <= FREQ_BANDS['MF'][1])
    mask_hf = (freqs > FREQ_BANDS['HF'][0])
    
    ccf_lf, lcc_lf, sda_lf, se_lf, csd_lf = calculate_statistical_indices(ref_mags[mask_lf], mags[mask_lf])
    ccf_mf, lcc_mf, sda_mf, se_mf, csd_mf = calculate_statistical_indices(ref_mags[mask_mf], mags[mask_mf])
    ccf_hf, lcc_hf, sda_hf, se_hf, csd_hf = calculate_statistical_indices(ref_mags[mask_hf], mags[mask_hf])
    
    # Extract Physical Peaks (HF Band)
    hf_freqs = freqs[mask_hf]
    hf_mags = mags[mask_hf]
    peaks, _ = find_peaks(hf_mags, prominence=3) 
    
    peak_freqs = hf_freqs[peaks]
    peak_mags = hf_mags[peaks]
    
    def safe_get(arr, idx):
        return arr[idx] if len(arr) > idx else np.nan

    return {
        'Unit_ID': unit_id,
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
# 3. POST-EXTRACTION VALIDATOR SUITE
# ==========================================
def validate_ml_dataset(csv_path):
    """
    Asserts mathematical logic and physical expectations of the final dataset.
    Protects against data leakage and silent calculation errors.
    """
    print(f"\n[VALIDATION] Verifying Dataset Integrity: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # CS CHECK 1: No NaNs in statistical indices
    stat_columns = [col for col in df.columns if any(x in col for x in ['CCF', 'LCC', 'SDA', 'SE', 'CSD'])]
    if df[stat_columns].isna().any().any():
        raise AssertionError("FAIL: NaNs detected in statistical calculations. Check zero-division catches.")
        
    # CS CHECK 2: Mathematical Bounds
    assert df['CCF_HF'].between(-1.0, 1.0).all(), "FAIL: CCF contains values outside [-1, 1]."
    assert (df['SDA_HF'] >= 0).all(), "FAIL: SDA contains negative values (Mathematically impossible)."
    assert (df['CSD_HF'] >= 0).all(), "FAIL: CSD contains negative values (Mathematically impossible)."

    # FRA CHECK 1: Ensure Baseline Leakage Prevention
    # The dataset should NEVER contain Trace_0.txt, as comparing a baseline to itself 
    # yields perfect scores and destroys ML model training.
    if (df['Run_ID'] == 'Trace_0.txt').any():
        raise AssertionError("FAIL: Baseline 'Trace_0.txt' leaked into the final ML dataset.")

    print("  [PASS] Mathematical Bounds Confirmed (-1 <= CCF <= 1, SDA/CSD >= 0).")
    print("  [PASS] Zero-Division & NaN Safety Confirmed.")
    print("  [PASS] Baseline Leakage successfully prevented.")
    print("[VALIDATION COMPLETE] Dataset ready for Machine Learning.\n")

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    all_rows = []

    for folder_name, (label, severity, fault_location) in DIR_CONFIG.items():
        target_folder_path = os.path.join(SCRIPT_DIR, folder_name)
        
        if not os.path.exists(target_folder_path):
            print(f"Warning: Directory '{target_folder_path}' not found. Skipping.")
            continue
            
        print(f"Processing condition: {folder_name}...")
        
        # Navigate into the specific Unit subfolders (e.g., Unit_0, Unit_1)
        unit_folders = glob.glob(os.path.join(target_folder_path, 'Unit_*'))
        
        for unit_path in unit_folders:
            unit_name = os.path.basename(unit_path)
            
            # FRA Requirement: Enforce strict LOCAL baselines
            local_baseline_path = os.path.join(unit_path, 'Trace_0.txt')
            if not os.path.exists(local_baseline_path):
                print(f"  -> Error: Local baseline missing in {unit_name}. Skipping unit.")
                continue
                
            ref_freqs, ref_mags = read_sim_data(local_baseline_path)
            if ref_freqs is None: continue # Skip if data was corrupted
            
            # Grab all sweeps for this specific unit
            txt_files = glob.glob(os.path.join(unit_path, '*.txt'))
            
            for filepath in txt_files:
                filename = os.path.basename(filepath)
                
                # Critical CS/FRA Rule: Never extract a baseline against itself
                if filename == 'Trace_0.txt':
                    continue
                
                freqs, mags = read_sim_data(filepath)
                if freqs is None: continue
                
                # Extract features
                feature_row = extract_features_single(freqs, mags, ref_mags, filename, unit_name)
                
                # Append Context Metadata
                feature_row['Source_Condition'] = folder_name
                feature_row['True_Label'] = label
                feature_row['Severity'] = severity
                feature_row['Location_of_fault'] = fault_location
                
                all_rows.append(feature_row)

    # 4. Final Compile and Export
    if all_rows:
        master_dataset = pd.DataFrame(all_rows)
        
        # Organize columns nicely
        front_cols = ['Source_Condition', 'Unit_ID', 'Run_ID', 'True_Label', 'Severity', 'Location_of_fault']
        remaining_cols = [c for c in master_dataset.columns if c not in front_cols]
        master_dataset = master_dataset[front_cols + remaining_cols]
        
        output_filepath = os.path.join(SCRIPT_DIR, OUTPUT_FILE)
        master_dataset.to_csv(output_filepath, index=False)
        print(f"\nExtraction complete. Saved to {output_filepath}")
        
        # 5. Run the strict CS/FRA Validation Suite
        validate_ml_dataset(output_filepath)
        
    else:
        print("\nNo data processed. Ensure DIR_CONFIG matches generated folders.")
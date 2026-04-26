import numpy as np
import pandas as pd
import os
import glob
from scipy.signal import find_peaks
import warnings
import random


# ==========================================
# 1. CONFIGURATION SECTION
# ==========================================
# CS FIX: Hardcoded DIR_CONFIG removed. Replaced with dynamic Class Mapping.
CLASS_LABELS = {
    'Healthy': 0,
    'Radial': 1,
    'Axial': 2,
    'LCP': 3,
    'TTSC': 4
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
    Splits a run into sub-bands and extracts features/peaks using Windowed Maxima.
    """
    mask_lf = (freqs >= FREQ_BANDS['LF'][0]) & (freqs < FREQ_BANDS['LF'][1])
    mask_mf = (freqs >= FREQ_BANDS['MF'][0]) & (freqs <= FREQ_BANDS['MF'][1])
    mask_hf = (freqs > FREQ_BANDS['HF'][0])
    
    ccf_lf, lcc_lf, sda_lf, se_lf, csd_lf = calculate_statistical_indices(ref_mags[mask_lf], mags[mask_lf])
    ccf_mf, lcc_mf, sda_mf, se_mf, csd_mf = calculate_statistical_indices(ref_mags[mask_mf], mags[mask_mf])
    ccf_hf, lcc_hf, sda_hf, se_hf, csd_hf = calculate_statistical_indices(ref_mags[mask_hf], mags[mask_hf])
    
    # ---------------------------------------------------------
    # NEW LOGIC: Windowed Maxima Peak Extraction
    # ---------------------------------------------------------
    # Define the 5 static frequency windows
    windows = [
        (100000, 250000),  # Window 1
        (250000, 400000),  # Window 2
        (400000, 600000),  # Window 3
        (600000, 800000),  # Window 4
        (800000, 1000000)  # Window 5
    ]
    
    peak_features = {}
    
    for i, (f_min, f_max) in enumerate(windows):
        window_mask = (freqs >= f_min) & (freqs < f_max)
        window_freqs = freqs[window_mask]
        window_mags = mags[window_mask]
        ref_window_mags = ref_mags[window_mask]
        
        if len(window_mags) > 0:
            # 1. Surgical Sub-Band Statistics
            ccf, lcc, sda, se, csd = calculate_statistical_indices(ref_window_mags, window_mags)
            peak_features[f'SDA_Win{i+1}'] = sda
            peak_features[f'CCF_Win{i+1}'] = ccf
            peak_features[f'CSD_Win{i+1}'] = csd
            
            # 2. Delta Peaks
            max_idx_fault = np.argmax(window_mags)
            max_idx_ref = np.argmax(ref_window_mags)
            peak_features[f'Delta_Peak_{i+1}_Hz'] = window_freqs[max_idx_fault] - window_freqs[max_idx_ref]
            peak_features[f'Delta_Peak_{i+1}_dB'] = window_mags[max_idx_fault] - ref_window_mags[max_idx_ref]

            # 3. Sub-Band Spectral Energy (Trapezoidal Integration)
            energy_ref = np.trapezoid(np.abs(ref_window_mags), window_freqs)
            energy_fault = np.trapezoid(np.abs(window_mags), window_freqs)
            peak_features[f'Delta_Energy_Win{i+1}'] = energy_fault - energy_ref


        else:
            peak_features[f'SDA_Win{i+1}'] = 0.0
            peak_features[f'CCF_Win{i+1}'] = 0.0
            peak_features[f'CSD_Win{i+1}'] = 0.0
            peak_features[f'Delta_Peak_{i+1}_Hz'] = 0.0
            peak_features[f'Delta_Peak_{i+1}_dB'] = 0.0
            peak_features[f'Delta_Energy_Win{i+1}'] = 0.0
    # ---------------------------------------------------------
    
    # Return the dictionary (combining stats and the new peak dictionary)
    base_features = {
        'Unit_ID': unit_id,
        'Run_ID': file_id,
        'CCF_LF': ccf_lf, 'LCC_LF': lcc_lf, 'SDA_LF': sda_lf, 'SE_LF': se_lf, 'CSD_LF': csd_lf,
        'CCF_MF': ccf_mf, 'LCC_MF': lcc_mf, 'SDA_MF': sda_mf, 'SE_MF': se_mf, 'CSD_MF': csd_mf,
        'CCF_HF': ccf_hf, 'LCC_HF': lcc_hf, 'SDA_HF': sda_hf, 'SE_HF': se_hf, 'CSD_HF': csd_hf,
    }
    
    # Merge the dictionaries
    base_features.update(peak_features)
    
    return base_features

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
    
    stat_columns = [col for col in df.columns if any(x in col for x in ['CCF', 'LCC', 'SDA', 'SE', 'CSD'])]
    if df[stat_columns].isna().any().any():
        raise AssertionError("FAIL: NaNs detected in statistical calculations. Check zero-division catches.")
        
    assert df['CCF_HF'].between(-1.0, 1.0).all(), "FAIL: CCF contains values outside [-1, 1]."
    assert (df['SDA_HF'] >= 0).all(), "FAIL: SDA contains negative values (Mathematically impossible)."
    assert (df['CSD_HF'] >= 0).all(), "FAIL: CSD contains negative values (Mathematically impossible)."

    if (df['Run_ID'] == 'Trace_0.txt').any():
        raise AssertionError("FAIL: Baseline 'Trace_0.txt' leaked into the final ML dataset.")

    print("  [PASS] Mathematical Bounds Confirmed (-1 <= CCF <= 1, SDA/CSD >= 0).")
    print("  [PASS] Zero-Division & NaN Safety Confirmed.")
    print("  [PASS] Baseline Leakage successfully prevented.")
    print("[VALIDATION COMPLETE] Dataset ready for Machine Learning.\n")

# ==========================================
# 4. MAIN EXECUTION (Dynamic Crawler)
# ==========================================
if __name__ == "__main__":
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(SCRIPT_DIR, "Dataset")
    
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"FATAL: The 'Dataset' directory was not found at {DATASET_DIR}. Run the generator first.")
        
    all_rows = []

    # Dynamically crawl all folders inside the Dataset directory
    for folder_name in os.listdir(DATASET_DIR):
        target_folder_path = os.path.join(DATASET_DIR, folder_name)
        
        if not os.path.isdir(target_folder_path):
            continue # Skip loose files
            
        print(f"Processing condition: {folder_name}...")
        
        # ---------------------------------------------------------
        # PARSING METADATA FROM FOLDER NAME
        # Example: Simulation_Radial_Moderate_Loc_8-9-10
        # ---------------------------------------------------------
        if folder_name == "Simulation_Healthy":
            fault_type = "Healthy"
            severity = "0%"
            fault_location = "None"
        else:
            try:
                parts = folder_name.split('_')
                fault_type = parts[1]        # e.g., 'Radial'
                severity = parts[2]          # e.g., 'Moderate'
                fault_location = parts[4]    # e.g., '8-9-10'
            except IndexError:
                print(f"  -> Error: Folder name '{folder_name}' does not match expected format. Skipping.")
                continue

        # Get the numeric label for the ML pipeline
        label = CLASS_LABELS.get(fault_type, -1)
        
        # ---------------------------------------------------------
        # DATA EXTRACTION LOOP
        # ---------------------------------------------------------

        # Grab all folders, then randomly shuffle them before picking 20
        # Grab absolutely every unit folder to build the Master CSV
        unit_folders = glob.glob(os.path.join(target_folder_path, 'Unit_*'))
        
        for unit_path in unit_folders:
            unit_name = os.path.basename(unit_path)
            
            local_baseline_path = os.path.join(unit_path, 'Trace_0.txt')
            if not os.path.exists(local_baseline_path):
                print(f"  -> Error: Local baseline missing in {unit_name}. Skipping unit.")
                continue
                
            ref_freqs, ref_mags = read_sim_data(local_baseline_path)
            if ref_freqs is None: continue 
            
            txt_files = glob.glob(os.path.join(unit_path, '*.txt'))
            
            for filepath in txt_files:
                filename = os.path.basename(filepath)
                
                if filename == 'Trace_0.txt':
                    continue
                
                freqs, mags = read_sim_data(filepath)
                if freqs is None: continue
                
                # Append Context Metadata to ensure Global Uniqueness
                global_unit_id = f"{folder_name}_{unit_name}"
                feature_row = extract_features_single(freqs, mags, ref_mags, filename, global_unit_id)
                
                feature_row['Source_Condition'] = folder_name
                feature_row['True_Label'] = label
                feature_row['Severity'] = severity
                feature_row['Location_of_fault'] = fault_location
                
                all_rows.append(feature_row)

    # 4. Final Compile and Export
    if all_rows:
        master_dataset = pd.DataFrame(all_rows)
        
        front_cols = ['Source_Condition', 'Unit_ID', 'Run_ID', 'True_Label', 'Severity', 'Location_of_fault']
        remaining_cols = [c for c in master_dataset.columns if c not in front_cols]
        master_dataset = master_dataset[front_cols + remaining_cols]
        
        output_filepath = os.path.join(SCRIPT_DIR, OUTPUT_FILE)
        master_dataset.to_csv(output_filepath, index=False)
        print(f"\nExtraction complete. Saved to {output_filepath}")
        
        # 5. Run the strict CS/FRA Validation Suite
        validate_ml_dataset(output_filepath)
        
    else:
        print("\nNo data processed. Ensure the Dataset folder is populated.")
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

USE_DAMPING_COMPENSATION = True
USE_PEAK_REFINEMENT = True   # Set False to revert to raw argmax/argmin


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

def refine_extremum(freqs, mags, idx):
    """
    Refine peak/valley position using parabolic interpolation on three points.
    The math for finding the vertex works identically for both maxima and minima.
    Returns interpolated frequency (Hz).
    """
    if idx == 0 or idx == len(mags)-1:
        return freqs[idx]
    a = mags[idx-1]
    b = mags[idx]
    c = mags[idx+1]
    denom = (a - 2*b + c)
    if denom == 0:
        offset = 0.0
    else:
        offset = (a - c) / (2.0 * denom)
    offset = max(-0.5, min(0.5, offset))   # safe clamping
    # Assume linear frequency spacing (your simulator uses uniform steps in Hz)
    step = freqs[1] - freqs[0]
    interp_freq = freqs[idx] + offset * step
    return interp_freq

def compensate_damping(freqs, mags, poly_order=2):
    """
    Removes a smooth polynomial baseline from the magnitude spectrum.
    This accentuates resonances that are otherwise buried in a rising/falling trend.
    Returns compensated magnitudes (same shape as mags).
    """
    if not USE_DAMPING_COMPENSATION:
        return mags  # No change
    
    # Fit polynomial to the magnitude (dB) as function of log(frequency)
    log_freqs = np.log10(freqs + 1e-12)  # avoid log(0)
    coeffs = np.polyfit(log_freqs, mags, deg=poly_order)
    baseline = np.polyval(coeffs, log_freqs)
    compensated = mags - baseline
    # Optional: re-normalise to keep the range similar to original
    compensated = compensated - np.min(compensated) + np.min(mags)  # preserve offset
    return compensated

def extract_features_single(freqs, mags, ref_mags, file_id, unit_id):
    """
    Splits a run into sub-bands and extracts features/peaks/valleys using Windowed Maxima/Minima.
    If USE_DAMPING_COMPENSATION is True, extremum detection is performed on
    damped-compensated magnitude spectra (baseline subtraction).
    """
    mask_lf = (freqs >= FREQ_BANDS['LF'][0]) & (freqs < FREQ_BANDS['LF'][1])
    mask_mf = (freqs >= FREQ_BANDS['MF'][0]) & (freqs <= FREQ_BANDS['MF'][1])
    mask_hf = (freqs > FREQ_BANDS['HF'][0])
    
    ccf_lf, lcc_lf, sda_lf, se_lf, csd_lf = calculate_statistical_indices(ref_mags[mask_lf], mags[mask_lf])
    ccf_mf, lcc_mf, sda_mf, se_mf, csd_mf = calculate_statistical_indices(ref_mags[mask_mf], mags[mask_mf])
    ccf_hf, lcc_hf, sda_hf, se_hf, csd_hf = calculate_statistical_indices(ref_mags[mask_hf], mags[mask_hf])
    
    # ---------------------------------------------------------
    # Windowed Extrema (Peaks & Valleys) Extraction 
    # ---------------------------------------------------------
    windows = [
        (10, 10000),       # Window 1: Deep Core/Inductive
        (10000, 100000),   # Window 2: Main HV Resonances
        (100000, 300000),  # Window 3: Main LV Resonances
        (300000, 600000),  # Window 4: Capacitive Tail
        (600000, 1000000)  # Window 5: Extreme Frequencies
    ]
    
    peak_features = {}
    
    for i, (f_min, f_max) in enumerate(windows):
        window_mask = (freqs >= f_min) & (freqs < f_max)
        window_freqs = freqs[window_mask]
        window_mags = mags[window_mask]
        ref_window_mags = ref_mags[window_mask]
        
        if len(window_mags) > 0:
            # 1. Sub-band statistical indices (always on raw data)
            ccf, lcc, sda, se, csd = calculate_statistical_indices(ref_window_mags, window_mags)
            peak_features[f'SDA_Win{i+1}'] = sda
            peak_features[f'CCF_Win{i+1}'] = ccf
            peak_features[f'CSD_Win{i+1}'] = csd
            
            # Apply compensation to both fault and reference magnitudes for extrema detection
            comp_mags = compensate_damping(window_freqs, window_mags)
            comp_ref_mags = compensate_damping(window_freqs, ref_window_mags)
            
            # 2a. Delta Peaks (Resonances - finding the SINGLE absolute maximum)
            max_idx_fault = np.argmax(comp_mags)
            max_idx_ref   = np.argmax(comp_ref_mags)

            if USE_PEAK_REFINEMENT:
                f_peak_fault = refine_extremum(window_freqs, comp_mags, max_idx_fault)
                f_peak_ref   = refine_extremum(window_freqs, comp_ref_mags, max_idx_ref)
            else:
                f_peak_fault = window_freqs[max_idx_fault]
                f_peak_ref   = window_freqs[max_idx_ref]

            peak_features[f'Delta_Peak_{i+1}_Hz'] = f_peak_fault - f_peak_ref
            peak_features[f'Delta_Peak_{i+1}_dB'] = window_mags[max_idx_fault] - ref_window_mags[max_idx_ref]
            
            # 2b. Delta Valleys (Anti-resonances - finding the SINGLE absolute minimum)
            min_idx_fault = np.argmin(comp_mags)
            min_idx_ref   = np.argmin(comp_ref_mags)

            if USE_PEAK_REFINEMENT:
                f_val_fault = refine_extremum(window_freqs, comp_mags, min_idx_fault)
                f_val_ref   = refine_extremum(window_freqs, comp_ref_mags, min_idx_ref)
            else:
                f_val_fault = window_freqs[min_idx_fault]
                f_val_ref   = window_freqs[min_idx_ref]

            peak_features[f'Delta_Valley_{i+1}_Hz'] = f_val_fault - f_val_ref
            peak_features[f'Delta_Valley_{i+1}_dB'] = window_mags[min_idx_fault] - ref_window_mags[min_idx_ref]

            
            # 3. Sub-band spectral energy (still on raw data, unchanged)
            energy_ref = np.trapezoid(np.abs(ref_window_mags), window_freqs)
            energy_fault = np.trapezoid(np.abs(window_mags), window_freqs)
            peak_features[f'Delta_Energy_Win{i+1}'] = energy_fault - energy_ref
        
        else:
            # Empty window fallback
            for suffix in ['SDA', 'CCF', 'CSD']:
                peak_features[f'{suffix}_Win{i+1}'] = 0.0
            peak_features[f'Delta_Peak_{i+1}_Hz'] = 0.0
            peak_features[f'Delta_Peak_{i+1}_dB'] = 0.0
            peak_features[f'Delta_Valley_{i+1}_Hz'] = 0.0
            peak_features[f'Delta_Valley_{i+1}_dB'] = 0.0
            peak_features[f'Delta_Energy_Win{i+1}'] = 0.0
    
    # ---------------------------------------------------------
    
    base_features = {
        'Unit_ID': unit_id,
        'Run_ID': file_id,
        'CCF_LF': ccf_lf, 'LCC_LF': lcc_lf, 'SDA_LF': sda_lf, 'SE_LF': se_lf, 'CSD_LF': csd_lf,
        'CCF_MF': ccf_mf, 'LCC_MF': lcc_mf, 'SDA_MF': sda_mf, 'SE_MF': se_mf, 'CSD_MF': csd_mf,
        'CCF_HF': ccf_hf, 'LCC_HF': lcc_hf, 'SDA_HF': sda_hf, 'SE_HF': se_hf, 'CSD_HF': csd_hf,
    }
    
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
        
    CONFIGS_TO_PROCESS = ['HV', 'LV']

    for config_name in CONFIGS_TO_PROCESS:
        config_path = os.path.join(DATASET_DIR, config_name)
        
        if not os.path.exists(config_path):
            print(f"\n[SKIP] Directory not found for {config_name} at {config_path}. Skipping...")
            continue
            
        print("\n" + "="*50)
        print(f" EXTRACTING FEATURES FOR {config_name} DATASET")
        print("="*50)

        all_rows = []

        # Dynamically crawl all folders inside the specific config directory (HV or LV)
        for folder_name in os.listdir(config_path):
            target_folder_path = os.path.join(config_path, folder_name)
            
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
                    global_unit_id = f"{config_name}_{folder_name}_{unit_name}"
                    feature_row = extract_features_single(freqs, mags, ref_mags, filename, global_unit_id)
                    
                    feature_row['Source_Condition'] = folder_name
                    feature_row['True_Label'] = label
                    feature_row['Severity'] = severity
                    feature_row['Location_of_fault'] = fault_location
                    
                    all_rows.append(feature_row)

        # Final Compile and Export for this specific config
        if all_rows:
            master_dataset = pd.DataFrame(all_rows)
            
            front_cols = ['Source_Condition', 'Unit_ID', 'Run_ID', 'True_Label', 'Severity', 'Location_of_fault']
            remaining_cols = [c for c in master_dataset.columns if c not in front_cols]
            master_dataset = master_dataset[front_cols + remaining_cols]
            
            output_filepath = os.path.join(SCRIPT_DIR, f"MASTER_ML_Dataset_{config_name}.csv")
            master_dataset.to_csv(output_filepath, index=False)
            print(f"\nExtraction complete for {config_name}. Saved to {output_filepath}")
            
            # Run the strict CS/FRA Validation Suite on the generated file
            validate_ml_dataset(output_filepath)
            
        else:
            print(f"\nNo data processed for {config_name}. Ensure the Dataset/{config_name} folder is populated.")
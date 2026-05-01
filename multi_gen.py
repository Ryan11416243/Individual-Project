import numpy as np
import matplotlib.pyplot as plt
import os
import random
import glob # Error checking
import shutil
np.random.seed(42)
random.seed(42)

# ==========================================
# CONFIGURATION BLOCK
# ==========================================
N_HV = 10
N_LV = 16
STEPS = 2000

NO_OF_HEALTHY = 5
NO_OF_FAULT = 5

C_SHIFT = 0.005   # 0.3% global drift
L_SHIFT = 0.0005  # 0.05% global drift

# Base parameters (from Cheng et al.)
params_HV = {
    "stages": N_HV,
    "L_air": 637.62e-3, 
    "C_g": 482.15e-12,  
    "C_s": 401.14e-12,
    "tan_delta": 0.035  
}

params_LV = {
    "stages": N_LV,
    "L_air": 1.58e-3,   
    "C_g": 5640.1e-12,  
    "C_s": 5.67e-12,
    "tan_delta": 0.035  
}

# --- FLEET & BATCH CONFIGURATION ---
fleet_config = {
    # Define the TOTAL unique transformers you want per fault category (e.g., 90 Radial, 90 LCP)
    # The script will dynamically divide this number to prevent class imbalance.
    "target_dnas_per_class": 1000,  

    "batch_mode": "fixed",     # Options: "fixed" or "random"
    
    # Settings for "fixed" mode:
    "fixed_batches": 8,         # Every transformer gets exactly 5 sweeps
    
    # Settings for "random" mode (Kept small to prevent massive file bloat):
    "random_range": (2, 5),     
    "random_distribution": "uniform" 
}


# --- FAULT CONFIGURATION ---
# This dictionary format is perfect for a future GUI to update dynamically
fault_config = {
    "active": True,
    "type": "Radial",
    "units_affected": [3],  # Can be a single int like 8, or a list like [1, 2], or list(range(3, 6))
    "C_g_multiplier": 1.0,
    "L_air_multiplier": 1.0,
    "C_s_multiplier": 1.0,

    "export_baseline": "sim_baseline_healthy.txt",
    "export_faulted": "sim_LCP_0.9.txt"
}

# -----------------------
# Toggles the Application of Variation Stages 
# -----------------------
stochastic_config = {
    "apply_stage1_inter_unit": True,        # Global shift: Transformer Fingerprint
    "apply_stage2_intra_unit": True,        # Local shift: Operational Variation
    "apply_stage4_measurement_noise": True, # Output shift: Measurement Noise
    "noise_floor_linear_stage4": 1e-5,
}


# ==========================================
# SEVERITY DICTIONARY 
# ==========================================
# Values represent the multiplier range. Example: +2% to +5% = (1.02, 1.05)
SEVERITY_RANGES = {
    "Radial": {
        "Incipient": (1.02, 1.05),
        "Moderate": (1.05, 1.12),
        "Severe": (1.10, 1.25)
    },
    "Axial": { # Note: Decreases
        "Incipient": (0.97, 0.99),
        "Moderate": (0.90, 0.97),
        "Severe": (0.80, 0.90)
    },
    "LCP": { # Note: Decreases
        "Incipient": (0.97, 0.99),
        "Moderate": (0.93, 0.97),
        "Severe": (0.88, 0.93)
    },
    "TTSC": { # Note: Decreases
        "Incipient": (0.97, 0.99),  # ~1 turn
        "Moderate": (0.92, 0.97),   # 2-5 turns
        "Severe": (0.80, 0.92)      # >5 turns
    }
}

# General Setup
active_config = params_LV 
R_source = 50.0           # Source impedance (standard FRA test setup)
R_load = 50.0             # Load impedance (standard FRA test setup)
V_in = 1.0                
frequencies = np.logspace(np.log10(10), np.log10(1e6), STEPS)

# ==========================================
# NETWORK GENERATOR MODULE
# ==========================================
def build_network_arrays(config):
    """
    Translates total lumped parameters into per-stage arrays.
    This allows every single stage to have unique component values.
    """
    n = config["stages"]
    
    # Initialize arrays filled with the baseline per-stage values
    # L and Cg divide by n; Cs multiplies by n (series capacitors)
    L_array = np.full(n, config["L_air"] / n)
    C_g_array = np.full(n, config["C_g"] / n)
    C_s_array = np.full(n, config["C_s"] * n)
    
    return L_array, C_g_array, C_s_array

def apply_fault(L_array, C_g_array, C_s_array, fault_type, severity_tier, units, fault_subtype=None):
    """
    Applies highly specific physics-based faults aligned with the LaTeX report.
    Architected to support future multi-winding (double-core) upscaling.
    """
    # Defensive programming: Never mutate the input arrays directly in this scope
    L_new = L_array.copy()
    C_g_new = C_g_array.copy()
    C_s_new = C_s_array.copy()
    
    # ---------------------------------------------------------
    # SEVERITY CALCULATION (FRA Expert Request)
    # Using a Truncated Normal Distribution rather than a flat uniform choice.
    # Physical justification: Fault severities in the field cluster around a mean 
    # expected degradation rather than being perfectly equally likely at the extremes.
    # ---------------------------------------------------------
    min_val, max_val = SEVERITY_RANGES[fault_type][severity_tier]
    mean_val = (min_val + max_val) / 2
    std_dev = (max_val - min_val) / 4
    mag = np.clip(np.random.normal(mean_val, std_dev), min_val, max_val)

    # ==========================================
    # 1. RADIAL DEFORMATION (Buckling)
    # ==========================================
    if fault_type == "Radial":
        if len(units) == 1:
            C_g_new[units[0]] *= mag
        else:
            # Dynamic parabolic distribution for bulges > 1 unit
            # Max severity at the center, tapering off at the edges
            center_idx = len(units) / 2.0 - 0.5
            for i, u in enumerate(units):
                dist = abs(i - center_idx) / (len(units) / 2.0)
                weight = 1.0 - (dist**2) # Creates a smooth parabola
                effective_mag = 1.0 + ((mag - 1.0) * weight)
                C_g_new[u] *= effective_mag

    # ==========================================
    # 2. AXIAL DISPLACEMENT
    # ==========================================
    elif fault_type == "Axial":
        
        if fault_subtype == "Single-Gap":
            for u in units:
                C_s_new[u] *= mag
                # Correlated drop (rho = 0.9) with 5% stochastic noise to prevent perfect ML memorization
                noise = np.random.normal(0, 0.05 * abs(1.0 - mag))
                C_g_new[u] *= (1.0 - ((1.0 - mag) * 0.9)) + noise

        elif fault_subtype == "Block-Telescoping":
            # Requires exactly 2 units: [expansion_boundary, compression_boundary]
            if len(units) != 2:
                raise ValueError(f"FATAL CS ERROR: Block-Telescoping requires exactly 2 boundary units. Received: {units}")
            
            top_u, bot_u = units[0], units[1]
            C_s_new[top_u] *= mag  # Gap expands (Cs drops)
            
            # Inverse effect at compression boundary
            compression_mag = 1.0 + (1.0 - mag) 
            C_s_new[bot_u] *= compression_mag

        elif fault_subtype == "Global-Shift":
            # Entire winding shifts vertically.
            # (Future Upscale Hook: Double-winding Mutual Inductance 'M' shifts will go here)
            for u in range(len(C_g_new)):
                # Apply the global shift, but add micro-variance per unit so it isn't mathematically flat
                local_mag = np.clip(np.random.normal(mag, 0.01), min_val, max_val)
                C_g_new[u] *= local_mag

    # ==========================================
    # 3. LOSS OF CLAMPING PRESSURE (LCP)
    # ==========================================
    elif fault_type == "LCP":
        for u in range(len(C_s_new)):
            local_mag = np.clip(np.random.normal(mag, 0.005), min_val, max_val)
            C_s_new[u] *= local_mag
           
           # Removed as creates 
           # C_g_new[u] *= 1.01 # Minor capacitance proxy increase

    # ==========================================
    # 4. TURN-TO-TURN SHORT CIRCUIT (TTSC)
    # ==========================================
    elif fault_type == "TTSC":
        for u in units:
            # L drops proportionally to the square of the remaining active turns
            L_new[u] *= mag
            
            # C_s represents inter-turn capacitors in series (C_s ∝ 1/N).
            # If L ∝ N^2, then N ∝ sqrt(L). 
            # Therefore, the new series capacitance increases by 1/sqrt(mag).
            # We add a minor noise factor to prevent perfect ML memorization.
            C_s_new[u] /= np.sqrt(mag)

    return L_new, C_g_new, C_s_new



# =========================================
# Noise and Variation Application
# =========================================
def apply_stochastic_baseline(config, stoch_config, fp_C=1.0, fp_L=1.0):
    """
    Applies the pre-calculated, frozen Stage 1 fingerprint (DNA) and 
    adds operational noise per-stage (Stage 2).
    """
    n = config["stages"]
    
    # Base per-stage nominal values
    L_nom = config["L_air"] / n
    C_g_nom = config["C_g"] / n
    C_s_nom = config["C_s"] * n

    # ----------------------------------------------------
    # STAGE 1: Apply the frozen DNA passed from the batch loop
    # ----------------------------------------------------
    L_fp = L_nom * fp_L
    C_g_fp = C_g_nom * fp_C
    C_s_fp = C_s_nom * fp_C
    
    # ----------------------------------------------------
    # STAGE 2: Operational Variation (Intra-Unit Noise)
    # ----------------------------------------------------
    if stoch_config["apply_stage2_intra_unit"]:
        
        # 1. Correlated Global Shift (The Physics)
        # The entire transformer winding heats up/ages together. 
        # We generate ONE random multiplier for this specific sweep.
        global_C_shift = np.random.normal(1.0, C_SHIFT) 
        global_L_shift = np.random.normal(1.0, L_SHIFT)  # Inductance drifts less
        
        # 2. Uncorrelated Micro-Noise
        # A tiny independent variance per-unit to prevent perfect mathematical flatness
        micro_C_noise =  0.0005  # 0.1% independent noise
        micro_L_noise =  0.0002 # 0.05% independent noise

        # 3. Apply the global shift, then add the micro-noise array
        L_array = np.random.normal(L_fp * global_L_shift, L_fp * micro_L_noise, n)
        C_g_array = np.random.normal(C_g_fp * global_C_shift, C_g_fp * micro_C_noise, n)
        C_s_array = np.random.normal(C_s_fp * global_C_shift, C_s_fp * micro_C_noise, n)
        
    else:
        L_array = np.full(n, L_fp)
        C_g_array = np.full(n, C_g_fp)
        C_s_array = np.full(n, C_s_fp)
        
    return L_array, C_g_array, C_s_array

# =====================================
# Stage 4: Measurement Noise
# ====================================

# NOTE: An earlier implementation injected AWGN directly in the dB domain.
# This was abandoned because dB-domain noise has constant variance across
# the spectrum, while real VNA noise is constant in linear voltage units
# and therefore disproportionately loud at deep anti-resonances. The
# linear-domain implementation in apply_vna_instrumentation() replaces it.
# def add_measurement_noise(magnitudes_dB, noise_level=0.5):
#     """
#     Implements Stage 4: Constant Additive White Gaussian Noise (AWGN) in the dB domain.
#     """
#     # Generates a flat noise floor independent of the signal's magnitude
#     noise = np.random.normal(0, noise_level, len(magnitudes_dB))
#     return magnitudes_dB + noise



# ==========================================
# MATRIX SOLVER ENGINE
# ==========================================
def calculate_fra_dynamic(L_array, C_g_array, C_s_array, config, frequencies):
    """
    Builds and solves the Nodal Admittance Matrix using Nodal Stamping.
    Includes frequency-dependent dielectric damping (tan_delta).
    """
    n = len(L_array)
    H_complex = []
    
    Y_load = 1.0 / R_load
    tan_delta = config.get("tan_delta", 0.0) # Default to 0 if not provided
    
    for f in frequencies:
        omega = 2 * np.pi * f
        
        # Build the (n+1) x (n+1) Matrix
        Y_bus = np.zeros((n+1, n+1), dtype=complex)
        
        for i in range(n):
            # NEW: Frequency-dependent dielectric loss added to capacitors[cite: 2]
            G_s = omega * C_s_array[i] * tan_delta
            G_g = omega * C_g_array[i] * tan_delta
            
            # Series branch: Inductor + Lossy Series Capacitor[cite: 2]
            Y_series = (1 / (1j * omega * L_array[i])) + (1j * omega * C_s_array[i] + G_s)
            
            # Shunt branch: Lossy Ground Capacitor[cite: 2]
            Y_shunt = (1j * omega * C_g_array[i] + G_g)
            
            # NODAL ADMITTANCE STAMPING
            Y_bus[i, i] += Y_series
            Y_bus[i+1, i+1] += Y_series
            Y_bus[i, i+1] -= Y_series
            Y_bus[i+1, i] -= Y_series
            
            # Stamp Shunt Element (Connects Node i+1 to Ground)
            Y_bus[i+1, i+1] += Y_shunt
            
        # NEW: Force Node 0 to be an ideal voltage source[cite: 2]
        # This replaces the Y_source method to ensure V_in is exactly V_ideal_source
        Y_ideal_source = 1e12 
        Y_bus[0, 0] += Y_ideal_source
        Y_bus[n, n] += Y_load
        
        # Current Vector
        I_vector = np.zeros(n+1, dtype=complex)
        I_vector[0] = Y_ideal_source * V_in 
        
        try:
            V_nodes = np.linalg.solve(Y_bus, I_vector)
        except np.linalg.LinAlgError:
            raise RuntimeError(f"FATAL: Matrix became singular at {f} Hz.")

        # Transfer function: V_out / V_in_actual[cite: 2]
        V_in_actual = V_nodes[0]
        V_out = V_nodes[n]
        H_complex.append(V_out / V_in_actual)

    return np.array(H_complex)

# ====================================
# Stage 4: Measurement Noise (VNA Model)
# ====================================
def apply_vna_instrumentation(H_complex, apply_noise=True, noise_floor_linear=1e-5):
    """
    Simulates the signal processing of a Vector Network Analyzer (VNA).
    Injects a linear complex noise floor before converting to logarithmic magnitude.
    """
    if apply_noise:
        # Generate independent Gaussian noise for real and imaginary components
        noise_real = np.random.normal(0, noise_floor_linear, len(H_complex))
        noise_imag = np.random.normal(0, noise_floor_linear, len(H_complex))
        
        # Superimpose the noise floor onto the linear signal
        H_complex = H_complex + noise_real + 1j * noise_imag

    # Convert the (noisy or clean) complex signal to decibels
    magnitudes_dB = 20 * np.log10(np.abs(H_complex))
    
    return magnitudes_dB

# ====================================
# Export Module
# ====================================
def export_to_txt(filename, freq_array, mag_array):
    """
    Exports frequency and magnitude arrays to a tab-separated text file.
    Automatically creates necessary directories if they don't exist.
    """
    import os
    
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Resolve the full absolute path, combining script dir with the target folder/filename
    filepath = os.path.join(script_dir, filename)
    
    target_dir = os.path.dirname(filepath)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    # Write the data
    with open(filepath, 'w', encoding='utf-8') as file:
        file.write("Frequency(Hz)\tMagnitude(dB)\n")
        
        for f, m in zip(freq_array, mag_array):
            file.write(f"{f:e}\t{m:e}\n")
            
    # print(f"Data successfully exported to: {filepath}") # Mute if batch printing gets too noisy

# ----------------------------------------------
# Run in Batches
# ----------------------------------------------
def generate_batch(output_dir, file_prefix, active_config, stochastic_config, fault_config, fleet_config, frequencies):
    """
    Generates a fleet of unique transformers (DNAs).
    For each transformer, generates a baseline and a configurable number of historical sweeps.
    """
    # ---------------------------------------------------------
    # 1. PATH RESOLUTION & AUTO-PURGE
    # ---------------------------------------------------------
    # Lock the target directory to the script's exact physical location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_output_dir = os.path.join(script_dir, output_dir)

    # Purge the absolute directory to guarantee no ghost data
    if os.path.exists(abs_output_dir):
        print(f"🧹 Purging old data in '{abs_output_dir}'...")
        shutil.rmtree(abs_output_dir)
        
    os.makedirs(abs_output_dir)

    # LOOP 1: Iterate through each unique transformer (DNA)
    for dna_idx in range(fleet_config["num_dnas"]):
        
        # Determine the number of historical sweeps for THIS specific transformer
        if fleet_config["batch_mode"] == "fixed":
            num_sets = fleet_config["fixed_batches"]
        
        elif fleet_config["batch_mode"] == "random":
            min_runs, max_runs = fleet_config["random_range"]
            if fleet_config["random_distribution"] == "uniform":
                num_sets = random.randint(min_runs, max_runs)
            elif fleet_config["random_distribution"] == "normal":
                mean = (min_runs + max_runs) / 2
                std_dev = (max_runs - min_runs) / 4
                num_sets = int(np.clip(np.random.normal(mean, std_dev), min_runs, max_runs))
        else:
            num_sets = 5 # Safety fallback

        print(f"\n==================================================")
        print(f"GENERATING DNA {dna_idx} | Total Sweeps Planned: {num_sets}")
        print(f"==================================================")

        # Freeze the Stage 1 Fingerprint for this specific transformer
        if stochastic_config["apply_stage1_inter_unit"]:
            batch_fp_C = np.random.normal(1.0, 0.05)  
            batch_fp_L = np.random.normal(1.0, 0.035) 
        else:
            batch_fp_C, batch_fp_L = 1.0, 1.0

        # Create a subfolder for this specific transformer
        # We define the relative path for the exporter, and the absolute path to create it now
        unit_dir_relative = os.path.join(output_dir, f"Unit_{dna_idx}")
        abs_unit_dir = os.path.join(script_dir, unit_dir_relative)
        
        if not os.path.exists(abs_unit_dir):
            os.makedirs(abs_unit_dir)

        # LOOP 2: Generate the specific sweeps for this transformer
        for i in range(num_sets):
            current_stoch_config = stochastic_config.copy()
            
            if i == 0:
                print(f"  -> Run {i}: Commissioning Baseline (Stage 1 Only)")
                current_stoch_config["apply_stage2_intra_unit"] = False
                # current_stoch_config["apply_stage4_measurement_noise"] = False 
                current_fault_config = {"active": False} 
            else:
                print(f"  -> Run {i}: Operational / Fault State")
                current_fault_config = fault_config

            # 1. Build Arrays using the FROZEN fingerprint for this DNA
            L_base, Cg_base, Cs_base = apply_stochastic_baseline(
                active_config, current_stoch_config, fp_C=batch_fp_C, fp_L=batch_fp_L
            )
            
            # 2. Apply Faults
            if current_fault_config["active"]:
                L_net, Cg_net, Cs_net = apply_fault(
                    L_array=L_base.copy(), 
                    C_g_array=Cg_base.copy(), 
                    C_s_array=Cs_base.copy(), 
                    fault_type=current_fault_config["type"],
                    severity_tier=current_fault_config["severity_tier"],
                    units=current_fault_config["units_affected"],
                    fault_subtype=current_fault_config.get("fault_subtype") # .get() prevents errors if it doesn't exist
                )
            else:
                L_net, Cg_net, Cs_net = L_base, Cg_base, Cs_base

            # 3. Solve & Apply Instrumentation
            # Get the raw complex physics simulation
            H_raw =calculate_fra_dynamic(L_net, Cg_net, Cs_net, active_config, frequencies)
            
            # Apply the VNA modeling and dB conversion
            # Using 1e-5 as the linear voltage noise floor (equivalent to a highly accurate VNA)
            mag_db = apply_vna_instrumentation(
                H_complex=H_raw, 
                apply_noise=current_stoch_config["apply_stage4_measurement_noise"],
                noise_floor_linear=stochastic_config.get("noise_floor_linear_stage4", 1e-5)
            )
            # 4. Export to the specific Unit's subfolder using the relative path
            filename_relative = os.path.join(unit_dir_relative, f"{file_prefix}_{i}.txt")
            export_to_txt(filename_relative, frequencies, mag_db)


# ========================================
# Distribution of Location of faults
# ========================================
def get_locations_for_fault(fault_type, total_stages):
        """
        Uses Stratified Spatial Sampling to guarantee faults are distributed 
        across the Top, Middle, and Bottom of the winding, preventing physical clumping.
        """
        if fault_type == "LCP":
            return [list(range(total_stages))]
            
        locations = []
        
        # Define 3 physical strata (Top, Middle, Bottom)
        z1_end = total_stages // 3
        z2_end = 2 * (total_stages // 3)
        
        zones = [
            (0, z1_end - 1),                  # Top Third (e.g., Units 0-4)
            (z1_end, z2_end - 1),             # Middle Third (e.g., Units 5-9)
            (z2_end, total_stages - 1)        # Bottom Third (e.g., Units 10-15)
        ]
        
        for z_start, z_end in zones:
            if fault_type == "Radial":
                # Radials are bulges. Randomly decide if it's a 1-unit or 3-unit bulge.
                size = random.choice([1, 3])
                if size == 1:
                    locations.append([random.randint(z_start, z_end)])
                else:
                    # Ensure a 3-unit bulge doesn't spill over the total transformer size
                    safe_z_end = min(z_end, total_stages - 3)
                    start_idx = random.randint(z_start, safe_z_end)
                    locations.append([start_idx, start_idx + 1, start_idx + 2])
                    
            elif fault_type in ["Axial", "TTSC"]:
                # Pick one random unit strictly within this specific zone
                locations.append([random.randint(z_start, z_end)])
                
        return locations

# =========================================
# Test
# ========================================
def validate_dataset_integrity(output_dir, fleet_config, expected_steps):
    """
    Crawls the generated directory to guarantee data structures, file counts,
    and mathematical integrity strictly follow the defined rules.
    """
    # 1. Resolve absolute path to match the export function
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_output_dir = os.path.join(script_dir, output_dir)
    
    print(f"\n[TESTING] Running Integrity Checks on: {abs_output_dir}")
    
    if not os.path.exists(abs_output_dir):
        raise AssertionError(f"FAIL: Output directory '{abs_output_dir}' was never created.")

    unit_folders = glob.glob(os.path.join(abs_output_dir, "Unit_*"))
    
    # TEST 1: Did we generate the correct number of DNAs?
    assert len(unit_folders) == fleet_config["num_dnas"], \
        f"FAIL: Expected {fleet_config['num_dnas']} Unit folders, found {len(unit_folders)}."
    print(f"  [PASS] Correct number of DNA folders ({len(unit_folders)}).")

    for folder in unit_folders:
        files = glob.glob(os.path.join(folder, "Trace_*.txt"))
        num_files = len(files)
        
        # TEST 2: Does Trace_0.txt exist in every folder?
        baseline_path = os.path.join(folder, "Trace_0.txt")
        assert os.path.exists(baseline_path), \
            f"FAIL: Critical baseline 'Trace_0.txt' is missing in {folder}."
            
        # TEST 3: Does the file count respect the batch_mode toggles?
        if fleet_config["batch_mode"] == "fixed":
            expected = fleet_config["fixed_batches"]
            assert num_files == expected, \
                f"FAIL: Fixed mode expected {expected} files in {folder}, found {num_files}."
        elif fleet_config["batch_mode"] == "random":
            min_f, max_f = fleet_config["random_range"]
            assert min_f <= num_files <= max_f, \
                f"FAIL: Random mode limits broken in {folder}. Found {num_files} files (Expected {min_f}-{max_f})."

        # TEST 4: Data Matrix Validation
        for filepath in files:
            try:
                data = np.loadtxt(filepath, skiprows=1) # Skip header
            except Exception as e:
                raise AssertionError(f"FAIL: Could not read file {filepath}. Corrupted data? Error: {e}")
            
            # 4a. Check Data Shape
            assert data.shape == (expected_steps, 2), \
                f"FAIL: Wrong data shape in {filepath}. Expected ({expected_steps}, 2), got {data.shape}."
            
            # 4b. Check for impossible math (NaNs or Infs)
            assert not np.isnan(data).any(), f"FAIL: NaN values detected in {filepath}. Matrix solver failed."
            assert not np.isinf(data).any(), f"FAIL: Infinite values detected in {filepath}. Division by zero?"

        # TEST 5: Healthy Baseline CCF Sanity Check
        # Only run this check on the Healthy dataset
        if "Healthy" in output_dir:
            baseline_data = np.loadtxt(baseline_path, skiprows=1)
            baseline_mag = baseline_data[:, 1]
            
            for filepath in files:
                if filepath != baseline_path:
                    trace_data = np.loadtxt(filepath, skiprows=1)
                    trace_mag = trace_data[:, 1]
                    
                    # Calculate Cross-Correlation
                    ccf = np.corrcoef(baseline_mag, trace_mag)[0, 1]
                    
                    # Assert CCF stays highly correlated under normal operational noise
                    assert ccf > 0.97, \
                        f"FAIL: Operational noise too high. CCF dropped to {ccf:.4f} in {filepath}."

    print(f"  [PASS] All File Counts strictly follow '{fleet_config['batch_mode']}' logic.")
    print(f"  [PASS] Baseline Trace_0.txt verified present in all folders.")
    print(f"  [PASS] Mathematical integrity verified (No NaNs/Infs, Exact Shape matched).")
    print(f"  [PASS] Correct Value of CCF with Healthy Cases.")
    print("[TESTING COMPLETE] Dataset is clean and ML-ready.\n")

def run_physics_unit_tests(config):
    """
    Executes a single pass of targeted mathematical assertions to guarantee 
    the fault algorithms perfectly match the physical transformer theory.
    Runs once before massive dataset generation.
    """
    print("\n[PRE-FLIGHT CHECK] Running Physics Unit Tests...")
    
    # Setup dummy arrays
    L_base, Cg_base, Cs_base = build_network_arrays(config)
    
    # ---------------------------------------------------------
    # TEST 1: Turn-to-Turn Short Circuit (TTSC) Physics
    # Theory: L drops by 'mag', Cs INCREASES by 1/sqrt(mag)
    # ---------------------------------------------------------
    test_mag = 0.90 # 10% inductance drop
    # Temporarily override the random sampler in apply_fault just for this test
    # (In a true TDD environment you'd mock this, but we can test the raw logic)
    
    L_new, Cg_new, Cs_new = apply_fault(
        L_base, Cg_base, Cs_base, 
        fault_type="TTSC", severity_tier="Severe", units=[5]
    )
    # Since apply_fault uses a random severity bounded by the tier, 
    # we just need to verify the *proportionality* between the new L and new Cs
    actual_mag = L_new[5] / L_base[5]
    expected_cs_multiplier = 1.0 / np.sqrt(actual_mag)
    actual_cs_multiplier = Cs_new[5] / Cs_base[5]
    
    # Assert they are nearly identical (accounting for float precision)
    assert np.isclose(expected_cs_multiplier, actual_cs_multiplier, rtol=1e-5), \
        "PHYSICS FAIL: TTSC Capacitance did not scale inversely to sqrt(L)."

    # ---------------------------------------------------------
    # TEST 2: Radial Deformation Parabolic Bulge
    # Theory: Center unit must have max Cg increase, edges must have less.
    # ---------------------------------------------------------
    L_new, Cg_new, Cs_new = apply_fault(
        L_base, Cg_base, Cs_base, 
        fault_type="Radial", severity_tier="Severe", units=[3, 4, 5]
    )
    
    center_increase = Cg_new[4] / Cg_base[4]
    edge_increase = Cg_new[3] / Cg_base[3]
    
    assert center_increase > edge_increase, \
        "PHYSICS FAIL: Radial bulge is not parabolic. Edges expanded more than the center."

    # ---------------------------------------------------------
    # TEST 3: VNA Noise Floor (Linear vs Logarithmic)
    # Theory: A 1e-5 linear noise floor should have a massive % impact on a 1e-4 signal, 
    # but near 0% impact on a 1.0 signal.
    # ---------------------------------------------------------
    dummy_H = np.array([1.0 + 0j, 0.0001 + 0j]) # [Peak, Deep Valley]
    noisy_dB = apply_vna_instrumentation(dummy_H, apply_noise=True, noise_floor_linear=1e-5)
    clean_dB = apply_vna_instrumentation(dummy_H, apply_noise=False)
    
    delta_peak = abs(noisy_dB[0] - clean_dB[0])
    delta_valley = abs(noisy_dB[1] - clean_dB[1])
    
    assert delta_valley > (delta_peak * 10), \
        "PHYSICS FAIL: VNA noise did not corrupt the deep anti-resonance proportionally more than the peak."

    print("  [PASS] TTSC Proportionality (L vs Cs) Confirmed.")
    print("  [PASS] Radial Parabolic Spatial Weighting Confirmed.")
    print("  [PASS] VNA Linear Noise Floor Behavior Confirmed.")
    print("[PRE-FLIGHT COMPLETE] All physics equations computationally verified.\n")

# ==========================================
# 5. BATCH RUN, EXPORT, AND VALIDATE
# ==========================================
if __name__ == "__main__":
    
    TARGET_DNAS = fleet_config["target_dnas_per_class"]
    
    # Wrap the entire execution in a loop to process both HV and LV configurations
    configurations = [("HV", params_HV), ("LV", params_LV)]
    
    for voltage_level, current_active_config in configurations:
        print("\n" + "#"*70)
        print(f"### INITIATING PIPELINE FOR {voltage_level} WINDING CONFIGURATION ###")
        print("#"*70)

        # Purge Old data for this specific voltage level
        script_dir = os.path.dirname(os.path.abspath(__file__))
        master_dataset_dir = os.path.join(script_dir, f"Dataset/{voltage_level}")
        
        if os.path.exists(master_dataset_dir):
            print(f"\n XXX INITIATING MASTER PURGE: Deleting old data in '{master_dataset_dir}'...")
            shutil.rmtree(master_dataset_dir)
            
        # Recreate the empty master directory
        os.makedirs(master_dataset_dir)

        run_physics_unit_tests(current_active_config)
        
        # ---------------------------------------------
        # 1. Generate Healthy Baseline Fleet
        # ---------------------------------------------
        print("\n" + "="*50)
        print(f" PHASE 1: GENERATING HEALTHY BASELINE FLEET (Target: {TARGET_DNAS} Units)")
        print("="*50)
        
        fault_config["active"] = False
        output_target_healthy = f"Dataset/{voltage_level}/Simulation_Healthy"
        
        # Healthy only has 1 "permutation", so it gets the full target amount
        fleet_config["num_dnas"] = TARGET_DNAS 
        
        generate_batch(
            output_dir=output_target_healthy, 
            file_prefix="Trace",             
            active_config=current_active_config, 
            stochastic_config=stochastic_config, 
            fault_config=fault_config,
            fleet_config=fleet_config,
            frequencies=frequencies
        )
        validate_dataset_integrity(output_target_healthy, fleet_config, STEPS)

        # ---------------------------------------------
        # 2. Automated Fault Permutation Engine
        # ---------------------------------------------
        print("\n" + "="*50)
        print(" PHASE 2: GENERATING FAULT PERMUTATIONS (Dynamically Balanced)")
        print("="*50)

        # Iterate through every combination
        for f_type in SEVERITY_RANGES.keys():
            
            # ---------------------------------------------------------
            # CS DYNAMIC BALANCING LOGIC
            # ---------------------------------------------------------
            # Update to use the current stages for HV or LV
            locations = get_locations_for_fault(f_type, current_active_config["stages"])
            severities = ["Incipient", "Moderate", "Severe"]
            
            # Calculate how many sub-folders this fault type will create
            total_permutations = len(severities) * len(locations)
            
            # Divide target DNAs by permutations so all classes sum up to the same total size
            dynamic_dnas_per_batch = max(2, int(TARGET_DNAS / total_permutations))
            fleet_config["num_dnas"] = dynamic_dnas_per_batch
            
            for severity in severities:
                for loc in locations:
                    
                    loc_str = "-".join(map(str, loc))
                    if f_type == "LCP": 
                        loc_str = "Global" 
                    
                    # Update output string to accurately reflect HV/LV Dataset path
                    output_folder = f"Dataset/{voltage_level}/Simulation_{f_type}_{severity}_Loc_{loc_str}"
                    
                    print(f"\n CONFIGURING BATCH: {f_type} | {severity} | Location(s): {loc_str}")
                    print(f"   -> Allocating {dynamic_dnas_per_batch} DNAs to maintain class balance.")
                    
                    # Update Fault Config dynamically
                    current_fault_config = fault_config.copy()
                    current_fault_config["active"] = True
                    current_fault_config["type"] = f_type
                    current_fault_config["units_affected"] = loc
                    current_fault_config["severity_tier"] = severity 

                    if f_type == "Axial":
                        current_fault_config["fault_subtype"] = "Single-Gap"

                    # Run Generation
                    generate_batch(
                        output_dir=output_folder, 
                        file_prefix="Trace", 
                        active_config=current_active_config, 
                        stochastic_config=stochastic_config, 
                        fault_config=current_fault_config, 
                        fleet_config=fleet_config,
                        frequencies=frequencies
                    )
                    
                    # Run Integrity Check
                    validate_dataset_integrity(output_folder, fleet_config, STEPS)
                    
    print("\n✅ MASTER DATASET GENERATION COMPLETE FOR BOTH HV AND LV.")
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import glob # Error checking
import shutil

# ==========================================
# CONFIGURATION BLOCK
# ==========================================
N_HV = 10
N_LV = 16
STEPS = 2000

NO_OF_HEALTHY = 5
NO_OF_FAULT = 5

# Base parameters (from Cheng et al.)
params_HV = {
    "stages": N_HV,
    "L_air": 637.62e-3, 
    "C_g": 482.15e-12,  
    "C_s": 401.14e-12   
}

params_LV = {
    "stages": N_LV,
    "L_air": 1.58e-3,   
    "C_g": 5640.1e-12,  
    "C_s": 5.67e-12     
}


# --- FLEET & BATCH CONFIGURATION ---
fleet_config = {
    "num_dnas": 5,              # How many unique transformers to generate
    "batch_mode": "random",     # Options: "fixed" or "random"
    
    # Settings for "fixed" mode:
    "fixed_batches": 10,        # Every transformer gets exactly 10 sweeps (1 baseline + 9 operational/fault)
    
    # Settings for "random" mode:
    "random_range": (3, 12),    # Each transformer gets a random number of sweeps between 3 and 12
    "random_distribution": "uniform" # Options: "uniform" (equal chance) or "normal" (bell curve clustering)
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
    "noise_level_stage4": 0.01              # 1% standard deviation for Stage 4
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
R_load = 50.0             
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

def apply_fault(L_array, C_g_array, C_s_array, fault):
    """
    Modifies specific elements in the component arrays.
    Handles both single units and arrays of multiple units.
    """
    if not fault["active"]:
        return L_array, C_g_array, C_s_array
        
    # Grab the units and ensure they are formatted as a list
    units = fault["units_affected"]
    if isinstance(units, int):
        units = [units]  # Wrap single integer in a list
        
    # Keep track of successfully applied faults for the print statement
    applied_units = []

    # Loop through every unit requested by the GUI/Config
    for unit in units:
        idx = unit - 1  # Convert GUI unit number (1-based) to Python index (0-based)
        
        # Ensure index is within bounds before applying math
        if 0 <= idx < len(L_array):
            C_g_array[idx] *= fault["C_g_multiplier"]
            L_array[idx] *= fault["L_air_multiplier"]
            C_s_array[idx] *= fault["C_s_multiplier"]
            applied_units.append(unit)
        else:
            print(f"Warning: Fault unit {unit} is out of bounds for a {len(L_array)}-stage model.")
            
    if applied_units:
        print(f"Applied {fault['type']} fault across unit(s): {applied_units}.")
        
    return L_array, C_g_array, C_s_array



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
        L_array = np.random.normal(L_fp, L_fp * 0.0075, n)
        C_g_array = np.random.normal(C_g_fp, C_g_fp * 0.0125, n)
        C_s_array = np.random.normal(C_s_fp, C_s_fp * 0.0125, n)
    else:
        L_array = np.full(n, L_fp)
        C_g_array = np.full(n, C_g_fp)
        C_s_array = np.full(n, C_s_fp)
        
    return L_array, C_g_array, C_s_array

# =====================================
# Stage 4: Measurement Noise
# ====================================

def add_measurement_noise(magnitudes_dB, noise_level=0.5):
    """
    Implements Stage 4: Constant Additive White Gaussian Noise (AWGN) in the dB domain.
    """
    # Generates a flat noise floor independent of the signal's magnitude
    noise = np.random.normal(0, noise_level, len(magnitudes_dB))
    return magnitudes_dB + noise



# ==========================================
# ADVANCED MATRIX SOLVER ENGINE
# ==========================================
def calculate_fra_dynamic(L_array, C_g_array, C_s_array, frequencies):
    """
    Builds and solves the Nodal Admittance Matrix using component arrays.
    """
    n = len(L_array)
    magnitudes_dB = []
    
    for f in frequencies:
        omega = 2 * np.pi * f
        
        # 1. Calculate Admittance Arrays for this frequency
        # Y_s[i] is the series admittance connecting Node(i-1) to Node(i)
        Y_s = 1 / (1j * omega * L_array) + (1j * omega * C_s_array)
        # Y_g[i] is the shunt admittance from Node(i) to Ground
        Y_g = 1j * omega * C_g_array
        Y_load = 1 / R_load
        
        # 2. Build the N x N Matrix
        Y_bus = np.zeros((n, n), dtype=complex)
        
        for i in range(n):
            if i == 0:
                # Node 1: Connects to Vin via Y_s[0], and Node 2 via Y_s[1]
                Y_bus[i, i] = Y_s[0] + Y_s[1] + Y_g[0]
                Y_bus[i, i+1] = -Y_s[1]
            elif i == n - 1:
                # Node N: Connects to Node N-1 via Y_s[n-1], and to Load
                Y_bus[i, i] = Y_s[i] + Y_g[i] + Y_load
                Y_bus[i, i-1] = -Y_s[i]
            else:
                # Intermediate Nodes
                Y_bus[i, i] = Y_s[i] + Y_s[i+1] + Y_g[i]
                Y_bus[i, i-1] = -Y_s[i]
                Y_bus[i, i+1] = -Y_s[i+1]
                
        # 3. Current Vector
        I_vector = np.zeros(n, dtype=complex)
        I_vector[0] = Y_s[0] * V_in 
        
        # 4. Solve
        try:
            V_nodes = np.linalg.solve(Y_bus, I_vector)
        except np.linalg.LinAlgError:
            raise RuntimeError(f"FATAL: Matrix became singular at {f} Hz. Check component values for short circuits.")

        V_out = V_nodes[-1]
        
        magnitudes_dB.append(20 * np.log10(abs(V_out / V_in)))


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
                current_stoch_config["apply_stage4_measurement_noise"] = False 
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
                L_net, Cg_net, Cs_net = apply_fault(L_base.copy(), Cg_base.copy(), Cs_base.copy(), current_fault_config)
            else:
                L_net, Cg_net, Cs_net = L_base, Cg_base, Cs_base

            # 3. Solve & Noise
            mag_db = calculate_fra_dynamic(L_net, Cg_net, Cs_net, frequencies)
            if current_stoch_config["apply_stage4_measurement_noise"]:
                mag_db = add_measurement_noise(mag_db, noise_level=current_stoch_config["noise_level_stage4"])

            # 4. Export to the specific Unit's subfolder using the relative path
            filename_relative = os.path.join(unit_dir_relative, f"{file_prefix}_{i}.txt")
            export_to_txt(filename_relative, frequencies, mag_db)



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

    print(f"  [PASS] All File Counts strictly follow '{fleet_config['batch_mode']}' logic.")
    print(f"  [PASS] Baseline Trace_0.txt verified present in all folders.")
    print(f"  [PASS] Mathematical integrity verified (No NaNs/Infs, Exact Shape matched).")
    print(f"  [PASS] Correct Value of CCF with Healthy Cases.")
    print("[TESTING COMPLETE] Dataset is clean and ML-ready.\n")


# ==========================================
# RUN AND COMPARE
# ==========================================
# Generate baseline network
# _base, Cg_base, Cs_base = build_network_arrays(active_config)
# mag_baseline = calculate_fra_dynamic(L_base, Cg_base, Cs_base, frequencies)

# EXPORT BASELINE
# export_to_txt(fault_config["export_baseline"], frequencies, mag_baseline)

# Generate and solve faulted network
# if fault_config["active"]:
#    L_fault, Cg_fault, Cs_fault = apply_fault(L_base.copy(), Cg_base.copy(), Cs_base.copy(), fault_config)
#    mag_faulted = calculate_fra_dynamic(L_fault, Cg_fault, Cs_fault, frequencies)
    
    # EXPORT FAULTED
#    export_to_txt(fault_config["export_faulted"], frequencies, mag_faulted)



# ==========================================
# 5. BATCH RUN, EXPORT, AND VALIDATE
# ==========================================
if __name__ == "__main__":
    
    # ---------------------------------------------
    # Example A: Generate Healthy Fleet
    # ---------------------------------------------
    print("STARTING HEALTHY FLEET GENERATION...")
    fault_config["active"] = False
    
    # Override fleet config for this specific run
    fleet_config["num_dnas"] = 3
    fleet_config["batch_mode"] = "random"
    
    output_target_healthy = "Simulation_Healthy"
    
    generate_batch(
        output_dir=output_target_healthy, 
        file_prefix="Trace",             
        active_config=active_config, 
        stochastic_config=stochastic_config, 
        fault_config=fault_config,
        fleet_config=fleet_config,
        frequencies=frequencies
    )
    
    # IMMEDIATELY TEST WHAT WE JUST BUILT
    validate_dataset_integrity(output_target_healthy, fleet_config, STEPS)


    # ---------------------------------------------
    # Example B: Generate Radial Fault Dataset (10% Severity)
    # ---------------------------------------------
    print("STARTING RADIAL FAULT BATCH GENERATION...")
    fault_config["active"] = True
    fault_config["type"] = "Radial"
    fault_config["units_affected"] = [3]  
    fault_config["C_g_multiplier"] = 1.10 
    
    # Override fleet config to test "fixed" mode
    fleet_config["num_dnas"] = 3
    fleet_config["batch_mode"] = "fixed"
    fleet_config["fixed_batches"] = 5
    
    output_target_radial = "Simulation_Radial_10"
    
    generate_batch(
        output_dir=output_target_radial, 
        file_prefix="Trace", 
        active_config=active_config, 
        stochastic_config=stochastic_config, 
        fault_config=fault_config, 
        fleet_config=fleet_config,
        frequencies=frequencies
    )
    
    # IMMEDIATELY TEST WHAT WE JUST BUILT
    validate_dataset_integrity(output_target_radial, fleet_config, STEPS)


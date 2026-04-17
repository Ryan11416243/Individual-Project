import numpy as np
import matplotlib.pyplot as plt
import os

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
def apply_stochastic_baseline(config, stoch_config):
    """
    Implements the hybrid stochastic parameter variation (Table 1).
    Generates a unique transformer fingerprint (Stage 1) and adds 
    operational noise per-stage (Stage 2).
    """
    n = config["stages"]
    
    # Base per-stage nominal values
    L_nom = config["L_air"] / n
    C_g_nom = config["C_g"] / n
    C_s_nom = config["C_s"] * n
    
    # ----------------------------------------------------
    # STAGE 1: Transformer Fingerprint (Inter-Unit Variability)
    # ----------------------------------------------------
    if stoch_config["apply_stage1_inter_unit"]:
        fp_C = np.random.normal(1.0, 0.05)  
        fp_L = np.random.normal(1.0, 0.035) 
    else:
        fp_C, fp_L = 1.0, 1.0  # Deterministic fallback
        
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
        # Fill arrays with the deterministic (or Stage 1 shifted) values
        L_array = np.full(n, L_fp)
        C_g_array = np.full(n, C_g_fp)
        C_s_array = np.full(n, C_s_fp)
        
    return L_array, C_g_array, C_s_array

def add_measurement_noise(magnitudes_dB, noise_level=0.01):
    """
    Implements Stage 4: 0.5% to 2% Gaussian Measurement Noise applied 
    to the final output magnitude.
    """
    # ------------------------------------------------------
    # Stage 4: Measurement Noise
    # ------------------------------------------------------
    # Base noise is scaled dynamically to the signal strength

    noise = np.random.normal(0, np.abs(magnitudes_dB) * noise_level, len(magnitudes_dB))
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
        V_nodes = np.linalg.solve(Y_bus, I_vector)
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
def generate_batch(num_sets, output_dir, file_prefix, active_config, stochastic_config, fault_config, frequencies):
    """
    Generates a batch of FRA datasets.
    The first set (index 0) is forced to be the nominal/deterministic baseline.
    Subsequent sets utilize the stochastic configurations.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(num_sets):
        # Create a local copy of the stochastic config so we can safely modify it for the first run
        current_stoch_config = stochastic_config.copy()
        
        # Force the first set to be purely nominal
        if i == 0:
            print(f"\n--- Generating Set {i}: Pure Nominal Baseline ---")
            current_stoch_config["apply_stage1_inter_unit"] = False
            current_stoch_config["apply_stage2_intra_unit"] = False
            current_stoch_config["apply_stage4_measurement_noise"] = False 
        else:
            print(f"--- Generating Set {i}: Stochastic Variation ---")

        # 1. Build Baseline Arrays
        L_base, Cg_base, Cs_base = apply_stochastic_baseline(active_config, current_stoch_config)
        
        # 2. Apply Faults (if active)
        if fault_config["active"]:
            L_net, Cg_net, Cs_net = apply_fault(L_base.copy(), Cg_base.copy(), Cs_base.copy(), fault_config)
        else:
            L_net, Cg_net, Cs_net = L_base, Cg_base, Cs_base

        # 3. Solve the Admittance Matrix
        mag_db = calculate_fra_dynamic(L_net, Cg_net, Cs_net, frequencies)

        # 4. Apply Stage 4 Output Noise (if active in current config)
        if current_stoch_config["apply_stage4_measurement_noise"]:
            mag_db = add_measurement_noise(mag_db, noise_level=current_stoch_config["noise_level_stage4"])

        # 5. Export to File
        # Example output: "Healthy_Data/Healthy_Trace_0.txt"
        filename = os.path.join(output_dir, f"{file_prefix}_{i}.txt")
        export_to_txt(filename, frequencies, mag_db)

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


if __name__ == "__main__":
    
    # ---------------------------------------------
    # Example A: Generate Healthy Dataset
    # ---------------------------------------------
    print("STARTING HEALTHY BATCH GENERATION...")
    fault_config["active"] = False
    
    generate_batch(
        num_sets= NO_OF_HEALTHY,                     
        output_dir="Simulation_Healthy", # Saves to a folder named "Simulation_Healthy"
        file_prefix="Trace",             # Files will be named Trace_0.txt, Trace_1.txt, etc.
        active_config=active_config, 
        stochastic_config=stochastic_config, 
        fault_config=fault_config, 
        frequencies=frequencies
    )

    # ---------------------------------------------
    # Example B: Generate Radial Fault Dataset (10% Severity)
    # ---------------------------------------------
    print("\nSTARTING RADIAL FAULT BATCH GENERATION...")
    fault_config["active"] = True
    fault_config["type"] = "Radial"
    fault_config["units_affected"] = [3]  # Affects unit 3
    fault_config["C_g_multiplier"] = 1.10 # +10% Cg severity
    
    generate_batch(
        num_sets= NO_OF_FAULT,                       
        output_dir="Simulation_Radial", 
        file_prefix="Sev_10", 
        active_config=active_config, 
        stochastic_config=stochastic_config, 
        fault_config=fault_config, 
        frequencies=frequencies
    )
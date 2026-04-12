import numpy as np
import matplotlib.pyplot as plt
import re
import os

# ==========================================
# 1. PARSE LTSPICE TEXT DATA
# ==========================================
def load_ltspice_txt(filepath):
    """
    Parses LTSpice .txt export files with the format:
    Freq    (Magnitude_dB, Phase_deg)
    """
    ltspice_freq = []
    ltspice_mag = []
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}. Skipping LTSpice import.")
        return [], []
        
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            # Skip header lines or empty lines
            if line.startswith('Step') or line.startswith('Freq') or not line.strip():
                continue
            
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    # Part 0 is frequency: 1.00000000000000e+00
                    freq = float(parts[0])
                    
                    # Part 1 is the complex tuple: (-1.382887e-01dB,-5.598e-01)
                    # We use regex to extract the number between '(' and 'dB'
                    match = re.search(r'\(([^d]+)dB', parts[1])
                    if match:
                        mag = float(match.group(1))
                        ltspice_freq.append(freq)
                        ltspice_mag.append(mag)
                except ValueError:
                    continue # Skip lines that can't be parsed
                    
    return np.array(ltspice_freq), np.array(ltspice_mag)

# ==========================================
# 2. CONFIGURATION BLOCK
# ==========================================
N_HV = 10
N_LV = 16

params_LV = {
    "stages": N_LV,
    "L_air": 1.58e-3,    
    "C_g": 5640.1e-12,   
    "C_s": 5.67e-12      
}

active_config = params_LV  
R_load = 50.0              
V_in = 1.0                 

STEPS = 2000
# ==========================================
# 3. MATRIX SOLVER ENGINE
# ==========================================
def calculate_fra(config, frequencies):
    n = config["stages"]
    L_stage = config["L_air"] / n
    C_g_stage = config["C_g"] / n
    C_s_stage = config["C_s"] * n
    
    magnitudes_dB = []
    
    for f in frequencies:
        omega = 2 * np.pi * f
        Y_L = 1 / (1j * omega * L_stage)
        Y_Cs = 1j * omega * C_s_stage
        Y_series = Y_L + Y_Cs 
        
        Y_shunt = 1j * omega * C_g_stage
        Y_load = 1 / R_load
        
        Y_bus = np.zeros((n, n), dtype=complex)
        
        for i in range(n):
            if i == 0:
                Y_bus[i, i] = 2 * Y_series + Y_shunt
                Y_bus[i, i+1] = -Y_series
            elif i == n - 1:
                Y_bus[i, i] = Y_series + Y_shunt + Y_load
                Y_bus[i, i-1] = -Y_series
            else:
                Y_bus[i, i] = 2 * Y_series + Y_shunt
                Y_bus[i, i-1] = -Y_series
                Y_bus[i, i+1] = -Y_series
                
        I_vector = np.zeros(n, dtype=complex)
        I_vector[0] = Y_series * V_in 
        
        V_nodes = np.linalg.solve(Y_bus, I_vector)
        V_out = V_nodes[-1]
        
        transfer_function = abs(V_out / V_in)
        magnitudes_dB.append(20 * np.log10(transfer_function))
        
    return magnitudes_dB

# ==========================================
# 4. RUN SIMULATION & PLOT COMPARISON
# ==========================================
# Load the LTSpice Data First
lt_filepath = "Simulation_Py\ltspice_data.txt" 
lt_freq, lt_mag = load_ltspice_txt(lt_filepath)

# Match our Python simulation frequencies to the LTSpice sweep bounds
if len(lt_freq) > 0:
    freq_start = lt_freq[0]
    freq_end = lt_freq[-1]
else:
    freq_start = 10
    freq_end = 1e6

frequencies = np.logspace(np.log10(freq_start), np.log10(freq_end), STEPS)

print(f"Running Python matrix solver for {active_config['stages']} stages...")
py_magnitudes = calculate_fra(active_config, frequencies)

# Plot Both Datasets
plt.figure(figsize=(10, 6))

# Plot LTSpice Data (as a dashed red line or scatter points)
if len(lt_freq) > 0:
    plt.semilogx(lt_freq, lt_mag, color='red', linestyle='--', linewidth=2, label="LTSpice Export")

# Plot Python Data (as a solid blue line)
plt.semilogx(frequencies, py_magnitudes, color='blue', alpha=0.7, linewidth=2, label=f"Python Model ({active_config['stages']}-Stage)")

plt.title(f"FRA Simulation Comparison: Python vs. LTSpice")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.show()

# ==========================================
# Evaluation with Stat
# ==========================================


# Ensure arrays are numpy arrays for math operations
lt_mag_np = np.array(lt_mag)
py_mag_np = np.array(py_magnitudes)

# 1. Root Mean Square Error (RMSE)
rmse = np.sqrt(np.mean((lt_mag_np - py_mag_np)**2))

# 2. Maximum Absolute Error
max_error = np.max(np.abs(lt_mag_np - py_mag_np))

# 3. Cross-Correlation Coefficient (CCF)
mean_lt = np.mean(lt_mag_np)
mean_py = np.mean(py_mag_np)
numerator = np.sum((lt_mag_np - mean_lt) * (py_mag_np - mean_py))
denominator = np.sqrt(np.sum((lt_mag_np - mean_lt)**2) * np.sum((py_mag_np - mean_py)**2))
ccf = numerator / denominator

print(f"--- Evaluation Metrics ---")
print(f"RMSE: {rmse:.4f} dB")
print(f"Max Absolute Error: {max_error:.4f} dB")
print(f"Cross-Correlation (CCF): {ccf:.6f}")